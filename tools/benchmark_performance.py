"""Repeatable, isolated PolyBot timings; never writes an existing model or starts live training.

Run the same command before/after. CUDA timings synchronize at measurement
boundaries, include required diagnostic readbacks, and exclude simulator waits.
Results are synthetic workloads, not live lap/FPS or training-stability claims.
"""

from __future__ import annotations

import argparse
import ast
import copy
import cProfile
import functools
import gc
import hashlib
import io
import json
import os
import platform
import pstats
import socket
import subprocess
import tempfile
import textwrap
import time
import tracemalloc
from pathlib import Path
from threading import Thread
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.control.actions import ControlDemand
from polybot.environment.env import PolyTrackEnv
from polybot.environment.rewards import RewardContext, calculate_reward
from polybot.mock import MockSimulatorTransport
from polybot.protocol import Transition, request_message
from polybot.replay_swarm import interpolate_replay
from polybot.training.config import TrainingConfig
from polybot.training.replay_storage import load_replay_buffer, save_replay_buffer
from polybot.training.visual_replays import ReplaySample


def production_config() -> TrainingConfig:
    path = Path("profiles/training/simple-track-grtqc-contact-aware-30-uncertainty5e-1-explore5e-3.json")
    config = TrainingConfig.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
    config.backend = "mock"
    return config


def measure(operation: Any, count: int, synchronize: Any = lambda: None) -> dict[str, float]:
    for _ in range(3):
        operation()
    synchronize()
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    timings = []
    for _ in range(count):
        synchronize()
        started = time.perf_counter()
        operation()
        synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    wall = time.perf_counter() - wall_start
    return {
        "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)), "calls_per_s": count / wall,
        "process_cpu_percent_one_core": 100 * (time.process_time() - cpu_start) / wall,
    }


def core_benchmarks(count: int) -> dict[str, Any]:
    transport = MockSimulatorTransport()
    config = production_config()
    env = PolyTrackEnv(transport, frame_skip=30, action_adapter=GRTQCBackend().action_adapter(config),
                       expose_training_state=True)
    obs, _ = env.reset(seed=17)
    telemetry = env.latest_telemetry
    assert telemetry is not None
    result = transport.request(request_message(500, "reset", {"seed": 17, "track_id": "mock/straight"}))["result"]
    encoded = json.dumps(result)
    action = np.array([0., 1.], dtype=np.float32)
    transition = Transition.from_wire(result, lookahead_count=12)
    reward_context = RewardContext(
        transition=transition, action=ControlDemand(0., 1., 0.), previous_control=ControlDemand(0., 1., 0.),
        config=config.rewards, fixed_dt_s=1 / 120, previous_progress_m=0., highest_progress_m=0.,
        episode_start_progress_m=0.,
    )

    def step() -> None:
        nonlocal obs
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=17)

    # The direct mock request reset its episode; reset the owning env too.
    env.reset(seed=17)
    try:
        stats = {
            "observation": measure(telemetry.to_vector, count),
            "protocol_parse": measure(lambda: Transition.from_wire(result, lookahead_count=12), count),
            "reward": measure(lambda: calculate_reward(reward_context), count),
            "json_decode": measure(lambda: json.loads(encoded), count),
            "json_encode": measure(lambda: json.dumps(result, allow_nan=False, separators=(",", ":")), count),
            "environment_step": measure(step, count),
        }
        tracemalloc.start()
        for _ in range(500):
            step()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats["environment_python_memory_bytes"] = {"retained": current, "peak": peak}
        samples = [ReplaySample(i, i * .03, (i * .1, 0., 0.), (0., 0., 0., 1.)) for i in range(1200)]
        stats["replay_interpolation_1200_samples"] = measure(lambda: interpolate_replay(samples, 18.123), count)
        profiler = cProfile.Profile()
        profiler.runcall(lambda: [step() for _ in range(100)])
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats("cumtime").print_stats(15)
        stats["environment_profile"] = out.getvalue()
        return stats
    finally:
        env.close()


def learner_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    config = production_config()
    backend = GRTQCBackend()
    env = PolyTrackEnv(
        MockSimulatorTransport(), frame_skip=config.frame_skip,
        action_adapter=backend.action_adapter(config), expose_training_state=True,
    )
    model = backend.create_model(config, env, args.device)
    model.set_logger(configure(None, []))
    model.actor_unlocked = True
    replay = model.replay_buffer
    assert replay is not None
    rng = np.random.default_rng(17)
    # Production capacity, observation width, batch size, n-step horizon and
    # network. Synthetic bounded state/action values populate all slots.
    for array in (replay.observations, replay.next_observations):
        array[:] = rng.uniform(-1, 1, array.shape).astype(np.float32)
    replay.actions[:] = rng.uniform(-1, 1, replay.actions.shape)
    replay.rewards[:] = rng.normal(size=replay.rewards.shape)
    replay.dones[:] = 0
    replay.dones[1110::1111] = 1
    replay.full, replay.pos = True, 0
    model.set_actor_reference_observations(replay.observations[:1111, 0].copy())
    obs = replay.observations[100, 0].copy()
    sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result: dict[str, Any] = {"batch_size": config.grtqc.batch_size, "capacity": replay.buffer_size,
                              "n_step": config.grtqc.n_step_return, "observation_shape": list(obs.shape)}
    try:
        if args.compare_revision:
            result["paired"] = paired_learner(model, args, sync)
            model.actor_step_action_limit = config.grtqc.actor_step_action_limit
            if args.memory_updates:
                result["memory"] = learner_memory(model, args.memory_updates, sync)
            return result
        result["inference"] = measure(lambda: model.predict(obs, deterministic=True), args.count, sync)
        result["replay_sample"] = measure(lambda: replay.sample(config.grtqc.batch_size), args.count, sync)
        result["replay_insert"] = measure(lambda: replay.add(
            obs, obs, np.array([0., 1.], dtype=np.float32), np.array([1.], dtype=np.float32),
            np.array([False]), [{}],
        ), args.count)
        model._actor_evaluation_hold = True
        result["critic_update"] = measure(lambda: model.train(1, config.grtqc.batch_size), args.updates, sync)
        model._actor_evaluation_hold = False

        def actor_due() -> None:
            model._n_updates = model.actor_update_interval - 1
            model.train(1, config.grtqc.batch_size)

        result["actor_due_full_update"] = measure(actor_due, args.updates, sync)
        # Exercise the same production bisection with a deliberately binding
        # diagnostic cap, so this expensive branch cannot disappear by chance.
        model.actor_step_action_limit = 1e-7
        result["capped_actor_due_full_update"] = measure(actor_due, args.updates, sync)
        model.actor_step_action_limit = config.grtqc.actor_step_action_limit
        profiler = cProfile.Profile()
        sync()
        profiler.runcall(lambda: [actor_due() for _ in range(5)])
        sync()
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats("cumtime").print_stats(22)
        result["learner_profile"] = out.getvalue()
        if args.io:
            with tempfile.TemporaryDirectory(prefix="polybot-perf-") as directory:
                checkpoint = Path(directory) / "policy.zip"
                replay_file = Path(directory) / "replay.pkl"
                result["checkpoint_save"] = measure(lambda: model.save(checkpoint), 3, sync)
                result["checkpoint_load"] = measure(lambda: backend.load_model(checkpoint, None, args.device), 3, sync)
                result["replay_save"] = measure(lambda: save_replay_buffer(model, replay_file), 3, sync)
                result["replay_bytes"] = replay_file.stat().st_size
                result["replay_load"] = measure(lambda: load_replay_buffer(model, replay_file), 3, sync)
        if args.device == "cuda":
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        if args.memory_updates:
            result["memory"] = learner_memory(model, args.memory_updates, sync)
        return result
    finally:
        env.close()


def paired_learner(model: Any, args: argparse.Namespace, sync: Any) -> dict[str, Any]:
    """Interleave old/new updates under identical RNG, including the binding cap.

    Only the old train method is compiled from the requested repository revision;
    all math dependencies are the installed versions shared by both paths.
    """
    import polybot.algorithms.grtqc as module

    source = subprocess.check_output(
        ["git", "show", f"{args.compare_revision}:src/polybot/algorithms/grtqc.py"], text=True,
    )
    tree = ast.parse(source)
    node = next(m for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "GRTQC"
                for m in c.body if isinstance(m, ast.FunctionDef) and m.name == "train")
    function_source = textwrap.dedent("\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]))
    namespace = dict(vars(module))
    exec(compile(function_source, "<baseline-train>", "exec"), namespace)
    baseline = copy.deepcopy(model)
    report = {}
    for mode in ("critic", "actor", "capped_actor"):
        for candidate in (baseline, model):
            candidate._actor_evaluation_hold = mode == "critic"
            candidate.actor_step_action_limit = 1e-7 if mode == "capped_actor" else .0075
        timings: dict[str, list[float]] = {"baseline": [], "optimized": []}
        for iteration in range(args.updates + 3):
            order = ("baseline", "optimized") if iteration % 2 == 0 else ("optimized", "baseline")
            for name in order:
                candidate = baseline if name == "baseline" else model
                candidate._n_updates = candidate.actor_update_interval - 1
                np.random.seed(1000 + iteration)
                torch.manual_seed(1000 + iteration)
                sync()
                start = time.perf_counter()
                if name == "baseline":
                    namespace["train"](candidate, 1, model.batch_size)
                else:
                    candidate.train(1, model.batch_size)
                sync()
                if iteration >= 3:
                    timings[name].append((time.perf_counter() - start) * 1000)
        for key, value in baseline.policy.state_dict().items():
            if not torch.equal(value, model.policy.state_dict()[key]):
                raise RuntimeError(f"production-shaped differential update drift in {mode}: {key}")
        report[mode] = {name: {"median_ms": float(np.median(values)),
                              "p95_ms": float(np.percentile(values, 95)),
                              "p99_ms": float(np.percentile(values, 99))}
                        for name, values in timings.items()}
        report[mode]["policy_parameters_bitwise_equal"] = True
    if args.profile_operators:
        counts = {}
        for name, candidate in (("baseline", baseline), ("optimized", model)):
            candidate._actor_evaluation_hold = True
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                if name == "baseline":
                    namespace["train"](candidate, 1, model.batch_size)
                else:
                    candidate.train(1, model.batch_size)
                sync()
            counts[name] = {event.key: event.count for event in profiler.key_averages()
                            if event.key in ("aten::item", "aten::_local_scalar_dense", "aten::_to_copy")}
        report["critic_operator_counts"] = counts
    return report


def learner_memory(model: Any, updates: int, sync: Any) -> dict[str, Any]:
    """Steady-state growth after optimizer/reference warmup, without simulator history."""
    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None
    gc.collect()
    sync()
    tracemalloc.start()
    points = []
    for phase in range(3):
        if phase:
            for _ in range(updates):
                model.train(1, model.batch_size)
        gc.collect()
        sync()
        current, peak = tracemalloc.get_traced_memory()
        points.append({"updates": phase * updates, "python_retained_bytes": current, "python_peak_bytes": peak,
                       "rss_bytes": process.memory_info().rss if process else None,
                       "cuda_allocated_bytes": torch.cuda.memory_allocated() if model.device.type == "cuda" else None})
    tracemalloc.stop()
    return {"samples": points, "history_length": len(model._critic_loss_history),
            "replay_capacity": model.replay_buffer.buffer_size}


def transport_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    import polybot.transport as module

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server_factory = module.serve
    if args.compression != "auto":
        server_factory = functools.partial(
            server_factory, compression=None if args.compression == "none" else "deflate"
        )
    with patch.object(module, "serve", server_factory):
        transport = module.WebSocketServerTransport(port=port, connect_timeout_s=5)
        transport.start()

    def simulator() -> None:
        mock = MockSimulatorTransport()
        try:
            with connect(transport.endpoint) as connection:
                for raw in connection:
                    connection.send(json.dumps(mock.request(json.loads(raw))))
        except ConnectionClosed:
            pass

    thread = Thread(target=simulator, daemon=True)
    thread.start()
    config = production_config()
    backend = GRTQCBackend()
    env = PolyTrackEnv(transport, frame_skip=30, action_adapter=backend.action_adapter(config),
                       expose_training_state=True)
    model = backend.create_model(config, env, args.device)
    obs, _ = env.reset(seed=17)

    def control() -> None:
        nonlocal obs
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=17)

    try:
        return {"compression": args.compression,
                "hello_request_response": measure(lambda: transport.request(request_message(
                    999, "hello", {"protocol": "polybot.sim", "protocol_version": 2, "lookahead_count": 12},
                )), args.count),
                "telemetry_policy_control_roundtrip": measure(control, args.count)}
    finally:
        env.close()
        thread.join(timeout=3)


def gui_benchmarks(count: int) -> dict[str, Any]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from polybot.gui.log_viewer import LiveLogWindow

    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="polybot-log-perf-") as directory:
        path = Path(directory) / "run.jsonl"
        event = {"type": "episode", "episode": 1, "timesteps": 100, "progress": 1,
                 "elapsed_s": 33.324, "reward": 123, "events": ["finish"], "at": "2026-10-17T00:00:00"}
        line = json.dumps(event) + "\n"
        path.write_text(line * 5000, encoding="utf-8")
        window = LiveLogWindow(path)
        window.timer.stop()

        slices = []

        def backlog() -> None:
            window.position = 0
            window.events.clear()
            while window.position < path.stat().st_size:
                start = time.perf_counter()
                window.refresh()
                slices.append((time.perf_counter() - start) * 1000)

        result = {"log_backlog_drain_5000": measure(backlog, min(count, 10)),
                  "refresh_slice_median_ms": float(np.median(slices)),
                  "refresh_slice_p95_ms": float(np.percentile(slices, 95)),
                  "refresh_slice_max_ms": max(slices)}
        window.close()
        app.processEvents()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", choices=("core", "learner", "gui", "transport"),
                        default=["core", "learner", "gui"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--io", action="store_true", help="include full production-capacity replay save/load")
    parser.add_argument("--compare-revision", help="interleave original/optimized train methods with bitwise checks")
    parser.add_argument("--compression", choices=("auto", "none", "deflate"), default="auto")
    parser.add_argument("--memory-updates", type=int, default=0,
                        help="updates in each of two steady-state memory phases")
    parser.add_argument("--profile-operators", action="store_true",
                        help="count paired critic scalar-readback operators")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.count < 1 or args.updates < 1 or args.threads < 1 or args.memory_updates < 0:
        parser.error("counts, updates and threads must be positive; memory-updates must be nonnegative")
    torch.set_num_threads(args.threads)
    np.random.seed(17)
    torch.manual_seed(17)
    report: dict[str, Any] = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": args.device, "threads": args.threads,
        "benchmark_options": {key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "note": "Warm synthetic workload; CUDA synchronized at each measurement boundary. Other apps may use GPU.",
    }
    import polybot.algorithms.grtqc as learner_module
    report["learner_source_sha256"] = hashlib.sha256(Path(learner_module.__file__).read_bytes()).hexdigest()
    report["learner_package_path"] = str(learner_module.__file__)
    report["comparison_revision"] = args.compare_revision
    for section in args.sections:
        print(f"Measuring {section}...", flush=True)
        report[section] = (core_benchmarks(args.count) if section == "core" else
                           learner_benchmarks(args) if section == "learner" else
                           transport_benchmarks(args) if section == "transport" else gui_benchmarks(args.count))
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
