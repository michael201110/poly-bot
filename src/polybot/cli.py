"""Command-line entry points for smoke testing, training, and driving."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from polybot.controller import CenterlineController
from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.protocol import ProtocolViolation
from polybot.transport import SimulatorTransport, WebSocketServerTransport


def _common_arguments(
    parser: argparse.ArgumentParser,
    *,
    track_default: str | None = "mock/gentle-s",
    frame_skip_default: int | None = 4,
    max_steps_default: int | None = 2_000,
) -> None:
    track_help = "simulator track identifier"
    frame_skip_help = "physics ticks held per action"
    if track_default is None:
        track_help += " (default: mock/gentle-s for mock, current for WebSocket)"
    if frame_skip_default is None:
        frame_skip_help += " (default: 4 for mock, 10 for WebSocket)"
    max_steps_help = "Gymnasium episode limit"
    if max_steps_default is None:
        max_steps_help += " (default: 2000 for mock, 30000 for WebSocket)"
    parser.add_argument("--track", default=track_default, help=track_help)
    parser.add_argument("--seed", type=int, default=0, help="episode/trainer seed")
    parser.add_argument("--lookahead", type=int, default=12, help="future track sample count")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=frame_skip_default,
        help=frame_skip_help,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=max_steps_default,
        help=max_steps_help,
    )


def _websocket_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(host="127.0.0.1", port=8765)
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for the PolyTrack mod to connect",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for each simulator response",
    )


def _apply_backend_defaults(args: argparse.Namespace) -> None:
    if args.track is None:
        args.track = "current" if args.backend == "websocket" else "mock/gentle-s"
    if args.frame_skip is None:
        args.frame_skip = 10 if args.backend == "websocket" else 4
    if args.max_steps is None:
        args.max_steps = 30_000 if args.backend == "websocket" else 2_000


def _validate_websocket_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.host != "127.0.0.1" or args.port != 8765:
        parser.error("the PolyTrack mod currently requires --host 127.0.0.1 --port 8765")


def _make_transport(args: argparse.Namespace) -> SimulatorTransport:
    if getattr(args, "backend", "websocket") == "mock":
        return MockSimulatorTransport()

    transport = WebSocketServerTransport(
        args.host,
        args.port,
        connect_timeout_s=args.connect_timeout,
        request_timeout_s=args.request_timeout,
    )
    print(f"Waiting for the local PolyTrack mod at {transport.endpoint} ...", flush=True)
    return transport


def _episode_summary(
    episode: int,
    seed: int,
    info: dict[str, object],
    total_reward: float,
) -> dict[str, object]:
    return {
        "episode": episode,
        "seed": seed,
        "finished": "finish" in info["events"],
        "crashed": "crash" in info["events"],
        "elapsed_s": round(float(info["elapsed_s"]), 3),
        "progress_m": round(float(info["route_progress_m"]), 3),
        "reward": round(total_reward, 3),
    }


def _reset_drive_when_ready(
    env: PolyTrackEnv,
    *,
    seed: int,
    timeout_s: float,
) -> tuple[object, dict[str, object]]:
    """Wait through menu/reference states instead of terminating the driver."""

    deadline = time.monotonic() + timeout_s
    waiting_for: str | None = None
    while True:
        try:
            return env.reset(seed=seed)
        except ProtocolViolation as exc:
            message = str(exc)
            if not message.startswith(("game_not_ready:", "missing_reference:")):
                raise
            if time.monotonic() >= deadline:
                raise
            if message != waiting_for:
                print(
                    f"Waiting for PolyTrack: {message.split(':', 1)[1].strip()}",
                    flush=True,
                )
                waiting_for = message
            time.sleep(0.5)


def _terminal_restart_requested() -> bool:
    """Return true when R is waiting in an interactive Windows terminal."""

    if sys.platform != "win32":
        return False
    import msvcrt

    restart = False
    while msvcrt.kbhit():
        restart = msvcrt.getwch().lower() == "r" or restart
    return restart


def _bias_initial_policy_forward(model: object, strength: float) -> None:
    """Bias PPO's initial MultiDiscrete logits toward straight throttle."""

    if strength == 0:
        return
    import torch

    action_net = getattr(getattr(model, "policy", None), "action_net", None)
    bias = getattr(action_net, "bias", None)
    if bias is None or bias.numel() != 7:
        raise RuntimeError("forward bias requires a MultiDiscrete([3, 2, 2]) PPO policy")
    with torch.no_grad():
        # Logit layout: steer[-1,0,1], throttle[0,1], brake[0,1].
        bias[1] += strength * 0.5
        bias[4] += strength
        bias[5] += strength


def smoke_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the baseline against the mock simulator")
    _common_arguments(parser)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    env = PolyTrackEnv(
        MockSimulatorTransport(),
        track_id=args.track,
        lookahead_count=args.lookahead,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps,
    )
    controller = CenterlineController()
    summaries: list[dict[str, object]] = []
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                telemetry = env.latest_telemetry
                assert telemetry is not None
                _, reward, terminated, truncated, info = env.step(
                    controller.policy_action(telemetry)
                )
                total_reward += reward
            summaries.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "finished": "finish" in info["events"],
                    "crashed": "crash" in info["events"],
                    "steps": info["tick"] // args.frame_skip,
                    "elapsed_s": round(float(info["elapsed_s"]), 3),
                    "progress_m": round(float(info["route_progress_m"]), 3),
                    "reward": round(total_reward, 3),
                }
            )
    finally:
        env.close()

    print(json.dumps(summaries, indent=2))
    return 0 if all(bool(summary["finished"]) for summary in summaries) else 1


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO against a PolyBot simulator")
    _common_arguments(
        parser,
        track_default=None,
        frame_skip_default=None,
        max_steps_default=None,
    )
    parser.add_argument("--backend", choices=("mock", "websocket"), default="mock")
    _websocket_arguments(parser)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate; use a smaller value for late-stage fine-tuning",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.0,
        help="PPO entropy incentive for exploration; small values such as 0.005 work best",
    )
    parser.add_argument(
        "--forward-bias",
        type=float,
        default=1.5,
        help="initial PPO logit bias toward straight throttle and no brake; 0 disables it",
    )
    parser.add_argument("--model-out", type=Path, default=Path("models/polybot-ppo"))
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume training from an existing PPO model",
    )
    parser.add_argument(
        "--checkpoint-episodes",
        type=int,
        default=5,
        help="save the latest model plus an archive after this many episode restarts; 0 disables",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=None,
        help="optional TensorBoard output directory (requires tensorboard)",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="show Stable-Baselines progress UI (requires tqdm and rich)",
    )
    args = parser.parse_args(argv)
    _apply_backend_defaults(args)

    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.entropy_coef < 0:
        parser.error("--entropy-coef must be non-negative")
    if args.forward_bias < 0:
        parser.error("--forward-bias must be non-negative")
    if args.checkpoint_episodes < 0:
        parser.error("--checkpoint-episodes must be non-negative")
    _validate_websocket_arguments(parser, args)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        parser.error("training dependencies are missing; install with: pip install -e '.[train]'")
        raise AssertionError("unreachable") from exc

    transport = _make_transport(args)

    try:
        env = PolyTrackEnv(
            transport,
            track_id=args.track,
            lookahead_count=args.lookahead,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_steps,
            request_timeout_s=args.request_timeout,
        )
    except BaseException:
        transport.close()
        raise
    try:
        tensorboard_log = (
            str(args.tensorboard_log) if args.tensorboard_log is not None else None
        )
        if args.resume is None:
            model = PPO(
                "MlpPolicy",
                env,
                seed=args.seed,
                verbose=1,
                tensorboard_log=tensorboard_log,
                learning_rate=args.learning_rate,
                ent_coef=args.entropy_coef,
            )
            _bias_initial_policy_forward(model, args.forward_bias)
        else:
            model = PPO.load(
                str(args.resume),
                env=env,
                tensorboard_log=tensorboard_log,
                custom_objects={
                    "learning_rate": args.learning_rate,
                    "ent_coef": args.entropy_coef,
                },
            )

        class EpisodeCheckpointCallback(BaseCallback):
            def __init__(self, output: Path, every: int) -> None:
                super().__init__()
                self.output = output.with_suffix("")
                self.best_output = self.output.with_name(f"{self.output.name}-best")
                self.best_metadata = self.output.with_name(f"{self.output.name}-best.json")
                self.every = every
                self.episodes = 0
                self.next_checkpoint = every
                self.best_progress_ratio = 0.0
                self.clean_episode = True
                if self.best_metadata.exists():
                    try:
                        metadata = json.loads(self.best_metadata.read_text(encoding="utf-8"))
                        self.best_progress_ratio = float(metadata["progress_ratio"])
                    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                        pass

            def _on_step(self) -> bool:
                dones = self.locals.get("dones", ())
                self.episodes += sum(bool(done) for done in dones)
                for info, done in zip(self.locals.get("infos", ()), dones, strict=False):
                    reward_terms = info.get("reward_terms", {})
                    if reward_terms.get("barrier_launch", 0.0) < 0.0:
                        self.clean_episode = False
                    progress_m = float(info.get("route_progress_m", 0.0))
                    track_length_m = max(1.0, float(info.get("track_length_m", 1.0)))
                    progress_ratio = progress_m / track_length_m
                    is_finish = "finish" in info.get("events", ())
                    if (
                        self.clean_episode
                        and (
                            progress_ratio >= self.best_progress_ratio + 0.02
                            or is_finish
                        )
                    ):
                        self.output.parent.mkdir(parents=True, exist_ok=True)
                        self.model.save(str(self.best_output))
                        self.best_progress_ratio = max(
                            self.best_progress_ratio, progress_ratio
                        )
                        self.best_metadata.write_text(
                            json.dumps(
                                {
                                    "progress_m": progress_m,
                                    "track_length_m": track_length_m,
                                    "progress_ratio": self.best_progress_ratio,
                                    "finished": is_finish,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        print(
                            f"Saved best model at {progress_ratio:.1%} track progress.",
                            flush=True,
                        )
                    if done:
                        self.clean_episode = True
                if self.every and self.episodes >= self.next_checkpoint:
                    self.output.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save(str(self.output))
                    archive = self.output.with_name(
                        f"{self.output.name}-episode-{self.episodes}"
                    )
                    self.model.save(str(archive))
                    print(f"Checkpointed model after {self.episodes} episodes.", flush=True)
                    while self.next_checkpoint <= self.episodes:
                        self.next_checkpoint += self.every
                return True

        callback = EpisodeCheckpointCallback(args.model_out, args.checkpoint_episodes)
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                progress_bar=args.progress_bar,
                reset_num_timesteps=args.resume is None,
            )
        except KeyboardInterrupt:
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(args.model_out.with_suffix("")))
            print(f"Interrupted; saved model to {args.model_out.with_suffix('')}.zip")
            return 130
        except BaseException:
            emergency = args.model_out.with_suffix("").with_name(
                f"{args.model_out.with_suffix('').name}-emergency"
            )
            emergency.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(emergency))
            print(f"Training failed; saved emergency model to {emergency}.zip")
            raise
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        output = args.model_out.with_suffix("")
        model.save(str(output))
        print(f"Saved model to {output}.zip")
    finally:
        env.close()
    return 0


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO policy")
    _common_arguments(
        parser,
        track_default=None,
        frame_skip_default=None,
        max_steps_default=None,
    )
    parser.add_argument("model", type=Path, help="Stable-Baselines PPO model (.zip is optional)")
    parser.add_argument("--backend", choices=("mock", "websocket"), default="mock")
    _websocket_arguments(parser)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args(argv)
    _apply_backend_defaults(args)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    _validate_websocket_arguments(parser, args)
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        parser.error("training dependencies are missing; install with: pip install -e '.[train]'")
        raise AssertionError("unreachable") from exc

    transport = _make_transport(args)
    try:
        env = PolyTrackEnv(
            transport,
            track_id=args.track,
            lookahead_count=args.lookahead,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_steps,
            request_timeout_s=args.request_timeout,
        )
    except BaseException:
        transport.close()
        raise
    summaries: list[dict[str, object]] = []
    try:
        model = PPO.load(str(args.model), env=env)
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            info: dict[str, object] = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            summaries.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "finished": "finish" in info["events"],
                    "crashed": "crash" in info["events"],
                    "elapsed_s": round(float(info["elapsed_s"]), 3),
                    "progress_m": round(float(info["route_progress_m"]), 3),
                    "reward": round(total_reward, 3),
                }
            )
    finally:
        env.close()
    print(json.dumps(summaries, indent=2))
    return 0 if all(bool(summary["finished"]) for summary in summaries) else 1


def drive_main(argv: Sequence[str] | None = None) -> int:
    """Drive the currently selected local PolyTrack race with a policy."""

    parser = argparse.ArgumentParser(
        description="Drive the current PolyTrack race through the local-only AI bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _common_arguments(
        parser,
        track_default="current",
        frame_skip_default=10,
        max_steps_default=30_000,
    )
    _websocket_arguments(parser)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--model",
        type=Path,
        help="saved Stable-Baselines PPO model; omit to use the centreline controller",
    )
    policy.add_argument(
        "--centerline",
        action="store_true",
        help="explicitly use the built-in centreline controller (the default)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="number of races to drive")
    parser.add_argument(
        "--steering-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="invert the baseline controller if the car steers away from the ghost line",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=18.0,
        help="maximum baseline-controller speed in metres per second",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample PPO actions instead of using deterministic predictions",
    )
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.stochastic and args.model is None:
        parser.error("--stochastic requires --model")
    if args.target_speed <= 0:
        parser.error("--target-speed must be positive")
    _validate_websocket_arguments(parser, args)

    ppo_type = None
    if args.model is not None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            parser.error("PPO dependencies are missing; install with: pip install -e '.[train]'")
            raise AssertionError("unreachable") from exc
        ppo_type = PPO

    transport = _make_transport(args)
    try:
        env = PolyTrackEnv(
            transport,
            track_id=args.track,
            lookahead_count=args.lookahead,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_steps,
            request_timeout_s=args.request_timeout,
        )
    except BaseException:
        transport.close()
        raise
    controller = CenterlineController(
        steering_sign=args.steering_sign,
        max_speed_mps=args.target_speed,
    )
    summaries: list[dict[str, object]] = []
    try:
        model = ppo_type.load(str(args.model), env=env) if ppo_type is not None else None
        print("Press R in this terminal to restart the current run.", flush=True)
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            while True:
                observation, _ = _reset_drive_when_ready(
                    env,
                    seed=episode_seed,
                    timeout_s=args.connect_timeout,
                )
                total_reward = 0.0
                terminated = truncated = False
                info: dict[str, object] = {}
                restart_reason: str | None = None
                stuck_since: float | None = None
                last_status = 0.0
                while not (terminated or truncated):
                    step_started = time.monotonic()
                    if _terminal_restart_requested():
                        restart_reason = "manual restart"
                        break
                    if model is not None:
                        action, _ = model.predict(
                            observation,
                            deterministic=not args.stochastic,
                        )
                    else:
                        telemetry = env.latest_telemetry
                        assert telemetry is not None
                        action = controller.policy_action(telemetry)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    if not (terminated or truncated):
                        telemetry = env.latest_telemetry
                        assert telemetry is not None
                        if (
                            abs(telemetry.lateral_offset_m)
                            > telemetry.track_half_width_m * 1.1
                        ):
                            restart_reason = "off track"
                            break
                        speed = abs(telemetry.local_velocity_mps[2])
                        if telemetry.elapsed_s > 3.0 and speed < 0.5:
                            stuck_since = stuck_since or time.monotonic()
                            if time.monotonic() - stuck_since >= 2.0:
                                restart_reason = "stuck"
                                break
                        else:
                            stuck_since = None
                        now = time.monotonic()
                        if now - last_status >= 1.0:
                            print(
                                "drive "
                                f"speed={speed:5.2f}m/s "
                                f"offset={telemetry.lateral_offset_m:+6.2f}m "
                                f"heading={telemetry.heading_error_rad:+6.2f}rad "
                                f"steer={int(action[0]):+d}",
                                flush=True,
                            )
                            last_status = now
                    # The worker is intentionally unthrottled for training. A
                    # visible drive should instead track simulation time.
                    remaining = args.frame_skip * 0.001 - (time.monotonic() - step_started)
                    if remaining > 0:
                        time.sleep(remaining)
                if restart_reason is not None:
                    print(f"Restarting run ({restart_reason})...", flush=True)
                    continue
                summaries.append(_episode_summary(episode, episode_seed, info, total_reward))
                break
    except ProtocolViolation as exc:
        print(f"PolyBot stopped: {exc}", file=sys.stderr)
        return 2
    finally:
        env.close()

    print(json.dumps(summaries, indent=2))
    return 0
