"""Differential checks against the frozen pre-optimization numerical paths."""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.environment.env import PolyTrackEnv
from polybot.gui.log_tail import read_event_batch
from polybot.mock import MockSimulatorTransport
from polybot.protocol import Action, Telemetry
from polybot.training.config import GRTQCConfig, TrainingConfig

_spec = importlib.util.spec_from_file_location(
    "performance_reference", Path(__file__).parent / "fixtures" / "performance_reference.py",
)
assert _spec is not None and _spec.loader is not None
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)


def test_recorded_lap_observations_are_bitwise_identical():
    payload = json.loads((Path(__file__).parent / "fixtures" / "performance-telemetry.json").read_text())
    for values in payload["samples"]:
        sample = Telemetry(**{**values, "previous_action": Action(**values["previous_action"]),
                              "expert_action": Action(**values["expert_action"])})
        np.testing.assert_array_equal(sample.to_vector().view(np.uint32),
                                      reference.observation_reference(sample).view(np.uint32))


@pytest.mark.parametrize("lookahead_count", [0, 1, 12, 64])
def test_observations_are_bitwise_identical(lookahead_count):
    env = PolyTrackEnv(MockSimulatorTransport())
    try:
        env.reset(seed=17)
        telemetry = env.latest_telemetry
        rng = np.random.default_rng(12)
        for _ in range(100):
            # Include clipping boundaries and signed zero as well as random
            # values; keep float64 divisions before casting to the policy.
            points = rng.uniform(-1000, 1000, (lookahead_count, 4))
            if lookahead_count:
                points[0] = (500., -250., -0., 1.)
            sample = replace(telemetry, lookahead=tuple(map(tuple, points)), lookahead_mask=(1,) * lookahead_count)
            expected = reference.observation_reference(sample)
            actual = sample.to_vector()
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    finally:
        env.close()


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable",
))])
@pytest.mark.parametrize("mode", ["critic_only", "warmup_unlock", "actor", "capped_actor", "controller"])
def test_grtqc_updates_optimizer_rng_and_diagnostics_are_identical(device, mode):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    models, envs = [], []
    try:
        for _ in range(2):
            config = TrainingConfig(algorithm="grtqc", grtqc=GRTQCConfig(
                architecture="tiny", replay_capacity=128, batch_size=8, n_step_return=16,
                training_origin="transfer" if mode == "controller" else "scratch",
                critic_controller_state=True, critic_environment_state=True,
                critic_raw_actions=True, actor_verified_state_sampling=True,
                actor_controller_state=mode == "controller", actor_update_interval=2,
                actor_learning_rate=1e-3, actor_step_action_limit=1e-7 if mode == "capped_actor" else .1,
                actor_reference_drift_limit=.5, critic_warmup_updates=100000,
                policy_std_limit=.01 if mode == "controller" else 0., actor_uncertainty_coefficient=.5,
            ))
            backend = GRTQCBackend()
            env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend.action_adapter(config),
                               expose_training_state=True)
            envs.append(env)
            model = backend.create_model(config, env, device)
            model.set_logger(configure(None, []))
            rng = np.random.default_rng(19)
            replay = model.replay_buffer
            replay.observations[:] = rng.normal(size=replay.observations.shape)
            replay.next_observations[:] = rng.normal(size=replay.next_observations.shape)
            replay.actions[:] = rng.uniform(-1, 1, replay.actions.shape)
            replay.rewards[:] = rng.normal(size=replay.rewards.shape)
            replay.dones[10::11] = 1
            replay.full, replay.pos = True, 0
            model.set_actor_reference_observations(replay.observations[:32, 0].copy())
            model.actor_unlocked = mode not in {"critic_only", "warmup_unlock"}
            if mode == "warmup_unlock":
                # Exercise the readiness transition inside the update, after
                # the just-computed critic diagnostics enter its histories.
                model.critic_warmup_updates = 1
                model.critic_readiness_window = 2
                model.critic_readiness_relative_change = 1e6
                model.critic_mc_initialization_updates = 0
                model._critic_loss_history.extend([1e6, 1e6])
                model._disagreement_history.extend([1e6, 1e6])
            models.append(model)
        states, forwards = [], []
        for index, model in enumerate(models):
            np.random.seed(31)
            torch.manual_seed(31)
            calls = []
            hook = model.actor.register_forward_hook(lambda *args, calls=calls: calls.append(1))
            if index == 0:
                reference.train_reference(model, 4, 8)
            else:
                model.train(4, 8)
            hook.remove()
            forwards.append(len(calls))
            states.append((np.random.get_state(), torch.get_rng_state(),
                           torch.cuda.get_rng_state() if device == "cuda" else None))
        old, new = models
        assert_state_equal(old.policy.state_dict(), new.policy.state_dict())
        for optimizer in ("actor", "critic"):
            assert_state_equal(getattr(old, optimizer).optimizer.state_dict(),
                               getattr(new, optimizer).optimizer.state_dict())
        assert_state_equal(old.ent_coef_optimizer.state_dict(), new.ent_coef_optimizer.state_dict())
        assert_state_equal(old._training_diagnostics, new._training_diagnostics)
        assert_state_equal(states[0], states[1])
        assert list(old._critic_loss_history) == list(new._critic_loss_history)
        assert list(old._disagreement_history) == list(new._disagreement_history)
        assert old.actor_unlocked == new.actor_unlocked
        if mode == "warmup_unlock":
            assert new.actor_unlocked
        if mode != "critic_only":
            assert forwards[1] < forwards[0]
    finally:
        for env in envs:
            env.close()
        torch.set_num_threads(previous_threads)


def test_log_batches_preserve_order_partial_utf8_and_invalid_line_retry(tmp_path):
    path = tmp_path / "log.jsonl"
    lines = [json.dumps({"episode": i, "label": "λ"}, ensure_ascii=False).encode() + b"\n" for i in range(600)]
    path.write_bytes(b"".join(lines) + b'{"label":"\xce')
    position, collected = 0, []
    for _ in range(3):
        events, position, limited = read_event_batch(path, position)
        collected.extend(events)
        assert len(events) <= 256
    assert [event["episode"] for event in collected] == list(range(600))
    assert position == sum(map(len, lines))
    assert not limited
    with path.open("ab") as stream:
        stream.write(b'\xbb"}\n')
    events, position, _ = read_event_batch(path, position)
    assert events == [{"label": "λ"}]
    with path.open("ab") as stream:
        stream.write(b'bad JSON\n')
    assert read_event_batch(path, position) == ([], position, False)


def test_log_byte_budget_allows_one_large_event(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps({"large": "x" * 1000}) + "\n", encoding="utf-8")
    events, position, limited = read_event_batch(path, 0, max_bytes=10)
    assert len(events) == 1 and limited
    assert position == path.stat().st_size


def test_gui_catches_up_in_bounded_batches_and_handles_truncation(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from polybot.gui.log_viewer import LiveLogWindow

    app = QApplication.instance() or QApplication([])
    path = tmp_path / "log.jsonl"
    event = {"type": "episode", "episode": 1, "timesteps": 100, "progress": 1,
             "elapsed_s": 33., "reward": 1, "events": ["finish"]}
    path.write_text((json.dumps(event) + "\n") * 1000, encoding="utf-8")
    window = LiveLogWindow(path)
    try:
        assert 0 < window.position < path.stat().st_size
        assert window.catch_up_timer.isActive()
        for _ in range(10):
            app.processEvents()
        assert window.position == path.stat().st_size
        assert window.events.document().blockCount() <= 400
        path.write_text(json.dumps({**event, "episode": 2}) + "\n", encoding="utf-8")
        window.refresh()
        assert "Episode 2" in window.events.toPlainText()
    finally:
        window.close()
