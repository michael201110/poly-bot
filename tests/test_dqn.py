from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gymnasium import spaces

from polybot.algorithms.dqn import PhaseExplorationSchedule
from polybot.algorithms.registry import backend_for
from polybot.control.native_digital import NativeDigitalActionAdapter
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import ModelRegistry
from polybot.protocol import Action
from polybot.training.config import CurriculumConfig, DQNConfig, EvaluationConfig, TrainingConfig
from polybot.training.dqn_brake_stage import SIX_TO_NINE, expand_no_brake_checkpoint
from polybot.training.runner import TrainingRunner


@pytest.mark.parametrize("index,expected", [
    (0, Action(0, False, False)),
    (1, Action(0, True, False)),
    (2, Action(0, False, True)),
    (3, Action(-1, False, False)),
    (4, Action(-1, True, False)),
    (5, Action(-1, False, True)),
    (6, Action(1, False, False)),
    (7, Action(1, True, False)),
    (8, Action(1, False, True)),
])
def test_all_native_digital_actions_are_exact_and_stateless(index: int, expected: Action) -> None:
    adapter = NativeDigitalActionAdapter()
    assert adapter.action_space == spaces.Discrete(9)
    assert adapter.schema == "digital-discrete-9-v2"
    assert adapter.sequence is False
    assert set(vars(adapter)) == {"action_space", "brake_enabled", "schema"}
    for _ in range(2):
        applied = adapter.apply(np.int64(index), 30)
        assert applied.ticks == [expected] * 30
        assert applied.demand.steer == expected.steer
        assert applied.demand.throttle == expected.throttle
        assert applied.demand.brake == expected.brake
        assert not (expected.throttle and expected.brake)
        adapter.reset()


@pytest.mark.parametrize("index,expected", [
    (0, Action(0, False, False)), (1, Action(0, True, False)),
    (2, Action(-1, False, False)), (3, Action(-1, True, False)),
    (4, Action(1, False, False)), (5, Action(1, True, False)),
])
def test_no_brake_dqn_actions(index: int, expected: Action) -> None:
    adapter = NativeDigitalActionAdapter(brake_enabled=False)
    assert adapter.action_space == spaces.Discrete(6)
    assert adapter.schema == "digital-discrete-6-no-brake-v2"
    assert adapter.apply(index, 30).ticks == [expected] * 30


def test_dqn_step_holds_left_throttle_for_all_thirty_ticks_without_pwm(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("DQN must not call a PWM scheduler")

    monkeypatch.setattr("polybot.control.pwm.PwmSteering.generate", fail)
    monkeypatch.setattr("polybot.control.pwm.ContinuousPwmControls.generate", fail)
    monkeypatch.setattr("polybot.control.actions.decode_pwm_level", fail)
    transport = MockSimulatorTransport()
    env = PolyTrackEnv(
        transport, track_id="mock/straight", frame_skip=30,
        action_adapter=NativeDigitalActionAdapter(),
    )
    try:
        env.reset(seed=4)
        _, _, _, _, info = env.step(4)
        assert info["ticks_advanced"] == 30
        commands = [entry for entry in transport.command_log if entry["op"] == "step"]
        assert [entry["params"]["ticks"] for entry in commands] == [16, 14]
        assert all(entry["params"]["action"] == Action(-1, True, False).to_wire()
                   for entry in commands)
        assert info["requested_control_duty"] == {"steer": -1.0, "throttle": 1.0, "brake": 0.0}
        assert info["applied_control_fraction"] == {"steer": -1.0, "throttle": 1.0, "brake": 0.0}
    finally:
        env.close()


@pytest.mark.parametrize("changes", [
    {"replay_capacity": 0}, {"learning_starts": -1}, {"batch_size": 0},
    {"gamma": 0}, {"train_frequency": 0}, {"gradient_steps": 0},
    {"target_update_interval": 0}, {"exploration_fraction": 1.1},
    {"exploration_initial_eps": -0.1},
    {"exploration_final_eps": 0.5, "exploration_initial_eps": 0.2},
    {"action_set": "unknown"},
])
def test_dqn_config_rejects_invalid_values(changes: dict) -> None:
    with pytest.raises(ValueError, match="DQN"):
        DQNConfig(**changes)


def test_no_brake_policy_and_replay_transfer_to_full_dqn(tmp_path) -> None:
    early = TrainingConfig(
        algorithm="dqn", device="cpu", timesteps=32,
        evaluation=EvaluationConfig(32, 1), checkpoint_interval=0,
        output_root=tmp_path / "early", log_root=tmp_path / "logs",
        dqn=DQNConfig(action_set="no_brake", architecture="yosh_2020",
                      learning_starts=8, batch_size=8, replay_capacity=128),
    )
    source = TrainingRunner(early).run()
    later = replace(
        early, timesteps=16, output_root=tmp_path / "later",
        evaluation=EvaluationConfig(16, 1),
        dqn=replace(early.dqn, action_set="full"),
    )
    env = PolyTrackEnv(MockSimulatorTransport(), track_id=later.track_id,
                       action_adapter=NativeDigitalActionAdapter())
    try:
        destination = expand_no_brake_checkpoint(source, later, env, "cpu")
        from sb3_contrib import QRDQN

        old = QRDQN.load(source / "policy.zip", device="cpu")
        new = QRDQN.load(destination / "policy.zip", device="cpu")
        assert new.action_space == spaces.Discrete(9)
        assert new.num_timesteps == old.num_timesteps
        for quantile in range(old.policy.quantile_net.n_quantiles):
            for old_row, new_row in enumerate(SIX_TO_NINE):
                np.testing.assert_array_equal(
                    old.policy.quantile_net.quantile_net[-1].weight[quantile * 6 + old_row]
                    .detach().numpy(),
                    new.policy.quantile_net.quantile_net[-1].weight[quantile * 9 + new_row]
                    .detach().numpy(),
                )
        new.load_replay_buffer(destination / "replay.pkl")
        assert new.replay_buffer.size() >= old.num_timesteps - 1
        assert set(new.replay_buffer.actions[:new.replay_buffer.size(), 0, 0]) <= set(SIX_TO_NINE)
    finally:
        env.close()
    resumed = TrainingRunner(later).run(resume=destination)
    assert resumed == destination
    assert QRDQN.load(resumed / "policy.zip", device="cpu").num_timesteps > old.num_timesteps


def test_qrdqn_phase_exploration_reheats_and_decays_per_phase(tmp_path) -> None:
    cfg = DQNConfig(exploration_initial_eps=1.0, exploration_final_eps=0.1,
                    exploration_fraction=.5, architecture="tiny")
    schedule = PhaseExplorationSchedule(cfg, 100)
    assert schedule.initial == 1.0
    assert schedule(1.0) == 1.0
    assert schedule.advance(1) == pytest.approx(.982)
    assert schedule.steps == 1
    schedule.advance(50)
    assert schedule.steps == 50
    assert schedule(1.0) == pytest.approx(.1)

    training = TrainingConfig(algorithm="dqn", dqn=cfg, timesteps=10,
                              output_root=tmp_path / "models", log_root=tmp_path / "logs")
    backend = backend_for("dqn")
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend.action_adapter(training))
    try:
        model = backend.create_model(training, env, "cpu")
        backend.begin_phase(model, training, 100)
        assert model.exploration_rate == 1.0
        backend.advance_phase(model, 5)
        assert model.exploration_rate == model.exploration_schedule(0.9)
        assert model.exploration_rate < 1.0
        backend.begin_phase(model, training, 100)
        assert model.exploration_rate == 1.0
        assert backend.metrics(model)["replay_size"] == 0
    finally:
        env.close()


def test_qrdqn_curriculum_reheats_per_phase_and_keeps_replay(tmp_path) -> None:
    base = TrainingConfig(
        algorithm="dqn", device="cpu", timesteps=60,
        curriculum=CurriculumConfig("q4-full"),
        evaluation=EvaluationConfig(10, 1), checkpoint_interval=0,
        output_root=tmp_path / "models", log_root=tmp_path / "logs",
        dqn=DQNConfig(architecture="tiny", learning_starts=8, batch_size=8,
                      replay_capacity=128, train_frequency=1,
                      exploration_fraction=1.0),
    )
    events: list[dict] = []
    latest = TrainingRunner(base, events.append).run()
    phases = [event for event in events if event["type"] == "phase"]
    summaries = [event for event in events if event["type"] == "phase_summary"]
    assert [event["initial_epsilon"] for event in phases] == [1.0, 1.0]
    assert phases[0]["spawn_ratio"] == pytest.approx(.70)
    assert phases[0]["start_ratio"] == pytest.approx(.75)
    assert summaries[0]["epsilon"] == pytest.approx(.05)
    assert summaries[0]["replay_size"] == phases[1]["replay_size"]
    assert summaries[0]["actions_seen"]
    evaluations = [event["timesteps"] for event in events if event["type"] == "evaluation"]
    assert evaluations == sorted(evaluations)
    assert evaluations[-1] == 60
    from sb3_contrib import QRDQN

    model = QRDQN.load(latest / "policy.zip", device="cpu")
    assert model.num_timesteps == 60
    assert summaries[-1]["replay_size"] >= summaries[0]["replay_size"]
    assert summaries[-1]["replay_size"] > 0


def test_qrdqn_resume_into_curriculum_reheats_without_resetting_steps_or_replay(tmp_path) -> None:
    base = TrainingConfig(
        algorithm="dqn", device="cpu", timesteps=12,
        evaluation=EvaluationConfig(12, 1), checkpoint_interval=0,
        output_root=tmp_path / "models", log_root=tmp_path / "logs",
        dqn=DQNConfig(architecture="tiny", learning_starts=4, batch_size=8,
                      replay_capacity=128, train_frequency=1),
    )
    latest = TrainingRunner(base).run()
    registry = ModelRegistry(base.output_root)
    before = registry.read_metadata(latest).training_timesteps
    replay_before = latest / "replay.pkl"
    assert replay_before.is_file()
    curriculum = replace(
        base, timesteps=20, evaluation=EvaluationConfig(10, 1),
        curriculum=CurriculumConfig("section", .25, .5),
    )
    events: list[dict] = []
    TrainingRunner(curriculum, events.append).run(resume=latest)
    phase = next(event for event in events if event["type"] == "phase")
    assert phase["initial_epsilon"] == 1.0
    assert phase["spawn_ratio"] == pytest.approx(.20)
    metadata = registry.read_metadata(latest)
    assert metadata.training_timesteps == before + 20
    assert (latest / "replay.pkl").stat().st_size >= replay_before.stat().st_size
