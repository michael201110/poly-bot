from __future__ import annotations

from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.algorithms.tqc import TQCBackend
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.protocol import ProtocolViolation
from polybot.training.config import EvaluationConfig, GRTQCConfig, TQCConfig, TrainingConfig
from polybot.training.evaluation import EvaluationResult
from polybot.training.runner import TrainingRunner


def _config(algorithm: str) -> TrainingConfig:
    settings = {
        "tqc": {"tqc": TQCConfig(architecture="tiny", learning_starts=8, batch_size=8)},
        "grtqc": {"grtqc": GRTQCConfig(
            architecture="tiny", learning_starts=8, batch_size=8,
            train_frequency=1, critic_warmup_updates=100,
            critic_readiness_window=8,
        )},
    }
    return TrainingConfig(
        algorithm=algorithm, evaluation=EvaluationConfig(16, 5),
        **settings[algorithm],
    )


def _model(config: TrainingConfig):
    backend = TQCBackend() if config.algorithm == "tqc" else GRTQCBackend()
    env = PolyTrackEnv(
        MockSimulatorTransport(), action_adapter=backend.action_adapter(config),
    )
    return backend.create_model(config, env, "cpu"), env


def test_gated_actor_transfers_all_tqc_weights_with_near_identical_actions() -> None:
    source, source_env = _model(_config("tqc"))
    target, target_env = _model(_config("grtqc"))
    try:
        transfer = target.actor.load_state_dict(source.actor.state_dict(), strict=False)
        assert not transfer.unexpected_keys
        assert len(transfer.missing_keys) == 4
        assert all(".gate." in key for key in transfer.missing_keys)
        observations = np.random.default_rng(7).normal(
            size=(64, source.observation_space.shape[0]),
        ).astype(np.float32)
        with th.no_grad():
            inputs = th.as_tensor(observations)
            left = source.actor(inputs, deterministic=True)
            right = target.actor(inputs, deterministic=True)
        th.testing.assert_close(left, right, rtol=0, atol=0)
    finally:
        source_env.close()
        target_env.close()


def test_critic_updates_keep_transferred_actor_frozen_until_ready() -> None:
    model, env = _model(_config("grtqc"))
    try:
        before = {name: weight.clone() for name, weight in model.actor.state_dict().items()}
        model.learn(24)
        assert model.critic_updates_since_transfer > 0
        assert not model.actor_unlocked
        for name, weight in model.actor.state_dict().items():
            th.testing.assert_close(weight, before[name])
        metrics = GRTQCBackend().metrics(model)
        assert metrics["critic_disagreement"] is not None
        assert metrics["disagreement_penalty"] is not None
    finally:
        env.close()


def test_actor_step_backtracks_large_action_change() -> None:
    config = _config("grtqc")
    assert config.grtqc is not None
    config.grtqc.actor_learning_rate = 1e-3
    config.grtqc.actor_step_action_limit = 1e-5
    model, env = _model(config)
    try:
        reference_observations = np.random.default_rng(11).normal(
            size=(256, model.observation_space.shape[0]),
        ).astype(np.float32)
        model.set_actor_reference_observations(reference_observations)
        model.actor_unlocked = True
        model.learn(16)
        metrics = GRTQCBackend().metrics(model)
        assert metrics["actor_proposed_action_drift"] is not None
        assert metrics["actor_executed_action_drift"] is not None
        assert metrics["actor_proposed_action_drift"] > config.grtqc.actor_step_action_limit
        assert metrics["actor_executed_action_drift"] <= 1.1e-5
        assert metrics["actor_reference_action_drift"] <= 1.1e-5
        assert metrics["actor_reference_cumulative_action_drift"] <= (
            config.grtqc.actor_reference_drift_limit
        )
    finally:
        env.close()


def test_disabled_reference_cap_allows_learning_with_bounded_steps() -> None:
    config = _config("grtqc")
    config.grtqc.actor_learning_rate = 1e-3
    config.grtqc.actor_step_action_limit = 1e-4
    config.grtqc.actor_reference_drift_limit = 0.0
    model, env = _model(config)
    try:
        observations = np.random.default_rng(11).normal(
            size=(256, model.observation_space.shape[0]),
        ).astype(np.float32)
        model.set_actor_reference_observations(observations)
        model.actor_unlocked = True
        model.learn(24)
        metrics = GRTQCBackend().metrics(model)
        assert metrics["actor_reference_cumulative_action_drift"] > 0
        assert 0 < metrics["actor_executed_action_drift"] <= 1.01e-4
        assert metrics["actor_reference_action_drift"] <= 1.01e-4
    finally:
        env.close()


def test_correlated_exploration_resets_at_episode_end(monkeypatch) -> None:
    config = _config("grtqc")
    config.grtqc.exploration_correlation = 0.9
    model, env = _model(config)
    try:
        rng = np.random.default_rng(config.seed)
        first = model._rollout_noise((1, 2), 0.006)
        expected_first = np.sqrt(1 - 0.9**2) * rng.normal(0, 0.006, (1, 2))
        np.testing.assert_allclose(first, expected_first, rtol=1e-6)
        second = model._rollout_noise((1, 2), 0.006)
        expected_second = 0.9 * first + np.sqrt(1 - 0.9**2) * rng.normal(0, 0.006, (1, 2))
        np.testing.assert_allclose(second, expected_second, rtol=1e-6)
        monkeypatch.setattr("polybot.algorithms.tqc.SeededWarmupTQC._store_transition", lambda *a: None)
        model._store_transition(None, first, None, np.zeros(1), np.array([True]), [{}])
        np.testing.assert_array_equal(model._exploration_noise, np.zeros((1, 2)))
        model._rollout_noise((1, 2), 0.006)
        np.testing.assert_array_equal(model._rollout_noise((1, 2), 0), np.zeros((1, 2)))
    finally:
        env.close()


@pytest.mark.parametrize(
    ("laps", "expected"),
    [((24.2,) * 5, "champion"), ((24.3,) * 5, "rejected"), ((24.1,) * 4, "rejected")],
)
def test_promotion_requires_five_reliable_faster_laps(tmp_path, monkeypatch, laps, expected) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    result = EvaluationResult(
        episodes=5, finish_rate=len(laps) / 5,
        median_progress=1.0 if len(laps) == 5 else 0.9,
        mean_progress=1.0 if len(laps) == 5 else 0.9,
        best_lap_s=min(laps), median_lap_s=float(np.median(laps)),
        crash_rate=0.0, off_track_rate=0.0, stall_rate=0.0,
    )
    saved: list[str] = []
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: result)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path)
    runner._evaluate()
    assert expected in saved[0]


def test_promotion_uses_measured_frame_skip_reference(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    assert config.grtqc is not None
    config.grtqc.reference_lap_s = 24.616
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    result = EvaluationResult(5, 1.0, 1.0, 1.0, 24.5, 24.5, 0.0, 0.0, 0.0)
    saved = []
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: result)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path)
    runner._evaluate()
    assert saved == ["champion"]


def test_grtqc_keeps_contact_reduction_as_candidate_within_pace_tolerance(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    assert config.grtqc is not None
    config.grtqc.reference_lap_s = 24.616
    config.grtqc.champion_lap_tolerance_s = 0.15
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    previous = EvaluationResult(
        5, 1.0, 1.0, 1.0, 24.616, 24.616, 0.0, 0.0, 0.0,
        barrier_contact_steps=20,
    )
    current = EvaluationResult(
        5, 1.0, 1.0, 1.0, 24.893, 24.893, 0.0, 0.0, 0.0,
        barrier_contact_steps=10,
    )
    champion_dir = runner.registry.slot(config.track_name, "grtqc", "champion")
    champion_dir.mkdir(parents=True)
    (champion_dir / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner.registry, "read_metadata",
        lambda _path: SimpleNamespace(
            training_config={"rewards": config.to_dict()["rewards"]},
            reward_semantics="executed-controls-v1", evaluation=previous.to_dict(),
        ),
    )
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: current)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    saved = []
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path)
    runner._grtqc_weak_evaluations = 2

    runner._evaluate()

    assert saved == ["contact-candidate"]
    assert runner._grtqc_weak_evaluations == 0


def test_failed_candidate_restores_only_verified_actor_and_rewarms_critics(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    current = th.nn.Linear(2, 2)
    verified = th.nn.Linear(2, 2)
    current.optimizer = th.optim.Adam(current.parameters(), lr=1e-5)
    verified.optimizer = th.optim.Adam(verified.parameters(), lr=1e-5)
    critic = th.nn.Linear(2, 1)
    preserved = {key: value.clone() for key, value in critic.state_dict().items()}
    runner.model = SimpleNamespace(
        actor=current, critic=critic, actor_lr=1e-5, actor_unlocked=True,
        critic_warmup_updates=2100,
        critic_updates_since_transfer=2100,
        _critic_loss_history=deque([1.0]), _disagreement_history=deque([0.1]),
        log_ent_coef=None, num_timesteps=9000,
    )
    runner.device = SimpleNamespace(resolved="cpu")
    monkeypatch.setattr(runner.backend, "load_model", lambda *a, **k: SimpleNamespace(actor=verified))
    events = []
    monkeypatch.setattr(runner, "_emit", events.append)
    result = EvaluationResult(5, 0.0, 0.5, 0.5, None, None, 1.0, 0.0, 0.0)
    runner._recover_grtqc_actor(result, tmp_path / "rejected")
    for key, value in current.state_dict().items():
        th.testing.assert_close(value, verified.state_dict()[key])
    for key, value in critic.state_dict().items():
        th.testing.assert_close(value, preserved[key])
    assert runner.model.actor_lr == pytest.approx(1e-5)
    assert not runner.model.actor_unlocked
    assert runner.model.critic_updates_since_transfer == 1100
    assert events[0]["critic_replay_preserved"] is True
    assert events[0]["critic_cooldown_updates"] == 1000


def test_first_cleaner_candidate_is_saved_and_can_be_recovered_without_champion(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    references = []
    runner.model = SimpleNamespace(num_timesteps=5000, set_actor_reference_observations=references.append)
    previous = EvaluationResult(5, 1.0, 1.0, 1.0, 24.263, 24.263, 0.0, 0.0, 0.0, barrier_contact_steps=10)
    current = replace(previous, best_lap_s=24.4, median_lap_s=24.4, barrier_contact_steps=5)
    initialization = runner.registry.slot(config.track_name, "grtqc", "initialization")
    candidate = runner.registry.slot(config.track_name, "grtqc", "contact-candidate")
    initialization.mkdir(parents=True)
    (initialization / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner.registry, "read_metadata", lambda path: SimpleNamespace(
        evaluation=(previous if path == initialization else current).to_dict(),
    ))

    def evaluate(*args, observation_sink, **kwargs):
        observation_sink.append(np.zeros(3))
        return current

    def save(name, evaluation):
        assert name == "contact-candidate"
        candidate.mkdir(parents=True)
        (candidate / "metadata.json").write_text("{}", encoding="utf-8")
        return candidate

    monkeypatch.setattr("polybot.training.runner.evaluate_model", evaluate)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", save)
    runner._evaluate()
    assert runner._grtqc_verified_actor_source() == candidate
    assert len(references) == 1


@pytest.mark.parametrize("error", ["stale_episode: expired", "wrong_track: mismatch"])
def test_evaluation_retries_only_stale_episode_and_discards_partial_samples(tmp_path, monkeypatch, error) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    attempts = []

    def evaluate(*args, observation_sink, **kwargs):
        attempts.append(len(observation_sink))
        if len(attempts) == 1:
            observation_sink.append(np.zeros(3))
            raise ProtocolViolation(error)
        return EvaluationResult(5, 1.0, 1.0, 1.0, 24.263, 24.263, 0.0, 0.0, 0.0)

    monkeypatch.setattr("polybot.training.runner.evaluate_model", evaluate)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda *args: tmp_path)
    if error.startswith("stale_episode:"):
        assert runner._evaluate().finish_rate == 1.0
        assert attempts == [0, 0]
    else:
        with pytest.raises(ProtocolViolation, match="wrong_track"):
            runner._evaluate()
        assert attempts == [0]


def test_grtqc_waits_for_three_weak_evaluations_before_recovery(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000, actor_unlocked=True)
    weak = EvaluationResult(5, 0.0, 0.5, 0.5, None, None, 1.0, 0.0, 0.0)
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: weak)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: tmp_path / name)
    recovered = []
    monkeypatch.setattr(runner, "_recover_grtqc_actor", lambda *args: recovered.append(args))
    runner._evaluate()
    runner._evaluate()
    assert not recovered
    runner._evaluate()
    assert len(recovered) == 1


def test_rejected_grtqc_checkpoint_does_not_duplicate_replay(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace()
    monkeypatch.setattr(runner, "_metadata", lambda evaluation: None)
    monkeypatch.setattr(runner.registry, "write_metadata", lambda directory, metadata: None)
    flags = []
    monkeypatch.setattr(
        runner.backend, "save_model",
        lambda model, directory, *, resume: flags.append(resume),
    )
    runner._save("checkpoints/step-100-rejected")
    runner._save("latest")
    assert flags == [False, True]


def test_transport_disconnect_checkpoints_current_model(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=1234)
    saved = []
    events = []
    monkeypatch.setattr(runner, "_save", lambda name, evaluation=None: saved.append(name) or tmp_path)
    monkeypatch.setattr(runner, "_emit", events.append)

    runner._checkpoint_after_transport_failure(ConnectionError("player left race"))

    assert saved == ["latest"]
    assert events == [{
        "type": "transport_disconnected", "timesteps": 1234,
        "checkpoint": str(tmp_path), "error": "player left race",
    }]


def test_external_grtqc_initialization_is_kept_for_future_resumes(tmp_path) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    source = tmp_path / "prior-experiment" / "summer-1" / "grtqc" / "initialization"
    source.mkdir(parents=True)
    (source / "policy.zip").write_bytes(b"transferred-policy")
    (source / "metadata.json").write_text("{}", encoding="utf-8")
    (source / "transfer.json").write_text("{}", encoding="utf-8")

    preserved = runner._preserve_grtqc_initialization(source)

    assert preserved == runner.registry.slot(config.track_name, "grtqc", "initialization")
    assert (preserved / "policy.zip").read_bytes() == b"transferred-policy"
    assert (preserved / "transfer.json").is_file()
    assert runner._preserve_grtqc_initialization(source) == preserved

    (source / "policy.zip").write_bytes(b"different-policy")
    with pytest.raises(FileExistsError, match="GRTQC initialization differs"):
        runner._preserve_grtqc_initialization(source)
