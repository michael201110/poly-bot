from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import torch as th

from polybot.algorithms.registry import ALGORITHMS, backend_for
from polybot.control.native_digital import NativeDigitalActionAdapter
from polybot.environment.curriculum import build_plan
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import IncompatibleModelError, ModelMetadata, ModelRegistry
from polybot.protocol import Action
from polybot.training.adaptation import _policy_digest, candidate_diagnostics_pass
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    DQNConfig,
    EvaluationConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.devices import resolve_device
from polybot.training.evaluation import EvaluationResult, evaluate_model
from polybot.training.lap_analysis import discover_airborne_regions, sector_delta_map
from polybot.training.pace_config import champion_evaluation_config
from polybot.training.promotion import promote_directory
from polybot.training.runner import TrainingRunner, _evaluation_for_current_checkpoint
from polybot.training.section_optimizer import SectionOptimizer, section_windows, write_checkpoint
from polybot.training.wr_search import (
    _candidate_grid,
    compose_overlay_stack,
    confirmation_count_for_gain,
)


def configuration(tmp_path, algorithm: str) -> TrainingConfig:
    specific = {
        "ppo": {"ppo": PPOConfig(architecture="tiny", rollout_steps=32, batch_size=16, epochs=1)},
        "dqn": {"dqn": DQNConfig(architecture="tiny", learning_starts=8, batch_size=8,
                                 replay_capacity=1000, train_frequency=1, target_update_interval=16)},
        "tqc": {"tqc": TQCConfig(architecture="tiny", learning_starts=100,
                                 batch_size=16, replay_capacity=1000)},
    }[algorithm]
    return TrainingConfig(
        algorithm=algorithm, device="cpu", timesteps=32,
        evaluation=EvaluationConfig(32, 1), checkpoint_interval=16,
        output_root=tmp_path / "models", log_root=tmp_path / "logs", **specific,
    )


def test_websocket_environment_allows_cold_ghost_reference_startup(tmp_path) -> None:
    config = configuration(tmp_path, "ppo")
    config.backend = "websocket"
    runner = TrainingRunner(config, transport_factory=MockSimulatorTransport)
    env = runner._environment()
    try:
        assert env.request_timeout_s == 180.0
    finally:
        env.close()


def test_registry_contains_three_equal_backends() -> None:
    assert set(ALGORITHMS) == {"ppo", "dqn", "tqc"}
    with pytest.raises(ValueError, match="unknown algorithm"):
        backend_for("sac")


def test_config_roundtrip_and_algorithm_specific_validation(tmp_path) -> None:
    for name in ALGORITHMS:
        config = configuration(tmp_path, name)
        assert TrainingConfig.from_dict(config.to_dict()) == config


def test_pace_search_uses_champion_evaluation_semantics_and_requested_seed(tmp_path) -> None:
    champion_config = configuration(tmp_path, "tqc")
    champion_config.track_name = "Summer 1"
    champion_config.track_id = "current"
    champion_config.rewards = replace(
        champion_config.rewards, barrier_collision_impulse_threshold=1_000_000_000.0,
    )
    requested = TrainingConfig.from_dict(champion_config.to_dict())
    requested.seed = 12345
    requested.rewards = replace(requested.rewards, barrier_collision_impulse_threshold=0.0)
    requested.reward_profile = "different profile"

    metadata = ModelMetadata(
        algorithm="tqc", architecture="tiny", actor_parameters=0, critic_parameters=0,
        total_trainable_parameters=0, observation_schema="polybot.observation.v2",
        action_schema="continuous-pwm-v2", track_name=champion_config.track_name,
        track_id=champion_config.track_id, lookahead_count=champion_config.lookahead_count,
        reward_profile=champion_config.reward_profile, curriculum={},
        training_config=champion_config.to_dict(), training_timesteps=0, simulator_ticks=0,
        wall_seconds=0, seed=champion_config.seed, device="cpu", finishes=0, crashes=0,
    )
    resolved = champion_evaluation_config(requested, metadata)
    assert resolved.seed == requested.seed
    assert resolved.reward_profile == champion_config.reward_profile
    assert resolved.rewards.barrier_collision_impulse_threshold == 1_000_000_000.0
    assert resolved.tqc == champion_config.tqc

    requested.track_id = "another-track"
    with pytest.raises(ValueError, match="different track"):
        champion_evaluation_config(requested, metadata)
    with pytest.raises(ValueError, match="TQC settings"):
        TrainingConfig(algorithm="ppo", ppo=PPOConfig(), tqc=TQCConfig())
    with pytest.raises(ValueError, match="DQN"):
        TrainingConfig(algorithm="ppo", ppo=PPOConfig(), dqn=DQNConfig())
    old_v2 = configuration(tmp_path, "tqc").to_dict()
    del old_v2["dqn"]
    assert TrainingConfig.from_dict(old_v2).dqn is None
    with pytest.raises(ValueError, match="only v2"):
        TrainingConfig.from_dict({"schema": "polybot.config.v1"})
    old_ppo = configuration(tmp_path, "ppo").to_dict()
    old_ppo["ppo"]["pwm_levels"] = 41
    with pytest.raises(ValueError, match=r"continuous Box\(2\)"):
        TrainingConfig.from_dict(old_ppo)
    ppo_cfg = configuration(tmp_path, "ppo")
    legacy_ppo = ModelMetadata(
        algorithm="ppo", architecture="tiny", actor_parameters=0, critic_parameters=0,
        total_trainable_parameters=0, observation_schema="polybot.observation.v2",
        action_schema="pwm-multidiscrete-v2", track_name=ppo_cfg.track_name,
        track_id=ppo_cfg.track_id, lookahead_count=ppo_cfg.lookahead_count,
        reward_profile=None, curriculum={}, training_config=ppo_cfg.to_dict(),
        training_timesteps=0, simulator_ticks=0, wall_seconds=0, seed=0, device="cpu",
        finishes=0, crashes=0,
    )
    with pytest.raises(IncompatibleModelError, match="discrete PWM"):
        ModelRegistry(tmp_path).validate(legacy_ppo, ppo_cfg, "continuous-pwm-v2")


def test_curriculum_budget_is_global() -> None:
    for mode, phase_count in (
        ("full", 1), ("quarters", 5), ("quarters-randomised", 1),
        ("q4-full", 2),
    ):
        plan = build_plan(CurriculumConfig(mode), 103)
        assert len(plan.phases) == phase_count
        assert plan.total_steps == 103
        assert all(phase.steps > 0 for phase in plan.phases)
    assert build_plan(CurriculumConfig("section", .25, .5), 50).phases[0].end_ratio == .5
    assert build_plan(CurriculumConfig("timed", start_s=3, end_s=6), 50).phases[0].end_s == 6
    custom = CurriculumConfig("custom", phases=(
        CurriculumPhaseConfig("section", 30, .75, 1.0),
        CurriculumPhaseConfig("full", 70),
    ))
    assert build_plan(custom, 100).total_steps == 100
    with pytest.raises(ValueError, match="sum"):
        build_plan(custom, 101)
    cfg = TrainingConfig(curriculum=custom, timesteps=100, tqc=TQCConfig())
    assert TrainingConfig.from_dict(cfg.to_dict()) == cfg


def test_curriculum_sections_spawn_with_lead_in_and_keep_global_budget() -> None:
    expected = [(0.0, 0.0, 0.25), (0.20, 0.25, 0.50), (0.45, 0.50, 0.75),
                (0.70, 0.75, 1.0)]
    plan = build_plan(CurriculumConfig("quarters"), 1000)
    assert plan.total_steps == 1000
    for phase, (spawn, start, end) in zip(plan.phases[:4], expected, strict=True):
        assert (phase.spawn_ratio, phase.start_ratio, phase.end_ratio) == (spawn, start, end)
    randomised = build_plan(CurriculumConfig("quarters-randomised"), 100)
    assert randomised.total_steps == 100
    assert randomised.phases[0].env_kwargs()["curriculum_lead_in_ratio"] == 0.05
    q4_full = build_plan(CurriculumConfig("q4-full"), 100)
    assert (q4_full.phases[0].spawn_ratio, q4_full.phases[0].start_ratio,
            q4_full.phases[0].end_ratio) == (0.70, 0.75, 1.0)
    section = build_plan(CurriculumConfig("section", .25, .5), 20)
    assert (section.phases[0].spawn_ratio, section.phases[0].start_ratio) == (0.20, 0.25)
    custom = build_plan(CurriculumConfig("custom", phases=(
        CurriculumPhaseConfig("section", 20, .5, .75, lead_in_ratio=.1),
        CurriculumPhaseConfig("full", 30),
    )), 50)
    assert (custom.phases[0].spawn_ratio, custom.phases[0].start_ratio,
            custom.phases[0].end_ratio) == (.4, .5, .75)

    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", curriculum_random_quarters=True,
        action_adapter=NativeDigitalActionAdapter(),
    )
    try:
        _, info = env.reset(seed=17)
        assert info["curriculum_start_ratio"] - info["curriculum_spawn_ratio"] == pytest.approx(.05)
        assert info["curriculum_end_ratio"] - info["curriculum_start_ratio"] == pytest.approx(.25)
    finally:
        env.close()


def test_curriculum_reset_info_is_moving_and_section_relative() -> None:
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", frame_skip=4,
        curriculum_spawn_ratio=.20, curriculum_start_ratio=.25, curriculum_end_ratio=.5,
        action_adapter=NativeDigitalActionAdapter(),
    )
    try:
        _, reset = env.reset(seed=4)
        assert reset["ticks_advanced"] == 0
        assert reset["route_progress_m"] > 0
        assert reset["local_velocity_mps"][2] == pytest.approx(20.0)
        assert reset["previous_action"] == Action().to_wire()
        assert reset["actual_steering"] == 0
        assert reset["section_progress"] == 0
        assert reset["curriculum_in_lead_in"] is True
        _, _, _, _, info = env.step(1)
        assert info["ticks_advanced"] == 4
        assert info["section_progress"] == pytest.approx(0.0)
        assert info["curriculum_stage"] == "lead-in"
    finally:
        env.close()

    q1 = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight", frame_skip=4)
    try:
        _, reset = q1.reset(seed=4)
        assert reset["ticks_advanced"] == 0
        assert reset["route_progress_m"] == 0
        assert reset["local_velocity_mps"][2] == 0
        assert reset["curriculum_stage"] == "full track"
    finally:
        q1.close()


def test_champion_rank_uses_deterministic_finish_and_progress() -> None:
    weak = EvaluationResult(3, 0, .7, .75, None, None, 0, 0, 0)
    strong = EvaluationResult(3, 1/3, .5, .6, 25, 25, 1/3, 0, 0)
    assert strong.rank() > weak.rank()
    quicker = replace(strong, best_lap_s=22, median_lap_s=22)
    assert quicker.rank() > strong.rank()
    reliable = EvaluationResult(5, 1.0, 1.0, 1.0, 24.5, 25.0, 0, 0, 0)
    slower = replace(reliable, median_lap_s=25.1, mean_progress=1.000001)
    assert reliable.rank() > slower.rank()


def test_tqc_resume_applies_optimizer_rate_and_mutable_settings(tmp_path) -> None:
    initial = configuration(tmp_path, "tqc")
    initial.tqc = replace(initial.tqc, learning_rate=3e-4)
    backend = backend_for("tqc")
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id=initial.track_id,
        action_adapter=backend.action_adapter(initial),
    )
    try:
        model = backend.create_model(initial, env, "cpu")
        path = tmp_path / "saved"
        backend.save_model(model, path, resume=True)
        resumed = backend.load_model(path / "policy.zip", env, "cpu", resume=True)
        changed = replace(initial, tqc=replace(
            initial.tqc, learning_rate=1e-5, batch_size=32,
            train_frequency=4, gradient_steps=2, gamma=0.995, tau=0.006,
        ))
        backend.configure_resume(resumed, changed, "cpu")
        assert resumed.learning_rate == pytest.approx(1e-5)
        assert resumed.lr_schedule(0.5) == pytest.approx(1e-5)
        for optimizer in (
            resumed.actor.optimizer, resumed.critic.optimizer,
            resumed.ent_coef_optimizer,
        ):
            assert optimizer is not None
            assert all(group["lr"] == pytest.approx(1e-5)
                       for group in optimizer.param_groups)
        assert resumed.batch_size == 32
        assert resumed.train_freq.frequency == 4
        assert resumed.gradient_steps == 2
        assert resumed.gamma == pytest.approx(0.995)
        assert resumed.tau == pytest.approx(0.006)
    finally:
        env.close()


def test_ppo_resume_applies_current_optimizer_and_update_settings(tmp_path) -> None:
    initial = configuration(tmp_path, "ppo")
    initial.ppo = replace(initial.ppo, learning_rate=3e-4)
    backend = backend_for("ppo")
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id=initial.track_id,
        action_adapter=backend.action_adapter(initial),
    )
    try:
        model = backend.create_model(initial, env, "cpu")
        path = tmp_path / "ppo-saved"
        backend.save_model(model, path, resume=True)
        resumed = backend.load_model(path / "policy.zip", env, "cpu", resume=True)
        changed = replace(initial, ppo=replace(
            initial.ppo, learning_rate=1e-5, entropy_coefficient=1e-4,
            epochs=2, gamma=0.99, gae_lambda=0.9, target_kl=0.003,
        ))

        backend.configure_resume(resumed, changed, "cpu")

        assert resumed.learning_rate == pytest.approx(1e-5)
        assert resumed.lr_schedule(0.5) == pytest.approx(1e-5)
        assert all(group["lr"] == pytest.approx(1e-5)
                   for group in resumed.policy.optimizer.param_groups)
        assert resumed.ent_coef == pytest.approx(1e-4)
        assert resumed.n_epochs == 2
        assert resumed.batch_size == changed.ppo.batch_size
        assert resumed.gamma == pytest.approx(0.99)
        assert resumed.gae_lambda == pytest.approx(0.9)
        assert resumed.target_kl == pytest.approx(0.003)
    finally:
        env.close()


def test_champion_directory_promotion_is_complete_and_retains_backup(tmp_path) -> None:
    champion = tmp_path / "champion"
    staging = tmp_path / "candidate"
    champion.mkdir()
    staging.mkdir()
    for name in ("policy.zip", "replay.pkl", "metadata.json"):
        (champion / name).write_text("old", encoding="utf-8")
    (staging / "policy.zip").write_text("new", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        promote_directory(staging, champion, require_replay=True)
    assert (champion / "policy.zip").read_text(encoding="utf-8") == "old"
    for name in ("replay.pkl", "metadata.json"):
        (staging / name).write_text("new", encoding="utf-8")
    backup = promote_directory(staging, champion, require_replay=True)
    assert backup is not None
    assert (champion / "policy.zip").read_text(encoding="utf-8") == "new"
    assert (backup / "policy.zip").read_text(encoding="utf-8") == "old"


def test_champion_promotion_recovers_from_transient_windows_lock(tmp_path, monkeypatch) -> None:
    import polybot.training.promotion as promotion

    champion = tmp_path / "champion"
    staging = tmp_path / "candidate"
    champion.mkdir()
    staging.mkdir()
    for folder, text in ((champion, "old"), (staging, "new")):
        for name in ("policy.zip", "replay.pkl", "metadata.json"):
            (folder / name).write_text(text, encoding="utf-8")
    replace_real = promotion.os.replace
    attempts = 0

    def transient_replace(source, destination):
        nonlocal attempts
        if source == staging.resolve() and destination == champion.resolve() and attempts == 0:
            attempts += 1
            raise PermissionError("simulated scanner lock")
        return replace_real(source, destination)

    monkeypatch.setattr(promotion.os, "replace", transient_replace)
    monkeypatch.setattr(promotion.time, "sleep", lambda _: None)
    backup = promote_directory(staging, champion, require_replay=True)
    assert attempts == 1
    assert backup is not None
    assert (champion / "replay.pkl").read_text(encoding="utf-8") == "new"
    assert (backup / "replay.pkl").read_text(encoding="utf-8") == "old"


def test_pace_polish_resumes_champion_instead_of_latest(tmp_path) -> None:
    cfg = configuration(tmp_path, "tqc")
    TrainingRunner(cfg).run()
    registry = ModelRegistry(cfg.output_root)
    champion = registry.slot(cfg.track_name, "tqc", "champion")
    latest = registry.slot(cfg.track_name, "tqc", "latest")
    events = []
    short = replace(cfg, timesteps=4, checkpoint_interval=0)
    with pytest.raises(ValueError, match="champion, not latest"):
        TrainingRunner(short).run(resume=latest, pace_polish=True)
    TrainingRunner(short, events.append).run(pace_polish=True)
    started = next(event for event in events if event["type"] == "started")
    assert started["resume_source"] == str(champion)
    assert started["rollback_on_regression"] is True
    assert started["mode"] == "pace_polish"


def test_speed_search_requires_confirmation_and_records_section_promotion(
    tmp_path, monkeypatch,
) -> None:
    import polybot.training.speed_search as speed_search

    config = configuration(tmp_path, "tqc")
    TrainingRunner(config).run()
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion")
    original = registry.read_metadata(champion)
    baseline = EvaluationResult(5, 1.0, 1.0, 1.0, 20.0, 20.0, 0, 0, 0)
    saved_overlay = {
        "kind": "air_brake", "start": 0.7, "end": 0.8, "duty": 1.0, "taper": 0.003,
    }
    registry.write_metadata(champion, replace(
        original, evaluation=baseline.to_dict(), policy_overlays=[saved_overlay],
    ))
    config.backend = "websocket"
    config_path = tmp_path / "search.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    policy_before = (champion / "policy.zip").read_bytes()
    history_path = champion.parent / "pace-history.jsonl"
    history_before = history_path.read_text(encoding="utf-8").splitlines()
    faster = replace(baseline, median_lap_s=19.0, best_lap_s=19.0)
    failed = replace(baseline, finish_rate=0.0, median_lap_s=None, best_lap_s=None)
    evaluations = iter((faster, failed))
    observed_overlay_stacks = []

    def evaluate_with_saved_overlays(model, *args, **kwargs):
        observed_overlay_stacks.append(model.policy_overlays)
        return next(evaluations)

    monkeypatch.setattr(speed_search, "evaluate_model", evaluate_with_saved_overlays)
    speed_search.search(config_path, tmp_path / "rejected.jsonl", 1, 18.0)
    assert observed_overlay_stacks == [[saved_overlay], [saved_overlay]]
    assert (champion / "policy.zip").read_bytes() == policy_before
    assert history_path.read_text(encoding="utf-8").splitlines() == history_before

    evaluations = iter((faster, faster))
    speed_search.search(config_path, tmp_path / "accepted.jsonl", 1, 18.0, mode="section")
    assert registry.read_metadata(champion).evaluation["median_lap_s"] == 19.0
    assert registry.read_metadata(champion).policy_overlays == [saved_overlay]
    details = json.loads((champion / "speed-search.json").read_text(encoding="utf-8"))
    assert ModelRegistry(config.output_root).read_metadata(champion).critic_adaptation_required
    accepted_events = [json.loads(line) for line in (tmp_path / "accepted.jsonl").read_text().splitlines()]
    assert any(event.get("critic_adaptation_required") is True
               for event in accepted_events if event["type"] == "champion")
    assert details["parameters"]["speed_window"] is not None
    assert details["delta_s"] == pytest.approx(-1.0)
    assert len(details["speed_bias_schedule"]) == 1
    history = history_path.read_text(encoding="utf-8").splitlines()
    assert len(history) == len(history_before) + 1
    assert json.loads(history[-1])["source"] == "speed_search_section"
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend_for("tqc").action_adapter(config))
    try:
        restored = backend_for("tqc").load_model(champion / "policy.zip", env, "cpu", resume=True)
        assert restored.speed_bias_schedule == details["speed_bias_schedule"]
    finally:
        env.close()


def test_speed_search_prioritizes_start_acceleration_windows() -> None:
    from polybot.training.speed_search import _section_probes

    probes = _section_probes()
    assert probes[:6] == [
        (0.0, 0.05, 0.02), (0.0, 0.05, 0.05), (0.0, 0.05, 0.10),
        (0.0, 0.05, -0.02), (0.0, 0.05, -0.05), (0.0, 0.05, -0.10),
    ]
    assert (0.0, 0.20, 0.10) in probes
    assert (0.05, 0.15, -0.10) in probes
    assert (0.55, 0.75, 0.01) in probes


def test_speed_search_cli_supports_seeded_followup_batches() -> None:
    import sys
    from unittest.mock import patch

    from polybot.training.speed_search import main

    arguments = ["speed_search", "--config", "config.json", "--log", "run.jsonl",
                 "--mode", "section", "--seed-offset", "2000"]
    with patch.object(sys, "argv", arguments), \
         patch("polybot.training.speed_search.search") as search_mock:
        main()
    assert search_mock.call_args.args[-1] == 2000


@pytest.mark.parametrize("algorithm", ["ppo", "dqn", "tqc"])
def test_short_train_save_resume_and_evaluate(tmp_path, algorithm: str) -> None:
    config = configuration(tmp_path, algorithm)
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run()
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, algorithm, "champion")
    assert (latest / "policy.zip").is_file()
    assert (champion / "policy.zip").is_file()
    assert (champion / "metadata.json").is_file()
    assert (champion / "replay.pkl").exists() == (algorithm in {"dqn", "tqc"})
    assert (latest / "replay.pkl").exists() == (algorithm in {"dqn", "tqc"})
    metadata = registry.read_metadata(latest)
    assert metadata.training_config == config.to_dict()
    assert metadata.observation_schema == "polybot.observation.v2"
    assert metadata.evaluation["episodes"] == 1
    assert (metadata.actor_parameters == 0) == (algorithm == "dqn")
    assert metadata.critic_parameters > 0
    assert metadata.total_trainable_parameters >= metadata.actor_parameters
    if algorithm == "ppo":
        registry.write_metadata(latest, replace(metadata, polybot_version="2.0.0"))
        assert registry.read_metadata(latest).polybot_version == "2.0.0"
    assert any(event["type"] == "evaluation" for event in events)
    assert any(event["type"] == "champion" for event in events)
    assert list(config.log_root.glob("*.jsonl"))
    before = metadata.training_timesteps
    if algorithm == "dqn":
        backend = backend_for("dqn")
        env = PolyTrackEnv(MockSimulatorTransport(), track_id=config.track_id,
                           action_adapter=backend.action_adapter(config))
        try:
            loaded = backend.load_model(latest / "policy.zip", env, "cpu", resume=True)
            assert loaded.action_space.n == 9
            assert loaded.replay_buffer.size() > 0
            assert backend.metrics(loaded)["updates"] > 0
            deterministic, _ = loaded.predict(env.reset(seed=5)[0], deterministic=True)
            assert 0 <= int(deterministic) < 9
            best = backend.load_model(champion / "policy.zip", env, "cpu", resume=True)
            assert best.replay_buffer.size() > 0
        finally:
            env.close()
    TrainingRunner(config).run(resume=latest)
    assert registry.read_metadata(latest).training_timesteps > before
    with pytest.raises(IncompatibleModelError, match="track"):
        registry.validate(metadata, replace(config, track_id="mock/gentle-s"),
                          backend_for(algorithm).action_adapter(config).schema)


def test_latest_checkpoint_drops_stale_evaluation_after_unvalidated_updates() -> None:
    result = EvaluationResult(
        episodes=5, finish_rate=1.0, median_progress=1.0, mean_progress=1.0,
        best_lap_s=24.675, median_lap_s=24.675, crash_rate=0.0,
        off_track_rate=0.0, stall_rate=0.0,
    )

    assert _evaluation_for_current_checkpoint(
        result, current_steps=100, evaluated_steps=100,
    ) is result
    assert _evaluation_for_current_checkpoint(
        result, current_steps=101, evaluated_steps=100,
    ) is None


def test_older_tqc_champion_without_replay_can_continue(tmp_path) -> None:
    config = configuration(tmp_path, "tqc")
    TrainingRunner(config).run()
    champion = ModelRegistry(config.output_root).slot(config.track_name, "tqc", "champion")
    (champion / "replay.pkl").unlink()
    resumed = replace(
        config, timesteps=16, evaluation=EvaluationConfig(16, 1),
        tqc=replace(config.tqc, learning_starts=8),
    )
    runner = TrainingRunner(resumed)
    runner.run(resume=champion, fresh_replay=True)
    assert runner.model.replay_buffer.size() >= 16
    assert runner.model.learning_starts >= runner.model.num_timesteps
    assert runner.model._refill_replay_from_policy
    assert runner.model._n_updates == 0


def test_older_tqc_champion_refills_replay_before_rollback(tmp_path, monkeypatch) -> None:
    import polybot.training.runner as runner_module

    strong = EvaluationResult(1, 1.0, 1.0, 1.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    weak = EvaluationResult(1, 0.0, 0.2, 0.2, None, None, 0.0, 1.0, 0.0)
    evaluations = iter((strong, strong, weak))
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    config = replace(
        configuration(tmp_path, "tqc"), timesteps=16,
        evaluation=EvaluationConfig(16, 1),
    )
    TrainingRunner(config).run()
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion")
    (champion / "replay.pkl").unlink()
    resumed = replace(config, timesteps=32, tqc=replace(config.tqc, learning_starts=16))
    events: list[dict] = []
    latest = TrainingRunner(resumed, events.append).run(
        resume=champion, fresh_replay=True, rollback_to_champion=True
    )
    assert (champion / "replay.pkl").is_file()
    assert any(event["type"] == "champion_replay" for event in events)
    assert any(event["type"] == "rollback" and event["replay_source"] == "champion"
               for event in events)
    assert registry.read_metadata(latest).evaluation is None


def test_tqc_changed_rewards_require_and_refill_fresh_replay(tmp_path, monkeypatch) -> None:
    import polybot.training.runner as runner_module

    strong = EvaluationResult(1, 1.0, 1.0, 1.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: strong)
    config = replace(
        configuration(tmp_path, "tqc"), timesteps=16,
        evaluation=EvaluationConfig(16, 1),
    )
    TrainingRunner(config).run()
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion")
    paced = replace(
        config, rewards=replace(config.rewards, finish_target_s=20.0),
        tqc=replace(config.tqc, learning_starts=16),
    )
    with pytest.raises(ValueError, match="reward settings differ"):
        TrainingRunner(paced).run(resume=champion)
    events: list[dict] = []
    TrainingRunner(paced, events.append).run(
        resume=champion, fresh_replay=True, rollback_to_champion=True
    )
    assert registry.read_metadata(champion).training_config["rewards"] == paced.to_dict()["rewards"]
    assert (champion / "replay.pkl").is_file()
    assert any(event["type"] == "champion_replay" for event in events)


def test_old_tqc_reward_semantics_rejects_replay_even_with_same_coefficients(tmp_path) -> None:
    config = configuration(tmp_path, "tqc")
    TrainingRunner(config).run()
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion")
    metadata = registry.read_metadata(champion)
    registry.write_metadata(champion, replace(metadata, reward_semantics=None))
    with pytest.raises(ValueError, match="reward settings differ"):
        TrainingRunner(replace(config, timesteps=4, checkpoint_interval=0)).run(
            resume=champion, pace_polish=True
        )


def test_tqc_champion_anchor_caps_actor_action_drift() -> None:
    config = TrainingConfig(algorithm="tqc", tqc=TQCConfig(architecture="tiny"))
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        observation, _ = env.reset(seed=42)
        batch = th.as_tensor(np.stack([observation] * 16), device=model.device)
        model.anchor_to_current_policy(0.03)
        model._champion_observations = batch
        with th.no_grad():
            reference = model.actor(batch, deterministic=True).clone()
            model.actor.mu.bias.add_(th.tensor([1.0, -1.0]))
            assert (model.actor(batch, deterministic=True) - reference).abs().amax() > 0.03
        model._enforce_actor_anchor()
        with th.no_grad():
            drift = (model.actor(batch, deterministic=True) - reference).abs().amax().item()
        assert drift <= 0.0301
        assert model._anchor_action_drift <= 0.0301
    finally:
        env.close()


def test_speed_search_policy_cannot_skip_critic_adaptation_on_resume(tmp_path) -> None:
    config = configuration(tmp_path, "tqc")
    TrainingRunner(config).run()
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion")
    metadata = registry.read_metadata(champion)
    registry.write_metadata(champion, replace(metadata, critic_adaptation_required=True))
    with pytest.raises(ValueError, match="requires critic adaptation"):
        TrainingRunner(config).run(resume=champion, pace_polish=True)


def test_tqc_speed_bias_schedule_applies_only_inside_window(tmp_path) -> None:
    config = TrainingConfig(algorithm="tqc", tqc=TQCConfig(architecture="tiny"))
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        observation, _ = env.reset(seed=42)
        inside = observation.copy()
        inside[12] = 0.5
        outside = observation.copy()
        outside[12] = 0.1
        baseline, _ = model.predict(inside, deterministic=True)
        outside_baseline, _ = model.predict(outside, deterministic=True)
        model.speed_bias_schedule = [(0.3, 0.7, 0.2)]
        adjusted, _ = model.predict(inside, deterministic=True)
        outside_adjusted, _ = model.predict(outside, deterministic=True)
        assert adjusted[0] == baseline[0]
        assert adjusted[1] == pytest.approx(min(1.0, baseline[1] + 0.2))
        np.testing.assert_array_equal(outside_adjusted, outside_baseline)
        directory = tmp_path / "scheduled-policy"
        backend.save_model(model, directory)
        restored = backend.load_model(directory / "policy.zip", env, "cpu")
        assert restored.speed_bias_schedule == [list(window) for window in model.speed_bias_schedule]
        np.testing.assert_allclose(restored.predict(inside, deterministic=True)[0], adjusted)
    finally:
        env.close()


def test_tqc_replay_expansion_and_critic_only_freeze_actor_and_entropy() -> None:
    config = TrainingConfig(algorithm="tqc", device="cpu", tqc=TQCConfig(
        architecture="tiny", batch_size=8, learning_starts=100, replay_capacity=128,
    ))
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        model._adaptation_mode = "replay_expansion"
        model._adaptation_noise = np.asarray([0.01, 0.01], dtype=np.float32)
        actor_before = [parameter.detach().clone() for parameter in model.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in model.critic.parameters()]
        model.learn(16, progress_bar=False)
        assert all(th.equal(old, new) for old, new in zip(actor_before, model.actor.parameters(), strict=True))
        assert all(th.equal(old, new) for old, new in zip(critic_before, model.critic.parameters(), strict=True))
        assert model.replay_buffer.size() == 16
        assert model._adaptation_action_deviation
        assert float(np.mean(model._adaptation_action_deviation)) > 0.0

        entropy_before = model.log_ent_coef.detach().clone()
        actor_before = [parameter.detach().clone() for parameter in model.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in model.critic.parameters()]
        target_before = [parameter.detach().clone() for parameter in model.critic_target.parameters()]
        stats = model.train_critics(4, 8)
        assert all(th.equal(old, new) for old, new in zip(actor_before, model.actor.parameters(), strict=True))
        assert any(not th.equal(old, new) for old, new in zip(critic_before, model.critic.parameters(), strict=True))
        assert any(
            not th.equal(old, new)
            for old, new in zip(target_before, model.critic_target.parameters(), strict=True)
        )
        assert th.equal(entropy_before, model.log_ent_coef.detach())
        assert stats["critic_loss"] >= 0 and "q_perturbed_action_mean" in stats
    finally:
        env.close()


def test_tqc_independent_actor_and_critic_learning_rates() -> None:
    config = TrainingConfig(algorithm="tqc", device="cpu", tqc=TQCConfig(
        architecture="tiny", learning_rate=1e-4,
        actor_learning_rate=1e-6, critic_learning_rate=5e-5,
    ))
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend_for("tqc").action_adapter(config))
    try:
        model = backend_for("tqc").create_model(config, env, "cpu")
        assert model.actor.optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
        assert model.critic.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
        model._logger = Mock()
        model._update_learning_rate([
            model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer,
        ])
        assert model.actor.optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)
        assert model.critic.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)
        assert model.ent_coef_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    finally:
        env.close()


def test_adaptation_candidate_gate_checks_closed_loop_drift_and_speed() -> None:
    config = TrainingConfig(algorithm="tqc", tqc=TQCConfig(architecture="tiny"))
    reference = EvaluationResult(5, 1.0, 1.0, 1.0, 24.888, 24.888, 0.0, 0.0, 0.0)
    acceptable = replace(
        reference, best_lap_s=24.8885, median_lap_s=24.8885,
        max_position_deviation_m=0.2, max_progress_deviation_m=0.1,
        max_heading_deviation_rad=0.02,
        max_steering_disagreement=0.001, max_longitudinal_disagreement=0.002,
    )
    passed, detail = candidate_diagnostics_pass(acceptable, reference, config)
    assert passed and detail["lap_delta_s"] == pytest.approx(0.0005)
    assert detail["rejection_reasons"] == []
    paired = replace(acceptable, median_lap_s=24.90, lap_time_delta_s=-0.001)
    passed, detail = candidate_diagnostics_pass(paired, reference, config)
    assert passed and detail["lap_delta_s"] == pytest.approx(-0.001)

    unsafe = replace(acceptable, off_track_rate=0.2, max_position_deviation_m=6.0)
    passed, detail = candidate_diagnostics_pass(unsafe, reference, config)
    assert not passed
    assert "crash, off-track, or stall" in detail["rejection_reasons"]
    assert "position drift" in detail["rejection_reasons"]
    slower = replace(acceptable, median_lap_s=24.90)
    passed, detail = candidate_diagnostics_pass(slower, reference, config)
    assert not passed and "lap-time regression" in detail["rejection_reasons"]
    invalid = replace(acceptable, median_lap_s=float("nan"))
    passed, detail = candidate_diagnostics_pass(invalid, reference, config)
    assert not passed
    assert "non-finite evaluation metrics" in detail["rejection_reasons"]


def test_adaptation_source_digest_includes_effective_policy_overlays(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory, amount in ((first, 0.01), (second, 0.02)):
        (directory / "policy.zip").write_bytes(b"same network weights")
        (directory / "metadata.json").write_text(json.dumps({
            "policy_overlays": [{"kind": "drive_bias", "amount": amount}],
        }), encoding="utf-8")
    assert _policy_digest(first) != _policy_digest(second)


def test_local_replay_expansion_noise_default_is_sparse() -> None:
    assert TQCConfig().adaptation_noise_probability == pytest.approx(0.0001)


def _write_mock_tqc_champion(
    config: TrainingConfig, *, initial_replay_steps: int = 0,
) -> tuple[TrainingRunner, EvaluationResult]:
    runner = TrainingRunner(config)
    runner.device = resolve_device("cpu", algorithm="tqc")
    env = runner._environment()
    try:
        runner.model = runner.backend.create_model(config, env, "cpu")
        if initial_replay_steps:
            runner.model.learn(initial_replay_steps, reset_num_timesteps=False, progress_bar=False)
        runner.model.critic_adaptation_required = True
        runner.model.policy_overlays = [{
            "kind": "drive_bias", "start": 0.2, "end": 0.4,
            "amount": 0.001, "taper": 0.01,
        }]
        result = EvaluationResult(5, 1.0, 1.0, 1.0, 10.0, 10.0, 0.0, 0.0, 0.0)
        champion = runner.registry.slot(config.track_name, "tqc", "champion")
        runner.backend.save_model(runner.model, champion, resume=True)
        runner.registry.write_metadata(champion, runner._metadata(result))
        return runner, result
    finally:
        env.close()


def test_full_adaptation_requires_enough_new_replay_for_critic_batch(tmp_path) -> None:
    import polybot.training.adaptation as adaptation

    base = configuration(tmp_path, "tqc")
    config = replace(base, tqc=replace(base.tqc, adaptation_replay_steps=7, batch_size=8))
    with pytest.raises(ValueError, match="at least one critic batch"):
        adaptation.run_adaptation(config, "full")


def test_three_stage_adaptation_collects_then_critic_updates_atomically(
    tmp_path, monkeypatch,
) -> None:
    import polybot.training.adaptation as adaptation

    base = configuration(tmp_path, "tqc")
    config = replace(
        base, backend="mock", track_name="Mock straight", track_id="mock/straight",
        evaluation=EvaluationConfig(16, 2),
        tqc=replace(
            base.tqc, batch_size=8, replay_capacity=128,
            adaptation_replay_steps=16, adaptation_noise_probability=0.2,
            critic_adaptation_updates=4,
        ),
    )
    runner, result = _write_mock_tqc_champion(config, initial_replay_steps=8)
    champion = runner.registry.slot(config.track_name, "tqc", "champion")
    original_actor = [parameter.detach().clone() for parameter in runner.model.actor.parameters()]
    original_entropy = runner.model.log_ent_coef.detach().clone()
    original_critic = [parameter.detach().clone() for parameter in runner.model.critic.parameters()]
    monkeypatch.setattr(adaptation, "evaluate_model", lambda *args, **kwargs: result)
    events = []
    adaptation.run_adaptation(config, "full", events.append)

    metadata = runner.registry.read_metadata(champion)
    env = runner._environment()
    try:
        updated = runner.backend.load_model(champion / "policy.zip", env, "cpu", resume=True)
    finally:
        env.close()
    assert metadata.adaptation_stage == "critics_adapted"
    assert not metadata.critic_adaptation_required
    assert metadata.policy_overlays == updated.policy_overlays
    assert any(
        not th.equal(old, new)
        for old, new in zip(original_critic, updated.critic.parameters(), strict=True)
    )
    assert all(
        th.equal(old, new)
        for old, new in zip(original_actor, updated.actor.parameters(), strict=True)
    )
    assert th.equal(original_entropy, updated.log_ent_coef.detach())
    assert updated.replay_buffer.size() == 24
    assert (champion.parent / "champion-backup-1" / "metadata.json").is_file()
    assert not (champion.parent / "adaptation" / "working" / "working-source.json").exists()
    history = [json.loads(line) for line in (champion.parent / "pace-history.jsonl").read_text().splitlines()]
    assert history[-1]["source"] == "critic_adaptation"
    stages = [event.get("stage") for event in events if event.get("type") == "adaptation_stage"]
    assert stages == ["replay_expansion", "critic_adaptation", "candidate_promotion"]


def test_rejected_critic_candidate_keeps_champion_and_validated_replay_unchanged(
    tmp_path, monkeypatch,
) -> None:
    import polybot.training.adaptation as adaptation

    base = configuration(tmp_path, "tqc")
    config = replace(
        base, backend="mock", track_name="Mock straight", track_id="mock/straight",
        evaluation=EvaluationConfig(16, 2),
        tqc=replace(
            base.tqc, batch_size=8, replay_capacity=128,
            adaptation_replay_steps=16, adaptation_noise_probability=0.2,
            critic_adaptation_updates=4,
        ),
    )
    runner, result = _write_mock_tqc_champion(config)
    champion = runner.registry.slot(config.track_name, "tqc", "champion")
    original_policy = (champion / "policy.zip").read_bytes()
    unsafe = replace(result, off_track_rate=0.5, max_position_deviation_m=10.0)
    evaluations = iter((result, unsafe))
    monkeypatch.setattr(adaptation, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    with pytest.raises(RuntimeError, match="candidate rejected"):
        adaptation.run_adaptation(config, "full")

    metadata = runner.registry.read_metadata(champion)
    work = champion.parent / "adaptation" / "working"
    assert (champion / "policy.zip").read_bytes() == original_policy
    assert metadata.critic_adaptation_required
    assert runner.registry.read_metadata(work).adaptation_stage == "replay_expanded"
    assert (work / "working-source.json").is_file()


def test_final_candidate_gate_rolls_back_without_touching_tuned_champion(tmp_path, monkeypatch) -> None:
    import polybot.training.adaptation as adaptation

    base = configuration(tmp_path, "tqc")
    config = replace(
        base, backend="mock", track_name="Mock straight", track_id="mock/straight",
        evaluation=EvaluationConfig(16, 2),
        tqc=replace(
            base.tqc, batch_size=8, replay_capacity=128,
            adaptation_replay_steps=16, adaptation_noise_probability=0.2,
            critic_adaptation_updates=4,
        ),
    )
    runner, result = _write_mock_tqc_champion(config)
    champion = runner.registry.slot(config.track_name, "tqc", "champion")
    original_policy = (champion / "policy.zip").read_bytes()
    unsafe = replace(result, off_track_rate=0.2, max_position_deviation_m=8.0)
    evaluations = iter((result, result, unsafe))
    monkeypatch.setattr(adaptation, "evaluate_model", lambda *args, **kwargs: next(evaluations))

    with pytest.raises(RuntimeError, match="final closed-loop gate rejected"):
        adaptation.run_adaptation(config, "full")

    metadata = runner.registry.read_metadata(champion)
    work = champion.parent / "adaptation" / "working"
    assert (champion / "policy.zip").read_bytes() == original_policy
    assert metadata.critic_adaptation_required
    assert runner.registry.read_metadata(work).adaptation_stage == "critics_validated"
    assert (work / "working-source.json").is_file()


def test_three_adaptation_stages_can_resume_between_critic_and_promotion(tmp_path, monkeypatch) -> None:
    import polybot.training.adaptation as adaptation

    base = configuration(tmp_path, "tqc")
    config = replace(
        base, backend="mock", track_name="Mock straight", track_id="mock/straight",
        evaluation=EvaluationConfig(16, 2),
        tqc=replace(
            base.tqc, batch_size=8, replay_capacity=128,
            adaptation_replay_steps=16, adaptation_noise_probability=0.2,
            critic_adaptation_updates=4,
        ),
    )
    runner, result = _write_mock_tqc_champion(config)
    champion = runner.registry.slot(config.track_name, "tqc", "champion")
    source_policy = (champion / "policy.zip").read_bytes()
    monkeypatch.setattr(adaptation, "evaluate_model", lambda *args, **kwargs: result)

    adaptation.run_adaptation(config, "collect")
    adaptation.run_adaptation(config, "critics")
    work = champion.parent / "adaptation" / "working"
    assert (champion / "policy.zip").read_bytes() == source_policy
    assert runner.registry.read_metadata(work).adaptation_stage == "critics_validated"
    assert (work / "working-source.json").is_file()

    adaptation.run_adaptation(config, "promote")
    promoted = runner.registry.read_metadata(champion)
    assert promoted.adaptation_stage == "critics_adapted"
    assert not promoted.critic_adaptation_required
    assert (champion.parent / "champion-backup-1" / "metadata.json").is_file()


def test_closed_loop_evaluation_detects_compounding_drift_despite_small_action_delta() -> None:
    class Model:
        def __init__(self, steer: float) -> None:
            self.steer = steer
            self.policy = self

        def set_training_mode(self, _training: bool) -> None:
            pass

        def predict(self, _observation, deterministic=True):
            return np.asarray([self.steer, 0.0], dtype=np.float32), None

    class Env:
        def reset(self, seed=None):
            self.step_count = 0
            self.lateral = 0.0
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action):
            self.step_count += 1
            self.lateral += float(action[0]) * 10.0
            info = {
                "route_progress_m": float(self.step_count), "track_length_m": 10.0,
                "position_m": (float(self.step_count), self.lateral, 0.0),
                "heading_error_rad": self.lateral * 0.01,
                "local_velocity_mps": (0.0, 0.0, 20.0), "elapsed_s": self.step_count * 0.1,
                "events": ("finish",) if self.step_count == 10 else (),
            }
            return np.zeros(1, dtype=np.float32), 0.0, self.step_count == 10, False, info

        def close(self):
            pass

    result = evaluate_model(Model(0.01), Env, episodes=1, seed=2, reference_model=Model(0.0))
    assert result.max_steering_disagreement == pytest.approx(0.01)
    assert result.max_position_deviation_m == pytest.approx(1.0)


def test_sector_delta_map_and_airborne_region_discovery() -> None:
    def sample(progress, elapsed, *, airborne=False, speed=20.0):
        contacts = (0.0, 0.0, 0.0, 0.0) if airborne else (1.0, 1.0, 1.0, 1.0)
        return {
            "route_progress_m": progress * 100.0, "track_length_m": 100.0,
            "elapsed_s": elapsed, "speed_mps": speed,
            "wheel_contacts": contacts, "airborne": airborne,
            "position_m": (progress, 0.0, 1.0), "local_velocity_mps": (0, 0, speed),
            "quaternion_xyzw": (0, 0, 0, 1),
        }
    champion = [sample(0.0, 0.0), sample(0.5, 5.0), sample(1.0, 10.0)]
    candidate = [sample(0.0, 0.0), sample(0.5, 4.8), sample(1.0, 9.8)]
    splits = sector_delta_map(champion, candidate, 0.1)
    assert len(splits) == 10
    assert all(row["end"] > row["start"] for row in splits)
    assert splits[0]["delta_s"] == pytest.approx(-0.04)
    assert splits[0]["cumulative_delta_s"] == pytest.approx(-0.04)
    trace = [sample(0.0, 0.0), sample(0.2, 1.0), sample(0.25, 1.1, airborne=True),
             sample(0.3, 1.2, airborne=True), sample(0.35, 1.3)]
    regions = discover_airborne_regions([trace])
    assert len(regions) == 1
    assert regions[0]["start"] == pytest.approx(0.25)
    assert regions[0]["end"] == pytest.approx(0.3)
    repeated = discover_airborne_regions([trace, trace])
    assert len(repeated) == 1
    assert repeated[0]["lap_count"] == 2
    unfinished = discover_airborne_regions([[
        sample(0.2, 1.0), sample(0.25, 1.1, airborne=True),
        sample(0.3, 1.2, airborne=True),
    ]])
    assert unfinished[0]["landed"] == 0.0


def test_tqc_search_overlays_are_smooth_and_airbrake_requires_all_wheels_airborne() -> None:
    config = TrainingConfig(algorithm="tqc", device="cpu", tqc=TQCConfig(architecture="tiny"))
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        observation, _ = env.reset(seed=7)
        observation[12] = 0.5
        baseline, _ = model.predict(observation, deterministic=True)
        model.policy_overlays = [{"kind": "steer_bias", "start": 0.4, "end": 0.6,
                                  "amount": 0.002, "taper": 0.02}]
        center, _ = model.predict(observation, deterministic=True)
        assert center[0] - baseline[0] == pytest.approx(0.002)
        observation[12] = 0.4
        edge_baseline, _ = model.predict(observation, deterministic=True)
        edge, _ = model.predict(observation, deterministic=True)
        assert edge[0] - edge_baseline[0] == pytest.approx(0.0, abs=1e-7)

        model.policy_overlays = [{"kind": "air_brake", "start": 0.4, "end": 0.6,
                                  "duty": 0.05, "taper": 0.01}]
        model.policy_overlays = [{"kind": "air_brake", "start": 0.4, "end": 0.6,
                                  "duty": 0.05, "taper": 0.01}]
        observation[12] = 0.5
        observation[17:21] = 0.0
        airborne, _ = model.predict(observation, deterministic=True)
        assert model._air_brake_active and airborne[1] == pytest.approx(-0.05)
        outside = observation.copy()
        outside[12] = 0.3
        batched, _ = model.predict(np.stack((observation, outside)), deterministic=True)
        individual, _ = model.predict(outside, deterministic=True)
        np.testing.assert_allclose(batched[0], airborne)
        np.testing.assert_allclose(batched[1], individual)
        observation[19] = 1.0
        model.policy_overlays = []
        grounded_base, _ = model.predict(observation, deterministic=True)
        model.policy_overlays = [{"kind": "air_brake", "start": 0.4, "end": 0.6,
                                  "duty": 0.05, "taper": 0.01}]
        grounded, _ = model.predict(observation, deterministic=True)
        assert not model._air_brake_active
        np.testing.assert_allclose(grounded, grounded_base)

        for layer, index, value, operation in (
            ({"kind": "steer_gain", "amount": 1.005}, 0, 1.005, "gain"),
            ({"kind": "drive_bias", "amount": 0.005}, 1, 0.005, "bias"),
            ({"kind": "drive_gain", "amount": 0.995}, 1, 0.995, "gain"),
        ):
            observation[12] = 0.5
            model.policy_overlays = []
            original, _ = model.predict(observation, deterministic=True)
            model.policy_overlays = [{**layer, "start": 0.4, "end": 0.6, "taper": 0.01}]
            actual, _ = model.predict(observation, deterministic=True)
            expected = (original[index] * value if operation == "gain"
                        else original[index] + value)
            assert actual[index] == pytest.approx(expected)
    finally:
        env.close()


def test_tqc_overlay_transform_matches_live_policy_and_keeps_gradients(tmp_path) -> None:
    config = TrainingConfig(algorithm="tqc", device="cpu", tqc=TQCConfig(architecture="tiny"))
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        observations = np.stack([env.reset(seed=index)[0] for index in (3, 4)]).astype(np.float32)
        observations[:, 12] = 0.5
        observations[0, 17:21] = 0.0
        observations[1, 17:21] = 1.0
        model.speed_bias_schedule = [(0.3, 0.7, 0.2)]
        model.policy_overlays = [
            {"kind": "steer_bias", "start": 0.4, "end": 0.6, "amount": 0.002, "taper": 0.02},
            {"kind": "drive_gain", "start": 0.4, "end": 0.6, "amount": 1.1, "taper": 0.02},
            {"kind": "air_brake", "start": 0.4, "end": 0.6, "duty": 0.05, "taper": 0.01},
        ]
        obs = th.as_tensor(observations)
        raw = model.actor(obs, deterministic=True)
        training_actions = model._apply_overlays_to_actions(raw, obs).detach().numpy()
        live_actions = np.stack([model.predict(row, deterministic=True)[0] for row in observations])
        np.testing.assert_allclose(training_actions, live_actions, atol=1e-6)

        base = th.tensor([[0.2, 0.1], [0.2, 0.1]], requires_grad=True)
        transformed = model._apply_overlays_to_actions(base, obs)
        transformed.sum().backward()
        assert base.grad is not None
        assert base.grad[1, 1].abs() > 0
        assert base.grad[0, 0].abs() > 0
    finally:
        env.close()


def test_tqc_training_uses_post_overlay_actions_for_actor_and_target_critics(tmp_path) -> None:
    config = replace(
        configuration(tmp_path, "tqc"), timesteps=24,
        tqc=replace(configuration(tmp_path, "tqc").tqc, learning_starts=8,
                    train_frequency=1, batch_size=8, replay_capacity=128),
    )
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight",
                       action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        model.policy_overlays = [{
            "kind": "drive_bias", "start": 0.1, "end": 0.9, "amount": 0.01, "taper": 0.02,
        }]
        model.speed_bias_schedule = [(0.1, 0.9, 0.01)]
        model.learn(total_timesteps=24, progress_bar=False)
        assert model._n_updates > 0
        assert np.isfinite(model.logger.name_to_value["train/actor_loss"])
        assert np.isfinite(model.logger.name_to_value["train/critic_loss"])
    finally:
        env.close()


def test_wr_search_uses_small_coordinate_candidates_and_micro_confirmation() -> None:
    rows = _candidate_grid([(0.4, 0.45)], [], family="steering")
    assert rows[0]["kind"] == "steer_bias"
    assert rows[0]["amount"] == pytest.approx(-0.001)
    assert max(abs(row["amount"]) for row in rows if row["kind"] == "steer_bias") == 0.01
    assert confirmation_count_for_gain(
        0.009, extra_confirmation_threshold_s=0.01,
        minimum_confirmation_episodes=5, micro_confirmation_episodes=10,
    ) == 10
    assert confirmation_count_for_gain(
        0.01, extra_confirmation_threshold_s=0.01,
        minimum_confirmation_episodes=5, micro_confirmation_episodes=10,
    ) == 5
    first = {"kind": "steer_bias", "start": 0.4, "end": 0.45, "amount": -0.001}
    second = {**first, "amount": 0.001}
    independent = {"kind": "drive_gain", "start": 0.4, "end": 0.45, "amount": 1.002}
    stack = compose_overlay_stack([first, independent], second)
    assert stack == [independent, second]


def test_air_brake_search_includes_sustained_and_held_control_duties() -> None:
    region = {"start": 0.68, "end": 0.81, "duration_s": 2.97, "landed": 1.0}
    candidates = _candidate_grid([(0.68, 0.81)], [region], family="air_brake")
    assert [candidate["duty"] for candidate in candidates] == [
        0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0,
    ]
    previous = {"kind": "air_brake", "start": 0.6885, "end": 0.8126, "duty": 0.02}
    refined = {"kind": "air_brake", "start": 0.6886, "end": 0.8117, "duty": 0.2}
    assert compose_overlay_stack([previous], refined) == [refined]
    global_hold = {"kind": "air_brake", "start": 0.0, "end": 1.0, "duty": 1.0}
    assert compose_overlay_stack([previous, refined], global_hold) == [global_hold]
    candidates = _candidate_grid(
        [(0.2, 0.25), (0.7, 0.75)], [region], family="air_brake",
        include_global_air_brake=True,
    )
    assert candidates[0] == {
        "kind": "air_brake", "start": 0.0, "end": 1.0, "duty": 1.0, "taper": 0.003,
    }


def test_section_optimizer_windows_refine_only_prioritized_regions(tmp_path) -> None:
    coarse = section_windows(0, [])
    assert len(coarse) == 10
    assert coarse[0] == (0.0, 0.1) and coarse[-1] == (0.9, 1.0)
    refined = section_windows(1, [
        {"start": 0.5, "end": 0.6, "priority": 1},
        {"start": 0.2, "end": 0.3, "priority": 3},
    ])
    assert refined == [(0.2, 0.25), (0.25, 0.3), (0.5, 0.55), (0.55, 0.6)]
    state_path = tmp_path / "optimizer-state.json"
    state = {"section_index": 3, "candidate_index": 7, "champion_hash": "abc"}
    write_checkpoint(state_path, state)
    assert json.loads(state_path.read_text(encoding="utf-8")) == state
    assert list(tmp_path.glob("*.tmp")) == []


def test_section_optimizer_combo_mode_pairs_steering_and_drive() -> None:
    optimizer = object.__new__(SectionOptimizer)
    optimizer.candidate_mode = "combo"
    candidates = optimizer._build_candidates(0.2, 0.3)

    assert len(candidates) == 16
    assert all(len(candidate) == 2 for candidate in candidates)
    assert all({layer["kind"] for layer in candidate} == {"steer_bias", "drive_bias"}
               for candidate in candidates)
    assert all(layer["start"] == 0.2 and layer["end"] == 0.3
               for candidate in candidates for layer in candidate)
    assert len({(candidate[0]["amount"], candidate[1]["amount"])
                for candidate in candidates}) == 16


def test_adaptive_section_candidates_follow_profile_then_narrow(tmp_path) -> None:
    profile = {
        "arms": [{
            "id": "late_drive", "section": [0.75, 1.0],
            "kind": "speed_bias_schedule", "anchor": 0.0, "step": 0.01,
            "values": [0.035, 0.045, 0.052, 0.060, 0.070],
        }],
    }
    profile_path = tmp_path / "adaptive.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    optimizer = object.__new__(SectionOptimizer)
    optimizer.candidate_mode = "adaptive"
    optimizer.adaptive_arms = json.loads(profile_path.read_text())["arms"]
    optimizer.state = {"level_index": 0, "adaptive_memory": {}}
    initial = optimizer._build_candidates(0.75, 1.0)
    assert [row["amount"] for row in initial] == profile["arms"][0]["values"]
    optimizer.state["level_index"] = 1
    optimizer.state["adaptive_memory"]["late_drive"] = {"best_parameter_value": 0.052}
    refined = optimizer._build_candidates(0.75, 0.8)
    assert [row["amount"] for row in refined] == pytest.approx([0.042, 0.047, 0.052, 0.057, 0.062])
    assert all(row["kind"] == "speed_bias_schedule" for row in refined)


@pytest.mark.parametrize(
    ("parameter", "values", "window_start", "window_end", "expected_windows"),
    [
        ("start", [0.6786, 0.6836, 0.6886, 0.6936, 0.6986], 0.6886, 0.8117,
         [(value, 0.8117) for value in [0.6786, 0.6836, 0.6886, 0.6936, 0.6986]]),
        ("end", [0.8017, 0.8067, 0.8117, 0.8167, 0.8217], 0.6886, 0.8117,
         [(0.6886, value) for value in [0.8017, 0.8067, 0.8117, 0.8167, 0.8217]]),
    ],
)
def test_adaptive_air_brake_timing_candidates_shift_one_boundary(
    parameter, values, window_start, window_end, expected_windows,
) -> None:
    section = [0.65, 0.75] if parameter == "start" else [0.75, 0.85]
    arm = {
        "id": f"air_brake_{parameter}", "section": section, "kind": "air_brake",
        "parameter": parameter, "anchor": window_start if parameter == "start" else window_end,
        "step": 0.005, "values": values, "window_start": window_start,
        "window_end": window_end, "window_taper": 0.003,
        "fixed_parameters": {"duty": 1.0},
    }
    optimizer = object.__new__(SectionOptimizer)
    optimizer.candidate_mode = "adaptive"
    optimizer.adaptive_arms = [arm]
    optimizer.state = {"level_index": 0, "adaptive_memory": {}}

    candidates = optimizer._build_candidates(*section)

    assert [(row["start"], row["end"]) for row in candidates] == pytest.approx(expected_windows)
    assert all(row["kind"] == "air_brake" and row["duty"] == 1.0
               and row["taper"] == 0.003 for row in candidates)


def test_rejected_adaptive_schedule_trial_restores_schedule() -> None:
    class Model:
        policy_overlays = [{"kind": "air_brake", "start": 0.6, "end": 0.8, "duty": 1.0}]
        speed_bias_schedule = [(0.1, 0.2, 0.03)]

    baseline = EvaluationResult(1, 1.0, 1.0, 1.0, 24.263, 24.263, 0.0, 0.0, 0.0)
    screen = replace(baseline, median_lap_s=24.2, best_lap_s=24.2)
    failed_confirmation = replace(baseline, median_lap_s=24.3, best_lap_s=24.3)
    outcomes = iter((screen, failed_confirmation))
    optimizer = object.__new__(SectionOptimizer)
    optimizer.model = Model()
    optimizer.lap_traces = []
    optimizer.state = {
        "champion_lap_s": 24.263, "timing_floor_s": 0.001, "total_laps": 0,
        "total_trials": 0, "section_scores": {}, "rejected_candidates": [],
        "seed": 1,
    }
    optimizer.candidate_mode = "single"
    optimizer.evaluate = lambda *args, **kwargs: next(outcomes)
    optimizer.should_stop = lambda: False
    optimizer.emit = lambda *args, **kwargs: None
    optimizer.checkpoint = lambda: None
    accepted = optimizer._run_trial(0.75, 1.0, {
        "kind": "speed_bias_schedule", "start": 0.75, "end": 1.0, "amount": 0.052,
    })
    assert accepted is False
    assert optimizer.model.speed_bias_schedule == [(0.1, 0.2, 0.03)]
    assert optimizer.model.policy_overlays == Model.policy_overlays


def test_tqc_policy_overlay_survives_checkpoint_save_and_reload(tmp_path) -> None:
    config = TrainingConfig(algorithm="tqc", device="cpu", tqc=TQCConfig(architecture="tiny"))
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        model.policy_overlays = [{
            "kind": "air_brake", "start": 0.68, "end": 0.81, "duty": 0.02, "taper": 0.003,
        }]
        slot = tmp_path / "saved-model"
        backend.save_model(model, slot, resume=False)
        loaded = backend.load_model(slot / "policy.zip", env, "cpu")
        assert loaded.policy_overlays == model.policy_overlays
    finally:
        env.close()


def test_continue_best_stops_after_repeated_regressions(tmp_path, monkeypatch) -> None:
    import polybot.training.runner as runner_module

    config = replace(
        configuration(tmp_path, "tqc"), timesteps=80,
        evaluation=EvaluationConfig(16, 1),
    )
    strong = EvaluationResult(1, 1.0, 1.0, 1.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    weak = EvaluationResult(1, 0.0, 0.2, 0.2, None, None, 0.0, 1.0, 0.0)
    evaluations = iter((strong, weak, weak, weak))
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run(rollback_to_champion=True)
    assert ModelRegistry(config.output_root).read_metadata(latest).training_timesteps == 64
    assert any(event["type"] == "regression_stop" for event in events)
    assert events[-1]["type"] == "stopped"


def test_ppo_regression_restores_consistent_champion_and_stops_after_three_weaker_evals(
    tmp_path, monkeypatch,
) -> None:
    import polybot.training.runner as runner_module

    base = configuration(tmp_path, "ppo")
    config = replace(
        base, timesteps=160, evaluation=EvaluationConfig(32, 1), checkpoint_interval=0,
    )
    strong = EvaluationResult(1, 1.0, 1.0, 1.0, 24.263, 24.263, 0.0, 0.0, 0.0)
    weak = EvaluationResult(1, 0.0, 0.2, 0.2, None, None, 1.0, 0.0, 0.0)
    evaluations = iter((strong, weak, weak, weak))
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run(rollback_to_champion=True)
    metadata = ModelRegistry(config.output_root).read_metadata(latest)
    assert metadata.training_timesteps == 128
    assert metadata.evaluation["finish_rate"] == 1.0
    assert sum(event["type"] == "ppo_champion_restore" for event in events) == 3
    restore_events = [event for event in events if event["type"] == "ppo_champion_restore"]
    assert [event["rng_seed"] for event in restore_events] == [
        (config.seed + event["timesteps"]) % (2**32 - 1)
        for event in restore_events
    ]
    assert len({event["rng_seed"] for event in restore_events}) == len(restore_events)
    assert any(event["type"] == "regression_stop" for event in events)
    assert events[-1]["type"] == "stopped"


def test_ppo_small_progress_dips_keep_learning_but_preserve_champion(tmp_path, monkeypatch) -> None:
    import polybot.training.runner as runner_module

    base = configuration(tmp_path, "ppo")
    config = replace(
        base, timesteps=96, evaluation=EvaluationConfig(32, 1), checkpoint_interval=0,
    )
    strong = EvaluationResult(1, 0.0, 0.50, 0.50, None, None, 0.0, 0.0, 0.0)
    slightly_weaker = EvaluationResult(
        1, 0.0, 0.45, 0.45, None, None, 0.0, 0.0, 0.0,
    )
    evaluations = iter((strong, slightly_weaker, slightly_weaker))
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run(
        rollback_to_champion=True, ppo_rollback_progress_tolerance=0.10,
    )

    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "ppo", "champion")
    assert registry.read_metadata(champion).evaluation["median_progress"] == 0.50
    latest_metadata = registry.read_metadata(latest)
    assert latest_metadata.training_timesteps == 96
    assert latest_metadata.evaluation["median_progress"] == 0.45
    assert not any(event["type"] == "ppo_champion_restore" for event in events)


def test_slower_complete_laps_do_not_stop_best_model_training(tmp_path, monkeypatch) -> None:
    import polybot.training.runner as runner_module

    config = replace(
        configuration(tmp_path, "tqc"), timesteps=80,
        evaluation=EvaluationConfig(16, 1),
    )
    strong = EvaluationResult(1, 1.0, 1.0, 1.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    slower = replace(strong, best_lap_s=20.1, median_lap_s=20.1)
    evaluations = iter((strong, slower, slower, slower, slower))
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run(rollback_to_champion=True)
    assert ModelRegistry(config.output_root).read_metadata(latest).training_timesteps == 80
    assert sum(event["type"] == "rollback" for event in events) == 4
    assert all(not event["severe"] for event in events if event["type"] == "rollback")
    assert not any(event["type"] == "regression_stop" for event in events)
    assert events[-1]["type"] == "completed"


def test_near_champion_laps_keep_learning_without_replacing_champion(tmp_path, monkeypatch) -> None:
    import polybot.training.runner as runner_module

    base = configuration(tmp_path, "tqc")
    config = replace(
        base, timesteps=48, evaluation=EvaluationConfig(16, 1),
        tqc=replace(base.tqc, champion_lap_tolerance_s=0.2),
    )
    strong = EvaluationResult(1, 1.0, 1.0, 1.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    slower = replace(strong, best_lap_s=20.1, median_lap_s=20.1)
    evaluations = iter((strong, slower, slower))
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run(rollback_to_champion=True)
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion")
    assert registry.read_metadata(champion).evaluation["median_lap_s"] == 20.0
    assert registry.read_metadata(latest).evaluation["median_lap_s"] == 20.1
    assert sum(event["type"] == "lap_tolerance" for event in events) == 2
    assert not any(event["type"] == "rollback" for event in events)


@pytest.mark.parametrize("timesteps", (24, 32))
def test_continue_best_restores_champion_after_weaker_evaluation(
    tmp_path, monkeypatch, timesteps: int
) -> None:
    import polybot.training.runner as runner_module

    config = replace(
        configuration(tmp_path, "tqc"), timesteps=timesteps,
        evaluation=EvaluationConfig(16, 1),
    )
    strong = EvaluationResult(1, 1.0, 1.0, 1.0, 20.0, 20.0, 0.0, 0.0, 0.0)
    weak = EvaluationResult(1, 0.0, 0.2, 0.2, None, None, 0.0, 1.0, 0.0)
    evaluations = iter((strong, weak))
    monkeypatch.setattr(runner_module, "evaluate_model", lambda *args, **kwargs: next(evaluations))
    events: list[dict] = []
    latest = TrainingRunner(config, events.append).run(rollback_to_champion=True)
    registry = ModelRegistry(config.output_root)
    champion = registry.slot(config.track_name, "tqc", "champion")
    assert (champion / "replay.pkl").is_file()
    assert registry.read_metadata(champion).training_timesteps == 16
    assert registry.read_metadata(latest).training_timesteps == timesteps
    assert registry.read_metadata(latest).evaluation is None
    assert any(event["type"] == "rollback" and event["replay_source"] == "champion"
               for event in events)


def test_dqn_exploration_follows_full_budget_across_evaluation_chunks(tmp_path) -> None:
    config = replace(
        configuration(tmp_path, "dqn"),
        timesteps=1_000,
        evaluation=EvaluationConfig(16, 1),
        checkpoint_interval=0,
        dqn=replace(configuration(tmp_path, "dqn").dqn, exploration_fraction=1.0),
    )
    runner = None

    def on_event(event: dict) -> None:
        if event["type"] == "evaluation":
            runner.stop()

    runner = TrainingRunner(config, on_event)
    latest = runner.run()
    env = PolyTrackEnv(MockSimulatorTransport(), track_id=config.track_id,
                       action_adapter=backend_for("dqn").action_adapter(config))
    try:
        model = backend_for("dqn").load_model(latest / "policy.zip", env, "cpu")
        assert model.num_timesteps == 16
        assert model.exploration_rate > 0.9
    finally:
        env.close()


def test_tqc_warmup_replay_action_matches_executed_action(tmp_path) -> None:
    config = configuration(tmp_path, "tqc")
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight",
                       action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        action, replay = model._sample_action(config.tqc.learning_starts, n_envs=1)
        np.testing.assert_array_equal(action, replay)
        _, _ = env.reset(seed=0)
        _, _, _, _, info = env.step(action[0])
        assert info["requested_control_duty"]["throttle"] == pytest.approx(max(0, action[0, 1]))
        assert info["requested_control_duty"]["brake"] == pytest.approx(max(0, -action[0, 1]))
    finally:
        env.close()


def test_tqc_champion_replay_refill_uses_deterministic_policy(tmp_path) -> None:
    config = configuration(tmp_path, "tqc")
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight",
                       action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        observation, _ = env.reset(seed=17)
        model._last_obs = observation.reshape(1, -1)
        model._refill_replay_from_policy = True
        expected, _ = model.predict(observation, deterministic=True)
        action_a, replay_a = model._sample_action(config.tqc.learning_starts, n_envs=1)
        action_b, replay_b = model._sample_action(config.tqc.learning_starts, n_envs=1)
        np.testing.assert_allclose(action_a[0], expected, atol=1e-7)
        np.testing.assert_array_equal(action_a, action_b)
        np.testing.assert_array_equal(action_a, replay_a)
        np.testing.assert_array_equal(action_a, replay_b)
    finally:
        env.close()


def test_tqc_anchored_continuation_uses_deterministic_actions(tmp_path) -> None:
    config = configuration(tmp_path, "tqc")
    backend = backend_for("tqc")
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight",
                       action_adapter=backend.action_adapter(config))
    try:
        model = backend.create_model(config, env, "cpu")
        observation, _ = env.reset(seed=23)
        model._last_obs = observation.reshape(1, -1)
        model.num_timesteps = config.tqc.learning_starts + 1
        model.anchor_to_current_policy(0.005)

        expected, _ = model.predict(observation, deterministic=True)
        action_a, replay_a = model._sample_action(config.tqc.learning_starts, n_envs=1)
        action_b, replay_b = model._sample_action(config.tqc.learning_starts, n_envs=1)

        np.testing.assert_allclose(action_a[0], expected, atol=1e-7)
        np.testing.assert_array_equal(action_a, action_b)
        np.testing.assert_array_equal(action_a, replay_a)
        np.testing.assert_array_equal(action_a, replay_b)
    finally:
        env.close()


def test_saved_v2_metadata_rejects_v1(tmp_path) -> None:
    cfg = configuration(tmp_path, "ppo")
    path = ModelRegistry(cfg.output_root).slot(cfg.track_name, "ppo", "latest")
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(json.dumps({"schema": "polybot.model.v1"}))
    with pytest.raises(IncompatibleModelError, match="only v2"):
        ModelRegistry(cfg.output_root).read_metadata(path)
