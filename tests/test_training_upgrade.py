from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from polybot.env import (
    PolyTrackEnv,
    summer_1_pace_reward_config,
    summer_1_reward_config,
)
from polybot.gui.main import preferred_model, reward_breakdown, timed_curriculum_bounds
from polybot.mock import MockSimulatorTransport
from polybot.protocol import Telemetry
from polybot.pwm import PwmSteering
from polybot.training.config import (
    CurriculumConfig,
    TrainingConfig,
    architecture,
    estimate_ppo_parameters,
)
from polybot.training.devices import resolve_device
from polybot.training.manager import TrainingManager, is_retryable_simulator_error
from polybot.training.models import IncompatibleModelError, ModelMetadata, ModelRegistry, track_slug
from polybot.training.trainer import RollingStepRate


def test_xl_parameter_count() -> None:
    assert architecture("xl") == (1024, 1024, 512)
    assert estimate_ppo_parameters(Telemetry.vector_size(12), (41, 2, 2), "xl") == 3_340_334


def test_pwm_is_digital_distributed_deterministic_and_resets() -> None:
    pwm = PwmSteering()
    output = pwm.generate(0.5, 30)
    assert set(output) <= {-1, 0, 1}
    assert output.count(1) == 15
    assert output[:10] != [1] * 10
    pwm.reset()
    assert output == pwm.generate(0.5, 30)
    assert pwm.generate(-0.5, 4) == [0, -1, 0, -1]
    assert pwm.generate(0, 4) == [0] * 4
    assert pwm.generate(1, 4) == [1] * 4


def test_registry_is_track_scoped_and_checks_schema(tmp_path) -> None:
    assert track_slug(" Summer 1 ") == "summer-1"
    registry = ModelRegistry(tmp_path)
    summer = ModelMetadata("Summer 1", "summer-1", "xl", 123)
    registry.write_metadata(summer)
    assert registry.metadata_path("Summer 1").parent != registry.metadata_path("Other").parent
    with pytest.raises(IncompatibleModelError):
        registry.assert_compatible(summer, track_name="Other", action_schema=summer.action_schema)


def test_device_resolution_without_cuda() -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert resolve_device("auto", torch).resolved == "cpu"
    assert resolve_device("cpu", torch).resolved == "cpu"
    with pytest.raises(RuntimeError, match="unavailable|is_available"):
        resolve_device("cuda", torch)


def test_pwm_sequence_uses_one_protocol_round_trip() -> None:
    transport = MockSimulatorTransport()
    env = PolyTrackEnv(
        transport,
        track_id="mock/gentle-s",
        pwm_enabled=True,
        pwm_levels=41,
        frame_skip=16,
    )
    env.reset(seed=1)
    before = len(transport.command_log)
    env.step(np.array([30, 1, 0]))  # +50% steering
    requests = transport.command_log[before:]
    assert len(requests) == 1
    params = requests[0]["params"]
    assert len(params["actions"]) == 16
    assert {action["steer"] for action in params["actions"]} == {0.0, 1.0}
    env.close()


def test_summer_1_reward_profile_is_dense_and_balanced() -> None:
    rewards = summer_1_reward_config()
    assert rewards.progress_per_m > 0
    assert rewards.on_track_speed_per_m > 0
    assert rewards.checkpoint_bonus > 0
    assert rewards.finish_bonus > rewards.checkpoint_bonus
    assert rewards.crash_penalty < 0
    assert rewards.off_track_penalty < 0
    assert -100 < rewards.ground_slip_penalty_per_rad_s < 0
    assert rewards.progress_per_m == 2.0
    assert rewards.elapsed_cost_per_s == -0.2
    assert rewards.ground_brake_penalty_per_s == -3.0
    assert rewards.takeoff_target_speed_mps == 35.0
    assert rewards.imitation_bonus_per_s == 15.0
    assert rewards.checkpoint_speed_bonus_per_mps == 1.0
    assert rewards.checkpoint_speed_bonus_limit_mps == 45.0
    assert rewards.barrier_contact_penalty == -1000.0
    assert rewards.barrier_early_penalty == 0.0
    assert rewards.barrier_collision_impulse_threshold == 0.0
    assert rewards.failure_progress_clawback_per_m == 0.0
    assert rewards.failure_early_penalty == -2500.0
    assert rewards.airborne_roll_failure_penalty == -1000.0
    assert rewards.finish_bonus + rewards.finish_fast_bonus == 3000.0
    assert rewards.curriculum_section_bonus == 250.0
    assert (
        rewards.checkpoint_speed_bonus_per_mps * rewards.checkpoint_speed_bonus_limit_mps
        <= rewards.checkpoint_bonus
    )


def test_fresh_run_archive_preserves_latest_and_metadata(tmp_path) -> None:
    registry = ModelRegistry(tmp_path)
    directory = registry.initialise_track("Summer 1")
    (directory / "latest.zip").write_bytes(b"model")
    (directory / "latest.metadata.json").write_text("{}", encoding="utf-8")
    archived = registry.archive_latest("Summer 1")
    assert archived is not None
    assert archived.read_bytes() == b"model"
    assert archived.with_suffix(".metadata.json").exists()
    assert (directory / "latest.zip").read_bytes() == b"model"


def test_gui_prefers_latest_model_unless_current_selection_is_valid(tmp_path) -> None:
    best = tmp_path / "best.zip"
    latest = tmp_path / "latest.zip"
    models = [best, latest]

    assert preferred_model(models) == latest
    assert preferred_model(models, str(best)) == best
    assert preferred_model(models, "New model") == latest


def test_gui_validates_timed_curriculum_bounds() -> None:
    assert timed_curriculum_bounds("full", 10.0, 5.0) is None
    assert timed_curriculum_bounds("timed", 5.0, 12.5) == (5.0, 12.5)
    with pytest.raises(ValueError, match="start < end"):
        timed_curriculum_bounds("timed", 12.5, 5.0)


def test_gui_groups_episode_reward_terms() -> None:
    text = reward_breakdown(
        {
            "progress": 1200.0,
            "ghost_imitation": 50.0,
            "checkpoint": 100.0,
            "elapsed": -5.0,
            "ground_slip": -20.0,
            "barrier_contact": -1000.0,
            "failure_early": -500.0,
        }
    )

    assert "drive=+1200.0" in text
    assert "ghost=+50.0" in text
    assert "milestone=+100.0" in text
    assert "terminal=-1500.0" in text


def test_pace_profile_increases_time_pressure_without_removing_safety() -> None:
    balanced = summer_1_reward_config()
    pace = summer_1_pace_reward_config()
    assert pace.elapsed_cost_per_s < balanced.elapsed_cost_per_s
    assert pace.checkpoint_fast_bonus > balanced.checkpoint_fast_bonus
    assert pace.checkpoint_target_s < balanced.checkpoint_target_s
    assert pace.checkpoint_target_s == 8.0
    assert pace.checkpoint_speed_bonus_per_mps > 0
    assert pace.imitation_bonus_per_s == 20.0
    assert pace.checkpoint_speed_bonus_per_mps == 10.0
    assert pace.crash_penalty == balanced.crash_penalty
    assert pace.off_track_penalty == balanced.off_track_penalty


def test_xl_training_schedule_reduces_optimizer_batches() -> None:
    config = TrainingConfig()
    assert config.max_episode_seconds == 60.0
    assert config.learning_rate == 0.0001
    assert config.gamma == 0.9995
    assert config.gae_lambda == 0.995
    assert config.entropy_coefficient == 0.001
    assert config.rollout_steps == 8192
    assert config.batch_size == 1024
    assert config.ppo_epochs == 3
    assert config.rollout_steps // config.batch_size * config.ppo_epochs == 24
    with pytest.raises(ValueError, match="divide"):
        TrainingConfig(rollout_steps=8192, batch_size=300)


def test_step_rate_uses_recent_deltas_not_resumed_model_total() -> None:
    rate = RollingStepRate(window_s=5.0)

    assert rate.update(439_202, now=10.0) == 0.0
    assert rate.update(439_302, now=11.0) == pytest.approx(100.0)
    assert rate.update(439_802, now=16.0) == pytest.approx(100.0)
    assert rate.update(439_803, now=26.0) < 1.0


def test_randomised_quarters_choose_seeded_episode_sections() -> None:
    transport = MockSimulatorTransport()
    env = PolyTrackEnv(
        transport,
        track_id="mock/gentle-s",
        curriculum_random_quarters=True,
    )
    _, first = env.reset(seed=123)
    _, repeated = env.reset(seed=123)
    assert first["curriculum_quarter"] == repeated["curriculum_quarter"]
    assert first["curriculum_quarter"] in {1, 2, 3, 4}
    assert first["curriculum_end_ratio"] - first["curriculum_start_ratio"] == 0.25
    env.close()


def test_randomised_quarters_are_one_mixed_manager_phase() -> None:
    config = TrainingConfig()
    config.curriculum.mode = "quarters-randomised"
    assert [phase.mode for phase in TrainingManager(config).phases()] == ["quarters-randomised"]


def test_quarters_can_resume_from_q4_then_transition_to_full() -> None:
    config = TrainingConfig()
    config.curriculum.mode = "quarters-from-q4"

    phases = TrainingManager(config).phases()

    assert len(phases) == 2
    assert phases[0] == CurriculumConfig("section", 0.75, 1.0)
    assert phases[1] == CurriculumConfig()


def test_simulator_retry_only_accepts_transient_connection_errors() -> None:
    assert is_retryable_simulator_error(TimeoutError("simulator request timed out"))
    assert is_retryable_simulator_error(RuntimeError("adapter disconnected"))
    assert is_retryable_simulator_error(RuntimeError("stale_episode"))
    assert not is_retryable_simulator_error(ValueError("invalid batch size"))


def test_summer_profiles_penalise_ground_braking() -> None:
    assert summer_1_reward_config().ground_brake_penalty_per_s < 0
    assert (
        summer_1_pace_reward_config().ground_brake_penalty_per_s
        < summer_1_reward_config().ground_brake_penalty_per_s
    )
