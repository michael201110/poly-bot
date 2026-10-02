from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from polybot.algorithms.registry import backend_for
from polybot.environment.curriculum import build_plan
from polybot.training.config import PPOConfig, TrainingConfig
from polybot.training.evaluation import EvaluationResult
from polybot.training.runner import _freeze_ppo_actor
from polybot.training.teacher_student import (
    DaggerDataset,
    TeacherDataset,
    _align_ppo_config_to_checkpoint,
    _apply_ppo_training_section,
    _apply_reward_profile,
    _dagger_rounds_remain_after_reliable_gate,
    _dagger_seed_student,
    _ensure_ppo_teacher_anchor,
    _evaluation_confirms_target,
    _evaluation_rank,
    _gym_env,
    _initial_dagger_student,
    _partial_progress_gate_passed,
    _ppo_teacher_anchor_dir_for_source,
    _promote_ppo_champion_if_better,
    _run_with_stop_file,
    _seed_validated_ppo_champion,
    _should_resume_ppo_candidate,
    _should_resume_ppo_champion,
    _tqc_actor_graft_architecture,
    aggregate_teacher_datasets,
    collect_dagger_data,
    collect_teacher_data,
    dagger_action_error_report,
    imitation_metrics,
    pretrain_actor,
    split_aggregated_trajectories,
    split_trajectories,
)


def test_partial_stopped_evaluation_cannot_confirm_target_lap() -> None:
    assert not _evaluation_confirms_target({"target_reached": False}, 22.0)


def test_complete_evaluation_confirms_target_only_with_reliable_fast_lap() -> None:
    evaluation = {
        "episodes": 5, "finish_rate": 1.0, "median_progress": 1.0,
        "mean_progress": 1.0, "best_lap_s": 21.9, "median_lap_s": 21.95,
        "crash_rate": 0.0, "off_track_rate": 0.0, "stall_rate": 0.0,
    }
    assert _evaluation_confirms_target(evaluation, 22.0)
    evaluation["finish_rate"] = 0.8
    assert not _evaluation_confirms_target(evaluation, 22.0)


def test_ppo_continuation_uses_checkpoint_architecture_and_action_noise() -> None:
    config = TrainingConfig(algorithm="ppo", ppo=PPOConfig(architecture="standard"))
    metadata = SimpleNamespace(
        algorithm="ppo", architecture="tqc_compatible",
        training_config={"ppo": {"action_std": 0.05, "rollout_steps": 4096}},
    )

    _align_ppo_config_to_checkpoint(config, metadata)

    assert config.ppo.architecture == "tqc_compatible"
    assert config.ppo.action_std == 0.05
    assert config.ppo.rollout_steps == 4096

    _align_ppo_config_to_checkpoint(config, metadata, action_std=0.02)
    assert config.ppo.action_std == 0.02
    with pytest.raises(ValueError, match="rollout steps cannot change"):
        _align_ppo_config_to_checkpoint(config, metadata, rollout_steps=2048)


def test_ppo_training_section_changes_collection_not_full_track_evaluation() -> None:
    config = TrainingConfig(algorithm="ppo", ppo=PPOConfig())

    _apply_ppo_training_section(config, 0.0, 0.15, lead_in_ratio=0.05)

    phase = build_plan(config.curriculum, 8192).phases[0]
    assert (phase.mode, phase.spawn_ratio, phase.start_ratio, phase.end_ratio) == (
        "section", 0.0, 0.0, 0.15,
    )
    with pytest.raises(ValueError, match="both PPO training section bounds"):
        _apply_ppo_training_section(config, 0.0, None)


def test_exact_tqc_actor_init_can_select_full_trainable_compatible_policy() -> None:
    assert _tqc_actor_graft_architecture(None) == "tqc_residual"
    assert _tqc_actor_graft_architecture("tqc_compatible") == "tqc_compatible"
    with pytest.raises(ValueError, match="only tqc_compatible"):
        _tqc_actor_graft_architecture("standard")


def test_teacher_student_stop_file_interrupts_active_training(tmp_path: Path) -> None:
    stop_file = tmp_path / "stop.txt"
    started = threading.Event()
    stopped = threading.Event()

    class Runner:
        def stop(self) -> None:
            stopped.set()

        def run(self) -> Path:
            started.set()
            assert stopped.wait(timeout=2.0)
            return tmp_path / "latest"

    def request_stop() -> None:
        assert started.wait(timeout=1.0)
        stop_file.write_text("stop\n", encoding="utf-8")

    requester = threading.Thread(target=request_stop)
    requester.start()
    result = _run_with_stop_file(Runner(), stop_file)
    requester.join(timeout=1.0)

    assert result == tmp_path / "latest"
    assert stopped.is_set()


def test_named_reward_profile_overrides_teacher_reward_configuration() -> None:
    config = TrainingConfig(algorithm="ppo", ppo=PPOConfig())
    selected = _apply_reward_profile(config, "Summer 1 - 20s pace spin control")

    assert selected is config
    assert config.reward_profile == "Summer 1 - 20s pace spin control"
    assert config.rewards.finish_target_s == 20.0
    assert config.rewards.ground_spin_deadzone_radps == 5.0
    assert config.rewards.ground_spin_penalty_per_rad_s == -12.0
    assert config.rewards.ground_spin_min_grounded_wheels == 2
    assert config.rewards.barrier_collision_impulse_threshold == 1e9


@pytest.mark.parametrize(
    ("continue_after_reliable", "first_round", "next_round", "rounds_to_run", "expected"),
    [
        (False, 1, 2, 3, False),
        (True, 1, 2, 3, True),
        (True, 1, 4, 3, False),
        (True, 1, 2, None, False),
    ],
)
def test_reliable_dagger_baseline_does_not_skip_requested_rounds(
    continue_after_reliable: bool, first_round: int, next_round: int,
    rounds_to_run: int | None, expected: bool,
) -> None:
    assert _dagger_rounds_remain_after_reliable_gate(
        continue_after_reliable=continue_after_reliable,
        first_round=first_round,
        next_round=next_round,
        rounds_to_run=rounds_to_run,
    ) is expected
class _TeacherPolicy:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []

    def set_training_mode(self, _enabled: bool) -> None:
        pass

    def predict(self, _observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, None]:
        assert deterministic
        self.observations.append(np.array(_observation, copy=True))
        return np.array([0.137, 0.684], dtype=np.float32), None


class _Teacher:
    def __init__(self) -> None:
        self.policy = _TeacherPolicy()
        self.policy_overlays = [
            {"kind": "air_brake", "start": 0.0, "end": 1.0, "duty": 0.9, "taper": 0.01},
            {"kind": "drive_bias", "start": 0.0, "end": 1.0, "amount": 0.1},
        ]
        self.speed_bias_schedule = []
        self._air_brake_active = False

    def predict(self, observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, None]:
        action, state = self.policy.predict(observation, deterministic=deterministic)
        self._air_brake_active = True
        return np.array([action[0], -0.9], dtype=np.float32), state


class _OneStepLapEnv:
    def __init__(self) -> None:
        self.index = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict]:
        self.index = 0
        observation = np.zeros(105, dtype=np.float32)
        observation[12] = 0.5
        observation[17:21] = 0.0
        return observation, {"seed": seed}

    def step(self, _action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.index += 1
        return np.zeros(105, dtype=np.float32), 0.0, True, False, {
            "events": ["finish"], "route_progress_m": 100.0, "track_length_m": 100.0,
            "elapsed_s": 24.0, "local_velocity_mps": [0.0, 0.0, 30.0],
            "position_m": [0.0, 0.0, 0.0], "heading_error_rad": 0.0,
            "wheel_contacts": [1, 1, 1, 1],
        }


class _Student:
    class Policy:
        def set_training_mode(self, _enabled: bool) -> None:
            pass

    def __init__(self) -> None:
        self.policy = self.Policy()
        self.observations: list[np.ndarray] = []
        self.deterministic_flags: list[bool] = []

    def predict(self, observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, None]:
        self.deterministic_flags.append(deterministic)
        self.observations.append(np.array(observation, copy=True))
        return np.array([0.75, 0.25], dtype=np.float32), None


class _DaggerFailureEnv:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.index = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict]:
        del seed
        self.index = 0
        observation = np.zeros(105, dtype=np.float32)
        observation[12] = 0.1
        observation[17:21] = 1.0
        return observation, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.actions.append(np.array(action, copy=True))
        self.index += 1
        observation = np.zeros(105, dtype=np.float32)
        observation[12] = 0.1 + self.index * 0.05
        observation[17:21] = 1.0
        failed = self.index == 3
        return observation, 0.0, failed, False, {
            "events": ["crash"] if failed else [], "elapsed_s": self.index * 0.5,
            "route_progress_m": observation[12] * 100.0, "track_length_m": 100.0,
            "local_velocity_mps": [0.0, 0.0, 20.0], "position_m": [1.0, 2.0, 3.0],
            "heading_error_rad": 0.1, "wheel_contacts": [1, 1, 1, 1],
        }


def test_teacher_collection_records_raw_and_final_controls_without_mutation() -> None:
    teacher = _Teacher()
    stack_before = [layer.copy() for layer in teacher.policy_overlays]
    dataset = collect_teacher_data(
        teacher, _OneStepLapEnv(), successful_laps=2, max_attempts=2, seed=10,
        teacher_id="frozen-teacher",
    )
    assert dataset.successful_trajectory_ids == [0, 1]
    np.testing.assert_allclose(dataset.raw_actions, [[0.137, 0.684], [0.137, 0.684]])
    np.testing.assert_allclose(dataset.final_actions, [[0.137, 0.784], [0.137, 0.784]])
    np.testing.assert_allclose(dataset.driving_actions, [[0.137, -0.9], [0.137, -0.9]])
    assert teacher.policy_overlays == stack_before


def test_dagger_ppo_controls_while_teacher_labels_identical_student_states() -> None:
    teacher = _Teacher()
    student = _Student()
    env = _DaggerFailureEnv()
    stack_before = [layer.copy() for layer in teacher.policy_overlays]
    dataset = collect_dagger_data(
        student, teacher, env, episodes=2, dagger_round=1, seed=4,
        teacher_id="frozen-teacher",
    )
    # The final crash-triggering transition is excluded; the pre-failure recovery
    # observations remain, and the environment only ever receives PPO actions.
    assert len(dataset.observations) == 4
    assert dataset.collection_deterministic is True
    np.testing.assert_array_equal(dataset.sources, np.full(4, "dagger"))
    assert len(env.actions) == 6
    np.testing.assert_allclose(env.actions, np.tile([0.75, 0.25], (6, 1)))
    np.testing.assert_allclose(dataset.student_actions, np.tile([0.75, 0.25], (4, 1)))
    np.testing.assert_allclose(dataset.teacher_actions, np.tile([0.137, 0.784], (4, 1)))
    np.testing.assert_array_equal(dataset.episode_outcomes, np.full(4, "crash"))
    np.testing.assert_allclose(dataset.failure_progress, np.full(4, 0.2))
    # Both TQC queries in each iteration see precisely the observation PPO saw.
    student_observations = np.asarray(student.observations)
    np.testing.assert_array_equal(
        student_observations.reshape(2, 3, 105)[:, :2].reshape(-1, 105), dataset.observations
    )
    teacher_queries = np.asarray(teacher.policy.observations)
    np.testing.assert_array_equal(teacher_queries[::2], student_observations)
    np.testing.assert_array_equal(teacher_queries[1::2], student_observations)
    assert teacher.policy_overlays == stack_before
    assert student.deterministic_flags == [True] * 6


def test_stochastic_dagger_collection_is_explicit_and_recorded() -> None:
    student = _Student()
    dataset = collect_dagger_data(
        student, _Teacher(), _DaggerFailureEnv(), episodes=2,
        dagger_round=1, teacher_id="frozen-teacher", deterministic=False,
    )
    assert dataset.collection_deterministic is False
    assert student.deterministic_flags == [False] * 6


def test_stochastic_dagger_seed_controls_student_action_sampling() -> None:
    class SeededStudent(_Student):
        def set_random_seed(self, seed: int) -> None:
            self.rng = np.random.default_rng(seed)

        def predict(
            self, observation: np.ndarray, *, deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            super().predict(observation, deterministic=deterministic)
            if deterministic:
                return np.array([0.75, 0.25], dtype=np.float32), None
            return self.rng.uniform(-1, 1, size=2).astype(np.float32), None

    first = collect_dagger_data(
        SeededStudent(), _Teacher(), _DaggerFailureEnv(), episodes=1,
        dagger_round=1, seed=10, teacher_id="frozen-teacher", deterministic=False,
    )
    second = collect_dagger_data(
        SeededStudent(), _Teacher(), _DaggerFailureEnv(), episodes=1,
        dagger_round=2, seed=11, teacher_id="frozen-teacher", deterministic=False,
    )

    assert not np.array_equal(first.student_actions, second.student_actions)


def test_dagger_failure_window_retains_the_longer_lead_in() -> None:
    dataset = collect_dagger_data(
        _Student(), _Teacher(), _DaggerFailureEnv(), episodes=2,
        dagger_round=1, teacher_id="frozen-teacher", failure_window_s=0.5,
    )
    # At 0.5 seconds per transition, the 0.5-second window preserves only the
    # final pre-terminal action from each episode.
    assert len(dataset.observations) == 2
    np.testing.assert_allclose(dataset.time_to_failure_s, [0.5, 0.5])


def test_new_dagger_series_starts_from_teacher_pretrained_actor(tmp_path) -> None:
    class Registry:
        def slot(self, *_args):
            return tmp_path / "ppo-champion"

    output = tmp_path / "teacher-student"
    pretrained = output / "pretrained"
    pretrained.mkdir(parents=True)
    (pretrained / "policy.zip").touch()
    champion = tmp_path / "ppo-champion"
    champion.mkdir()
    (champion / "policy.zip").touch()
    assert _initial_dagger_student(output, Registry(), "Summer 1") == pretrained


def test_dagger_series_can_start_from_an_evaluated_ppo_checkpoint(tmp_path) -> None:
    class Registry:
        def slot(self, *_args):
            return tmp_path / "teacher-pretrained"

    output = tmp_path / "teacher-student"
    explicit = tmp_path / "ppo-champion"
    explicit.mkdir()
    (explicit / "policy.zip").touch()
    (explicit / "metadata.json").touch()

    assert _dagger_seed_student(explicit, output, Registry(), "Summer 1") == explicit


def test_dagger_seed_requires_a_complete_model_slot(tmp_path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="policy.zip and metadata.json"):
        _dagger_seed_student(missing, tmp_path / "output", object(), "Summer 1")


def test_ppo_teacher_anchor_is_an_immutable_snapshot(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "policy.zip").write_text("baseline policy", encoding="utf-8")
    (source / "metadata.json").write_text("{}", encoding="utf-8")
    anchor_dir = tmp_path / "run" / "ppo-teacher-anchor"

    anchor = _ensure_ppo_teacher_anchor(anchor_dir, source)
    (source / "policy.zip").write_text("updated candidate", encoding="utf-8")
    next_candidate = tmp_path / "next-candidate"
    next_candidate.mkdir()
    (next_candidate / "policy.zip").write_text("another candidate", encoding="utf-8")
    (next_candidate / "metadata.json").write_text("{}", encoding="utf-8")

    assert anchor == anchor_dir / "policy.zip"
    assert _ensure_ppo_teacher_anchor(anchor_dir, next_candidate) == anchor
    assert anchor.read_text(encoding="utf-8") == "baseline policy"


def test_ppo_teacher_anchor_path_tracks_the_starting_policy(tmp_path) -> None:
    base = tmp_path / "run" / "ppo-teacher-anchor"
    first = tmp_path / "first"
    second = tmp_path / "second"
    for folder, policy in ((first, b"first policy"), (second, b"second policy")):
        folder.mkdir()
        (folder / "policy.zip").write_bytes(policy)

    first_anchor = _ppo_teacher_anchor_dir_for_source(base, first)
    same_anchor = _ppo_teacher_anchor_dir_for_source(base, first)
    second_anchor = _ppo_teacher_anchor_dir_for_source(base, second)
    assert first_anchor == same_anchor
    assert first_anchor != second_anchor


def test_dagger_checkpoint_rank_prefers_progress_until_finishes_then_pace() -> None:
    assert _evaluation_rank({"finish_rate": 0.0, "median_progress": 0.39}) > (
        _evaluation_rank({"finish_rate": 0.0, "median_progress": 0.31})
    )
    assert _evaluation_rank({"finish_rate": 0.2, "median_lap_s": 24.1}) > (
        _evaluation_rank({"finish_rate": 0.0, "median_progress": 1.0})
    )
    assert _evaluation_rank({"finish_rate": 1.0, "median_lap_s": 23.5}) > (
        _evaluation_rank({"finish_rate": 1.0, "median_lap_s": 24.1})
    )


def test_partial_progress_gate_uses_the_selected_median_progress() -> None:
    assert not _partial_progress_gate_passed({"median_progress": 0.349}, 0.35)
    assert _partial_progress_gate_passed({"median_progress": 0.35}, 0.35)
    assert not _partial_progress_gate_passed({}, 0.35)


def test_ppo_only_resumes_champion_after_a_material_regression() -> None:
    champion = {"finish_rate": 0.0, "median_progress": 0.391}
    assert not _should_resume_ppo_champion(
        {"finish_rate": 0.0, "median_progress": 0.390}, champion,
    )
    assert _should_resume_ppo_champion(
        {"finish_rate": 0.0, "median_progress": 0.33}, champion,
    )
    assert _should_resume_ppo_champion(
        {"finish_rate": 0.0, "median_progress": 0.39},
        {"finish_rate": 0.2, "median_progress": 0.8},
    )
    assert _should_resume_ppo_champion(
        {}, {"finish_rate": 1.0, "median_progress": 1.0, "median_lap_s": 24.675},
    )


def test_ppo_candidate_gets_recovery_blocks_before_champion_rollback() -> None:
    candidate = {"finish_rate": 0.0, "median_progress": 0.53}
    champion = {"finish_rate": 1.0, "median_progress": 1.0, "median_lap_s": 24.263}
    assert not _should_resume_ppo_candidate(
        candidate, champion, consecutive_incomplete_blocks=1,
    )
    assert not _should_resume_ppo_candidate(
        candidate, champion, consecutive_incomplete_blocks=2,
    )
    assert _should_resume_ppo_candidate(
        candidate, champion, consecutive_incomplete_blocks=3,
    )
    assert _should_resume_ppo_candidate(
        {"finish_rate": 0.0, "median_progress": 0.14}, champion,
        consecutive_incomplete_blocks=1,
    )


def test_ppo_resume_gate_can_reject_any_measured_lap_slowdown() -> None:
    candidate = {"finish_rate": 1.0, "median_progress": 1.0, "median_lap_s": 25.253}
    champion = {"finish_rate": 1.0, "median_progress": 1.0, "median_lap_s": 25.025}

    assert _should_resume_ppo_champion(candidate, champion, lap_tolerance_s=0.0)
    assert not _should_resume_ppo_champion(candidate, champion, lap_tolerance_s=0.5)


def test_ppo_resume_gate_rejects_a_thirteen_point_progress_regression() -> None:
    candidate = {"finish_rate": 0.0, "median_progress": 0.391}
    champion = {"finish_rate": 0.0, "median_progress": 0.523}

    assert _should_resume_ppo_champion(
        candidate, champion, progress_tolerance=0.02,
    )
    assert not _should_resume_ppo_champion(
        candidate, champion, progress_tolerance=0.20,
    )


@pytest.mark.parametrize(
    ("candidate_progress", "incumbent_progress", "expected_promoted"),
    [(0.45, 0.40, True), (0.39, 0.40, False)],
)
def test_isolated_ppo_promotion_requires_better_evaluation(
    tmp_path, candidate_progress, incumbent_progress, expected_promoted,
) -> None:
    class Registry:
        def __init__(self, root, metadata):
            self.root = root
            self.metadata = metadata

        def slot(self, _track, _algorithm, _name):
            return self.root / "summer-1" / "ppo" / "champion"

        def read_metadata(self, _path):
            return self.metadata

    def metadata(progress):
        return SimpleNamespace(
            algorithm="ppo",
            evaluation={
                "episodes": 5, "finish_rate": 0.0,
                "median_progress": progress, "mean_progress": progress,
                "best_lap_s": None, "median_lap_s": None,
                "crash_rate": 1.0, "off_track_rate": 0.0, "stall_rate": 0.0,
            },
        )

    candidate_root = tmp_path / "isolated"
    candidate = candidate_root / "summer-1" / "ppo" / "champion"
    candidate.mkdir(parents=True)
    (candidate / "policy.zip").write_bytes(b"candidate")
    (candidate / "metadata.json").write_text("{}", encoding="utf-8")
    destination_root = tmp_path / "main"
    destination = destination_root / "summer-1" / "ppo" / "champion"
    destination.mkdir(parents=True)
    (destination / "policy.zip").write_bytes(b"incumbent")
    (destination / "metadata.json").write_text("{}", encoding="utf-8")

    promoted = _promote_ppo_champion_if_better(
        Registry(candidate_root, metadata(candidate_progress)),
        Registry(destination_root, metadata(incumbent_progress)),
        "Summer 1",
    )

    assert (promoted is not None) is expected_promoted
    expected_contents = b"candidate" if expected_promoted else b"incumbent"
    assert (destination / "policy.zip").read_bytes() == expected_contents


def test_validated_ppo_seed_requires_all_configured_finishes(tmp_path: Path) -> None:
    class Registry:
        def __init__(self, root: Path) -> None:
            self.root = root
            self.source_metadata = SimpleNamespace(
                algorithm="ppo", track_name="Summer 1", evaluation=None,
                finishes=0, crashes=0,
            )

        def slot(self, track: str, algorithm: str, name: str) -> Path:
            return self.root / "summer-1" / algorithm / name

        def read_metadata(self, _path: Path) -> SimpleNamespace:
            return self.source_metadata

        def write_metadata(self, path: Path, metadata: SimpleNamespace) -> None:
            (path / "metadata.json").write_text(json.dumps({
                "evaluation": metadata.evaluation,
                "finishes": metadata.finishes,
                "crashes": metadata.crashes,
            }), encoding="utf-8")

    root = tmp_path / "isolated"
    registry = Registry(root)
    source = root / "summer-1" / "ppo" / "teacher-student" / "pretrained"
    source.mkdir(parents=True)
    (source / "policy.zip").write_bytes(b"grafted policy")
    (source / "metadata.json").write_text("{}", encoding="utf-8")
    evaluation = EvaluationResult(
        episodes=5, finish_rate=1.0, median_progress=1.0, mean_progress=1.0,
        best_lap_s=24.263, median_lap_s=24.263, crash_rate=0.0,
        off_track_rate=0.0, stall_rate=0.0,
    )

    champion = _seed_validated_ppo_champion(registry, source, evaluation)

    assert champion == root / "summer-1" / "ppo" / "champion"
    saved = json.loads((champion / "metadata.json").read_text(encoding="utf-8"))
    assert saved["finishes"] == 5
    assert saved["evaluation"]["median_lap_s"] == 24.263
    assert (champion / "policy.zip").read_bytes() == b"grafted policy"

    unreliable = EvaluationResult(
        episodes=5, finish_rate=0.8, median_progress=0.9, mean_progress=0.9,
        best_lap_s=24.263, median_lap_s=24.263, crash_rate=0.2,
        off_track_rate=0.0, stall_rate=0.0,
    )
    assert _seed_validated_ppo_champion(registry, source, unreliable) is None


def test_dagger_archive_and_error_report_preserve_recovery_telemetry(tmp_path) -> None:
    dataset = collect_dagger_data(
        _Student(), _Teacher(), _DaggerFailureEnv(), episodes=2,
        dagger_round=2, seed=0, teacher_id="frozen-teacher",
    )
    path = tmp_path / "dagger-round-002.npz"
    dataset.save(path)
    loaded = DaggerDataset.load(path)
    assert loaded.collection_deterministic is True
    np.testing.assert_array_equal(loaded.student_actions, dataset.student_actions)
    np.testing.assert_array_equal(loaded.teacher_actions, dataset.teacher_actions)
    np.testing.assert_array_equal(loaded.airborne, np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(loaded.sources, np.full(4, "dagger"))
    report = dagger_action_error_report(loaded)
    assert report["samples"] == 4
    assert report["failure_progress"][0]["outcome"] == "crash"
    assert report["windows_5_percent"]["10-15%"]["mean_steering_error"] == pytest.approx(0.613)


def test_aggregation_keeps_nominal_data_and_balances_recovery_weights(tmp_path) -> None:
    nominal = TeacherDataset(
        observations=np.zeros((8, 105), dtype=np.float32),
        raw_actions=np.zeros((8, 2), dtype=np.float32),
        final_actions=np.ones((8, 2), dtype=np.float32) * 0.1,
        driving_actions=np.ones((8, 2), dtype=np.float32) * 0.1,
        trajectory_ids=np.repeat(np.arange(4), 2), progress=np.tile([0.1, 0.2], 4),
        elapsed_s=np.zeros(8), speed_mps=np.ones(8), position_m=np.zeros((8, 3)),
        heading_error_rad=np.zeros(8), wheel_contacts=np.ones((8, 4)),
        overlay_active=np.zeros(8, dtype=bool), successful_trajectory_ids=[0, 1, 2, 3],
        teacher_id="frozen-teacher",
    )
    nominal_path = tmp_path / "teacher-nominal.npz"
    nominal.save(nominal_path)
    before = nominal_path.read_bytes()
    dagger = collect_dagger_data(
        _Student(), _Teacher(), _DaggerFailureEnv(), episodes=2,
        dagger_round=1, seed=5, teacher_id="frozen-teacher",
    )
    # Make one trajectory a neutral control group so the measured-failure
    # window's sample weighting is observable.
    dagger.episode_outcomes[2:] = "finish"
    aggregate, report = aggregate_teacher_datasets(
        nominal, [dagger], nominal_weight=0.65, recovery_weight=0.35,
    )
    aggregate_path = tmp_path / "aggregate-round-001.npz"
    aggregate.save(aggregate_path)
    loaded = TeacherDataset.load(aggregate_path)
    assert len(loaded.observations) == len(nominal.observations) + len(dagger.observations)
    assert np.count_nonzero(loaded.sources == "nominal") == len(nominal.observations)
    assert np.count_nonzero(loaded.sources == "dagger") == len(dagger.observations)
    assert loaded.sample_weights[loaded.sources == "nominal"].sum() == pytest.approx(0.65)
    assert loaded.sample_weights[loaded.sources == "dagger"].sum() == pytest.approx(0.35)
    assert report["nominal_samples"] == len(nominal.observations)
    assert report["recovery_samples"] == len(dagger.observations)
    dagger_weights = loaded.sample_weights[loaded.sources == "dagger"]
    assert report["failure_focus_samples"] == 1
    assert dagger_weights[:2].mean() > dagger_weights[2:].mean()
    assert nominal_path.read_bytes() == before
    dagger.time_to_failure_s[0] = 3.0
    short_window, _ = aggregate_teacher_datasets(
        nominal, [dagger], nominal_weight=0.65, recovery_weight=0.35,
        failure_window_s=2.0,
    )
    long_window, _ = aggregate_teacher_datasets(
        nominal, [dagger], nominal_weight=0.65, recovery_weight=0.35,
        failure_window_s=4.0,
    )
    assert long_window.sample_weights[len(nominal.observations)] > (
        short_window.sample_weights[len(nominal.observations)]
    )
    train, validation = split_aggregated_trajectories(
        loaded.trajectory_ids, loaded.sources, seed=3,
    )
    assert set(loaded.sources[train]) == {"nominal", "dagger"}
    assert set(loaded.sources[validation]) == {"nominal", "dagger"}


def test_weighted_sampling_does_not_square_recovery_source_weights() -> None:
    class Policy(th.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp_extractor = SimpleNamespace(policy_net=th.nn.Identity())
            self.action_net = th.nn.Linear(105, 2, bias=False)
            self.log_std = th.nn.Parameter(th.zeros(2))
            th.nn.init.zeros_(self.action_net.weight)

        def set_training_mode(self, enabled: bool) -> None:
            self.train(enabled)

        def get_distribution(self, observation: th.Tensor) -> SimpleNamespace:
            mean = self.action_net(observation)
            return SimpleNamespace(
                distribution=SimpleNamespace(mean=mean),
                get_actions=lambda deterministic: th.tanh(mean),
            )

    class Model:
        def __init__(self) -> None:
            self.policy = Policy()
            self.device = th.device("cpu")

        def predict(self, observation: np.ndarray, *, deterministic: bool):
            assert deterministic
            with th.no_grad():
                obs = th.as_tensor(observation, dtype=th.float32)
                mean = self.policy.get_distribution(obs).distribution.mean
            return mean.cpu().numpy(), None

    nominal_count, recovery_count = 100, 1
    observations = np.zeros((nominal_count + recovery_count, 105), dtype=np.float32)
    observations[:, 0] = 1.0
    targets = np.vstack((
        np.ones((nominal_count, 2), dtype=np.float32),
        -np.ones((recovery_count, 2), dtype=np.float32),
    ))
    weights = np.concatenate((
        np.full(nominal_count, 0.6 / nominal_count, dtype=np.float32),
        np.asarray([0.4], dtype=np.float32),
    ))
    dataset = TeacherDataset(
        observations=observations, raw_actions=targets, final_actions=targets,
        driving_actions=targets, trajectory_ids=np.arange(len(targets)),
        progress=np.zeros(len(targets)), elapsed_s=np.zeros(len(targets)),
        speed_mps=np.zeros(len(targets)), position_m=np.zeros((len(targets), 3)),
        heading_error_rad=np.zeros(len(targets)), wheel_contacts=np.ones((len(targets), 4)),
        overlay_active=np.zeros(len(targets), dtype=bool),
        successful_trajectory_ids=list(range(len(targets))), teacher_id="teacher",
        sources=np.concatenate((
            np.full(nominal_count, "nominal"), np.asarray(["dagger"])
        )), sample_weights=weights,
    )
    model = Model()
    pretrain_actor(
        model, dataset, epochs=1, batch_size=1024, patience=1, seed=3,
        learning_rate=1e-2, train_indices=np.arange(len(targets)),
        validation_indices=np.arange(len(targets)), sample_weights=weights,
    )
    # A correct 60/40 weighted draw has a positive net target. If the loss
    # multiplies by the weights again, the single recovery row dominates.
    assert model.predict(observations[:1], deterministic=True)[0][0, 0] > 0


def test_validation_split_keeps_whole_trajectories_separate() -> None:
    trajectory_ids = np.repeat(np.arange(10), 5)
    train, validation = split_trajectories(trajectory_ids, list(range(10)), seed=17)
    assert not set(trajectory_ids[train]) & set(trajectory_ids[validation])
    assert len(train) + len(validation) == len(trajectory_ids)
    with pytest.raises(ValueError, match="two successful"):
        split_trajectories(trajectory_ids, [1])


def test_teacher_dataset_roundtrip_preserves_arrays(tmp_path) -> None:
    dataset = TeacherDataset(
        observations=np.zeros((4, 105), dtype=np.float32),
        raw_actions=np.zeros((4, 2), dtype=np.float32),
        final_actions=np.ones((4, 2), dtype=np.float32),
        driving_actions=np.full((4, 2), 0.25, dtype=np.float32),
        trajectory_ids=np.repeat([1, 2], 2),
        progress=np.linspace(0.1, 0.9, 4), elapsed_s=np.arange(4, dtype=float),
        speed_mps=np.arange(4, dtype=float), position_m=np.zeros((4, 3)),
        heading_error_rad=np.zeros(4), wheel_contacts=np.ones((4, 4)),
        overlay_active=np.array([False, True, False, True]),
        successful_trajectory_ids=[1, 2], teacher_id="frozen-teacher",
    )
    path = tmp_path / "teacher.npz"
    dataset.save(path)
    loaded = TeacherDataset.load(path)
    np.testing.assert_array_equal(loaded.final_actions, dataset.final_actions)
    assert loaded.successful_trajectory_ids == [1, 2]
    assert loaded.teacher_id == "frozen-teacher"


def test_actor_pretraining_reduces_offline_error_without_touching_critic(tmp_path) -> None:
    config = TrainingConfig(
        algorithm="ppo", device="cpu", track_name="Summer 1", track_id="mock/straight",
        ppo=PPOConfig(architecture="tiny", rollout_steps=32, batch_size=16, epochs=1),
    )
    env = _gym_env(config, 105)
    np.testing.assert_array_equal(env.observation_space.low, np.full(105, -5.0, dtype=np.float32))
    np.testing.assert_array_equal(env.observation_space.high, np.full(105, 5.0, dtype=np.float32))
    model = backend_for("ppo").create_model(config, env, "cpu")
    checkpoint = tmp_path / "checkpoint-space-regression.zip"
    model.save(str(checkpoint))
    try:
        loaded = backend_for("ppo").load_model(checkpoint, _gym_env(config, 105), "cpu")
        assert loaded.observation_space == env.observation_space
    finally:
        checkpoint.unlink(missing_ok=True)
    rng = np.random.default_rng(5)
    observations = np.zeros((4 * 96, 105), dtype=np.float32)
    observations[:, :2] = rng.uniform(-1, 1, (len(observations), 2))
    actions = (observations[:, :2] * 0.45).astype(np.float32)
    dataset = TeacherDataset(
        observations=observations, raw_actions=actions.copy(), final_actions=actions,
        driving_actions=actions.copy(),
        trajectory_ids=np.repeat(np.arange(4), 96), progress=np.tile(np.linspace(0, 1, 96), 4),
        elapsed_s=np.zeros(len(observations)), speed_mps=np.zeros(len(observations)),
        position_m=np.zeros((len(observations), 3)), heading_error_rad=np.zeros(len(observations)),
        wheel_contacts=np.ones((len(observations), 4)), overlay_active=np.zeros(len(observations), bool),
        successful_trajectory_ids=[0, 1, 2, 3], teacher_id="test",
    )
    train_indices, validation_indices = split_trajectories(
        dataset.trajectory_ids, dataset.successful_trajectory_ids, seed=0
    )
    del train_indices
    before = imitation_metrics(model, dataset, validation_indices)
    critic = {
        name: value.detach().clone() for name, value in model.policy.state_dict().items()
        if "value" in name
    }
    report = pretrain_actor(model, dataset, epochs=30, batch_size=128, patience=8, seed=0)
    after = report["validation"]
    assert report["critic_unchanged"] is True
    np.testing.assert_allclose(
        np.exp(model.policy.log_std.detach().cpu().numpy()), [0.15, 0.15], rtol=1e-5
    )
    assert after["steering_mse"] + after["longitudinal_mse"] < (
        before["steering_mse"] + before["longitudinal_mse"]
    )
    state = model.policy.state_dict()
    assert all(th.equal(value, state[name]) for name, value in critic.items())


def test_actor_pretraining_updates_tqc_residual_head_with_frozen_teacher_layers() -> None:
    config = TrainingConfig(
        algorithm="ppo", device="cpu", track_name="Summer 1", track_id="mock/straight",
        ppo=PPOConfig(
            architecture="tqc_residual", rollout_steps=32, batch_size=16, epochs=1,
        ),
    )
    env = _gym_env(config, 105)
    model = backend_for("ppo").create_model(config, env, "cpu")
    try:
        for module in (
            model.policy.features_extractor,
            model.policy.mlp_extractor.policy_net,
            model.policy.action_net,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        rng = np.random.default_rng(9)
        observations = np.zeros((4 * 32, 105), dtype=np.float32)
        observations[:, :2] = rng.uniform(-1, 1, (len(observations), 2))
        actions = np.tile(np.asarray([0.2, -0.1], dtype=np.float32), (len(observations), 1))
        dataset = TeacherDataset(
            observations=observations, raw_actions=actions.copy(), final_actions=actions,
            driving_actions=actions.copy(), trajectory_ids=np.repeat(np.arange(4), 32),
            progress=np.tile(np.linspace(0, 1, 32), 4), elapsed_s=np.zeros(len(observations)),
            speed_mps=np.zeros(len(observations)), position_m=np.zeros((len(observations), 3)),
            heading_error_rad=np.zeros(len(observations)), wheel_contacts=np.ones((len(observations), 4)),
            overlay_active=np.zeros(len(observations), dtype=bool),
            successful_trajectory_ids=[0, 1, 2, 3], teacher_id="test",
        )
        residual_before = {
            name: value.detach().clone()
            for name, value in model.policy.residual_action.state_dict().items()
        }
        indices = np.arange(len(observations))
        before = imitation_metrics(model, dataset, indices)
        report = pretrain_actor(
            model, dataset, epochs=30, batch_size=64, patience=8, seed=0,
            learning_rate=1e-3,
        )
        assert report["validation"]["steering_mse"] + report["validation"]["longitudinal_mse"] < (
            before["steering_mse"] + before["longitudinal_mse"]
        )
        assert any(
            not th.equal(value, model.policy.residual_action.state_dict()[name])
            for name, value in residual_before.items()
        )
    finally:
        env.close()


def test_squashed_actor_pretraining_uses_deterministic_action_semantics() -> None:
    config = TrainingConfig(
        algorithm="ppo", device="cpu", track_name="Summer 1", track_id="mock/straight",
        ppo=PPOConfig(
            architecture="tqc_residual", rollout_steps=32, batch_size=16, epochs=1,
        ),
    )
    env = _gym_env(config, 105)
    model = backend_for("ppo").create_model(config, env, "cpu")
    try:
        for module in (
            model.policy.features_extractor,
            model.policy.mlp_extractor.policy_net,
            model.policy.action_net,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        with th.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_(th.tensor([2.0, -2.0]))
        observations = np.zeros((4 * 16, 105), dtype=np.float32)
        observations[:, 12] = 0.5
        actions, _ = model.predict(observations, deterministic=True)
        final_actions = np.clip(
            actions + np.asarray([0.0, 0.2], dtype=np.float32), -1.0, 1.0,
        )
        model.policy_overlays = [{
            "kind": "drive_bias", "start": 0.0, "end": 1.0,
            "amount": 0.2, "taper": 0.01,
        }]
        dataset = TeacherDataset(
            observations=observations, raw_actions=actions.copy(), final_actions=final_actions,
            driving_actions=actions.copy(), trajectory_ids=np.repeat(np.arange(4), 16),
            progress=np.tile(np.linspace(0, 1, 16), 4), elapsed_s=np.zeros(len(observations)),
            speed_mps=np.zeros(len(observations)), position_m=np.zeros((len(observations), 3)),
            heading_error_rad=np.zeros(len(observations)), wheel_contacts=np.ones((len(observations), 4)),
            overlay_active=np.zeros(len(observations), dtype=bool),
            successful_trajectory_ids=[0, 1, 2, 3], teacher_id="test",
        )
        before = {
            name: value.detach().clone()
            for name, value in model.policy.residual_action.state_dict().items()
        }
        report = pretrain_actor(
            model, dataset, epochs=5, batch_size=32, patience=5, seed=0,
            learning_rate=1e-3,
        )
        assert report["validation"]["steering_mse"] < 1e-6
        assert report["validation"]["longitudinal_mse"] < 1e-6
        assert all(
            th.equal(value, model.policy.residual_action.state_dict()[name])
            for name, value in before.items()
        )
    finally:
        env.close()


def test_value_warmup_freezes_residual_actor_but_keeps_critic_trainable() -> None:
    config = TrainingConfig(
        algorithm="ppo", device="cpu", track_name="Summer 1", track_id="mock/straight",
        ppo=PPOConfig(
            architecture="tqc_residual", rollout_steps=32, batch_size=16, epochs=1,
        ),
    )
    env = _gym_env(config, 105)
    model = backend_for("ppo").create_model(config, env, "cpu")
    try:
        assert any(
            parameter.requires_grad for parameter in model.policy.residual_action.parameters()
        )
        _freeze_ppo_actor(model)

        actor_modules = (
            model.policy.features_extractor,
            model.policy.mlp_extractor.policy_net,
            model.policy.action_net,
            model.policy.residual_action,
        )
        assert all(
            not parameter.requires_grad
            for module in actor_modules for parameter in module.parameters()
        )
        assert not model.policy.log_std.requires_grad
        assert any(
            parameter.requires_grad
            for module in (model.policy.mlp_extractor.value_net, model.policy.value_net)
            for parameter in module.parameters()
        )
    finally:
        env.close()


def test_tqc_residual_gates_policy_exploration_outside_progress_window() -> None:
    config = TrainingConfig(
        algorithm="ppo", device="cpu", track_name="Summer 1", track_id="mock/straight",
        ppo=PPOConfig(
            architecture="tqc_residual", residual_progress_start=0.6,
            residual_progress_end=1.0, action_std=0.02,
            rollout_steps=32, batch_size=16, epochs=1,
        ),
    )
    env = _gym_env(config, 105)
    model = backend_for("ppo").create_model(config, env, "cpu")
    try:
        observations = th.zeros((3, 105), dtype=th.float32)
        observations[:, 12] = th.tensor([0.59, 0.6, 0.8])
        distribution = model.policy.get_distribution(observations).distribution
        stddev = distribution.stddev

        assert th.allclose(stddev[0], th.full_like(stddev[0], 1e-4), atol=1e-8)
        assert th.allclose(stddev[1:], th.full_like(stddev[1:], 0.02), atol=1e-7)

        th.manual_seed(0)
        sampled_actions, _, _ = model.policy(observations, deterministic=False)
        mean_actions, _, _ = model.policy(observations, deterministic=True)
        assert th.equal(sampled_actions[0], mean_actions[0])
        assert not th.equal(sampled_actions[1], mean_actions[1])
        assert th.isfinite(model.policy.evaluate_actions(observations, sampled_actions)[1]).all()
    finally:
        env.close()


def test_tqc_residual_gated_policy_completes_a_finite_ppo_update() -> None:
    class ProgressEnv(gym.Env):
        observation_space = spaces.Box(-5, 5, (105,), dtype=np.float32)
        action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

        def __init__(self) -> None:
            self.step_n = 0

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self.step_n = 0
            observation = np.zeros(105, dtype=np.float32)
            observation[12] = 0.55
            return observation, {}

        def step(self, action: np.ndarray):
            self.step_n += 1
            observation = np.zeros(105, dtype=np.float32)
            observation[12] = min(0.55 + self.step_n * 0.01, 1.0)
            return observation, float(observation[12]), False, self.step_n >= 32, {}

    config = TrainingConfig(
        algorithm="ppo", backend="mock", device="cpu",
        ppo=PPOConfig(
            architecture="tqc_residual", residual_progress_start=0.75,
            residual_progress_end=1.0, action_std=0.02, learning_rate=1e-5,
            rollout_steps=32, batch_size=16, epochs=1,
        ),
    )
    model = backend_for("ppo").create_model(config, ProgressEnv(), "cpu")
    model.policy.freeze_base_actor()
    model.learn(total_timesteps=32)

    metrics = model.logger.name_to_value
    for name in (
        "train/policy_gradient_loss", "train/value_loss", "train/approx_kl", "train/loss",
    ):
        assert np.isfinite(metrics[name]), name
