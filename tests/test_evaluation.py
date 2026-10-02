from __future__ import annotations

from dataclasses import replace

from polybot.training.evaluation import EvaluationResult, evaluate_model


def test_target_lap_requires_reliable_full_track_completion() -> None:
    reliable_sub_target = EvaluationResult(
        episodes=5, finish_rate=1.0, median_progress=1.0, mean_progress=1.0,
        best_lap_s=21.9, median_lap_s=22.1, crash_rate=0.0,
        off_track_rate=0.0, stall_rate=0.0,
    )
    unreliable_fast_outlier = EvaluationResult(
        episodes=5, finish_rate=0.2, median_progress=0.3, mean_progress=0.4,
        best_lap_s=21.8, median_lap_s=21.8, crash_rate=0.0,
        off_track_rate=0.8, stall_rate=0.0,
    )
    exact_target = EvaluationResult(
        episodes=5, finish_rate=1.0, median_progress=1.0, mean_progress=1.0,
        best_lap_s=22.0, median_lap_s=22.0, crash_rate=0.0,
        off_track_rate=0.0, stall_rate=0.0,
    )

    assert not reliable_sub_target.confirms_target_lap(22.0)
    genuinely_fast = replace(reliable_sub_target, median_lap_s=21.95)
    assert genuinely_fast.confirms_target_lap(22.0)
    assert not replace(genuinely_fast, episodes=1).confirms_target_lap(22.0)
    assert not unreliable_fast_outlier.confirms_target_lap(22.0)
    assert not exact_target.confirms_target_lap(22.0)


def test_evaluation_records_barrier_contacts_and_progress() -> None:
    class Policy:
        def set_training_mode(self, training: bool) -> None:
            pass

    class Model:
        policy = Policy()

        def predict(self, observation, deterministic: bool):
            return [0.0, 1.0], None

    class Env:
        def reset(self, *, seed: int):
            return [0.0], {}

        def step(self, action):
            return [0.0], 0.0, True, False, {
                "events": ("finish", "barrier_contact"),
                "route_progress_m": 100.0,
                "track_length_m": 100.0,
                "elapsed_s": 24.0,
                "barrier_contact_progress": [0.1, 0.99],
                "nonlanding_impulse_peak": 8_000.0,
                "air_brake_summary": {},
            }

        def close(self) -> None:
            pass

    transitions = []
    result = evaluate_model(Model(), Env, episodes=1, seed=1, transition_sink=transitions)

    assert result.crash_rate == 0
    assert len(transitions) == 1
    assert transitions[0]["done"] and not transitions[0]["timeout"]
    assert result.barrier_contact_steps == 2
    assert result.max_barrier_impulse == 8_000.0
    assert result.barrier_contact_progress == (0.1, 0.99)
