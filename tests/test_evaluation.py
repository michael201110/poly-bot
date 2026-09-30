from __future__ import annotations

from polybot.training.evaluation import EvaluationResult


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

    assert reliable_sub_target.confirms_target_lap(22.0)
    assert not unreliable_fast_outlier.confirms_target_lap(22.0)
    assert not exact_target.confirms_target_lap(22.0)
