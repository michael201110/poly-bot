"""Serial coarse-to-fine, lap-time-only search around a frozen TQC champion."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import git_commit
from polybot.training.config import TrainingConfig
from polybot.training.evaluation import evaluate_model
from polybot.training.lap_analysis import discover_airborne_regions, sector_delta_map
from polybot.training.pace_history import append_pace_history
from polybot.training.promotion import promote_directory
from polybot.training.runner import TrainingRunner

STEERING_BIASES = (-0.001, 0.001, -0.002, 0.002, -0.005, 0.005, -0.01, 0.01)
STEERING_GAINS = (0.998, 1.002, 0.995, 1.005)
DRIVE_BIASES = (-0.002, 0.002, -0.005, 0.005, -0.01, 0.01, -0.02, 0.02)
DRIVE_GAINS = (0.998, 1.002, 0.995, 1.005)
# ContinuousPwmControls interprets these values as pulse density. Small duties
# mostly tap the brake; include sustained and fully held braking for air-search.
AIR_BRAKE_DUTIES = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0)


def emit(history: Path, event: dict[str, Any]) -> None:
    record = {"timestamp": datetime.now(UTC).isoformat(), "git_commit": git_commit(), **event}
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def confirmation_count_for_gain(
    improvement_s: float, *, extra_confirmation_threshold_s: float,
    minimum_confirmation_episodes: int, micro_confirmation_episodes: int,
) -> int:
    if extra_confirmation_threshold_s < 0:
        raise ValueError("extra confirmation threshold cannot be negative")
    if minimum_confirmation_episodes < 5 or micro_confirmation_episodes < minimum_confirmation_episodes:
        raise ValueError("promotion requires at least five confirmation laps")
    return (
        micro_confirmation_episodes
        if improvement_s < extra_confirmation_threshold_s
        else minimum_confirmation_episodes
    )


def compose_overlay_stack(
    existing: list[dict[str, Any]], candidate: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace the same coordinate while retaining independent learned layers."""
    start, end = float(candidate["start"]), float(candidate["end"])
    retained = [
        layer for layer in existing
        if not (
            layer.get("kind") == candidate.get("kind")
            and abs(float(layer.get("start", -1)) - start) <= 1e-9
            and abs(float(layer.get("end", -1)) - end) <= 1e-9
        )
    ]
    return [*retained, candidate]


def search(
    config_path: Path, *, max_trials: int = 12, target_s: float,
    resolution: float = 0.05, confirmation_episodes: int = 5,
    micro_gain_s: float = 0.01, micro_confirmation_episodes: int = 10,
    family: str = "all",
    region: tuple[float, float] | None = None, stop_file: Path | None = None,
) -> dict[str, Any]:
    if max_trials < 0 or not 0 < resolution <= 0.2:
        raise ValueError("trials must be non-negative and sector resolution in (0, 0.2]")
    if family not in {"all", "steering", "drive", "air_brake"}:
        raise ValueError("unknown search parameter family")
    if region is not None and not (0 <= region[0] < region[1] <= 1):
        raise ValueError("search region must be an increasing progress window within [0, 1]")
    config = TrainingConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8-sig")))
    if config.algorithm != "tqc" or config.backend != "websocket":
        raise ValueError("WR pace search requires live websocket TQC")
    if confirmation_episodes < 5 or micro_confirmation_episodes < confirmation_episodes:
        raise ValueError("promotion requires five laps; micro-gains require at least as many")
    runner = TrainingRunner(config)
    champion_dir = runner.registry.slot(config.track_name, "tqc", "champion")
    champion_meta = runner.registry.read_metadata(champion_dir)
    if champion_meta.evaluation is None or champion_meta.evaluation.get("median_lap_s") is None:
        raise ValueError("a fully evaluated champion is required")
    baseline_lap = float(champion_meta.evaluation["median_lap_s"])
    mock = PolyTrackEnv(MockSimulatorTransport(), action_adapter=runner.backend.action_adapter(config))
    try:
        model = runner.backend.load_model(
            champion_dir / "policy.zip", mock, "cpu", resume=True
        )
    finally:
        mock.close()
    # WR search treats TQC as a read-only base policy. Its optimizers are never called.
    model.policy.set_training_mode(False)
    for parameter in model.policy.parameters():
        parameter.requires_grad_(False)
    model.policy_overlays = list(champion_meta.policy_overlays)
    history = champion_dir.parent / "wr-search-history.jsonl"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    traces: list[list[dict[str, Any]]] = []
    baseline = evaluate_model(
        model, runner._environment, episodes=10, seed=config.seed + 7_000_000,
        telemetry_sink=traces,
    )
    if baseline.finish_rate < 1.0 or baseline.median_lap_s is None:
        raise RuntimeError("champion failed its 10-lap deterministic search baseline")
    baseline_lap = float(baseline.median_lap_s)
    lap_times = [float(trace[-1]["elapsed_s"]) for trace in traces]
    timing_floor = (max(lap_times) - min(lap_times)) if lap_times else 0.0
    sectors = _median_sectors(traces, resolution)
    ranked = sorted(sectors, key=lambda row: (-row["candidate_s"], row["speed_mps"]))
    focus_regions = {(row["start"], row["end"]) for row in ranked[:3]}
    for row in sectors:
        row["focus"] = (row["start"], row["end"]) in focus_regions
    airborne = discover_airborne_regions(traces)
    _write_json(champion_dir.parent / "champion-sector-map.json", {
        "timestamp": datetime.now(UTC).isoformat(), "resolution": resolution,
        "champion_lap_s": baseline.median_lap_s, "lap_times_s": lap_times,
        "timing_range_s": timing_floor, "sectors": sectors,
        "airborne_regions": airborne,
    })
    emit(history, {
        "type": "baseline", "run_id": run_id, "champion_lap_s": baseline.median_lap_s,
        "target_lap_s": target_s, "gap_s": baseline.median_lap_s - target_s,
        "episodes": 10, "finish_rate": baseline.finish_rate,
        "timing_range_s": timing_floor, "measurement_floor_s": max(timing_floor, 0.001),
        "sector_resolution": resolution, "airborne_regions": airborne,
        "sectors": sectors,
    })

    # Slowest, low-speed sectors are a useful initial scan ranking, not a claim
    # about a WR ghost that PolyBot does not currently have.
    regions = [region] if region is not None else [(row["start"], row["end"]) for row in ranked[:3]]
    search_airborne = airborne
    if region is not None:
        search_airborne = [
            item for item in airborne if item["start"] < region[1] and item["end"] > region[0]
        ]
    candidates = _candidate_grid(regions, search_airborne, family=family)
    candidates = candidates[:max_trials]
    accepted = 0
    trial_id = 0
    champion_id = hashlib.sha256((champion_dir / "policy.zip").read_bytes()).hexdigest()
    for overlay in candidates:
        if stop_file is not None and stop_file.exists():
            emit(history, {"type": "stopped", "run_id": run_id,
                           "trials": trial_id, "accepted": accepted})
            break
        trial_id += 1
        parent_champion_id = champion_id
        old_stack = deepcopy(model.policy_overlays)
        combined = compose_overlay_stack(old_stack, overlay)
        trial_dir = champion_dir.parent / f"wr-search-{run_id}-trial-{trial_id:04d}-{uuid4().hex[:8]}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        _write_json(trial_dir / "candidate.json", {
            "parent_champion": parent_champion_id, "overlays": combined,
        })
        model.policy_overlays = combined
        candidate_trace: list[list[dict[str, Any]]] = []
        screened = evaluate_model(
            model, runner._environment, episodes=1, seed=config.seed + 7_000_000,
            telemetry_sink=candidate_trace,
        )
        deltas = sector_delta_map(traces[0], candidate_trace[0], resolution) if candidate_trace else []
        passed = (
            screened.finish_rate == 1.0 and screened.median_progress == 1.0
            and screened.crash_rate == 0 and screened.off_track_rate == 0 and screened.stall_rate == 0
            and screened.median_lap_s is not None and screened.median_lap_s < baseline_lap
        )
        confirmation = None
        if passed:
            gain = baseline_lap - float(screened.median_lap_s)
            required = confirmation_count_for_gain(
                gain, extra_confirmation_threshold_s=micro_gain_s,
                minimum_confirmation_episodes=confirmation_episodes,
                micro_confirmation_episodes=micro_confirmation_episodes,
            )
            confirmation_trace: list[list[dict[str, Any]]] = []
            confirmation = evaluate_model(
                model, runner._environment, episodes=required,
                seed=config.seed + 7_000_000, telemetry_sink=confirmation_trace,
            )
            passed = (
                confirmation.finish_rate == 1.0 and confirmation.median_progress == 1.0
                and confirmation.crash_rate == 0 and confirmation.off_track_rate == 0
                and confirmation.stall_rate == 0 and confirmation.median_lap_s is not None
                and confirmation.median_lap_s < baseline_lap
            )
            if passed:
                stage = champion_dir.parent / f".{champion_dir.name}-wr-staging-{uuid4().hex}"
                runner.backend.save_model(model, stage, resume=True)
                updated = replace(
                    champion_meta, evaluation=confirmation.to_dict(),
                    critic_adaptation_required=True,
                    adaptation_stage="critic_adaptation_required",
                    policy_overlays=combined,
                    saved_at=datetime.now(UTC).isoformat(), git_commit=git_commit(),
                )
                runner.registry.write_metadata(stage, updated)
                (stage / "best-pace-config.json").write_text(
                    json.dumps({"overlays": combined}, indent=2) + "\n", encoding="utf-8"
                )
                for filename in ("speed-search.json",):
                    if (champion_dir / filename).is_file():
                        shutil.copy2(champion_dir / filename, stage / filename)
                backup = promote_directory(stage, champion_dir, require_replay=True)
                if backup is not None:
                    # promote_directory already retains the complete previous champion.
                    champion_id = hashlib.sha256(
                        (champion_dir / "policy.zip").read_bytes()
                    ).hexdigest()
                champion_meta = updated
                baseline_lap = float(confirmation.median_lap_s)
                traces = confirmation_trace
                traces_for_sector = confirmation_trace
                sectors = _median_sectors(traces_for_sector, resolution)
                accepted += 1
                append_pace_history(
                    champion_dir, source="wr_policy_overlay_search", evaluation=confirmation,
                    model=model, reward_profile=config.reward_profile,
                    air_brake_bonus_per_s=config.rewards.airborne_brake_bonus_per_s,
                    learning_rate=None,
                )
            else:
                model.policy_overlays = old_stack
        else:
            model.policy_overlays = old_stack
        emit(history, {
            "type": "trial", "run_id": run_id, "trial_id": trial_id,
            "parent_champion": parent_champion_id, "parameter": overlay,
            "screen_lap_s": screened.median_lap_s, "screen_finish_rate": screened.finish_rate,
            "confirmation": confirmation.to_dict() if confirmation else None,
            "sector_deltas": deltas, "accepted": passed,
            "rejection_reason": None if passed else "not faster and reliable after confirmation",
        })
        shutil.rmtree(trial_dir, ignore_errors=True)
        if baseline_lap <= target_s:
            break

    best_config = {"target_lap_s": target_s, "overlays": champion_meta.policy_overlays,
                   "champion_lap_s": baseline_lap, "gap_s": baseline_lap - target_s}
    _write_json(champion_dir.parent / "best-pace-config.json", best_config)
    result = {
        "type": "completed", "run_id": run_id, "champion_lap_s": baseline_lap,
        "target_lap_s": target_s, "gap_s": baseline_lap - target_s,
        "trials": trial_id, "accepted": accepted, "measurement_floor_s": max(timing_floor, 0.001),
        "airborne_regions": airborne,
    }
    emit(history, result)
    return result


def _median_sectors(
    traces: list[list[dict[str, Any]]], resolution: float,
) -> list[dict[str, float]]:
    if not traces:
        return []
    maps = []
    for trace in traces:
        maps.append({(row["start"], row["end"]): row for row in sector_delta_map(trace, trace, resolution)})
    keys = sorted(set.intersection(*(set(item) for item in maps))) if maps else []
    rows = []
    for start, end in keys:
        speeds = [
            statistics.fmean(float(sample.get("speed_mps", 0)) for sample in trace
                             if start <= _progress(sample) <= end)
            for trace in traces
        ]
        times = [
            next(row["candidate_s"] for row in sector_delta_map(trace, trace, resolution)
                 if row["start"] == start and row["end"] == end)
            for trace in traces
        ]
        rows.append({"start": start, "end": end,
                     "candidate_s": float(statistics.median(times)),
                     "speed_mps": float(statistics.median(speeds))})
    return rows


def _progress(sample: dict[str, Any]) -> float:
    return float(sample.get("route_progress_m", 0)) / max(float(sample.get("track_length_m", 1)), 1e-9)


def _candidate_grid(
    regions: list[tuple[float, float]], airborne: list[dict[str, float]],
    *, family: str = "all",
) -> list[dict[str, Any]]:
    kinds = (
        ("steer_bias", STEERING_BIASES), ("steer_gain", STEERING_GAINS),
        ("drive_bias", DRIVE_BIASES), ("drive_gain", DRIVE_GAINS),
    )
    result: list[dict[str, Any]] = []
    for start, end in regions:
        for kind, values in kinds:
            if family == "air_brake":
                continue
            if family == "steering" and not kind.startswith("steer"):
                continue
            if family == "drive" and not kind.startswith("drive"):
                continue
            for value in values:
                result.append({"kind": kind, "start": start, "end": end,
                               "amount": value, "taper": 0.01})
    landed_regions = [region for region in airborne if region.get("landed", 0) >= 1]
    if landed_regions and family in {"all", "air_brake"}:
        longest = max(landed_regions, key=lambda region: region["duration_s"])
        for duty in AIR_BRAKE_DUTIES:
            result.append({"kind": "air_brake", "start": longest["start"],
                           "end": longest["end"], "duty": duty,
                           "taper": min(0.003, (longest["end"] - longest["start"]) / 2)})
    # Keep coordinate descent reproducible: each parameter family is tested
    # locally, and every accepted overlay becomes the parent for the next one.
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--family", choices=("all", "steering", "drive", "air_brake"), default="all")
    parser.add_argument("--region-start", type=float)
    parser.add_argument("--region-end", type=float)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--confirm", type=int, default=5)
    parser.add_argument("--micro-gain", type=float, default=0.01)
    parser.add_argument("--micro-confirm", type=int, default=10)
    args = parser.parse_args()
    if (args.region_start is None) != (args.region_end is None):
        parser.error("--region-start and --region-end must be provided together")
    selected_region = ((args.region_start, args.region_end)
                       if args.region_start is not None and args.region_end is not None else None)
    search(args.config, max_trials=args.trials, target_s=args.target,
           resolution=args.resolution, confirmation_episodes=args.confirm,
           micro_gain_s=args.micro_gain, micro_confirmation_episodes=args.micro_confirm,
           family=args.family, region=selected_region, stop_file=args.stop_file)


if __name__ == "__main__":
    main()
