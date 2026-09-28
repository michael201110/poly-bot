"""Search small TQC actor-output changes on the live Summer 1 simulator.

Every candidate is screened on one deterministic lap. A faster completed lap
must then pass the normal five-lap champion evaluation before it is saved.
The saved champion and replay are never used as scratch space.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import git_commit
from polybot.training.config import TrainingConfig
from polybot.training.evaluation import evaluate_model
from polybot.training.pace_history import append_pace_history
from polybot.training.promotion import promote_directory
from polybot.training.runner import TrainingRunner


def emit(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"at": datetime.now(UTC).isoformat(), **event}) + "\n")
    print(json.dumps(event), flush=True)


def search(
    config_path: Path, log_path: Path, max_trials: int, target_s: float,
    stop_file: Path | None = None, mode: str = "global",
) -> None:
    if max_trials < 1 or target_s <= 0:
        raise ValueError("trials and target seconds must be positive")
    if mode not in {"global", "section"}:
        raise ValueError("search mode must be global or section")
    config = TrainingConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8-sig")))
    if config.algorithm != "tqc" or config.backend != "websocket":
        raise ValueError("live speed search requires a websocket TQC profile")
    runner = TrainingRunner(config)
    champion_dir = runner.registry.slot(config.track_name, "tqc", "champion")
    champion = runner.registry.read_metadata(champion_dir)
    if champion.evaluation is None or champion.evaluation["median_lap_s"] is None:
        raise ValueError("a completed evaluated champion is required")

    mock = PolyTrackEnv(MockSimulatorTransport(), action_adapter=runner.backend.action_adapter(config))
    try:
        model = runner.backend.load_model(champion_dir / "policy.zip", mock, "cpu", resume=True)
    finally:
        mock.close()

    best_lap = float(champion.evaluation["median_lap_s"])
    best_actor = deepcopy(model.actor.state_dict())
    best_schedule = deepcopy(getattr(model, "speed_bias_schedule", []))
    rng = random.Random(config.seed + 25)
    run_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Prefer changes that preserve the existing route; exploration broadens later.
    probes = [
        (steer, 0.0, 1.0, 1.0)
        for steer in (-0.02, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01, 0.02)
    ] + [
        (0.0, drive, 1.0, 1.0)
        for drive in (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.04)
    ] + [
        (0.0, 0.0, gain, 1.0) for gain in (0.98, 0.99, 1.01, 1.02)
    ] + [
        (0.0, 0.0, 1.0, gain) for gain in (0.98, 0.99, 1.01, 1.02)
    ]
    windows = (
        (0.34, 0.55), (0.55, 0.75), (0.75, 1.0),
        (0.34, 1.0), (0.5, 1.0), (0.24, 0.36), (0.1, 0.22),
    )
    section_probes = [
        (start, end, amount)
        for start, end in windows
        for amount in (0.005, 0.01, -0.005, -0.01, 0.02, -0.02)
    ]
    emit(log_path, {"type": "started", "champion_lap_s": best_lap,
                    "target_s": target_s, "max_trials": max_trials, "mode": mode})

    for trial in range(1, max_trials + 1):
        if best_lap <= target_s or (stop_file is not None and stop_file.exists()):
            break
        if mode == "global":
            section = None
            if trial <= len(probes):
                steer_bias, drive_bias, steer_gain, drive_gain = probes[trial - 1]
            else:
                broad = trial % 7 == 0
                steer_bias = rng.gauss(0, 0.02 if broad else 0.005)
                drive_bias = rng.gauss(0, 0.05 if broad else 0.015)
                steer_gain = max(0.95, min(1.05, rng.gauss(1, 0.03 if broad else 0.01)))
                drive_gain = max(0.95, min(1.05, rng.gauss(1, 0.03 if broad else 0.01)))
        else:
            steer_bias = drive_bias = 0.0
            steer_gain = drive_gain = 1.0
            if trial <= len(section_probes):
                section = section_probes[trial - 1]
            else:
                start, end = rng.choice(windows)
                section = (start, end, rng.gauss(0, 0.03 if trial % 7 == 0 else 0.01))

        model.actor.load_state_dict(best_actor)
        model.speed_bias_schedule = deepcopy(best_schedule)
        with torch.no_grad():
            model.actor.mu.weight[0].mul_(steer_gain)
            model.actor.mu.weight[1].mul_(drive_gain)
            model.actor.mu.bias[0].add_(steer_bias)
            model.actor.mu.bias[1].add_(drive_bias)
        if section is not None:
            model.speed_bias_schedule.append(section)
        parameters = {"steer_bias": steer_bias, "drive_bias": drive_bias,
                      "steer_gain": steer_gain, "drive_gain": drive_gain,
                      "speed_window": section}
        try:
            screened = evaluate_model(
                model, runner._environment, episodes=1, seed=config.seed + 1_000_000
            )
            emit(log_path, {"type": "trial", "trial": trial, "parameters": parameters,
                            "evaluation": screened.to_dict()})
            if screened.finish_rate < 1 or screened.median_lap_s is None:
                continue
            if screened.median_lap_s >= best_lap:
                continue

            confirmed = evaluate_model(
                model, runner._environment, episodes=config.evaluation.episodes,
                seed=config.seed + 1_000_000,
            )
            if confirmed.finish_rate < 1 or confirmed.median_lap_s is None:
                emit(log_path, {"type": "rejected", "trial": trial,
                                "evaluation": confirmed.to_dict()})
                continue
            if confirmed.median_lap_s >= best_lap:
                continue

            staging = champion_dir.parent / f"speed-search-{run_tag}-trial-{trial}"
            if staging.exists():
                raise FileExistsError(staging)
            runner.backend.save_model(model, staging, resume=True)
            updated = replace(
                champion, evaluation=confirmed.to_dict(),
                critic_adaptation_required=True,
                adaptation_stage="critic_adaptation_required",
                saved_at=datetime.now(UTC).isoformat(), git_commit=git_commit(),
            )
            runner.registry.write_metadata(staging, updated)
            (staging / "speed-search.json").write_text(json.dumps({
                "method": "live actor-output parameter search",
                "trial": trial, "parameters": parameters,
                "previous_champion_median_lap_s": best_lap,
                "candidate_median_lap_s": confirmed.median_lap_s,
                "delta_s": confirmed.median_lap_s - best_lap,
                "speed_bias_schedule": model.speed_bias_schedule,
                "evaluation": confirmed.to_dict(),
            }, indent=2) + "\n", encoding="utf-8")
            promote_directory(staging, champion_dir, require_replay=True)
            append_pace_history(
                champion_dir,
                source=f"speed_search_{mode}", evaluation=confirmed, model=model,
                reward_profile=config.reward_profile,
                air_brake_bonus_per_s=config.rewards.airborne_brake_bonus_per_s,
                learning_rate=config.tqc.learning_rate if config.tqc else None,
            )
            champion = updated
            best_lap = float(confirmed.median_lap_s)
            best_actor = deepcopy(model.actor.state_dict())
            best_schedule = deepcopy(model.speed_bias_schedule)
            emit(log_path, {"type": "champion", "trial": trial, "lap_s": best_lap,
                            "parameters": parameters,
                            "critic_adaptation_required": True})
        except Exception as exc:
            emit(log_path, {"type": "error", "trial": trial, "error": repr(exc)})
            raise
    emit(log_path, {"type": "completed", "best_lap_s": best_lap,
                    "target_met": best_lap <= target_s,
                    "stopped_by_user": stop_file is not None and stop_file.exists()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--target", type=float, default=25.0)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--mode", choices=("global", "section"), default="global")
    args = parser.parse_args()
    search(args.config, args.log, args.trials, args.target, args.stop_file, args.mode)


if __name__ == "__main__":
    main()
