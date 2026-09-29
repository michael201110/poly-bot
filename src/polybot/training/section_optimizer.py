"""Persistent, full-lap-validated section-wise search around a TQC champion."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
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
from polybot.training.wr_search import (
    _candidate_grid,
    _write_json,
    compose_overlay_stack,
)

LEVELS = (0.10, 0.05, 0.02, 0.01)
TIMING_LAPS = 10


def section_windows(level_index: int, refine_sections: list[dict[str, float]]) -> list[tuple[float, float]]:
    """Return the coarse sweep or evenly split windows selected for refinement."""
    level = LEVELS[min(max(int(level_index), 0), len(LEVELS) - 1)]
    if level_index == 0:
        return [(round(i * level, 6), round((i + 1) * level, 6)) for i in range(10)]
    selected = sorted(refine_sections, key=lambda row: (-float(row.get("priority", 0)), row["start"]))
    output = []
    for parent in selected:
        width = parent["end"] - parent["start"]
        if width <= level + 1e-9:
            output.append((parent["start"], parent["end"]))
        else:
            count = max(1, round(width / level))
            output.extend((round(parent["start"] + i * width / count, 6),
                           round(parent["start"] + (i + 1) * width / count, 6))
                          for i in range(count))
    return output


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically replace persisted optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SectionOptimizer:
    """Run/resume sequential section sweeps; every candidate is a normal full lap."""

    def __init__(
        self, config_path: Path, *, target_s: float = 22.262,
        max_runtime_seconds: float | None = None, stop_file: Path | None = None,
        state_path: Path | None = None, max_trials: int | None = None,
        skip_file: Path | None = None, refine_file: Path | None = None,
    ) -> None:
        self.config = TrainingConfig.from_dict(
            json.loads(config_path.read_text(encoding="utf-8-sig"))
        )
        if self.config.algorithm != "tqc" or self.config.backend != "websocket":
            raise ValueError("section optimizer requires a live websocket TQC champion")
        if max_runtime_seconds is not None and max_runtime_seconds <= 0:
            raise ValueError("runtime budget must be positive")
        self.target_s = target_s
        self.deadline = time.monotonic() + max_runtime_seconds if max_runtime_seconds else None
        self.stop_file = stop_file
        self.skip_file = skip_file
        self.refine_file = refine_file
        self.max_trials = max_trials
        self.runner = TrainingRunner(self.config)
        self.champion = self.runner.registry.slot(self.config.track_name, "tqc", "champion")
        self.root = self.champion.parent
        self.state_path = state_path or self.root / "optimizer-state.json"
        self.history = self.root / "section-search-history.jsonl"
        self.started = time.monotonic()
        self.state = self._load_state()
        self.prior_runtime = float(self.state.get("runtime_seconds", 0.0))
        self.model = self._load_model()
        self.lap_traces: list[list[dict[str, Any]]] = []

    def _load_model(self):
        mock = PolyTrackEnv(
            MockSimulatorTransport(),
            action_adapter=self.runner.backend.action_adapter(self.config),
        )
        try:
            model = self.runner.backend.load_model(self.champion / "policy.zip", mock, "cpu", resume=True)
        finally:
            mock.close()
        model.policy.set_training_mode(False)
        for parameter in model.policy.parameters():
            parameter.requires_grad_(False)
        model.policy_overlays = list(self.meta.policy_overlays)
        return model

    @property
    def meta(self):
        return self.runner.registry.read_metadata(self.champion)

    def _load_state(self) -> dict[str, Any]:
        meta = self.runner.registry.read_metadata(self.champion)
        if meta.evaluation is None or meta.evaluation.get("median_lap_s") is None:
            raise ValueError("a fully evaluated champion is required")
        current_hash = _hash(self.champion / "policy.zip")
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("champion_hash") != current_hash:
                # An external promotion invalidates this section's entry state.
                state.update(champion_hash=current_hash, champion_lap_s=float(meta.evaluation["median_lap_s"]))
                state.update(level_index=0, section_index=0, candidate_index=0, candidates=[], sweep_done=False,
                             timing_floor_s=None, baseline_mean_s=None, baseline_std_s=None,
                             section_scores={}, refine_sections=[], completed_sections=[])
                state["accepted_overlays"] = list(meta.policy_overlays)
        else:
            state = {
                "schema": "polybot.section_optimizer.v1", "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                "champion_hash": current_hash, "champion_lap_s": float(meta.evaluation["median_lap_s"]),
                "target_lap_s": self.target_s, "level_index": 0, "section_index": 0,
                "candidate_index": 0, "candidates": [], "completed_sections": [], "section_scores": {},
                "accepted_overlays": list(meta.policy_overlays), "rejected_candidates": [],
                "total_trials": 0, "total_laps": 0, "runtime_seconds": 0.0,
                "best_lap_s": float(meta.evaluation["median_lap_s"]), "timing_floor_s": None,
                "sweep_done": False, "seed": self.config.seed + 71_000_000,
            }
        state["target_lap_s"] = self.target_s
        return state

    def emit(self, kind: str, **values: Any) -> None:
        event = {
            "type": kind, "timestamp": datetime.now(UTC).isoformat(), "run_id": self.state["run_id"],
            "runtime_seconds": round(time.monotonic() - self.started, 3),
            "champion_hash": self.state["champion_hash"], "champion_lap_s": self.state["champion_lap_s"],
            **values,
        }
        self.history.parent.mkdir(parents=True, exist_ok=True)
        with self.history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    def checkpoint(self) -> None:
        self.state["runtime_seconds"] = round(
            self.prior_runtime + time.monotonic() - self.started, 3
        )
        self.state["updated_at"] = datetime.now(UTC).isoformat()
        write_checkpoint(self.state_path, self.state)

    def should_stop(self) -> str | None:
        if self.stop_file is not None and self.stop_file.exists():
            return "stop_file"
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return "runtime_budget"
        if self.max_trials is not None and self.state["total_trials"] >= self.max_trials:
            return "trial_budget"
        if float(self.state["champion_lap_s"]) <= self.target_s:
            return "target_reached"
        return None

    def evaluate(self, episodes: int, *, telemetry_sink=None):
        # Retry a failed bridge evaluation a bounded number of times. evaluate_model
        # closes its environment in finally; restarting the whole lap avoids using
        # a partial episode as evidence.
        last_error = None
        for attempt in range(3):
            try:
                return evaluate_model(
                    self.model, self.runner._environment, episodes=episodes,
                    seed=int(self.state["seed"]), telemetry_sink=telemetry_sink,
                )
            except (OSError, TimeoutError, ConnectionError, RuntimeError) as exc:
                last_error = exc
                self.emit("evaluation_retry", attempt=attempt + 1, episodes=episodes, error=str(exc))
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"evaluation failed after three full retries: {last_error}")

    def measure_baseline(self) -> None:
        if self.state.get("timing_floor_s") is not None:
            return
        self.lap_traces = []
        result = self.evaluate(TIMING_LAPS, telemetry_sink=self.lap_traces)
        if result.finish_rate != 1 or result.median_lap_s is None:
            raise RuntimeError("champion failed the 10-lap optimizer baseline")
        times = [float(row[-1]["elapsed_s"]) for row in self.lap_traces if row]
        mean = statistics.fmean(times)
        deviation = statistics.pstdev(times) if len(times) > 1 else 0.0
        floor = max(max(times) - min(times), 3.0 * deviation, 0.001)
        self.state.update(timing_floor_s=floor, baseline_mean_s=mean,
                          baseline_std_s=deviation, total_laps=self.state["total_laps"] + TIMING_LAPS)
        self.emit("baseline_measured", median_lap_s=result.median_lap_s, mean_lap_s=mean,
                  std_lap_s=deviation, measurement_floor_s=floor, laps=TIMING_LAPS)
        self.checkpoint()

    def _build_candidates(self, start: float, end: float) -> list[dict[str, Any]]:
        regions = [(start, end)]
        airborne = discover_airborne_regions(self.lap_traces)
        near = [r for r in airborne if r["start"] < end and r["end"] > start and r.get("landed", 0) >= 1]
        rows = _candidate_grid(regions, [], family="all")
        # Air brake controls are only generated for airborne intervals that
        # overlap this section. The action wrapper independently enforces the
        # all-four-wheels-airborne gate at runtime.
        for flight in near:
            air_start = max(start, float(flight["start"]))
            air_end = min(end, float(flight["end"]))
            if air_end <= air_start:
                continue
            for duty in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.35, 0.5, 0.75, 1.0):
                rows.append({"kind": "air_brake", "start": air_start, "end": air_end,
                             "duty": duty, "taper": min(0.005, (air_end - air_start) / 2)})
        # Include progressively finer micro values as windows shrink; always keep
        # boundaries smooth and scale taper to the actual window width.
        width = end - start
        for kind, values in (("steer_bias", (-0.003, 0.003)), ("steer_gain", (0.990, 1.010)),
                             ("drive_bias", (-0.003, 0.003)), ("drive_gain", (0.990, 1.010))):
            for value in values:
                rows.append({"kind": kind, "start": start, "end": end, "amount": value,
                             "taper": min(0.01, width / 2)})
        for row in rows:
            row["taper"] = min(float(row.get("taper", 0.005)), width / 2, 0.02)
        # deterministic order, no duplicate coordinate/value candidates
        unique = {}
        for row in rows:
            key = tuple(sorted(row.items()))
            unique[key] = row
        return list(unique.values())

    def _sections(self) -> list[tuple[float, float]]:
        return section_windows(int(self.state["level_index"]), self.state.get("refine_sections", []))

    def _promote(self, combined: list[dict[str, Any]], result, laps: list[list[dict[str, Any]]]) -> None:
        stage = self.root / f".{self.champion.name}-section-staging-{uuid4().hex}"
        self.runner.backend.save_model(self.model, stage, resume=True)
        metadata = replace(
            self.meta, evaluation=result.to_dict(), policy_overlays=combined,
            critic_adaptation_required=True, adaptation_stage="critic_adaptation_required",
            saved_at=datetime.now(UTC).isoformat(), git_commit=git_commit(),
        )
        self.runner.registry.write_metadata(stage, metadata)
        _write_json(stage / "best-pace-config.json", {
            "target_lap_s": self.target_s, "overlays": combined,
            "champion_lap_s": result.median_lap_s,
        })
        for filename in ("speed-search.json",):
            if (self.champion / filename).is_file():
                shutil.copy2(self.champion / filename, stage / filename)
        promote_directory(stage, self.champion, require_replay=True)
        self.state.update(
            champion_hash=_hash(self.champion / "policy.zip"),
            champion_lap_s=float(result.median_lap_s),
            best_lap_s=min(float(self.state["best_lap_s"]), float(result.median_lap_s)),
            accepted_overlays=combined,
        )
        append_pace_history(
            self.champion, source="section_policy_overlay_search", evaluation=result,
            model=self.model, reward_profile=self.config.reward_profile,
            air_brake_bonus_per_s=self.config.rewards.airborne_brake_bonus_per_s,
            learning_rate=None,
        )
        self.checkpoint()
        self.emit("champion_promoted", median_lap_s=result.median_lap_s, overlays=combined)

    def _run_trial(self, start: float, end: float, candidate: dict[str, Any]) -> bool:
        before = deepcopy(self.model.policy_overlays)
        combined = compose_overlay_stack(before, candidate)
        self.model.policy_overlays = combined
        trace: list[list[dict[str, Any]]] = []
        screened = self.evaluate(1, telemetry_sink=trace)
        self.state["total_laps"] += 1
        self.state["total_trials"] += 1
        deltas = (
            sector_delta_map(self.lap_traces[0], trace[0], max(0.01, end - start))
            if self.lap_traces and trace else []
        )
        base = float(self.state["champion_lap_s"])
        floor = float(self.state["timing_floor_s"] or 0.001)
        pass_screen = (
            screened.finish_rate == 1 and screened.median_progress == 1
            and screened.crash_rate == 0 and screened.off_track_rate == 0
            and screened.stall_rate == 0 and screened.median_lap_s is not None
            and screened.median_lap_s < base
        )
        confirmation = None
        accepted = False
        reason = "screen_regression_or_invalid_finish"
        if pass_screen and not self.should_stop():
            gain = base - float(screened.median_lap_s)
            confirmations = 10 if gain < max(0.02, floor * 2) else 5
            self.emit("confirmation_started", section=[start, end], candidate=candidate,
                      screen_lap_s=screened.median_lap_s, laps=confirmations)
            confirmation_trace: list[list[dict[str, Any]]] = []
            confirmation = self.evaluate(confirmations, telemetry_sink=confirmation_trace)
            self.state["total_laps"] += confirmations
            accepted = (
                confirmation.finish_rate == 1 and confirmation.median_progress == 1
                and confirmation.crash_rate == 0 and confirmation.off_track_rate == 0
                and confirmation.stall_rate == 0 and confirmation.median_lap_s is not None
                and confirmation.median_lap_s < base - floor
            )
            reason = "confirmed_faster_and_reliable" if accepted else "confirmation_below_noise_or_invalid"
            if accepted:
                self._promote(combined, confirmation, confirmation_trace)
                self.lap_traces = confirmation_trace
                self.state["section_scores"].setdefault(f"{start:.3f}-{end:.3f}", {"gains": 0, "near_misses": 0})
                self.state["section_scores"][f"{start:.3f}-{end:.3f}"]["gains"] += 1
            else:
                self.model.policy_overlays = before
        else:
            self.model.policy_overlays = before
        if not accepted and pass_screen:
            self.state["section_scores"].setdefault(f"{start:.3f}-{end:.3f}", {"gains": 0, "near_misses": 0})
            self.state["section_scores"][f"{start:.3f}-{end:.3f}"]["near_misses"] += 1
            self.state["rejected_candidates"].append({"section": [start, end], "candidate": candidate,
                                                        "screen_lap_s": screened.median_lap_s})
        self.emit("trial", section=[start, end], resolution=end-start, candidate=candidate,
                  screen_lap_s=screened.median_lap_s, confirmation=confirmation.to_dict() if confirmation else None,
                  accepted=accepted, rejection_reason=None if accepted else reason, sector_deltas=deltas,
                  total_trials=self.state["total_trials"], total_laps=self.state["total_laps"])
        return accepted

    def run(self) -> dict[str, Any]:
        self.emit("optimizer_resumed" if self.state.get("total_trials") else "optimizer_started",
                  target_lap_s=self.target_s)
        self.measure_baseline()
        if not self.lap_traces:
            self.lap_traces = []
            self.evaluate(TIMING_LAPS, telemetry_sink=self.lap_traces)
            self.state["total_laps"] += TIMING_LAPS
            self.checkpoint()
        while True:
            stop = self.should_stop()
            if stop:
                kind = "target_reached" if stop == "target_reached" else "optimizer_stopped"
                self.checkpoint()
                self.emit(kind, reason=stop, total_trials=self.state["total_trials"],
                          total_laps=self.state["total_laps"])
                return {"reason": stop, **self.state}
            sections = self._sections()
            if not sections:
                self.checkpoint()
                self.emit("sweep_completed", total_trials=self.state["total_trials"],
                          reason="no_promising_sections_to_refine")
                return {"reason": "sweep_completed", **self.state}
            if self.state["section_index"] >= len(sections):
                if self.state["level_index"] < len(LEVELS) - 1:
                    # Refine sections that made gains or had screen-passing near misses.
                    hot = []
                    for key, score in self.state["section_scores"].items():
                        start, end = (float(value) for value in key.split("-"))
                        priority = score.get("gains", 0) * 2 + score.get("near_misses", 0)
                        if priority:
                            hot.append({"start": start, "end": end, "priority": priority})
                    if not hot:
                        self.checkpoint()
                        self.emit("sweep_completed", total_trials=self.state["total_trials"],
                                  reason="no_promising_sections_to_refine")
                        return {"reason": "sweep_completed", **self.state}
                    self.state.update(level_index=self.state["level_index"] + 1, section_index=0,
                                      candidate_index=0, candidates=[], refine_sections=hot)
                    self.emit("refinement_started", level=LEVELS[self.state["level_index"]], sections=hot)
                    self.checkpoint()
                    continue
                self.checkpoint()
                self.emit("sweep_completed", total_trials=self.state["total_trials"],
                          reason="maximum_refinement_depth_reached")
                return {"reason": "sweep_completed", **self.state}
            start, end = sections[self.state["section_index"]]
            if self.refine_file is not None and self.refine_file.exists():
                self.refine_file.unlink(missing_ok=True)
                self.state.update(
                    level_index=min(self.state["level_index"] + 1, len(LEVELS) - 1),
                    refine_sections=[{"start": start, "end": end, "priority": 100}],
                    section_index=0, candidate_index=0, candidates=[],
                )
                self.emit("refinement_started", section=[start, end],
                          level=LEVELS[self.state["level_index"]], forced=True)
                self.checkpoint()
                continue
            if self.skip_file is not None and self.skip_file.exists():
                self.skip_file.unlink(missing_ok=True)
                self.state["completed_sections"].append({"section": [start, end], "skipped": True})
                self.state.update(section_index=self.state["section_index"] + 1,
                                  candidate_index=0, candidates=[])
                self.emit("section_completed", section=[start, end], skipped=True)
                self.checkpoint()
                continue
            if not self.state["candidates"]:
                self.state["candidates"] = self._build_candidates(start, end)
                self.state["candidate_index"] = 0
                self.emit("section_started", section=[start, end], resolution=end-start,
                          level=LEVELS[self.state["level_index"]], candidate_count=len(self.state["candidates"]))
                self.checkpoint()
            candidates = self.state["candidates"]
            if self.state["candidate_index"] >= len(candidates):
                key = f"{start:.3f}-{end:.3f}"
                self.state["completed_sections"].append({
                    "section": [start, end], "level": LEVELS[self.state["level_index"]],
                    "champion_lap_s": self.state["champion_lap_s"],
                })
                self.state["section_index"] += 1
                self.state["candidate_index"] = 0
                self.state["candidates"] = []
                self.emit("section_completed", section=[start, end], score=self.state["section_scores"].get(key, {}))
                self.checkpoint()
                continue
            if self.should_stop():
                self.checkpoint()
                continue
            candidate = candidates[self.state["candidate_index"]]
            try:
                self._run_trial(start, end, candidate)
            except (OSError, TimeoutError, ConnectionError, RuntimeError) as exc:
                self.model.policy_overlays = list(self.meta.policy_overlays)
                self.emit("trial_error", section=[start, end], candidate=candidate, error=str(exc))
                self.checkpoint()
                raise
            self.state["candidate_index"] += 1
            self.checkpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target", type=float, default=22.262)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--skip-file", type=Path)
    parser.add_argument("--refine-file", type=Path)
    args = parser.parse_args()
    budget = args.max_runtime_seconds if args.max_runtime_seconds is not None else (
        args.hours * 3600 if args.hours is not None else None
    )
    result = SectionOptimizer(args.config, target_s=args.target, max_runtime_seconds=budget,
                              stop_file=args.stop_file, state_path=args.state,
                              max_trials=args.max_trials, skip_file=args.skip_file,
                              refine_file=args.refine_file).run()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
