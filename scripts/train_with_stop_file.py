"""Run a PolyBot config until completion or a clean stop-file request."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import replace
from pathlib import Path

from polybot.environment.curriculum import build_plan
from polybot.models.registry import ModelRegistry
from polybot.protocol import ProtocolViolation
from polybot.training.config import CurriculumConfig, CurriculumPhaseConfig, TrainingConfig
from polybot.training.runner import TrainingRunner

RECONNECT_DELAY_S = 3.0


def remaining_config(config: TrainingConfig, completed: int) -> TrainingConfig | None:
    """Reconnect within the original budget and curriculum, without replaying phases."""
    if completed >= config.timesteps:
        return None
    phases = []
    for phase in build_plan(config.curriculum, config.timesteps).phases:
        if completed >= phase.steps:
            completed -= phase.steps
            continue
        phases.append(CurriculumPhaseConfig(
            mode=phase.mode, steps=phase.steps - completed,
            start_ratio=phase.start_ratio, end_ratio=phase.end_ratio,
            start_s=phase.start_s, end_s=phase.end_s, lead_in_ratio=phase.lead_in_ratio,
        ))
        completed = 0
    return replace(
        config, timesteps=sum(phase.steps for phase in phases),
        curriculum=CurriculumConfig("custom", phases=tuple(phases)),
    )


def should_continue(config: TrainingConfig, *, requested: bool, target_reached: bool, stop_file: Path) -> bool:
    """Continue another training budget only for an explicitly supervised scratch run."""
    if requested and (config.algorithm != "grtqc" or config.grtqc is None
                      or config.grtqc.training_origin != "scratch"):
        raise ValueError("continue-until-target is only supported for scratch GRTQC")
    return requested and not target_reached and not stop_file.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fresh-replay", action="store_true")
    parser.add_argument("--rollback-to-champion", action="store_true")
    parser.add_argument("--pace-polish", action="store_true")
    parser.add_argument(
        "--retry-transport", action="store_true",
        help="reconnect and resume the saved learner after simulator interruptions",
    )
    parser.add_argument(
        "--continue-until-target", action="store_true",
        help="start another saved scratch-GRTQC training budget until its confirmed lap target",
    )
    parser.add_argument(
        "--allow-ppo-reward-change", action="store_true",
        help="resume PPO weights under an intentional new reward profile; resets Adam moments",
    )
    args = parser.parse_args()
    config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
    if args.continue_until_target and (config.algorithm != "grtqc" or config.grtqc is None
                                       or config.grtqc.training_origin != "scratch"):
        raise SystemExit("--continue-until-target requires scratch GRTQC")
    if args.stop_file.exists():
        raise SystemExit(f"remove the stop file before starting: {args.stop_file}")
    target_reached = threading.Event()

    def emit(event: dict) -> None:
        if event.get("type") == "target_reached":
            target_reached.set()
        print(json.dumps(event, allow_nan=False), flush=True)

    runner = TrainingRunner(config, emit)
    finished = threading.Event()

    def watch_stop_file() -> None:
        while not finished.wait(0.5):
            if args.stop_file.exists():
                print(json.dumps({"type": "stop_requested", "stop_file": str(args.stop_file)}), flush=True)
                runner.stop()
                return

    watcher = threading.Thread(target=watch_stop_file, name="polybot-stop-file", daemon=True)
    watcher.start()
    try:
        resume, fresh_replay = args.resume, args.fresh_replay
        registry = ModelRegistry(config.output_root)
        initial_steps = registry.read_metadata(resume).training_timesteps if resume is not None else 0
        while True:
            try:
                latest = runner.run(
                    resume=resume, fresh_replay=fresh_replay,
                    rollback_to_champion=args.rollback_to_champion, pace_polish=args.pace_polish,
                    allow_ppo_reward_change=args.allow_ppo_reward_change,
                )
            except (ConnectionError, TimeoutError, ProtocolViolation) as exc:
                if not args.retry_transport or (
                    isinstance(exc, ProtocolViolation) and not str(exc).startswith("stale_episode:")
                ):
                    raise
                resume = registry.slot(config.track_name, config.algorithm, "latest")
                if not (resume / "metadata.json").is_file():
                    raise
                fresh_replay = False
                if config.algorithm == "grtqc" and config.grtqc.training_origin != "scratch":
                    initialization = registry.slot(config.track_name, "grtqc", "initialization")
                    # An interrupted transfer check must repeat its five-lap
                    # fidelity gate rather than resume an unverified seed.
                    if registry.read_metadata(initialization).evaluation is None:
                        resume, fresh_replay = initialization, True
                print(json.dumps({
                    "type": "launcher_reconnect", "error": str(exc), "resume": str(resume),
                }), flush=True)
                if args.stop_file.exists():
                    break
                finished.wait(RECONNECT_DELAY_S)
                if args.stop_file.exists():
                    break
                completed = max(0, registry.read_metadata(resume).training_timesteps - initial_steps)
                retry_config = remaining_config(config, completed)
                if retry_config is None:
                    break
                initial_steps = registry.read_metadata(resume).training_timesteps
                runner = TrainingRunner(
                    retry_config, emit,
                )
                continue
            print(json.dumps({
                "type": "launcher_completed", "latest": str(latest),
                "champion": str(registry.slot(config.track_name, config.algorithm, "champion")),
            }), flush=True)
            if not should_continue(
                config, requested=args.continue_until_target, target_reached=target_reached.is_set(),
                stop_file=args.stop_file,
            ):
                break
            resume = registry.slot(config.track_name, config.algorithm, "latest")
            fresh_replay = False
            initial_steps = registry.read_metadata(resume).training_timesteps
            target_reached.clear()
            print(json.dumps({
                "type": "training_budget_continued", "resume": str(resume),
                "training_timesteps": initial_steps, "next_budget": config.timesteps,
            }), flush=True)
            runner = TrainingRunner(config, emit)
    finally:
        finished.set()
        watcher.join(timeout=1.0)


if __name__ == "__main__":
    main()
