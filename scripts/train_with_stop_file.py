"""Run a PolyBot config until completion or a clean stop-file request."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig
from polybot.training.runner import TrainingRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--fresh-replay", action="store_true")
    parser.add_argument("--rollback-to-champion", action="store_true")
    parser.add_argument("--pace-polish", action="store_true")
    parser.add_argument(
        "--allow-ppo-reward-change", action="store_true",
        help="resume PPO weights under an intentional new reward profile; resets Adam moments",
    )
    args = parser.parse_args()
    config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
    if args.stop_file.exists():
        raise SystemExit(f"remove the stop file before starting: {args.stop_file}")
    runner = TrainingRunner(config, lambda event: print(json.dumps(event, allow_nan=False), flush=True))
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
        latest = runner.run(
            resume=args.resume, fresh_replay=args.fresh_replay,
            rollback_to_champion=args.rollback_to_champion, pace_polish=args.pace_polish,
            allow_ppo_reward_change=args.allow_ppo_reward_change,
        )
        print(json.dumps({
            "type": "launcher_completed", "latest": str(latest),
            "champion": str(ModelRegistry(config.output_root).slot(config.track_name, config.algorithm, "champion")),
        }), flush=True)
    finally:
        finished.set()
        watcher.join(timeout=1.0)


if __name__ == "__main__":
    main()
