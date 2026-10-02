"""Live-screen actor-update diagnostics; confirm every finishing policy over five laps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as th

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.training.config import TrainingConfig
from polybot.training.devices import resolve_device
from polybot.training.evaluation import evaluate_model
from polybot.training.metrics import EventSink
from polybot.training.runner import TrainingRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, help="run the five-paired-lap transfer gate first")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"preserve existing live results: {args.output}")
    th.set_num_threads(1)
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    config = TrainingConfig.from_dict(diagnostic["config"])
    runner = TrainingRunner(config)
    runner.sink = EventSink(args.output.with_suffix(".events.jsonl"))
    backend = GRTQCBackend()
    records = []
    if args.initialization:
        runner.device = resolve_device("cpu", algorithm="grtqc")
        runner.model = backend.load_model(args.initialization / "policy.zip", None, "cpu")
        result = runner._verify_grtqc_initialization(args.initialization)
        records.append({"initialization": str(args.initialization), "paired": result.to_dict()})
        args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(records[-1]), flush=True)
    for snapshot in diagnostic["snapshots"]:
        model = backend.load_model(Path(snapshot["policy"]), None, "cpu")
        screen = evaluate_model(model, runner._environment, episodes=1, seed=config.seed + 1_000_000)
        confirmed = (
            evaluate_model(model, runner._environment, episodes=5, seed=config.seed + 1_000_000)
            if screen.finish_rate == 1.0 else None
        )
        record = {
            **{key: snapshot[key] for key in ("actor_updates", "update_fraction") if key in snapshot},
            "policy": snapshot["policy"],
            "screen": screen.to_dict(), "confirmation": confirmed.to_dict() if confirmed else None,
        }
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)
    runner.sink.close()


if __name__ == "__main__":
    main()
