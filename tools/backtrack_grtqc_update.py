"""Build smaller fractions of a learned actor update for live validation.

No control targets or overlays are authored. These policy-only snapshots must
pass live completion and timing checks before becoming resumable candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as th

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig


def backtrack(reference: Path, proposal: Path, output: Path, fractions: list[float]) -> dict:
    if output.exists():
        raise FileExistsError(f"preserve existing diagnostic: {output}")
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("learned update fractions must be in (0, 1]")
    th.set_num_threads(1)
    registry, backend = ModelRegistry(), GRTQCBackend()
    metadata = registry.read_metadata(reference)
    config = TrainingConfig.from_dict(metadata.training_config)
    if config.algorithm != "grtqc":
        raise ValueError("update backtracking requires a GRTQC reference")
    for directory in (reference, proposal):
        registry.validate(registry.read_metadata(directory), config, backend.action_adapter(config).schema)
    baseline = backend.load_model(reference / "policy.zip", None, "cpu")
    candidate = backend.load_model(proposal / "policy.zip", None, "cpu")
    if (
        baseline.policy_overlays != candidate.policy_overlays
        or baseline.speed_bias_schedule != candidate.speed_bias_schedule
    ):
        raise ValueError("update backtracking requires identical saved output transforms")
    starting = baseline.actor.state_dict()
    proposed = {name: value.clone() for name, value in candidate.actor.state_dict().items()}
    if starting.keys() != proposed.keys() or any(
        value.shape != starting[name].shape for name, value in proposed.items()
    ):
        raise ValueError("update backtracking requires compatible actor parameters")
    output.mkdir(parents=True)
    snapshots = []
    for fraction in sorted(set(fractions), reverse=True):
        candidate.actor.load_state_dict({
            name: starting[name] + fraction * (value - starting[name])
            for name, value in proposed.items()
        })
        directory = output / f"fraction-{fraction:g}"
        backend.save_model(candidate, directory)
        snapshots.append({"update_fraction": fraction, "policy": str(directory / "policy.zip")})
    report = {
        "reference": str(reference), "proposal": str(proposal), "config": config.to_dict(),
        "snapshots": snapshots, "live_validation": "pending",
        "note": "Actor parameter interpolation only; critics and output transforms are unchanged.",
    }
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[1, .5, .25, .125, .0625, .03125, .015625])
    args = parser.parse_args()
    print(json.dumps(backtrack(args.reference, args.proposal, args.output, args.fractions), indent=2))


if __name__ == "__main__":
    main()
