"""Fingerprint raw/physical/PWM controls from an existing policy and matching recorded inputs.

Run under baseline and optimized PYTHONPATH and compare the hashes. This tool
only reads the checkpoint and observations; it never pads a historical schema,
loads replay into a learner, saves a policy, or connects to a simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from polybot.algorithms.registry import backend_for
from polybot.control.actions import ContinuousActionAdapter, DigitalActionAdapter


def fingerprint(checkpoint: Path, observations: Path, algorithm: str, limit: int) -> dict:
    model = backend_for(algorithm).load_model(checkpoint, None, "cpu")
    with np.load(observations, allow_pickle=False) as recording:
        inputs = recording["observations"][:limit].copy()
    if inputs.ndim != 2 or inputs.shape[1:] != model.observation_space.shape:
        raise ValueError(f"recorded shape {inputs.shape} does not match {model.observation_space}; no padding allowed")
    raw, _ = model.policy.predict(inputs, deterministic=True)
    transformed, _ = model.predict(inputs, deterministic=True)
    adapter = (
        DigitalActionAdapter() if isinstance(model.action_space, spaces.MultiDiscrete) else ContinuousActionAdapter()
    )
    pulses = []
    for action in transformed:
        pulses.extend((tick.steer, tick.throttle, tick.brake) for tick in adapter.apply(action, 30).ticks)
    return {
        "algorithm": algorithm, "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "observations": str(observations), "samples": len(inputs), "width": inputs.shape[1],
        "raw_actions_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "transformed_actions_sha256": hashlib.sha256(transformed.tobytes()).hexdigest(),
        "pwm_sha256": hashlib.sha256(np.asarray(pulses, dtype=np.int8).tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("grtqc", "tqc", "ppo"), required=True)
    parser.add_argument("--limit", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    torch.set_num_threads(1)
    report = fingerprint(args.checkpoint, args.observations, args.algorithm, args.limit)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
