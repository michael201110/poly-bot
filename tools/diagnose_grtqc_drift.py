"""Measure deterministic action drift of a GRTQC candidate on saved TQC states."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.algorithms.tqc import TQCBackend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--initialization", type=Path, default=Path("models/summer-1/grtqc/initialization"))
    parser.add_argument("--source", type=Path, default=Path(
        "models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion"
    ))
    args = parser.parse_args()
    baseline = GRTQCBackend().load_model(args.initialization / "policy.zip", None, "cpu")
    candidate = GRTQCBackend().load_model(args.candidate / "policy.zip", None, "cpu")
    source = TQCBackend().load_model(args.source / "policy.zip", None, "cpu")
    source.load_replay_buffer(str(args.source / "replay.pkl"))
    replay = source.replay_buffer
    assert replay is not None
    indices = np.linspace(0, replay.size() - 1, 8192, dtype=np.int64)
    observations = np.asarray(replay.observations[indices, 0], dtype=np.float32)

    def inputs(policy, batch):
        extra = policy.observation_space.shape[0] - batch.shape[1]
        if extra not in (0, 4):
            raise ValueError("drift audit only supports the known controller-state suffix")
        # This actor-only audit does not invent replay for the critics.
        return np.pad(batch, ((0, 0), (0, extra))) if extra else batch

    def drift(model):
        differences = []
        for batch in np.array_split(observations, 64):
            expected, _ = baseline.predict(inputs(baseline, batch), deterministic=True)
            actual, _ = model.predict(inputs(model, batch), deterministic=True)
            differences.append(np.abs(expected - actual))
        return np.concatenate(differences)

    differences = drift(candidate)
    only_linear = deepcopy(baseline)
    only_gates = deepcopy(baseline)
    linear_state = only_linear.actor.state_dict()
    gate_state = only_gates.actor.state_dict()
    for name, value in candidate.actor.state_dict().items():
        if ".gate." in name:
            gate_state[name] = value
        else:
            linear_state[name] = value
    only_linear.actor.load_state_dict(linear_state)
    only_gates.actor.load_state_dict(gate_state)
    linear_differences = drift(only_linear)
    gate_differences = drift(only_gates)
    report = {
        "candidate": str(args.candidate),
        "timesteps": candidate.num_timesteps,
        "actor_unlocked": candidate.actor_unlocked,
        "actor_lr": candidate.actor_lr,
        "action_abs_max": differences.max(axis=0).tolist(),
        "action_abs_p99": np.percentile(differences, 99, axis=0).tolist(),
        "action_abs_mean": differences.mean(axis=0).tolist(),
        "linear_only_abs_max": linear_differences.max(axis=0).tolist(),
        "gates_only_abs_max": gate_differences.max(axis=0).tolist(),
        "gate_parameters": {
            name: float((parameter - baseline.actor.state_dict()[name]).abs().max())
            for name, parameter in candidate.actor.state_dict().items()
            if ".gate." in name
        },
        "speed_windows_equal": candidate.speed_bias_schedule == baseline.speed_bias_schedule,
        "overlays_equal": candidate.policy_overlays == baseline.policy_overlays,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
