"""Compare saved GRTQC values with observed returns from completed replay laps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th

from polybot.algorithms.grtqc import GRTQCBackend


def audit(directory: Path) -> dict:
    th.set_num_threads(1)
    model = GRTQCBackend().load_model(directory / "policy.zip", None, "cpu")
    model.load_replay_buffer(str(directory / "replay.pkl"))
    replay = model.replay_buffer
    if replay is None or replay.n_envs != 1:
        raise ValueError("audit requires single-environment replay")
    order = np.arange(replay.size())
    if replay.full:
        order = np.concatenate((order[replay.pos:], order[:replay.pos]))
    observations = replay.observations[order, 0]
    next_observations = replay.next_observations[order, 0]
    rewards = replay.rewards[order, 0]
    dones = replay.dones[order, 0].astype(bool)
    timeouts = replay.timeouts[order, 0].astype(bool)
    returns = np.full(len(order), np.nan)
    start = 0
    complete_start = not replay.full
    finished = 0
    for end in range(len(order)):
        discontinuity = end + 1 < len(order) and not np.allclose(
            next_observations[end], observations[end + 1], rtol=0, atol=1e-6,
        )
        if dones[end]:
            if complete_start and not timeouts[end] and rewards[end] > 0:
                future = 0.0
                for index in range(end, start - 1, -1):
                    future = rewards[index] + model.gamma * future
                    returns[index] = future
                finished += 1
            start, complete_start = end + 1, True
        elif discontinuity:
            # Evaluation and curriculum switches can reset without a stored
            # terminal transition. Never combine returns across that reset.
            start, complete_start = end + 1, True
    valid = np.flatnonzero(np.isfinite(returns))
    if not len(valid):
        raise ValueError("replay has no fully observed successful episodes")
    estimates = []
    with th.no_grad():
        for batch in np.array_split(valid, max(1, (len(valid) + 255) // 256)):
            obs = th.as_tensor(observations[batch], dtype=th.float32)
            actions = th.as_tensor(replay.actions[order[batch], 0], dtype=th.float32)
            estimates.append(model.critic(obs, actions).mean(dim=(1, 2)).numpy())
    values = np.concatenate(estimates)
    expected = returns[valid]
    progress = observations[valid, 12]
    bins = []
    for lower, upper in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1), strict=True):
        selected = (progress >= lower) & (progress < upper)
        if selected.any():
            bins.append({
                "progress": [round(float(lower), 1), round(float(upper), 1)],
                "samples": int(selected.sum()),
                "mean_q": float(values[selected].mean()),
                "mean_observed_return": float(expected[selected].mean()),
            })
    return {
        "checkpoint": str(directory), "timesteps": model.num_timesteps,
        "critic_updates": model.critic_updates_since_transfer,
        "actor_unlocked": model.actor_unlocked, "gamma": model.gamma,
        "completed_laps": finished, "samples": len(valid),
        "mean_q": float(values.mean()), "mean_observed_return": float(expected.mean()),
        "mean_absolute_error": float(np.abs(values - expected).mean()),
        "progress_bins": bins,
        "interpretation": (
            "Observed discounted driving return is a calibration diagnostic, not an exact soft-Q target. "
            "It excludes entropy, and recorded behavior differs from the stochastic target policy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint)
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
