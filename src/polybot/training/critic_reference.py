"""Complete on-policy driving returns for frozen-policy critic initialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CriticReferenceSamples:
    observations: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    episodes: int


def on_policy_returns(
    replay: Any, reference: Any, gamma: float, *, max_samples: int = 8192,
) -> CriticReferenceSamples:
    """Select whole episodes whose recorded actions match the frozen policy.

    Include both successful and failed complete episodes. Incomplete episodes,
    timeouts and mismatched behavior cannot supply this policy's MC returns.
    The tolerance permits the measured batched/single CPU inference difference.
    """
    if replay.n_envs != 1 or not 0 < gamma <= 1 or max_samples < 1:
        raise ValueError("critic references need single-environment replay and valid sampling settings")
    if replay.observations.shape[2:] != reference.observation_space.shape:
        raise ValueError("critic reference and replay observation layouts differ")
    order = np.arange(replay.size())
    if replay.full:
        order = np.concatenate((order[replay.pos:], order[:replay.pos]))
    selected, targets = [], []
    resets = np.zeros(max(0, len(order) - 1), dtype=bool)
    for offset in range(0, len(order) - 1, 1024):
        stop = min(offset + 1024, len(order) - 1)
        previous = replay.next_observations[order[offset:stop], 0].reshape(stop - offset, -1)
        following = replay.observations[order[offset + 1:stop + 1], 0].reshape(stop - offset, -1)
        resets[offset:stop] = ~np.isclose(previous, following, rtol=0, atol=1e-6).all(axis=1)
    start, complete_start, episodes = 0, not replay.full, 0
    for end, row in enumerate(order):
        if replay.dones[row, 0] or replay.timeouts[row, 0]:
            if complete_start and not replay.timeouts[row, 0]:
                rows = order[start:end + 1]
                predictor = reference.policy if getattr(reference, "critic_raw_actions", False) else reference
                predicted, _ = predictor.predict(replay.observations[rows, 0], deterministic=True)
                if np.allclose(predicted, replay.actions[rows, 0], rtol=0, atol=3e-6):
                    discounted = np.empty(len(rows), dtype=np.float32)
                    future = 0.0
                    for offset in range(len(rows) - 1, -1, -1):
                        future = float(replay.rewards[rows[offset], 0]) + gamma * future
                        discounted[offset] = future
                    selected.extend(rows.tolist())
                    targets.extend(discounted.tolist())
                    episodes += 1
            start, complete_start = end + 1, True
        elif end + 1 < len(order) and resets[end]:
            start, complete_start = end + 1, True
    if not selected:
        raise ValueError("replay contains no complete episodes matching the frozen policy")
    selected_array = np.asarray(selected, dtype=np.int64)
    targets_array = np.asarray(targets, dtype=np.float32)
    if len(selected_array) > max_samples:
        subsample = np.linspace(0, len(selected_array) - 1, max_samples, dtype=np.int64)
        selected_array, targets_array = selected_array[subsample], targets_array[subsample]
    return CriticReferenceSamples(
        replay.observations[selected_array, 0].copy(), replay.actions[selected_array, 0].copy(),
        targets_array[:, None], episodes,
    )
