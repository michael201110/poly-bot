"""Single registration point for supported reinforcement learning algorithms."""

from __future__ import annotations

from polybot.algorithms.base import AlgorithmBackend
from polybot.algorithms.ppo import PPOBackend
from polybot.algorithms.tqc import TQCBackend

ALGORITHMS: dict[str, AlgorithmBackend] = {
    "ppo": PPOBackend(),
    "tqc": TQCBackend(),
}


def backend_for(name: str) -> AlgorithmBackend:
    try:
        return ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(f"unknown algorithm: {name}") from exc
