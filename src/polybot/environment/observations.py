"""Shared policy observation schema for every algorithm."""

from __future__ import annotations

import numpy as np

from polybot.protocol import Telemetry

SCHEMA = "polybot.observation.v2"


def observe(telemetry: Telemetry) -> np.ndarray:
    """45 normalized state values, then 4 values and 1 mask per lookahead point."""
    return telemetry.to_vector()


def size(lookahead_count: int) -> int:
    return Telemetry.vector_size(lookahead_count)
