"""Shared policy observation schema for every algorithm."""

from __future__ import annotations

import numpy as np

from polybot.protocol import Telemetry

SCHEMA = "polybot.observation.v2"
CONTROLLER_STATE_SCHEMA = "polybot.observation.v2.pwm-state"


def schema_for(config: object) -> str:
    settings = getattr(config, "grtqc", None)
    return CONTROLLER_STATE_SCHEMA if getattr(settings, "critic_controller_state", False) else SCHEMA


def observe(telemetry: Telemetry) -> np.ndarray:
    """45 normalized state values, then 4 values and 1 mask per lookahead point."""
    return telemetry.to_vector()


def size(lookahead_count: int) -> int:
    return Telemetry.vector_size(lookahead_count)
