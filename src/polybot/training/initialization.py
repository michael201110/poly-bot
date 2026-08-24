"""Policy initialization helpers for MultiDiscrete driving actions."""

from __future__ import annotations

from typing import Any


def apply_forward_bias(model: Any, strength: float = 1.5) -> None:
    """Favor neutral steering, throttle on, and brake off in an SB3 policy."""

    if strength < 0:
        raise ValueError("forward bias strength must be non-negative")
    if strength == 0:
        return
    import torch

    action_space = getattr(getattr(model, "policy", None), "action_space", None)
    dimensions = tuple(int(value) for value in getattr(action_space, "nvec", ()))
    if len(dimensions) != 3 or dimensions[1:] != (2, 2) or dimensions[0] % 2 == 0:
        raise RuntimeError("forward bias requires MultiDiscrete([odd steering levels, 2, 2])")
    bias = getattr(getattr(model.policy, "action_net", None), "bias", None)
    if bias is None or bias.numel() != sum(dimensions):
        raise RuntimeError("policy action head does not match its action space")
    steering_levels = dimensions[0]
    centre = steering_levels // 2
    throttle_on = steering_levels + 1
    brake_off = steering_levels + 2
    with torch.no_grad():
        bias[centre] += strength * 0.5
        bias[throttle_on] += strength
        bias[brake_off] += strength
