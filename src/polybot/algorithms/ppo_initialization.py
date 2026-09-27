"""Policy initialization helpers for MultiDiscrete driving actions."""

from __future__ import annotations

from typing import Any


def apply_forward_bias(
    model: Any, strength: float = 1.5, *, steering_strength: float = 0.0
) -> None:
    """Favor low steering, throttle on, and brake off in an SB3 policy."""

    if strength < 0 or steering_strength < 0:
        raise ValueError("forward and steering bias strengths must be non-negative")
    if strength == 0 and steering_strength == 0:
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
        if steering_strength:
            steering_values = torch.linspace(
                -1.0,
                1.0,
                steering_levels,
                device=bias.device,
                dtype=bias.dtype,
            )
            bias[:steering_levels] -= steering_strength * steering_values.abs()
        bias[centre] += strength * 0.5
        bias[throttle_on] += strength
        bias[brake_off] += strength
