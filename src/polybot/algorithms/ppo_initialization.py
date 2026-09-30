"""Policy initialization helpers for continuous driving actions."""

from __future__ import annotations

from typing import Any


def apply_forward_bias(
    model: Any, strength: float = 1.5, *, steering_strength: float = 0.0
) -> None:
    """Bias signed longitudinal output forward and reduce initial steering noise."""

    if strength < 0 or steering_strength < 0:
        raise ValueError("forward and steering bias strengths must be non-negative")
    if strength == 0 and steering_strength == 0:
        return
    import torch

    action_space = getattr(getattr(model, "policy", None), "action_space", None)
    if getattr(action_space, "shape", None) != (2,):
        raise RuntimeError("forward bias requires a continuous Box(2) action space")
    bias = getattr(getattr(model.policy, "action_net", None), "bias", None)
    if bias is None or bias.numel() != 2:
        raise RuntimeError("policy action head does not match its action space")
    with torch.no_grad():
        bias[1] += strength * 0.5
        if steering_strength and hasattr(model.policy, "log_std"):
            model.policy.log_std[0] -= steering_strength
