"""Utilities for widening Stable-Baselines actor-critic MLPs without drift."""

from __future__ import annotations

from typing import Any

import torch


def widen_actor_critic_policy(source: Any, target: Any) -> None:
    """Embed a two-layer actor-critic policy into wider target layers exactly."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    branches = ("policy_net", "value_net")
    for branch in branches:
        first_weight = f"mlp_extractor.{branch}.0.weight"
        first_bias = f"mlp_extractor.{branch}.0.bias"
        second_weight = f"mlp_extractor.{branch}.2.weight"
        second_bias = f"mlp_extractor.{branch}.2.bias"
        old_width = source_state[first_bias].shape[0]
        new_width = target_state[first_bias].shape[0]
        if new_width <= old_width:
            raise ValueError("target policy must be wider than source policy")
        if source_state[first_weight].shape[1] != target_state[first_weight].shape[1]:
            raise ValueError("source and target observation features do not match")

        target_state[first_weight][:old_width].copy_(source_state[first_weight])
        target_state[first_bias][:old_width].copy_(source_state[first_bias])

        # Old second-layer neurons must not see the new first-layer neurons.
        # New second-layer neurons retain random inputs but initially have zero
        # output weights, allowing them to begin learning without policy drift.
        target_state[second_weight][:old_width].zero_()
        target_state[second_weight][:old_width, :old_width].copy_(
            source_state[second_weight]
        )
        target_state[second_bias][:old_width].copy_(source_state[second_bias])

    for head in ("action_net", "value_net"):
        weight = f"{head}.weight"
        bias = f"{head}.bias"
        old_width = source_state[weight].shape[1]
        target_state[weight].zero_()
        target_state[weight][:, :old_width].copy_(source_state[weight])
        target_state[bias].copy_(source_state[bias])

    target.load_state_dict(target_state)


@torch.no_grad()
def policy_outputs(policy: Any, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return action logits and values for migration-equivalence checks."""

    features = policy.extract_features(observations)
    latent_policy, latent_value = policy.mlp_extractor(features)
    return policy.action_net(latent_policy), policy.value_net(latent_value)
