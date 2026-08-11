from __future__ import annotations

import torch
from stable_baselines3.common.policies import ActorCriticPolicy

from polybot.model_migration import policy_outputs, widen_actor_critic_policy


def test_widened_policy_preserves_logits_and_values() -> None:
    from gymnasium import spaces

    observation_space = spaces.Box(-1.0, 1.0, shape=(81,))
    action_space = spaces.MultiDiscrete([3, 2, 2])
    source = ActorCriticPolicy(
        observation_space,
        action_space,
        lambda _: 3e-4,
        net_arch={"pi": [64, 64], "vf": [64, 64]},
    )
    target = ActorCriticPolicy(
        observation_space,
        action_space,
        lambda _: 3e-4,
        net_arch={"pi": [128, 128], "vf": [128, 128]},
    )
    observations = torch.randn(32, 81)
    source_outputs = policy_outputs(source, observations)

    widen_actor_critic_policy(source, target)
    target_outputs = policy_outputs(target, observations)

    torch.testing.assert_close(target_outputs[0], source_outputs[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(target_outputs[1], source_outputs[1], rtol=0, atol=1e-6)
