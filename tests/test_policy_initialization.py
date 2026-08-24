from __future__ import annotations

import numpy as np
import pytest

from polybot.mock import make_mock_env
from polybot.training.initialization import apply_forward_bias


def test_pwm_forward_bias_targets_centre_throttle_and_no_brake() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    env = make_mock_env(pwm_enabled=True, pwm_levels=41)
    model = PPO("MlpPolicy", env, n_steps=2, batch_size=2)
    before = model.policy.action_net.bias.detach().cpu().numpy().copy()
    apply_forward_bias(model, 1.5)
    delta = model.policy.action_net.bias.detach().cpu().numpy() - before
    expected = np.zeros(45)
    expected[20] = 0.75
    expected[42] = 1.5
    expected[43] = 1.5
    np.testing.assert_allclose(delta, expected)
    env.close()
