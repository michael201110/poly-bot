from __future__ import annotations

import numpy as np
import pytest

from polybot.distillation import distill_policy
from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.training.config import policy_kwargs


def test_distillation_reduces_policy_loss() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    teacher_env = PolyTrackEnv(
        MockSimulatorTransport(), pwm_enabled=True, pwm_levels=5
    )
    student_env = PolyTrackEnv(
        MockSimulatorTransport(), pwm_enabled=True, pwm_levels=5
    )
    try:
        teacher = PPO(
            "MlpPolicy",
            teacher_env,
            policy_kwargs=policy_kwargs("legacy"),
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        student = PPO(
            "MlpPolicy",
            student_env,
            policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        observations = np.random.default_rng(7).normal(
            size=(64, teacher_env.observation_space.shape[0])
        ).astype(np.float32)

        metrics = distill_policy(
            teacher,
            student,
            observations,
            action_dims=(5, 2, 2),
            epochs=4,
            batch_size=16,
            learning_rate=1e-3,
        )

        assert metrics.final_loss < metrics.initial_loss
        assert 0.0 <= metrics.action_agreement <= 1.0
        assert metrics.observations == 64
    finally:
        teacher_env.close()
        student_env.close()
