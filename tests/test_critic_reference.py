from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer

from polybot.training.critic_reference import on_policy_returns


def _replay(capacity=32):
    space = spaces.Box(-100, 100, shape=(2,), dtype=np.float32)
    return ReplayBuffer(capacity, space, space, device="cpu"), SimpleNamespace(
        observation_space=space,
        predict=lambda observations, deterministic: (np.zeros((len(observations), 2)), None),
    )


def _add(replay, current, following, reward, *, done=False, noisy=False, timeout=False):
    replay.add(
        np.array([[current, 0]], dtype=np.float32), np.array([[following, 0]], dtype=np.float32),
        np.array([[0.001 if noisy else 0, 0]], dtype=np.float32),
        np.array([reward], dtype=np.float32), np.array([done]), [{"TimeLimit.truncated": timeout}],
    )


def test_complete_policy_returns_include_failures_and_reject_noise_timeouts_and_resets():
    replay, reference = _replay()
    _add(replay, 0, 1, 1)
    _add(replay, 1, 2, 2)
    _add(replay, 2, 3, 8, done=True)
    _add(replay, 0, 1, -4, done=True)
    _add(replay, 0, 1, 1)
    _add(replay, 1, 2, 1, noisy=True, done=True)
    _add(replay, 0, 1, 5, done=True, timeout=True)
    _add(replay, 0, 9, 3)  # Interrupted before a terminal transition.
    _add(replay, 0, 1, 6, done=True)  # New physical reset, not its successor.
    samples = on_policy_returns(replay, reference, 0.5)
    assert samples.episodes == 3
    np.testing.assert_array_equal(samples.returns[:, 0], [4, 6, 8, -4, 6])
    np.testing.assert_array_equal(samples.observations[:, 0], [0, 1, 2, 0, 0])
    np.testing.assert_array_equal(samples.actions, np.zeros((5, 2)))
    subset = on_policy_returns(replay, reference, 0.5, max_samples=3)
    assert subset.episodes == 3
    np.testing.assert_array_equal(subset.returns[:, 0], [4, 8, 6])


def test_wrapped_replay_discards_the_unproven_initial_episode():
    replay, reference = _replay(4)
    for _ in range(3):
        _add(replay, 0, 1, 2)
        _add(replay, 1, 2, 4, done=True)
    assert replay.full
    samples = on_policy_returns(replay, reference, 0.5)
    assert samples.episodes == 1
    np.testing.assert_array_equal(samples.returns[:, 0], [4, 4])


def test_reference_matching_allows_measured_cpu_inference_roundoff():
    replay, reference = _replay()
    _add(replay, 0, 1, 1, done=True)
    reference.predict = lambda observations, deterministic: (
        np.full((len(observations), 2), 4.1e-6, dtype=np.float32), None,
    )

    samples = on_policy_returns(replay, reference, 0.99)

    assert samples.episodes == 1
    assert len(samples.observations) == 1


def test_reference_returns_reject_incompatible_or_absent_on_policy_data():
    replay, reference = _replay()
    _add(replay, 0, 1, 1, noisy=True, done=True)
    with pytest.raises(ValueError, match="no complete episodes"):
        on_policy_returns(replay, reference, 0.99)
    reference.observation_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    with pytest.raises(ValueError, match="layouts differ"):
        on_policy_returns(replay, reference, 0.99)
