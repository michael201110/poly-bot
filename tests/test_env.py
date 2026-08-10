from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from polybot.controller import CenterlineController
from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport


def make_env(**kwargs: object) -> PolyTrackEnv:
    return PolyTrackEnv(MockSimulatorTransport(), **kwargs)


def test_gymnasium_contract() -> None:
    env = make_env(track_id="mock/straight")
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_reset_is_deterministic_for_same_seed() -> None:
    env = make_env()
    try:
        first, _ = env.reset(seed=42)
        second, _ = env.reset(seed=42)
        np.testing.assert_array_equal(first, second)
    finally:
        env.close()


def test_throttle_produces_progress_reward() -> None:
    env = make_env(track_id="mock/straight", frame_skip=4)
    try:
        env.reset(seed=0)
        _, reward, terminated, truncated, info = env.step(np.asarray([1, 1, 0], dtype=np.int64))
        assert info["reward_terms"]["progress"] > 0
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_wrapper_time_limit_truncates_without_termination() -> None:
    env = make_env(track_id="mock/straight", max_episode_steps=1)
    try:
        env.reset(seed=0)
        _, _, terminated, truncated, info = env.step(np.asarray([1, 0, 0], dtype=np.int64))
        assert not terminated
        assert truncated
        assert info["wrapper_time_limit"] is True
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.asarray([1, 0, 0], dtype=np.int64))
    finally:
        env.close()


def test_frame_skip_matches_individual_fixed_ticks() -> None:
    held_action = np.asarray([2, 1, 0], dtype=np.int64)
    batched = make_env(track_id="mock/straight", frame_skip=4)
    individual = make_env(track_id="mock/straight", frame_skip=1)
    try:
        batched.reset(seed=7)
        individual.reset(seed=7)
        batched.step(held_action)
        for _ in range(4):
            individual.step(held_action)
        assert batched.latest_telemetry is not None
        assert individual.latest_telemetry is not None
        np.testing.assert_allclose(
            batched.latest_telemetry.position_m,
            individual.latest_telemetry.position_m,
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            batched.latest_telemetry.local_velocity_mps,
            individual.latest_telemetry.local_velocity_mps,
            rtol=0,
            atol=1e-12,
        )
    finally:
        batched.close()
        individual.close()


@pytest.mark.parametrize("track_id", ["mock/straight", "mock/gentle-s"])
def test_baseline_completes_training_tracks(track_id: str) -> None:
    env = make_env(track_id=track_id, max_episode_steps=2_000)
    controller = CenterlineController()
    try:
        env.reset(seed=0)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            assert env.latest_telemetry is not None
            _, _, terminated, truncated, info = env.step(
                controller.policy_action(env.latest_telemetry)
            )
        assert "finish" in info["events"]
        assert "crash" not in info["events"]
    finally:
        env.close()
