"""Gymnasium environment for PolyTrack-compatible simulators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from polybot.protocol import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    Action,
    ProtocolViolation,
    Telemetry,
    Transition,
    request_message,
    response_result,
)
from polybot.transport import SimulatorTransport


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Reward coefficients kept independent from the game adapter."""

    progress_per_m: float = 0.05
    step_cost: float = -0.01
    checkpoint_bonus: float = 1.0
    finish_bonus: float = 25.0
    crash_penalty: float = -3.0
    off_track_penalty: float = -0.05
    action_change_penalty: float = -0.002
    max_forward_progress_per_step_m: float = 10.0
    max_reverse_progress_per_step_m: float = 3.0


class PolyTrackEnv(gym.Env[np.ndarray, np.ndarray]):
    """Synchronous Gymnasium wrapper around a PolyTrack simulator adapter."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        transport: SimulatorTransport,
        *,
        track_id: str = "mock/gentle-s",
        lookahead_count: int = 12,
        frame_skip: int = 4,
        max_episode_steps: int = 2_000,
        reward_config: RewardConfig | None = None,
        request_timeout_s: float = 10.0,
    ) -> None:
        super().__init__()
        if lookahead_count < 1:
            raise ValueError("lookahead_count must be positive")
        if frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if not track_id:
            raise ValueError("track_id cannot be empty")

        self.transport = transport
        self.track_id = track_id
        self.lookahead_count = lookahead_count
        self.frame_skip = frame_skip
        self.max_episode_steps = max_episode_steps
        self.reward_config = reward_config or RewardConfig()
        self.request_timeout_s = request_timeout_s

        self.action_space = spaces.MultiDiscrete(np.asarray([3, 2, 2], dtype=np.int64))
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(Telemetry.vector_size(lookahead_count),),
            dtype=np.float32,
        )

        self._next_request_id = 0
        self._handshake_complete = False
        self._closed = False
        self._episode_id: str | None = None
        self._episode_steps = 0
        self._previous_progress_m = 0.0
        self._previous_action = Action()
        self._episode_done = True
        self.latest_telemetry: Telemetry | None = None
        self.simulator_capabilities: Mapping[str, Any] = {}

    def _exchange(self, op: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("environment is closed")
        request_id = self._next_request_id
        self._next_request_id += 1
        request = request_message(request_id, op, params)
        response = self.transport.request(request, timeout_s=self.request_timeout_s)
        return response_result(response, expected_id=request_id)

    def _handshake(self) -> None:
        if self._handshake_complete:
            return
        result = self._exchange(
            "hello",
            {
                "protocol": PROTOCOL_NAME,
                "protocol_version": PROTOCOL_VERSION,
                "lookahead_count": self.lookahead_count,
            },
        )
        if result.get("protocol") != PROTOCOL_NAME:
            raise ProtocolViolation("simulator hello returned an unsupported protocol")
        if result.get("protocol_version") != PROTOCOL_VERSION:
            raise ProtocolViolation("simulator hello returned an unsupported protocol version")
        if result.get("lookahead_count") != self.lookahead_count:
            raise ProtocolViolation("simulator cannot provide the requested lookahead count")
        fixed_dt_s = result.get("fixed_dt_s")
        if (
            isinstance(fixed_dt_s, bool)
            or not isinstance(fixed_dt_s, (int, float))
            or not 0 < fixed_dt_s <= 1
        ):
            raise ProtocolViolation("simulator fixed_dt_s must be in (0, 1]")
        max_ticks = result.get("max_ticks_per_step")
        if (
            isinstance(max_ticks, bool)
            or not isinstance(max_ticks, int)
            or max_ticks < self.frame_skip
        ):
            raise ProtocolViolation("simulator cannot advance the requested frame_skip")
        self.simulator_capabilities = dict(result)
        self._handshake_complete = True

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._handshake()
        options = options or {}
        track_id = options.get("track_id", self.track_id)
        if not isinstance(track_id, str) or not track_id:
            raise ValueError("options['track_id'] must be a non-empty string")
        simulator_seed = (
            int(seed)
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.int32).max))
        )
        result = self._exchange(
            "reset",
            {
                "seed": simulator_seed,
                "track_id": track_id,
            },
        )
        transition = Transition.from_wire(result, lookahead_count=self.lookahead_count)
        if transition.ticks_advanced != 0:
            raise ProtocolViolation("reset must not advance simulation ticks")

        self._episode_id = transition.episode_id
        self._episode_steps = 0
        self._previous_progress_m = transition.telemetry.route_progress_m
        self._previous_action = transition.telemetry.previous_action
        self._episode_done = False
        self.latest_telemetry = transition.telemetry
        observation = transition.telemetry.to_vector()
        info = self._info(transition, reward_terms=None, simulator_seed=simulator_seed)
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_id is None or self._episode_done:
            raise RuntimeError("reset() must be called before step() or after an episode ends")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside {self.action_space}")
        decoded_action = Action.from_policy(action)
        result = self._exchange(
            "step",
            {
                "episode_id": self._episode_id,
                "action": decoded_action.to_wire(),
                "ticks": self.frame_skip,
            },
        )
        transition = Transition.from_wire(result, lookahead_count=self.lookahead_count)
        if transition.episode_id != self._episode_id:
            raise ProtocolViolation("simulator returned a stale or unexpected episode_id")
        if transition.ticks_advanced > self.frame_skip:
            raise ProtocolViolation("simulator advanced more ticks than requested")

        reward, reward_terms = self._reward(transition, decoded_action)
        self._episode_steps += 1
        events = set(transition.events)
        terminated = "finish" in events or "crash" in events
        truncated = "time_limit" in events or self._episode_steps >= self.max_episode_steps
        self._episode_done = terminated or truncated
        self._previous_progress_m = transition.telemetry.route_progress_m
        self._previous_action = decoded_action
        self.latest_telemetry = transition.telemetry

        observation = transition.telemetry.to_vector()
        info = self._info(transition, reward_terms=reward_terms)
        if truncated and "time_limit" not in events:
            info["wrapper_time_limit"] = True
        return observation, reward, terminated, truncated, info

    def _reward(self, transition: Transition, action: Action) -> tuple[float, dict[str, float]]:
        config = self.reward_config
        raw_delta = transition.telemetry.route_progress_m - self._previous_progress_m
        progress_delta = float(
            np.clip(
                raw_delta,
                -config.max_reverse_progress_per_step_m,
                config.max_forward_progress_per_step_m,
            )
        )
        events = transition.events
        terms = {
            "progress": config.progress_per_m * progress_delta,
            "step": config.step_cost,
            "checkpoint": config.checkpoint_bonus * events.count("checkpoint"),
            "finish": config.finish_bonus if "finish" in events else 0.0,
            "crash": config.crash_penalty if "crash" in events else 0.0,
            "off_track": config.off_track_penalty if "off_track" in events else 0.0,
            "action_change": config.action_change_penalty
            * (
                abs(action.steer - self._previous_action.steer)
                + int(action.throttle != self._previous_action.throttle)
                + int(action.brake != self._previous_action.brake)
            ),
        }
        reward = float(sum(terms.values()))
        if not np.isfinite(reward):
            raise ProtocolViolation("reward became non-finite")
        return reward, terms

    def _info(
        self,
        transition: Transition,
        *,
        reward_terms: Mapping[str, float] | None,
        simulator_seed: int | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            **transition.telemetry.to_info(),
            "tick": transition.tick,
            "ticks_advanced": transition.ticks_advanced,
            "events": transition.events,
            "simulator_info": dict(transition.simulator_info),
        }
        if reward_terms is not None:
            info["reward_terms"] = dict(reward_terms)
        if simulator_seed is not None:
            info["simulator_seed"] = simulator_seed
        return info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.transport.close()
        super().close()
