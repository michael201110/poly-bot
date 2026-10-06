"""Best-effort live AI telemetry overlay settings and frame packaging."""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from polybot.environment.observations import FEATURE_SCHEMA

_LOG = logging.getLogger(__name__)
HUD_SETTINGS_SCHEMA = "polybot.ai-overlay-settings.v1"
HUD_FRAME_SCHEMA = "polybot.ai-overlay-frame.v1"


@dataclass(frozen=True, slots=True)
class AIOverlaySettings:
    enabled: bool = True
    preset: str = "compact"
    scale: float = 1.0
    show_episode_status: bool = True
    show_labels: bool = True
    show_observations: bool = True
    show_controls: bool = True
    show_reward_breakdown: bool = True
    show_event_popups: bool = True
    lookahead_points: int = 3

    def __post_init__(self) -> None:
        for name in (
            "enabled", "show_episode_status", "show_observations", "show_controls",
            "show_reward_breakdown", "show_event_popups", "show_labels",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.preset not in {"compact", "full"}:
            raise ValueError("preset must be 'compact' or 'full'")
        if not math.isfinite(self.scale) or not 0.5 <= self.scale <= 1.5:
            raise ValueError("scale must be finite and from 0.5 to 1.5")
        if (
            isinstance(self.lookahead_points, bool)
            or not isinstance(self.lookahead_points, int)
            or not 0 <= self.lookahead_points <= 12
        ):
            raise ValueError("lookahead_points must be an integer from 0 to 12")


class AIOverlaySettingsStore:
    def __init__(self, path: str | Path = Path("config") / "ai-overlay.json") -> None:
        self.path = Path(path)

    def load(self) -> AIOverlaySettings:
        if not self.path.is_file():
            return AIOverlaySettings()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read AI overlay settings from {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != HUD_SETTINGS_SCHEMA:
            raise ValueError(f"unsupported or malformed AI overlay settings in {self.path}")
        settings = payload.get("settings")
        if not isinstance(settings, dict):
            raise ValueError(f"AI overlay settings in {self.path} must contain an object")
        allowed = {field.name for field in fields(AIOverlaySettings)}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"unknown AI overlay settings: {', '.join(sorted(unknown))}")
        try:
            return AIOverlaySettings(**settings)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid AI overlay settings in {self.path}: {exc}") from exc

    def save(self, settings: AIOverlaySettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = {"schema": HUD_SETTINGS_SCHEMA, "settings": asdict(settings)}
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.path)


class AIOverlayTelemetryWrapper(gym.Wrapper):
    """Ship existing policy/state/reward values to a capable bridge."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        core = object.__getattribute__(self, "_core")
        return getattr(core, name)

    def __init__(
        self,
        env: gym.Env,
        *,
        settings_provider: Callable[[], AIOverlaySettings],
        context_provider: Callable[[], Mapping[str, Any]],
        frame_skip: int,
        reward_scale_provider: Callable[[], float],
    ) -> None:
        super().__init__(env)
        self._core = env.unwrapped
        self._settings_provider = settings_provider
        self._context_provider = context_provider
        self._frame_skip = frame_skip
        self._reward_scale_provider = reward_scale_provider
        self._episode_number = 0
        self._decision_count = 0
        self._episode_return = 0.0
        self._last_features: list[dict[str, Any]] | None = None
        self._policy_observation: np.ndarray | None = None
        self._warned_schema = False
        self._warned_frame = False
        self._overlay_was_enabled = False

    def _set_capture(self, enabled: bool) -> None:
        setter = getattr(self._core, "set_hud_feature_capture", None)
        if callable(setter):
            setter(enabled)

    def _queue_frame(self, frame: dict[str, Any]) -> None:
        publisher = getattr(self._core, "publish_hud_frame", None)
        if not callable(publisher):
            publisher = getattr(self._core, "set_hud_frame", None)
        if not callable(publisher):
            if not self._warned_frame:
                _LOG.warning("AI HUD frame transport is unavailable; training will continue")
                self._warned_frame = True
            return
        try:
            publisher(frame)
        except (TypeError, ValueError, OverflowError) as exc:
            if not self._warned_frame:
                _LOG.warning("Could not package an AI HUD frame; training will continue: %s", exc)
                self._warned_frame = True

    @staticmethod
    def _action_values(action: Any, info: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw = info.get("raw_policy_action", action)
        try:
            values = np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return []
        if not np.isfinite(values).all():
            return []
        names = (
            ("steer", "longitudinal")
            if values.size == 2
            else ("steer", "throttle", "brake")
            if values.size == 3
            else tuple(f"action_{index}" for index in range(values.size))
        )
        return [
            {"label": name.replace("_", " ").title(), "value": float(value)}
            for name, value in zip(names, values, strict=True)
        ]

    def _make_frame(
        self,
        *,
        action: Any = None,
        info: Mapping[str, Any] | None = None,
        raw_reward: float | None = None,
        reset: bool = False,
        features: list[dict[str, Any]] | None = None,
        use_current_features: bool = True,
    ) -> dict[str, Any]:
        settings = self._settings_provider()
        state = dict(self._context_provider())
        telemetry_info = dict(info or {})
        mode = str(state.get("mode", "evaluation"))
        scale = float(self._reward_scale_provider())
        if not math.isfinite(scale):
            raise ValueError("HUD reward scale must be finite")
        raw = float(raw_reward or 0.0)
        learner_reward = raw * scale
        events = telemetry_info.get("events", ())
        if isinstance(events, str):
            events = (events,)
        events = [str(event) for event in events if isinstance(event, str)]
        status = "ready" if reset else "running"
        if "finish" in events:
            status = "finished"
        elif any(event in events for event in (
            "crash", "airborne_roll_failure", "off_track",
        )):
            status = "failed"
        elif any(event in events for event in ("time_limit", "timeout")):
            status = "timeout"
        result: dict[str, Any] = {
            "schema": HUD_FRAME_SCHEMA,
            "enabled": settings.enabled,
            "settings": asdict(settings),
            "mode": mode,
            "status": status,
            "run_id": state.get("run_id"),
            "track": state.get("track_name", ""),
            "algorithm": state.get("algorithm", ""),
            "model_slot": state.get("model_slot"),
            "policy_checkpoint_step": state.get("policy_checkpoint_step"),
            "training_step": state.get("training_step") if mode == "training" else None,
            "episode": int(state.get("episode", self._episode_number)),
            "episode_total": state.get("episode_total"),
            "decision_count": self._decision_count,
            "frame_skip": self._frame_skip,
            "simulator_tick": telemetry_info.get("tick"),
            "elapsed_simulation_s": telemetry_info.get("elapsed_s"),
            "progress_m": telemetry_info.get("route_progress_m"),
            "track_length_m": telemetry_info.get("track_length_m"),
            "lap_time_s": telemetry_info.get("elapsed_s") if "finish" in events else None,
            "events": events,
            "reset": reset,
        }
        active_features = self._last_features if use_current_features else features
        if settings.show_observations and active_features is not None:
            result["observation_schema"] = state.get("observation_schema", "")
            result["feature_schema"] = FEATURE_SCHEMA
            result["features"] = active_features
        elif settings.show_observations and getattr(
            self._core, "last_observation_feature_error", None
        ):
            message = str(self._core.last_observation_feature_error)
            result["observation_schema_error"] = message
            if not self._warned_schema:
                _LOG.warning("AI HUD cannot describe policy inputs; training will continue: %s", message)
                self._warned_schema = True
        if settings.show_controls and action is not None:
            result["controls"] = {
                "model_output": self._action_values(action, telemetry_info),
                "transformed_output": self._action_values(
                    telemetry_info.get("transformed_policy_action", action), {},
                ),
                "adapter_demand": telemetry_info.get("requested_control_duty"),
                "applied": telemetry_info.get("applied_control_fraction"),
            }
        if settings.show_reward_breakdown:
            terms = telemetry_info.get("reward_terms", {})
            groups = telemetry_info.get("reward_groups", {})
            raw_terms = dict(terms) if isinstance(terms, Mapping) else {}
            raw_groups = dict(groups) if isinstance(groups, Mapping) else {}
            result["reward"] = {
                "raw_step": raw,
                "learner_step": learner_reward,
                "episode_raw": self._episode_return,
                "episode_learner": self._episode_return * scale,
                "scale": scale,
                "terms": raw_terms,
                "groups": raw_groups,
                "learner_terms": {
                    name: float(value) * scale for name, value in raw_terms.items()
                },
                "learner_groups": {
                    name: float(value) * scale for name, value in raw_groups.items()
                },
            }
        return result

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        settings = self._settings_provider()
        self._set_capture(settings.enabled and settings.show_observations)
        observation, info = self.env.reset(**kwargs)
        self._episode_number += 1
        self._decision_count = 0
        self._episode_return = 0.0
        self._last_features = getattr(self._core, "last_observation_features", None)
        self._policy_observation = np.asarray(observation, dtype=np.float32).copy()
        self._overlay_was_enabled = settings.enabled
        if settings.enabled:
            self._queue_frame(self._make_frame(info=info, reset=True))
        else:
            self._queue_frame({"enabled": False})
        return observation, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        settings = self._settings_provider()
        self._set_capture(settings.enabled and settings.show_observations)
        if not settings.enabled:
            if self._overlay_was_enabled:
                self._queue_frame({"enabled": False})
            self._overlay_was_enabled = False
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._decision_count += 1
            self._episode_return += float(reward)
            self._last_features = None
            self._policy_observation = np.asarray(observation, dtype=np.float32).copy()
            return observation, reward, terminated, truncated, info
        self._overlay_was_enabled = True
        if settings.enabled and settings.show_observations and self._last_features is None:
            capture = getattr(self._core, "capture_hud_features", None)
            if callable(capture) and self._policy_observation is not None:
                try:
                    self._last_features = capture(self._policy_observation)
                except (TypeError, ValueError, OverflowError) as exc:
                    if not self._warned_schema:
                        _LOG.warning(
                            "Could not describe the current policy input; training will continue: %s",
                            exc,
                        )
                        self._warned_schema = True
        policy_features = self._last_features
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._decision_count += 1
        raw_reward = float(reward)
        self._episode_return += raw_reward
        self._last_features = getattr(self._core, "last_observation_features", None)
        self._policy_observation = np.asarray(observation, dtype=np.float32).copy()
        frame = self._make_frame(
            action=action,
            info=info,
            raw_reward=raw_reward,
            features=policy_features,
            use_current_features=False,
        )
        if terminated or truncated:
            if frame["status"] == "running":
                frame["status"] = "ended"
        self._queue_frame(frame)
        return observation, reward, terminated, truncated, info


__all__ = [
    "AIOverlaySettings",
    "AIOverlaySettingsStore",
    "AIOverlayTelemetryWrapper",
    "HUD_FRAME_SCHEMA",
    "HUD_SETTINGS_SCHEMA",
]
