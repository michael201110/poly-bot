"""Compressed, asynchronous visual replay storage for training episodes."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Full, Queue
from threading import Thread
from typing import Any

import gymnasium as gym
import numpy as np

REPLAY_SCHEMA = "polybot.visual-replay.v1"
INDEX_SCHEMA = "polybot.visual-replay-index.v1"
_REQUIRED_ARRAYS = ("ticks", "elapsed_s", "position_m", "quaternion_xyzw")
_LOG = logging.getLogger(__name__)


class ReplayFormatError(ValueError):
    """Raised when a replay index or episode payload is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class ReplaySample:
    """A transform at an arbitrary simulation time, not necessarily a policy step."""

    tick: int
    elapsed_s: float
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    wheel_state: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.tick, bool)
            or not isinstance(self.tick, (int, np.integer))
            or self.tick < 0
        ):
            raise ValueError("replay sample tick must be a non-negative integer")
        if not math.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("replay sample elapsed_s must be finite and non-negative")
        if len(self.position_m) != 3 or len(self.quaternion_xyzw) != 4:
            raise ValueError("replay sample position and quaternion must have lengths 3 and 4")
        if not all(math.isfinite(value) for value in (*self.position_m, *self.quaternion_xyzw)):
            raise ValueError("replay sample transform values must be finite")
        if sum(value * value for value in self.quaternion_xyzw) < 1e-12:
            raise ValueError("replay sample quaternion cannot be zero")
        if self.wheel_state is not None and (
            len(self.wheel_state) != 42 or not all(math.isfinite(value) for value in self.wheel_state)
        ):
            raise ValueError("replay wheel state must contain 42 finite numbers")

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> ReplaySample:
        """Build a visual sample from the environment's existing telemetry info."""
        try:
            position = tuple(float(value) for value in info["position_m"])
            quaternion = tuple(float(value) for value in info["quaternion_xyzw"])
            tick = info["tick"]
            elapsed_s = float(info["elapsed_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("environment info is missing replay transform telemetry") from exc
        if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)):
            raise ValueError("environment replay tick must be an integer")
        wheels = info.get("simulator_info", {}).get("replay_wheels")
        return cls(int(tick), elapsed_s, position, quaternion,
                   tuple(float(value) for value in wheels) if wheels is not None else None)


@dataclass(slots=True)
class ReplayPayload:
    metadata: dict[str, Any]
    samples: list[ReplaySample]
    decision_ticks: list[int] | None = None
    decision_elapsed_s: list[float] | None = None
    observations: list[np.ndarray] | np.ndarray | None = None
    actions: list[np.ndarray] | np.ndarray | None = None
    hud_frames: list[dict[str, Any]] | None = None


def _metadata_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    required = {
        "run_id", "episode_id", "algorithm", "track_id", "track_name",
        "training_step", "training_step_start", "training_step_end",
        "episode_length_decisions", "episode_length_ticks", "status",
        "final_progress_m", "frame_skip", "sample_count", "file",
    }
    missing = required.difference(metadata)
    if missing:
        raise ReplayFormatError(f"replay metadata is missing: {', '.join(sorted(missing))}")
    for name in ("run_id", "episode_id", "algorithm", "track_id", "track_name", "status", "file"):
        if not isinstance(metadata[name], str) or not metadata[name]:
            raise ReplayFormatError(f"replay metadata {name} must be a non-empty string")
    if "track_slug" in metadata and (
        not isinstance(metadata["track_slug"], str) or not metadata["track_slug"]
    ):
        raise ReplayFormatError("replay metadata track_slug must be non-empty text")
    for name in (
        "training_step", "training_step_start", "training_step_end",
        "episode_length_decisions", "episode_length_ticks", "frame_skip", "sample_count",
    ):
        value = metadata[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReplayFormatError(f"replay metadata {name} must be a non-negative integer")
    if metadata["frame_skip"] < 1:
        raise ReplayFormatError("replay metadata frame_skip must be positive")
    if not isinstance(metadata["final_progress_m"], (int, float)) or not math.isfinite(
        metadata["final_progress_m"]
    ):
        raise ReplayFormatError("replay metadata final_progress_m must be finite")
    if metadata["training_step"] != metadata["training_step_start"]:
        raise ReplayFormatError("training_step must equal the episode's global start step")
    if metadata["training_step_end"] < metadata["training_step_start"]:
        raise ReplayFormatError("replay end step precedes start step")
    if metadata["sample_count"] < 1:
        raise ReplayFormatError("replay must contain at least one transform sample")
    return dict(metadata)


def load_replay_index(directory: str | Path) -> list[dict[str, Any]]:
    """Read searchable replay metadata; old run directories naturally return no entries."""
    path = Path(directory) / "index.json"
    if not path.is_file():
        return []
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayFormatError(f"cannot read replay index {path}") from exc
    if not isinstance(index, dict) or index.get("schema") != INDEX_SCHEMA:
        raise ReplayFormatError(f"unsupported or incomplete replay index {path}")
    entries = index.get("episodes")
    if not isinstance(entries, list):
        raise ReplayFormatError(f"replay index {path} has no episode list")
    if any(not isinstance(entry, dict) for entry in entries):
        raise ReplayFormatError(f"replay index {path} contains an invalid episode entry")
    return [_metadata_entry(entry) for entry in entries]


def load_replay_episode(
    directory: str | Path, entry: dict[str, Any], *, include_optional: bool = True,
    include_hud: bool = False,
) -> ReplayPayload:
    """Load and validate one indexed NPZ without permitting object deserialization."""
    metadata = _metadata_entry(entry)
    filename = metadata["file"]
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".npz"):
        raise ReplayFormatError("replay index contains an invalid payload filename")
    path = Path(directory) / filename
    try:
        with np.load(path, allow_pickle=False) as archive:
            if any(name not in archive for name in _REQUIRED_ARRAYS):
                raise ReplayFormatError(f"replay payload {path} is missing transform arrays")
            ticks = np.asarray(archive["ticks"], dtype=np.int64)
            elapsed_s = np.asarray(archive["elapsed_s"], dtype=np.float64)
            position = np.asarray(archive["position_m"], dtype=np.float32)
            quaternion = np.asarray(archive["quaternion_xyzw"], dtype=np.float32)
            count = int(metadata["sample_count"])
            wheels = np.asarray(archive["wheel_state"], dtype=np.float32) if "wheel_state" in archive else None
            if wheels is not None and (wheels.shape != (count, 42) or not np.isfinite(wheels).all()):
                raise ReplayFormatError(f"replay payload {path} has invalid wheel state")
            if ticks.shape != (count,) or elapsed_s.shape != (count,):
                raise ReplayFormatError(f"replay payload {path} has incomplete sample arrays")
            if position.shape != (count, 3) or quaternion.shape != (count, 4):
                raise ReplayFormatError(f"replay payload {path} has invalid transform dimensions")
            if not (
                np.isfinite(elapsed_s).all()
                and np.isfinite(position).all()
                and np.isfinite(quaternion).all()
            ):
                raise ReplayFormatError(f"replay payload {path} contains non-finite transforms")
            samples = [
                ReplaySample(
                    int(tick), float(seconds), tuple(map(float, xyz)), tuple(map(float, quat)),
                    tuple(map(float, wheels[index])) if wheels is not None else None,
                )
                for index, (tick, seconds, xyz, quat) in enumerate(zip(ticks, elapsed_s, position, quaternion, strict=True))
            ]
            optional: dict[str, np.ndarray | None] = {
                name: None for name in ("decision_ticks", "decision_elapsed_s", "observations", "actions")
            }
            if include_optional:
                for name in optional:
                    optional[name] = np.asarray(archive[name]) if name in archive else None
            hud_frames: list[dict[str, Any]] | None = None
            if include_hud and "hud_frames" in archive:
                try:
                    decoded = [json.loads(str(value)) for value in archive["hud_frames"].tolist()]
                except (TypeError, json.JSONDecodeError) as exc:
                    raise ReplayFormatError(f"replay payload {path} has invalid HUD telemetry") from exc
                if any(not isinstance(frame, dict) for frame in decoded):
                    raise ReplayFormatError(f"replay payload {path} has invalid HUD telemetry")
                hud_frames = decoded
            decision_ticks = optional["decision_ticks"]
            decision_elapsed = optional["decision_elapsed_s"]
            if decision_ticks is not None and (
                decision_ticks.ndim != 1 or decision_elapsed is None
                or decision_elapsed.shape != decision_ticks.shape
            ):
                raise ReplayFormatError(f"replay payload {path} has invalid decision sample arrays")
            for name in ("observations", "actions"):
                array = optional[name]
                if array is not None and (
                    array.ndim != 2 or not np.isfinite(array).all()
                ):
                    raise ReplayFormatError(f"replay payload {path} has invalid {name} array")
    except ReplayFormatError:
        raise
    except (OSError, ValueError, EOFError, KeyError) as exc:
        raise ReplayFormatError(f"cannot load replay payload {path}") from exc
    return ReplayPayload(
        metadata=metadata,
        samples=samples,
        decision_ticks=optional["decision_ticks"].astype(np.int64).tolist()
        if optional["decision_ticks"] is not None else None,
        decision_elapsed_s=optional["decision_elapsed_s"].astype(np.float64).tolist()
        if optional["decision_elapsed_s"] is not None else None,
        observations=optional["observations"] if optional["observations"] is not None else None,
        actions=optional["actions"] if optional["actions"] is not None else None,
        hud_frames=hud_frames,
    )


class AsyncReplayWriter:
    """One bounded background writer; submitted episodes never block the trainer."""

    def __init__(
        self,
        directory: str | Path,
        *,
        run_metadata: dict[str, Any],
        max_pending: int = 8,
    ) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.directory = Path(directory)
        self.run_metadata = dict(run_metadata)
        self._queue: Queue[ReplayPayload | None] = Queue(maxsize=max_pending)
        self._closed = False
        self._thread = Thread(
            target=self._write_loop, name="polybot-visual-replay-writer", daemon=True,
        )
        self._thread.start()

    def submit(self, payload: ReplayPayload) -> bool:
        if self._closed:
            _LOG.warning("Visual replay recording is closed; dropping %s", payload.metadata.get("episode_id"))
            return False
        try:
            self._queue.put_nowait(payload)
        except Full:
            _LOG.warning(
                "Visual replay queue is full; dropping episode %s",
                payload.metadata.get("episode_id"),
            )
            return False
        return True

    def close(self, *, timeout_s: float = 30.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=timeout_s)
        except Full:
            _LOG.warning("Visual replay writer did not drain before shutdown")
            return
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            _LOG.warning("Visual replay writer is still busy after %.1f seconds", timeout_s)

    def _write_loop(self) -> None:
        entries: list[dict[str, Any]] = []
        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                try:
                    entry = self._write_episode(payload)
                    entries.append(entry)
                    self._write_index(entries)
                except Exception:
                    _LOG.warning(
                        "Could not persist visual replay episode %s; training will continue",
                        payload.metadata.get("episode_id"),
                        exc_info=True,
                    )
            finally:
                self._queue.task_done()

    def _write_episode(self, payload: ReplayPayload) -> dict[str, Any]:
        metadata = _metadata_entry(payload.metadata)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / metadata["file"]
        temporary = path.with_suffix(path.suffix + ".tmp")
        arrays: dict[str, np.ndarray] = {
            "ticks": np.asarray([sample.tick for sample in payload.samples], dtype=np.int64),
            "elapsed_s": np.asarray([sample.elapsed_s for sample in payload.samples], dtype=np.float64),
            "position_m": np.asarray([sample.position_m for sample in payload.samples], dtype=np.float32),
            "quaternion_xyzw": np.asarray(
                [sample.quaternion_xyzw for sample in payload.samples], dtype=np.float32,
            ),
        }
        if all(sample.wheel_state is not None for sample in payload.samples):
            arrays["wheel_state"] = np.asarray([sample.wheel_state for sample in payload.samples], dtype=np.float32)
        for name, values, dtype in (
            ("decision_ticks", payload.decision_ticks, np.int64),
            ("decision_elapsed_s", payload.decision_elapsed_s, np.float64),
            ("observations", payload.observations, np.float32),
            ("actions", payload.actions, np.float32),
        ):
            if values is not None:
                arrays[name] = np.asarray(values, dtype=dtype)
        if payload.hud_frames is not None:
            arrays["hud_frames"] = np.asarray(
                [json.dumps(frame, allow_nan=False, separators=(",", ":")) for frame in payload.hud_frames],
                dtype=np.str_,
            )
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return metadata

    def _write_index(self, entries: list[dict[str, Any]]) -> None:
        index = {
            "schema": INDEX_SCHEMA,
            "run": self.run_metadata,
            "episodes": entries,
        }
        temporary = self.directory / "index.json.tmp"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(index, stream, allow_nan=False, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.directory / "index.json")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


@dataclass(slots=True)
class _ActiveEpisode:
    episode_id: str
    training_step_start: int
    start_tick: int
    samples: list[ReplaySample]
    decision_ticks: list[int]
    decision_elapsed_s: list[float]
    observations: list[np.ndarray]
    actions: list[np.ndarray]
    hud_frames: list[dict[str, Any]]
    episode_return: float
    reward_terms: dict[str, float]
    reward_groups: dict[str, float]
    last_sample_elapsed_s: float
    decisions: int = 0
    final_info: dict[str, Any] | None = None


class VisualReplaySession:
    """Collect policy-step telemetry and submit finalized episodes asynchronously."""

    def __init__(
        self,
        writer: AsyncReplayWriter,
        *,
        run_id: str,
        algorithm: str,
        track_id: str,
        track_name: str,
        track_slug: str = "",
        frame_skip: int,
        sample_hz: float,
        record_observations: bool,
        training_step_provider: Any,
        reward_scale: float = 1.0,
        advance_training_steps: bool = True,
        hud_context_provider: Any = None,
    ) -> None:
        if not math.isfinite(sample_hz) or sample_hz <= 0:
            raise ValueError("visual replay sample_hz must be finite and positive")
        self.writer = writer
        self.run_id = run_id
        self.algorithm = algorithm
        self.track_id = track_id
        self.track_name = track_name
        self.track_slug = track_slug
        self.frame_skip = frame_skip
        self.sample_interval_s = 1.0 / sample_hz
        self.sample_hz = sample_hz
        self.record_observations = record_observations
        self.training_step_provider = training_step_provider
        self.advance_training_steps = advance_training_steps
        self.hud_context_provider = hud_context_provider
        if not math.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("visual replay reward_scale must be finite and positive")
        self.reward_scale = reward_scale
        self.global_step: int | None = None
        self.next_episode_number = 1
        self.active: _ActiveEpisode | None = None

    def reset(self, info: dict[str, Any]) -> None:
        if self.active is not None:
            self.finish(status="interrupted")
        if self.global_step is None:
            self.global_step = int(self.training_step_provider())
        sample = ReplaySample.from_info(info)
        episode_id = f"episode-{self.next_episode_number:06d}"
        self.next_episode_number += 1
        self.active = _ActiveEpisode(
            episode_id=episode_id,
            training_step_start=self.global_step,
            start_tick=sample.tick,
            samples=[sample],
            decision_ticks=[],
            decision_elapsed_s=[],
            observations=[],
            actions=[],
            hud_frames=[],
            episode_return=0.0,
            reward_terms={},
            reward_groups={},
            last_sample_elapsed_s=sample.elapsed_s,
            final_info=dict(info),
        )

    def step(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        info: dict[str, Any],
        reward: float = 0.0,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self.active is None:
            return
        episode = self.active
        sample = ReplaySample.from_info(info)
        episode.decisions += 1
        episode.final_info = dict(info)
        episode.episode_return += float(reward)
        for key, totals in (("reward_terms", episode.reward_terms), ("reward_groups", episode.reward_groups)):
            for name, value in info.get(key, {}).items():
                totals[name] = totals.get(name, 0.0) + float(value)
        hud_frame = _replay_hud_frame(
            info, action, float(reward), episode.episode_return,
            algorithm=self.algorithm, track_name=self.track_name,
            episode=self.next_episode_number - 1,
            decision_count=episode.decisions,
            training_step=episode.training_step_start + (episode.decisions if self.advance_training_steps else 0),
            frame_skip=self.frame_skip,
            reward_scale=self.reward_scale,
        )
        if self.record_observations:
            hud_frame.update({
                "feature_schema": "polybot.observation-features.v1",
                "observation_schema": info.get("replay_observation_schema"),
                "features": info.get("replay_policy_features", []),
                "observation_source": "recorded",
                "observation_elapsed_s": info.get("replay_observation_elapsed_s"),
                "observation_tick": info.get("replay_observation_tick"),
                "telemetry": info.get("replay_telemetry", {}),
            })
            hud_frame["settings"]["show_observations"] = True
            hud_frame["settings"]["lookahead_points"] = 12
        velocity = info.get("local_velocity_mps")
        hud_frame.update({
            "speed_mps": float(np.linalg.norm(velocity)) if velocity is not None else None,
            "reference_speed_mps": info.get("ghost_target_speed_mps"),
            "checkpoint_index": info.get("checkpoint_index"),
            "lateral_offset_m": info.get("lateral_offset_m"),
            "heading_error_rad": info.get("heading_error_rad"),
        })
        hud_frame.update(info.get("replay_race_context", {}))
        hud_frame["reward"].update({
            "episode_terms": dict(episode.reward_terms),
            "episode_groups": dict(episode.reward_groups),
            "episode_learner_terms": {name: value * self.reward_scale for name, value in episode.reward_terms.items()},
            "episode_learner_groups": {name: value * self.reward_scale for name, value in episode.reward_groups.items()},
        })
        if self.hud_context_provider is not None:
            hud_frame.update(self.hud_context_provider())
        episode.hud_frames.append(hud_frame)
        if self.record_observations:
            episode.decision_ticks.append(sample.tick)
            episode.decision_elapsed_s.append(sample.elapsed_s)
            episode.observations.append(np.asarray(observation, dtype=np.float32).reshape(-1).copy())
            episode.actions.append(np.asarray(action, dtype=np.float32).reshape(-1).copy())
        if (
            sample.elapsed_s - episode.last_sample_elapsed_s >= self.sample_interval_s
            or terminated or truncated
        ):
            if sample.tick != episode.samples[-1].tick or sample.elapsed_s != episode.samples[-1].elapsed_s:
                episode.samples.append(sample)
                episode.last_sample_elapsed_s = sample.elapsed_s
            else:
                episode.samples[-1] = sample
        if self.advance_training_steps:
            self.global_step = int(self.global_step) + 1
        if terminated or truncated:
            self.finish(
                info,
                status=_episode_status(info.get("events", ()), truncated=truncated),
            )

    def finish(self, info: dict[str, Any] | None = None, *, status: str = "interrupted") -> None:
        episode = self.active
        if episode is None:
            return
        if episode.decisions == 0 and status == "interrupted":
            self.active = None
            return
        final_info = dict(info or episode.final_info or {})
        try:
            final_sample = ReplaySample.from_info(final_info)
            if (
                final_sample.tick != episode.samples[-1].tick
                or final_sample.elapsed_s != episode.samples[-1].elapsed_s
            ):
                episode.samples.append(final_sample)
        except (KeyError, TypeError, ValueError):
            _LOG.warning("Visual replay %s has no usable final transform", episode.episode_id, exc_info=True)
        if not episode.samples:
            self.active = None
            return
        events = set(final_info.get("events", ()))
        track_length = float(final_info.get("track_length_m", 0.0) or 0.0)
        progress_m = float(final_info.get("route_progress_m", 0.0) or 0.0)
        end_step = int(self.global_step if self.global_step is not None else episode.training_step_start)
        capabilities = final_info.get("simulator_info", {})
        game_version = capabilities.get("game_version") if isinstance(capabilities, dict) else None
        if not game_version:
            game_version = final_info.get("game_version")
        metadata: dict[str, Any] = {
            "schema": REPLAY_SCHEMA,
            "run_id": self.run_id,
            "episode_id": episode.episode_id,
            "algorithm": self.algorithm,
            "track_id": self.track_id,
            "track_name": self.track_name,
            **({"track_slug": self.track_slug} if self.track_slug else {}),
            "training_step": episode.training_step_start,
            "training_step_start": episode.training_step_start,
            "training_step_end": end_step,
            "policy_step": episode.training_step_start,
            "episode_length_decisions": episode.decisions,
            "episode_length_ticks": max(0, episode.samples[-1].tick - episode.start_tick),
            "status": status,
            "events": sorted(events),
            "final_progress_m": progress_m,
            "final_progress_ratio": progress_m / track_length if track_length > 0 else 0.0,
            "lap_time_s": float(final_info["elapsed_s"]) if "finish" in events else None,
            "frame_skip": self.frame_skip,
            "sample_hz_limit": self.sample_hz,
            "sample_count": len(episode.samples),
            "simulator": capabilities.get("simulator") if isinstance(capabilities, dict) else None,
            "simulator_version": (
                capabilities.get("simulator_version") or game_version
                if isinstance(capabilities, dict) else game_version
            ),
            "game_version": game_version,
            "observations_recorded": self.record_observations,
            "created_at": datetime.now(UTC).isoformat(),
            "file": f"{episode.episode_id}.npz",
        }
        payload = ReplayPayload(
            metadata=metadata,
            samples=episode.samples,
            decision_ticks=episode.decision_ticks if self.record_observations else None,
            decision_elapsed_s=episode.decision_elapsed_s if self.record_observations else None,
            observations=episode.observations if self.record_observations else None,
            actions=episode.actions if self.record_observations else None,
            hud_frames=episode.hud_frames,
        )
        self.active = None
        self.writer.submit(payload)

    def close(self) -> None:
        if self.active is not None:
            self.finish(status="interrupted")

    def shutdown(self) -> None:
        self.close()
        self.writer.close()


def _replay_hud_frame(
    info: dict[str, Any], action: np.ndarray, reward: float, episode_return: float, *,
    algorithm: str, track_name: str, episode: int, decision_count: int,
    training_step: int, frame_skip: int,
    reward_scale: float,
) -> dict[str, Any]:
    """Capture the reward and control values that the live training HUD displays."""
    values = np.asarray(info.get("raw_policy_action", action), dtype=np.float64).reshape(-1)
    names = ("steer", "longitudinal") if values.size == 2 else (
        ("steer", "throttle", "brake") if values.size == 3
        else tuple(f"action_{index}" for index in range(values.size))
    )
    model_output = [
        {"label": name.replace("_", " ").title(), "value": float(value)}
        for name, value in zip(names, values, strict=True)
        if math.isfinite(float(value))
    ]
    transformed = np.asarray(info.get("transformed_policy_action", action), dtype=np.float64).reshape(-1)
    transformed_output = [
        {"label": name.replace("_", " ").title(), "value": float(value)}
        for name, value in zip(names, transformed, strict=True)
        if math.isfinite(float(value))
    ]
    terms = info.get("reward_terms", {})
    groups = info.get("reward_groups", {})
    terms = dict(terms) if isinstance(terms, dict) else {}
    groups = dict(groups) if isinstance(groups, dict) else {}
    status = "running"
    events = info.get("events", ())
    if "finish" in events:
        status = "finished"
    elif any(event in events for event in ("crash", "airborne_roll_failure", "off_track")):
        status = "failed"
    elif any(event in events for event in ("time_limit", "timeout")):
        status = "timeout"
    return {
        "schema": "polybot.ai-overlay-frame.v1",
        "enabled": True,
        "settings": {
            "preset": "compact", "scale": 1.0, "show_episode_status": True,
            "show_labels": True, "show_observations": False, "show_controls": True,
            "show_reward_breakdown": True, "show_event_popups": True,
            "lookahead_points": 0,
        },
        "mode": "replay",
        "status": status,
        "track": track_name,
        "algorithm": algorithm,
        "training_step": training_step,
        "episode": episode,
        "decision_count": decision_count,
        "frame_skip": frame_skip,
        "simulator_tick": info.get("tick"),
        "elapsed_simulation_s": info.get("elapsed_s"),
        "progress_m": info.get("route_progress_m"),
        "track_length_m": info.get("track_length_m"),
        "lap_time_s": info.get("elapsed_s") if status == "finished" else None,
        "events": list(events) if not isinstance(events, str) else [events],
        "controls": {
            "model_output": model_output,
            "transformed_output": transformed_output,
            "adapter_demand": info.get("requested_control_duty"),
            "applied": info.get("applied_control_fraction"),
        },
        "reward": {
            "raw_step": reward,
            "learner_step": reward * reward_scale,
            "episode_raw": episode_return,
            "episode_learner": episode_return * reward_scale,
            "scale": reward_scale,
            "terms": terms,
            "groups": groups,
            "learner_terms": {name: float(value) * reward_scale for name, value in terms.items()},
            "learner_groups": {name: float(value) * reward_scale for name, value in groups.items()},
        },
    }


class VisualReplayCaptureWrapper(gym.Wrapper):
    """Capture existing environment telemetry without changing returned values."""

    def __init__(self, env: gym.Env, session: VisualReplaySession) -> None:
        super().__init__(env)
        self.session = session
        self._recording_failed = False
        self._last_observation: np.ndarray | None = None
        self._last_observation_tick: int | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        result = self.env.reset(**kwargs)
        self._last_observation = np.asarray(result[0], dtype=np.float32).copy()
        self._last_observation_tick = result[1].get("tick")
        self._capture(lambda: self.session.reset(self._replay_info(result[1])))
        return result

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation = self._last_observation
        policy_features = None
        core = self.env.unwrapped
        input_state = getattr(core, "latest_telemetry", None)
        input_tick = self._last_observation_tick
        if self.session.record_observations and observation is not None:
            capture = getattr(core, "capture_hud_features", None)
            if callable(capture):
                def capture_features() -> None:
                    nonlocal policy_features
                    policy_features = capture(observation)
                self._capture(capture_features)
        result = self.env.step(action)
        def record_step() -> None:
            capture_info = self._replay_info(result[4])
            if self.session.record_observations:
                from polybot.environment.observations import (
                    SCHEMA, CONTROLLER_STATE_SCHEMA, TRAINING_STATE_SCHEMA, size,
                )
                capture_info["replay_policy_features"] = policy_features or []
                capture_info["replay_observation_schema"] = (
                    TRAINING_STATE_SCHEMA if getattr(core, "expose_training_state", False)
                    else CONTROLLER_STATE_SCHEMA if observation is not None and observation.size > size(len(input_state.lookahead) if input_state is not None else 12)
                    else SCHEMA
                )
                capture_info["replay_observation_elapsed_s"] = getattr(input_state, "elapsed_s", None)
                capture_info["replay_observation_tick"] = input_tick
                telemetry = getattr(core, "latest_telemetry", None)
                if telemetry is not None:
                    from dataclasses import asdict
                    capture_info["replay_telemetry"] = asdict(telemetry)
            if observation is not None:
                self.session.step(
                    observation,
                    np.asarray(action, dtype=np.float32),
                    capture_info,
                    reward=float(result[1]),
                    terminated=result[2],
                    truncated=result[3],
                )
        self._capture(record_step)
        self._last_observation = np.asarray(result[0], dtype=np.float32).copy()
        self._last_observation_tick = result[4].get("tick")
        return result

    def close(self) -> None:
        self._capture(self.session.close)
        self.env.close()

    def _capture(self, operation: Any) -> None:
        if self._recording_failed:
            return
        try:
            operation()
        except Exception:
            self._recording_failed = True
            _LOG.warning("Visual replay capture failed; training will continue", exc_info=True)

    def _replay_info(self, info: dict[str, Any]) -> dict[str, Any]:
        capture_info = dict(info)
        capabilities = getattr(self.env.unwrapped, "simulator_capabilities", {})
        simulator_info = dict(capture_info.get("simulator_info", {}))
        for name in ("simulator", "game_version"):
            if name in capabilities:
                simulator_info[name] = capabilities[name]
        capture_info["simulator_info"] = simulator_info
        line = getattr(self.env.unwrapped, "racing_line", None)
        if isinstance(line, dict):
            reference = {"name": line.get("source_name"), "target_time_s": line.get("lap_time_s")}
            position = info.get("position_m")
            candidates = [p for p in line.get("points", ())
                          if p.get("checkpoint_index") == info.get("checkpoint_index")]
            if position is not None and candidates:
                nearest = min(candidates, key=lambda p: sum(
                    (a-b)**2 for a,b in zip(position, p["position_m"], strict=True)
                ))
                reference["elapsed_s"] = nearest["tick"] / 1000.0
                reference["delta_s"] = float(info.get("elapsed_s", 0)) - reference["elapsed_s"]
                reference["delta_method"] = "nearest saved position"
            capture_info["replay_race_context"] = {"reference": reference}
        return capture_info


def _episode_status(events: Any, *, truncated: bool) -> str:
    event_set = set(events)
    if "finish" in event_set:
        return "finished"
    if "time_limit" in event_set:
        return "timeout"
    if event_set.intersection({
        "crash", "off_track", "stalled", "airborne_roll_failure", "off_track_landing",
    }):
        return "failed"
    return "truncated" if truncated else "terminated"
