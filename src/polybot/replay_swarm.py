"""Replay selection, colour mapping, and timestamp interpolation utilities."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from polybot.protocol import PROTOCOL_NAME, PROTOCOL_VERSION, ProtocolViolation, request_message, response_result
from polybot.training.visual_replays import (
    ReplayFormatError,
    ReplayPayload,
    ReplaySample,
    load_replay_episode,
    load_replay_index,
)
from polybot.transport import WebSocketServerTransport

FAILED_STATUSES = frozenset({"failed", "timeout"})
MAX_REPLAY_SAMPLE_COUNT = 500_000
REPLAY_CHUNK_SAMPLES = 256
MAX_REPLAY_GHOSTS = 500
MAX_REPLAY_TOTAL_SAMPLES = 250_000
MAX_REPLAY_PAYLOAD_BYTES = 32 * 1024 * 1024
_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")
_NAMED_COLORS = {
    "red": "#ff0000",
    "orange": "#ff8000",
    "yellow": "#ffff00",
    "lime": "#80ff00",
    "green": "#00ff00",
}


@dataclass(frozen=True, slots=True)
class ColorStop:
    step: int
    color: str

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise ValueError("color stop step must be an integer")
        parse_color(self.color)


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    replay_directory: Path
    metadata: dict[str, Any]

    @property
    def training_step(self) -> int:
        return int(self.metadata["training_step_start"])


@dataclass(frozen=True, slots=True)
class SelectedReplayPayload:
    selection: ReplaySelection
    payload: ReplayPayload
    color: str

    @property
    def episode_key(self) -> str:
        return f"{self.selection.metadata['run_id']}:{self.selection.metadata['episode_id']}"


@dataclass(frozen=True, slots=True)
class InterpolatedTransform:
    elapsed_s: float
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    opacity: float = 1.0


@dataclass(frozen=True, slots=True)
class ReplayPlaybackOptions:
    speed: float = 1.0
    opacity: float = 0.5
    color: str = "#ffffff"
    end_behavior: str = "fade"
    fade_duration_s: float = 0.75

    def __post_init__(self) -> None:
        if (
            isinstance(self.speed, bool)
            or not isinstance(self.speed, (int, float))
            or not math.isfinite(self.speed)
            or not 0.1 <= self.speed <= 8
        ):
            raise ValueError("playback speed must be between 0.1 and 8")
        if (
            isinstance(self.opacity, bool)
            or not isinstance(self.opacity, (int, float))
            or not math.isfinite(self.opacity)
            or not 0 <= self.opacity <= 1
        ):
            raise ValueError("opacity must be between 0 and 1")
        if not isinstance(self.color, str):
            raise ValueError("color must be text in #RRGGBB form")
        object.__setattr__(self, "color", "#{:02x}{:02x}{:02x}".format(*parse_color(self.color)))
        if self.end_behavior not in {"disappear", "freeze", "fade"}:
            raise ValueError("end_behavior must be disappear, freeze, or fade")
        if (
            isinstance(self.fade_duration_s, bool)
            or not isinstance(self.fade_duration_s, (int, float))
            or not math.isfinite(self.fade_duration_s)
            or not 0 <= self.fade_duration_s <= 10
        ):
            raise ValueError("fade duration must be between 0 and 10 seconds")


def parse_color(value: str) -> tuple[int, int, int]:
    """Parse #RRGGBB or a supported basic gradient color name."""
    color = _NAMED_COLORS.get(value.lower(), value)
    match = _COLOR_RE.fullmatch(color)
    if match is None:
        raise ValueError(f"invalid color {value!r}; use #RRGGBB or a basic gradient color")
    encoded = match.group(1)
    return tuple(int(encoded[offset : offset + 2], 16) for offset in (0, 2, 4))


def replay_sample_chunks(
    samples: Sequence[ReplaySample],
    *,
    chunk_size: int = REPLAY_CHUNK_SAMPLES,
) -> Iterator[dict[str, Any]]:
    """Validate and encode one trajectory into bounded JSON-friendly chunks."""
    _validate_replay_samples(samples)
    if isinstance(chunk_size, bool) or not 1 <= chunk_size <= REPLAY_CHUNK_SAMPLES:
        raise ValueError(f"chunk_size must be from 1 to {REPLAY_CHUNK_SAMPLES}")
    for start in range(0, len(samples), chunk_size):
        rows = [
            [
                int(sample.tick),
                float(sample.elapsed_s),
                *map(float, sample.position_m),
                *map(float, sample.quaternion_xyzw),
            ]
            for sample in samples[start : start + chunk_size]
        ]
        yield {"start": start, "samples": rows}


def _validate_replay_samples(samples: Sequence[ReplaySample]) -> None:
    if not samples or len(samples) > MAX_REPLAY_SAMPLE_COUNT:
        raise ValueError(f"replay must contain from 1 to {MAX_REPLAY_SAMPLE_COUNT} samples")
    previous: ReplaySample | None = None
    for sample in samples:
        if previous is not None and (sample.tick < previous.tick or sample.elapsed_s < previous.elapsed_s):
            raise ValueError("replay samples must be ordered by tick and elapsed_s")
        if sample.tick > 2**53 - 1:
            raise ValueError("replay sample tick exceeds the safe JSON integer range")
        previous = sample


def _chunk_payload_bytes(samples: Sequence[ReplaySample]) -> int:
    return sum(
        len(json.dumps(chunk["samples"], separators=(",", ":"), allow_nan=False).encode("utf-8")) + 8
        for chunk in replay_sample_chunks(samples)
    )


def _send_replay_request(
    transport: WebSocketServerTransport,
    request_id: int,
    operation: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    response = transport.request(request_message(request_id, operation, params))
    return dict(response_result(response, expected_id=request_id))


def send_replay_playback(
    transport: WebSocketServerTransport,
    *,
    action: str,
    payload: ReplayPayload | None,
    options: ReplayPlaybackOptions,
    seek_seconds: float | None = None,
    update_settings: Sequence[str] = (),
) -> dict[str, Any]:
    """Send one validated replay or a playback control over the existing bridge."""
    if action not in {
        "play",
        "load",
        "resume",
        "pause",
        "restart",
        "seek",
        "configure",
        "clear",
    }:
        raise ValueError(f"unsupported replay action: {action}")
    if action in {"play", "load"} and payload is None:
        raise ValueError(f"{action} requires a selected replay payload")
    if action == "seek" and (
        seek_seconds is None
        or isinstance(seek_seconds, bool)
        or not isinstance(seek_seconds, (int, float))
        or not math.isfinite(seek_seconds)
        or seek_seconds < 0
    ):
        raise ValueError("seek requires a finite, non-negative --seek-seconds value")
    if action in {"play", "load"}:
        assert payload is not None
        _validate_replay_samples(payload.samples)
    setting_values = {
        "speed": options.speed,
        "opacity": options.opacity,
        "color": options.color,
        "end_behavior": options.end_behavior,
        "fade_duration": options.fade_duration_s,
    }
    if any(name not in setting_values for name in update_settings):
        raise ValueError("unsupported playback setting")
    if action == "configure" and not update_settings:
        raise ValueError("configure requires at least one playback setting")

    request_id = 0
    hello = _send_replay_request(
        transport,
        request_id,
        "hello",
        {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "lookahead_count": 1,
        },
    )
    if hello.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolViolation("PolyTrack bridge returned an incompatible protocol version")
    request_id += 1

    result: dict[str, Any]
    if action in {"play", "load"}:
        assert payload is not None
        chunks = replay_sample_chunks(payload.samples)
        payload_bytes = _chunk_payload_bytes(payload.samples)
        _send_replay_request(
            transport,
            request_id,
            "visual_replay_begin",
            {
                "sample_count": len(payload.samples),
                "color": options.color,
                "speed": options.speed,
                "opacity": options.opacity,
                "end_behavior": options.end_behavior,
                "fade_duration_s": options.fade_duration_s,
                "payload_bytes": payload_bytes,
            },
        )
        request_id += 1
        for chunk in chunks:
            _send_replay_request(
                transport,
                request_id,
                "visual_replay_chunk",
                chunk,
            )
            request_id += 1
        result = _send_replay_request(
            transport,
            request_id,
            "visual_replay_commit",
            {"autoplay": action == "play"},
        )
        request_id += 1
    else:
        result = {}
    controls: list[tuple[str, dict[str, Any]]] = []
    if action == "resume":
        controls.append(("visual_replay_play", {}))
    elif action not in {"play", "load", "configure"}:
        controls.append(
            (
                f"visual_replay_{action}",
                {"seconds": seek_seconds} if action == "seek" else {},
            )
        )
    for name in update_settings:
        value = setting_values[name]
        controls.append(
            (
                f"visual_replay_{name}",
                {"value": value},
            )
        )
    for operation, params in controls:
        result = _send_replay_request(transport, request_id, operation, params)
        request_id += 1
    return result


def send_replay_swarm(
    transport: WebSocketServerTransport,
    *,
    action: str,
    payloads: Sequence[SelectedReplayPayload] = (),
    options: ReplayPlaybackOptions,
    seek_seconds: float | None = None,
    update_settings: Sequence[str] = (),
) -> dict[str, Any]:
    """Transfer selected trajectories once, then let the main-thread renderer play them."""
    supported_actions = {
        "play",
        "load",
        "resume",
        "pause",
        "restart",
        "seek",
        "configure",
        "clear",
        "status",
    }
    if action not in supported_actions:
        raise ValueError(f"unsupported replay action: {action}")
    if action in {"play", "load"} and not payloads:
        raise ValueError(f"{action} requires at least one selected replay")
    if len(payloads) > MAX_REPLAY_GHOSTS:
        raise ValueError(f"swarm exceeds maximum ghost count ({MAX_REPLAY_GHOSTS})")
    if action == "seek" and (
        seek_seconds is None
        or isinstance(seek_seconds, bool)
        or not isinstance(seek_seconds, (int, float))
        or not math.isfinite(seek_seconds)
        or seek_seconds < 0
    ):
        raise ValueError("seek requires a finite, non-negative --seek-seconds value")
    setting_values = {
        "speed": options.speed,
        "opacity": options.opacity,
        "color": options.color,
        "end_behavior": options.end_behavior,
        "fade_duration": options.fade_duration_s,
    }
    if any(name not in setting_values for name in update_settings):
        raise ValueError("unsupported playback setting")
    if action == "configure" and not update_settings:
        raise ValueError("configure requires at least one playback setting")
    if len({item.episode_key for item in payloads}) != len(payloads):
        raise ValueError("swarm contains duplicate replay episodes")

    total_samples = 0
    payload_bytes = 0
    for item in payloads:
        _validate_replay_samples(item.payload.samples)
        total_samples += len(item.payload.samples)
        if total_samples > MAX_REPLAY_TOTAL_SAMPLES:
            raise ValueError(f"swarm exceeds maximum total sample count ({MAX_REPLAY_TOTAL_SAMPLES})")
        payload_bytes += _chunk_payload_bytes(item.payload.samples)
        if payload_bytes > MAX_REPLAY_PAYLOAD_BYTES:
            raise ValueError(f"swarm exceeds maximum encoded payload size ({MAX_REPLAY_PAYLOAD_BYTES} bytes)")
        if len(item.episode_key) > 128:
            raise ValueError("replay run/episode identifier exceeds 128 characters")

    request_id = 0
    hello = _send_replay_request(
        transport,
        request_id,
        "hello",
        {
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "lookahead_count": 1,
        },
    )
    if hello.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolViolation("PolyTrack bridge returned an incompatible protocol version")
    request_id += 1

    result: dict[str, Any] = {}
    if action in {"play", "load"}:
        _send_replay_request(
            transport,
            request_id,
            "visual_replay_swarm_begin",
            {
                "ghost_count": len(payloads),
                "total_sample_count": total_samples,
                "payload_bytes": payload_bytes,
                "speed": options.speed,
                "opacity": options.opacity,
                "end_behavior": options.end_behavior,
                "fade_duration_s": options.fade_duration_s,
            },
        )
        request_id += 1
        for item in payloads:
            _send_replay_request(
                transport,
                request_id,
                "visual_replay_swarm_episode_begin",
                {
                    "episode_id": item.episode_key,
                    "training_step_start": item.selection.training_step,
                    "sample_count": len(item.payload.samples),
                    "color": item.color,
                },
            )
            request_id += 1
            for chunk in replay_sample_chunks(item.payload.samples):
                _send_replay_request(
                    transport,
                    request_id,
                    "visual_replay_swarm_chunk",
                    {"episode_id": item.episode_key, **chunk},
                )
                request_id += 1
            _send_replay_request(
                transport,
                request_id,
                "visual_replay_swarm_episode_commit",
                {"episode_id": item.episode_key},
            )
            request_id += 1
        result = _send_replay_request(
            transport,
            request_id,
            "visual_replay_swarm_commit",
            {"autoplay": action == "play"},
        )
        request_id += 1
    controls: list[tuple[str, dict[str, Any]]] = []
    if action == "resume":
        controls.append(("visual_replay_play", {}))
    elif action == "seek":
        controls.append(("visual_replay_seek", {"seconds": seek_seconds}))
    elif action in {"pause", "restart", "clear", "status"}:
        operation = "status" if action == "status" else action
        controls.append((f"visual_replay_{operation}", {}))
    for name in update_settings:
        controls.append(("visual_replay_" + name, {"value": setting_values[name]}))
    for operation, params in controls:
        result = _send_replay_request(transport, request_id, operation, params)
        request_id += 1
    return result


class ColorScale:
    """Piecewise-linear RGB scale from training step to gradient color."""

    def __init__(
        self,
        minimum_step: int = 0,
        maximum_step: int = 1_000_000,
        stops: Sequence[ColorStop] | None = None,
    ) -> None:
        if isinstance(minimum_step, bool) or not isinstance(minimum_step, int):
            raise ValueError("minimum color step must be an integer")
        if isinstance(maximum_step, bool) or not isinstance(maximum_step, int):
            raise ValueError("maximum color step must be an integer")
        if maximum_step <= minimum_step:
            raise ValueError("maximum color step must be greater than minimum color step")
        if stops is None:
            span = maximum_step - minimum_step
            self.stops = (
                ColorStop(minimum_step, "#ff0000"),
                ColorStop(minimum_step + span // 4, "#ff8000"),
                ColorStop(minimum_step + span // 2, "#ffff00"),
                ColorStop(minimum_step + (span * 3) // 4, "#80ff00"),
                ColorStop(maximum_step, "#00ff00"),
            )
        else:
            ordered = tuple(sorted(stops, key=lambda stop: stop.step))
            if len(ordered) < 2 or len({stop.step for stop in ordered}) != len(ordered):
                raise ValueError("at least two color stops with distinct steps are required")
            self.stops = ordered
        self.minimum_step = self.stops[0].step
        self.maximum_step = self.stops[-1].step

    def color(self, step: int | float) -> tuple[int, int, int]:
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step):
            raise ValueError("training step must be finite")
        value = min(float(self.maximum_step), max(float(self.minimum_step), float(step)))
        for left, right in zip(self.stops, self.stops[1:], strict=False):
            if value <= right.step:
                fraction = (value - left.step) / (right.step - left.step)
                left_rgb = parse_color(left.color)
                right_rgb = parse_color(right.color)
                return tuple(int(round(a + fraction * (b - a))) for a, b in zip(left_rgb, right_rgb, strict=True))
        return parse_color(self.stops[-1].color)

    def hex_color(self, step: int | float) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.color(step))

    def bucket(self, step: int | float) -> str:
        if len(self.stops) == 5 and tuple(parse_color(stop.color) for stop in self.stops) == (
            (255, 0, 0),
            (255, 128, 0),
            (255, 255, 0),
            (128, 255, 0),
            (0, 255, 0),
        ):
            ratio = (float(step) - self.minimum_step) / (self.maximum_step - self.minimum_step)
            index = min(4, max(0, int(math.floor(ratio * 4 + 0.5))))
            return ("red", "orange", "yellow", "lime", "green")[index]
        rgb = self.color(step)
        stop = min(self.stops, key=lambda item: _rgb_distance(rgb, parse_color(item.color)))
        return stop.color


def _rgb_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def parse_color_stops(values: Sequence[str]) -> tuple[ColorStop, ...]:
    """Parse repeatable CLI values in STEP:COLOR form."""
    stops = []
    for value in values:
        try:
            raw_step, color = value.split(":", maxsplit=1)
            stops.append(ColorStop(int(raw_step), color))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid color stop {value!r}; expected STEP:#RRGGBB") from exc
    return tuple(stops)


def filter_replays(
    entries: Sequence[dict[str, Any]],
    *,
    steps: tuple[int, int] | None = None,
    episodes: tuple[int, int] | None = None,
    finished_only: bool = False,
    failed_only: bool = False,
) -> list[dict[str, Any]]:
    """Filter index entries using episode-start policy decisions as training age."""
    if finished_only and failed_only:
        raise ValueError("finished_only and failed_only cannot both be selected")
    if steps is not None and (steps[0] < 0 or steps[1] < steps[0]):
        raise ValueError("step range must be non-negative and ordered")
    if episodes is not None and (episodes[0] < 0 or episodes[1] < episodes[0]):
        raise ValueError("episode range must be non-negative and ordered")

    selected = []
    for entry in entries:
        step = int(entry["training_step_start"])
        if steps is not None and not steps[0] <= step <= steps[1]:
            continue
        if episodes is not None:
            episode_number = _episode_number(entry["episode_id"])
            if episode_number is None or not episodes[0] <= episode_number <= episodes[1]:
                continue
        status = entry["status"]
        if finished_only and status != "finished":
            continue
        if failed_only and status not in FAILED_STATUSES:
            continue
        selected.append(entry)
    return selected


def _episode_number(episode_id: str) -> int | None:
    match = re.search(r"(\d+)$", episode_id)
    return int(match.group(1)) if match else None


def stratified_sample(
    entries: Sequence[dict[str, Any]],
    max_cars: int,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Choose one seeded entry per equal-rank stratum across sorted training ages."""
    if max_cars < 1:
        raise ValueError("max_cars must be positive")
    ordered = sorted(
        entries,
        key=lambda entry: (
            int(entry["training_step_start"]),
            str(entry["run_id"]),
            str(entry["episode_id"]),
        ),
    )
    if len(ordered) <= max_cars:
        return ordered
    rng = random.Random(seed)
    selected = []
    for stratum in range(max_cars):
        start = (stratum * len(ordered)) // max_cars
        end = ((stratum + 1) * len(ordered)) // max_cars
        selected.append(ordered[rng.randrange(start, end)])
    return selected


def select_replays(
    directories: Sequence[Path],
    *,
    steps: tuple[int, int] | None = None,
    episodes: tuple[int, int] | None = None,
    finished_only: bool = False,
    failed_only: bool = False,
    max_cars: int = 250,
    seed: int = 0,
) -> tuple[int, list[ReplaySelection]]:
    """Read only indexes, filter their metadata, then apply seeded sampling."""
    indexed: list[ReplaySelection] = []
    for directory in directories:
        indexed.extend(ReplaySelection(directory, metadata) for metadata in load_replay_index(directory))
    matches = filter_replays(
        [item.metadata for item in indexed],
        steps=steps,
        episodes=episodes,
        finished_only=finished_only,
        failed_only=failed_only,
    )
    match_keys = {(entry["run_id"], entry["episode_id"]) for entry in matches}
    matching = [item for item in indexed if (item.metadata["run_id"], item.metadata["episode_id"]) in match_keys]
    if len(matching) <= max_cars:
        return len(indexed), sorted(
            matching,
            key=lambda item: (item.training_step, item.metadata["episode_id"]),
        )
    selected_metadata = stratified_sample(matches, max_cars, seed=seed)
    selected_keys = {(entry["run_id"], entry["episode_id"]) for entry in selected_metadata}
    return len(indexed), sorted(
        (item for item in matching if (item.metadata["run_id"], item.metadata["episode_id"]) in selected_keys),
        key=lambda item: (item.training_step, item.metadata["episode_id"]),
    )


def require_single_replay(
    selected: Sequence[ReplaySelection],
    *,
    max_cars: int,
) -> ReplaySelection:
    if max_cars != 1:
        raise ValueError("in-game playback supports exactly one selected replay; set --max-cars 1")
    if len(selected) != 1:
        raise ValueError(f"expected exactly one matching replay, found {len(selected)}")
    return selected[0]


def resolve_replay_directories(run_path: str | Path) -> list[Path]:
    """Resolve an index directory, Stage 2 replay run, or parent containing runs."""
    path = Path(run_path)
    if (path / "index.json").is_file():
        return [path]
    candidates = sorted(
        {index.parent for index in path.glob("visual_replays/*/index.json")}
        | {index.parent for index in path.glob("*/index.json")}
    )
    return candidates


def interpolate_position(
    left: Sequence[float],
    right: Sequence[float],
    fraction: float,
) -> tuple[float, float, float]:
    if len(left) != 3 or len(right) != 3:
        raise ValueError("positions must have three components")
    t = _unit_fraction(fraction)
    return tuple(float(a + t * (b - a)) for a, b in zip(left, right, strict=True))


def interpolate_quaternion(
    left: Sequence[float],
    right: Sequence[float],
    fraction: float,
) -> tuple[float, float, float, float]:
    """Shortest-arc quaternion SLERP with normalized linear fallback near coincidence."""
    if len(left) != 4 or len(right) != 4:
        raise ValueError("quaternions must have four components")
    t = _unit_fraction(fraction)
    q0 = np.asarray(left, dtype=np.float64)
    q1 = np.asarray(right, dtype=np.float64)
    if not np.isfinite(q0).all() or not np.isfinite(q1).all():
        raise ValueError("quaternion components must be finite")
    norm0, norm1 = float(np.linalg.norm(q0)), float(np.linalg.norm(q1))
    if norm0 < 1e-12 or norm1 < 1e-12:
        raise ValueError("quaternions cannot be zero")
    q0 /= norm0
    q1 /= norm1
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        result /= np.linalg.norm(result)
    else:
        angle = math.acos(dot)
        sine = math.sin(angle)
        result = math.sin((1.0 - t) * angle) / sine * q0 + math.sin(t * angle) / sine * q1
    return tuple(float(value) for value in result)


def _unit_fraction(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("interpolation fraction must be finite")
    return min(1.0, max(0.0, value))


def interpolate_replay(
    samples: Sequence[ReplaySample],
    playback_elapsed_s: float,
    *,
    end_behavior: str = "fade",
    fade_duration_s: float = 0.75,
) -> InterpolatedTransform | None:
    """Interpolate one trajectory at shared swarm time, relative to its first sample."""
    if not samples:
        raise ValueError("replay trajectory must contain samples")
    if end_behavior not in {"disappear", "freeze", "fade"}:
        raise ValueError("end_behavior must be disappear, freeze, or fade")
    if not math.isfinite(playback_elapsed_s) or playback_elapsed_s < 0:
        raise ValueError("playback time must be finite and non-negative")
    if not math.isfinite(fade_duration_s) or fade_duration_s < 0:
        raise ValueError("fade duration must be finite and non-negative")
    for previous, current in zip(samples, samples[1:], strict=False):
        if current.elapsed_s < previous.elapsed_s:
            raise ValueError("replay sample times must be ordered")
    first_time = samples[0].elapsed_s
    last_time = samples[-1].elapsed_s
    local_time = first_time + playback_elapsed_s
    if local_time <= first_time:
        endpoint = samples[0]
        alpha = 1.0
    elif local_time >= last_time:
        endpoint = samples[-1]
        if local_time == last_time or end_behavior == "freeze":
            alpha = 1.0
        elif end_behavior == "disappear":
            return None
        elif fade_duration_s == 0 or local_time >= last_time + fade_duration_s:
            return None
        else:
            alpha = 1.0 - (local_time - last_time) / fade_duration_s
    else:
        endpoint = None
        alpha = 1.0
    if endpoint is not None:
        return InterpolatedTransform(
            elapsed_s=local_time,
            position_m=endpoint.position_m,
            quaternion_xyzw=endpoint.quaternion_xyzw,
            opacity=alpha,
        )
    for left, right in zip(samples, samples[1:], strict=False):
        if left.elapsed_s <= local_time <= right.elapsed_s:
            if right.elapsed_s == left.elapsed_s:
                fraction = 1.0
            else:
                fraction = (local_time - left.elapsed_s) / (right.elapsed_s - left.elapsed_s)
            return InterpolatedTransform(
                elapsed_s=local_time,
                position_m=interpolate_position(left.position_m, right.position_m, fraction),
                quaternion_xyzw=interpolate_quaternion(
                    left.quaternion_xyzw,
                    right.quaternion_xyzw,
                    fraction,
                ),
            )
    raise ValueError("could not locate replay interpolation interval")


def selection_report(
    total_indexed: int,
    selected: Sequence[ReplaySelection],
    *,
    matching_count: int,
    color_scale: ColorScale,
) -> dict[str, Any]:
    steps = [item.training_step for item in selected]
    colors: dict[str, int] = {}
    for item in selected:
        bucket = color_scale.bucket(item.training_step)
        colors[bucket] = colors.get(bucket, 0) + 1
    statuses: dict[str, int] = {}
    for item in selected:
        status = item.metadata["status"]
        statuses[status] = statuses.get(status, 0) + 1
    sorted_steps = sorted(steps)
    median = (
        (
            float(sorted_steps[len(sorted_steps) // 2])
            if len(sorted_steps) % 2
            else (sorted_steps[len(sorted_steps) // 2 - 1] + sorted_steps[len(sorted_steps) // 2]) / 2.0
        )
        if sorted_steps
        else None
    )
    failures = sum(item.metadata["status"] in FAILED_STATUSES for item in selected)
    return {
        "total_indexed_episodes": total_indexed,
        "episodes_matching_filters": matching_count,
        "selected_episode_count": len(selected),
        "selected_step_range": [min(steps), max(steps)] if steps else None,
        "training_step_min": min(steps) if steps else None,
        "training_step_max": max(steps) if steps else None,
        "training_step_median": median,
        "finish_count": statuses.get("finished", 0),
        "failure_count": failures,
        "status_counts": statuses,
        "color_buckets": colors,
        "selected_episodes": [
            {
                "run_id": item.metadata["run_id"],
                "episode_id": item.metadata["episode_id"],
                "training_step_start": item.training_step,
                "status": item.metadata["status"],
                "color": color_scale.hex_color(item.training_step),
                "color_bucket": color_scale.bucket(item.training_step),
                "replay_file": str(item.replay_directory / item.metadata["file"]),
            }
            for item in selected
        ],
    }


def _parse_range(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(":", maxsplit=1)
        result = int(left), int(right)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("range must be MIN:MAX using integers") from exc
    if result[0] < 0 or result[1] < result[0]:
        raise argparse.ArgumentTypeError("range must be non-negative and ordered")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polybot-replay-swarm",
        description="Inspect or play selected visual replays as renderer-only PolyTrack ghosts.",
    )
    parser.add_argument("--run", type=Path, required=True, help="replay run directory or parent containing replay runs")
    parser.add_argument("--steps", type=_parse_range, help="inclusive episode-start training-step range MIN:MAX")
    parser.add_argument("--episodes", type=_parse_range, help="inclusive numeric episode ID range MIN:MAX")
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument("--finished-only", action="store_true")
    filter_group.add_argument("--failed-only", action="store_true", help="include failed and timed-out episodes")
    parser.add_argument("--max-cars", type=int, default=100, help=f"maximum ghosts to load (1-{MAX_REPLAY_GHOSTS})")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--color-min-step", type=int, default=0)
    parser.add_argument("--color-max-step", type=int, default=1_000_000)
    parser.add_argument(
        "--action",
        choices=("play", "load", "resume", "pause", "restart", "seek", "configure", "clear", "status"),
        default="play",
    )
    parser.add_argument("--seek-seconds", type=float, help="playback offset used with --action seek")
    parser.add_argument("--speed", type=float, help="playback speed from 0.1 to 8")
    parser.add_argument("--opacity", type=float, help="ghost opacity from 0 to 1")
    parser.add_argument("--color", help="override selected replay color using #RRGGBB")
    parser.add_argument(
        "--end-behavior",
        choices=("disappear", "freeze", "fade"),
        help="what the ghost does after the final sample",
    )
    parser.add_argument("--fade-duration", type=float, help="fade duration in seconds")
    parser.add_argument("--port", type=int, default=8765, help="existing local PolyBot bridge port")
    parser.add_argument(
        "--color-stop",
        action="append",
        default=[],
        metavar="STEP:COLOR",
        help="custom color stop; repeat, e.g. --color-stop 0:#ff0000 --color-stop 500000:#ffff00",
    )
    parser.add_argument("--dry-run", action="store_true", help="print selection report only; no game connection")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_cars < 1 or args.max_cars > MAX_REPLAY_GHOSTS:
        parser.error(f"--max-cars must be from 1 to {MAX_REPLAY_GHOSTS}")
    if args.sample_seed < 0:
        parser.error("--sample-seed must be non-negative")
    try:
        stops = parse_color_stops(args.color_stop) if args.color_stop else None
        color_scale = ColorScale(args.color_min_step, args.color_max_step, stops)
        directories = resolve_replay_directories(args.run)
        entries = [(directory, metadata) for directory in directories for metadata in load_replay_index(directory)]
        matched_metadata = filter_replays(
            [metadata for _, metadata in entries],
            steps=args.steps,
            episodes=args.episodes,
            finished_only=args.finished_only,
            failed_only=args.failed_only,
        )
        matched_keys = {(entry["run_id"], entry["episode_id"]) for entry in matched_metadata}
        matching_count = len(matched_metadata)
        matching = [
            ReplaySelection(directory, metadata)
            for directory, metadata in entries
            if (metadata["run_id"], metadata["episode_id"]) in matched_keys
        ]
        if len(matching) > args.max_cars:
            picked = stratified_sample(
                matched_metadata,
                args.max_cars,
                seed=args.sample_seed,
            )
            picked_keys = {(entry["run_id"], entry["episode_id"]) for entry in picked}
            selected = [
                item for item in matching if (item.metadata["run_id"], item.metadata["episode_id"]) in picked_keys
            ]
        else:
            selected = matching
        selected.sort(key=lambda item: (item.training_step, item.metadata["episode_id"]))
        report = selection_report(
            len(entries),
            selected,
            matching_count=matching_count,
            color_scale=color_scale,
        )
        report["run"] = str(args.run)
        report["selection_step_filter"] = list(args.steps) if args.steps else None
        report["sample_seed"] = args.sample_seed
        print(json.dumps(report, indent=2, allow_nan=False))
        if args.dry_run:
            return 0
        payloads: list[SelectedReplayPayload] = []
        if args.action in {"play", "load"}:
            if not selected:
                raise ValueError(f"{args.action} requires at least one episode matching the selection")
            if len(selected) > MAX_REPLAY_GHOSTS:
                raise ValueError(f"selection exceeds maximum ghost count ({MAX_REPLAY_GHOSTS})")
            indexed_total = sum(item.metadata["sample_count"] for item in selected)
            if indexed_total > MAX_REPLAY_TOTAL_SAMPLES:
                raise ValueError(f"selection exceeds maximum total sample count ({MAX_REPLAY_TOTAL_SAMPLES})")
            for item in selected:
                payload = load_replay_episode(
                    item.replay_directory,
                    item.metadata,
                    include_optional=False,
                )
                payloads.append(
                    SelectedReplayPayload(
                        item,
                        payload,
                        args.color or color_scale.hex_color(item.training_step),
                    )
                )
        options = ReplayPlaybackOptions(
            speed=1.0 if args.speed is None else args.speed,
            opacity=0.5 if args.opacity is None else args.opacity,
            color=args.color or "#ffffff",
            end_behavior=args.end_behavior or "fade",
            fade_duration_s=0.75 if args.fade_duration is None else args.fade_duration,
        )
        transport = WebSocketServerTransport(
            port=args.port,
            connect_timeout_s=60.0,
            request_timeout_s=35.0,
        )
        try:
            result = send_replay_swarm(
                transport,
                action=args.action,
                payloads=payloads,
                options=options,
                seek_seconds=args.seek_seconds,
                update_settings=tuple(
                    name
                    for name, value in (
                        ("speed", args.speed),
                        ("opacity", args.opacity),
                        ("color", args.color),
                        ("end_behavior", args.end_behavior),
                        ("fade_duration", args.fade_duration),
                    )
                    if value is not None
                ),
            )
        finally:
            transport.close()
        print(
            json.dumps(
                {
                    "action": args.action,
                    "selected_episode_count": len(payloads),
                    "result": result,
                },
                indent=2,
                allow_nan=False,
            )
        )
    except (ConnectionError, TimeoutError, ProtocolViolation) as exc:
        print(f"polybot-replay-swarm: bridge command failed: {exc}", file=sys.stderr)
        return 3
    except (ReplayFormatError, ValueError, OSError) as exc:
        print(f"polybot-replay-swarm: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
