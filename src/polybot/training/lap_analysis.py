"""Split, trajectory, and airborne-region analysis for WR pace search."""

from __future__ import annotations

import statistics
from typing import Any

import numpy as np


def _progress(sample: dict[str, Any]) -> float:
    length = max(float(sample.get("track_length_m", 1.0)), 1e-9)
    return float(sample.get("route_progress_m", 0.0)) / length


def sector_times(samples: list[dict[str, Any]], resolution: float = 0.05) -> dict[float, float]:
    if not 0 < resolution <= 0.2:
        raise ValueError("sector resolution must be in (0, 0.2]")
    if len(samples) < 2:
        return {}
    boundaries = np.arange(0.0, 1.0, resolution).tolist() + [1.0]
    result: dict[float, float] = {}
    for boundary in boundaries:
        for previous, current in zip(samples, samples[1:], strict=False):
            p0, p1 = _progress(previous), _progress(current)
            if p0 <= boundary <= p1 and p1 > p0:
                mix = (boundary - p0) / (p1 - p0)
                t0, t1 = float(previous.get("elapsed_s", 0)), float(current.get("elapsed_s", 0))
                result[round(boundary, 6)] = t0 + mix * (t1 - t0)
                break
        if boundary == 0 and not result:
            result[0.0] = float(samples[0].get("elapsed_s", 0))
        if boundary == 1.0:
            result[1.0] = float(samples[-1].get("elapsed_s", 0))
    return result


def sector_delta_map(
    champion: list[dict[str, Any]], candidate: list[dict[str, Any]],
    resolution: float = 0.05,
) -> list[dict[str, float]]:
    before = sector_times(champion, resolution)
    after = sector_times(candidate, resolution)
    rows = []
    for start in sorted(set(before) & set(after)):
        if start >= 1.0:
            continue
        end = round(min(1.0, start + resolution), 6)
        if end not in before or end not in after:
            continue
        c_delta = before[end] - before[start]
        n_delta = after[end] - after[start]
        candidate_samples = [s for s in candidate if start <= _progress(s) <= end]
        champion_samples = [s for s in champion if start <= _progress(s) <= end]
        c_speed = statistics.fmean(float(s.get("speed_mps", 0)) for s in champion_samples) if champion_samples else 0.0
        n_speed = (
            statistics.fmean(float(s.get("speed_mps", 0)) for s in candidate_samples)
            if candidate_samples else 0.0
        )
        rows.append({
            "start": start, "end": end,
            "champion_s": c_delta, "candidate_s": n_delta,
            "delta_s": n_delta - c_delta,
            "cumulative_delta_s": (after[end] - before[end]),
            "speed_delta_mps": n_speed - c_speed,
        })
    return rows


def discover_airborne_regions(
    traces: list[list[dict[str, Any]]], *, minimum_duration_s: float = 0.02,
) -> list[dict[str, float]]:
    regions: list[dict[str, float]] = []
    for trace in traces:
        active: list[dict[str, Any]] = []
        for sample in trace:
            if sample.get("airborne"):
                active.append(sample)
            elif active:
                _append_air_region(regions, active, minimum_duration_s, landed=True)
                active = []
        if active:
            _append_air_region(regions, active, minimum_duration_s, landed=False)
    if not regions:
        return []
    clusters: list[list[dict[str, float]]] = []
    for region in sorted(regions, key=lambda item: item["start"]):
        if not clusters or region["start"] > max(item["end"] for item in clusters[-1]):
            clusters.append([region])
        else:
            clusters[-1].append(region)
    return [_median_air_region(cluster) for cluster in clusters]


def _median_air_region(regions: list[dict[str, float]]) -> dict[str, float]:
    keys = regions[0].keys()
    summary = {key: float(statistics.median(region[key] for region in regions)) for key in keys}
    quaternion = np.asarray([summary[f"landing_q{axis}"] for axis in "xyzw"], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if norm > 1e-9:
        quaternion /= norm
    for axis, value in zip("xyzw", quaternion, strict=True):
        summary[f"landing_q{axis}"] = float(value)
    summary["lap_count"] = float(len(regions))
    return summary


def _append_air_region(
    output: list[dict[str, float]], samples: list[dict[str, Any]], minimum_duration_s: float,
    *, landed: bool,
) -> None:
    first, last = samples[0], samples[-1]
    duration = float(last.get("elapsed_s", 0)) - float(first.get("elapsed_s", 0))
    if duration < minimum_duration_s:
        return
    position = last.get("position_m", (0, 0, 0))
    velocity = np.asarray(last.get("local_velocity_mps", (0, 0, 0)), dtype=float)
    quaternion = last.get("quaternion_xyzw", (0, 0, 0, 1))
    output.append({
        "start": _progress(first), "end": _progress(last), "duration_s": duration,
        "takeoff_speed_mps": float(first.get("speed_mps", 0)),
        "landing_speed_mps": float(last.get("speed_mps", 0)),
        "landing_position_x_m": float(position[0]), "landing_position_y_m": float(position[1]),
        "landing_position_z_m": float(position[2]),
        "landing_longitudinal_speed_mps": float(velocity[2]) if velocity.size > 2 else 0.0,
        "landing_qx": float(quaternion[0]), "landing_qy": float(quaternion[1]),
        "landing_qz": float(quaternion[2]), "landing_qw": float(quaternion[3]),
        "landed": float(landed),
    })
