"""Short, human-readable training event descriptions for the GUI."""

from __future__ import annotations

from datetime import datetime
from typing import Any

EVENT_NAMES = {
    "airborne_roll_failure": "unstable landing",
    "barrier_contact": "barrier impact",
    "crash": "crash",
    "finish": "finished",
    "off_track": "off track",
    "stalled": "stalled",
    "time_limit": "time limit",
}


def _time(event: dict[str, Any]) -> str:
    stamp = event.get("at")
    if not stamp:
        return ""
    try:
        return datetime.fromisoformat(stamp).astimezone().strftime("%H:%M:%S  ")
    except ValueError:
        return ""


def _number(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{value:,.1f}{suffix}"


def _percent(value: Any) -> str:
    return "—" if value is None else f"{value:.1%}"


def format_event(event: dict[str, Any]) -> str | None:
    """Summarise one JSONL event without hiding the full on-disk diagnostics."""
    kind = event.get("type")
    if kind == "progress":
        return None  # The status gauges update more often than the event history.
    prefix = _time(event)
    steps = event.get("timesteps")
    step = f" · step {steps:,}" if isinstance(steps, int) else ""

    if kind == "plan":
        return f"{prefix}Plan · {event['total_steps']:,} steps across {len(event['phases'])} phase(s)"
    if kind == "started":
        gpu = f" ({event['gpu_name']})" if event.get("gpu_name") else ""
        return f"{prefix}Started {event['algorithm'].upper()} on {event['device'].upper()}{gpu}"
    if kind == "phase":
        return f"{prefix}Phase {event['index']} · {event['mode']} · {event['steps']:,} planned steps"
    if kind == "episode":
        reasons = ", ".join(EVENT_NAMES.get(name, name.replace("_", " "))
                            for name in event.get("events", ())) or "ended"
        line = (
            f"{prefix}Episode {event['episode']}{step} · {_percent(event.get('progress'))} of track"
            f" · {reasons} · reward {event['reward']:+,.1f}"
        )
        if event.get("elapsed_s") is not None:
            line += f" · {_number(event['elapsed_s'], 's')}"
        groups = event.get("reward_groups") or {}
        meaningful = sorted(
            ((name, value) for name, value in groups.items() if abs(value) >= 0.05),
            key=lambda pair: abs(pair[1]), reverse=True,
        )[:3]
        if meaningful:
            line += "\n    Main rewards: " + " · ".join(
                f"{name} {value:+,.1f}" for name, value in meaningful
            )
        return line
    if kind == "evaluation":
        lap = _number(event.get("best_lap_s"), "s")
        return (
            f"{prefix}Evaluation{step} · finish {_percent(event.get('finish_rate'))}"
            f" · median progress {_percent(event.get('median_progress'))} · best lap {lap}"
        )
    if kind == "champion":
        return f"{prefix}New champion{step} · best evaluated policy saved"
    if kind == "checkpoint":
        return f"{prefix}Checkpoint{step} · training state saved"
    if kind in {"completed", "stopped"}:
        return f"{prefix}{'Completed' if kind == 'completed' else 'Stopped safely'}{step} · latest saved"
    return f"{prefix}{str(kind or 'Event').replace('_', ' ').capitalize()}{step}"
