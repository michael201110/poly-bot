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
        source = event.get("resume_source")
        source_slot = str(source).replace("\\", "/").split("/")[-1] if source else ""
        resumed = f" · resumed {source_slot}" if source else ""
        refill = " · refilling replay" if event.get("fresh_replay") else ""
        restore = " · champion rollback enabled" if event.get("rollback_on_regression") else ""
        return (
            f"{prefix}Started {event['algorithm'].upper()} on {event['device'].upper()}"
            f"{gpu}{resumed}{refill}{restore}"
        )
    if kind == "rollback":
        return (
            f"{prefix}Restored champion{step} · evaluation {_percent(event['evaluated_progress'])}"
            f" versus champion {_percent(event['champion_progress'])}"
            f" · replay from {event['replay_source']}"
        )
    if kind == "champion_replay":
        return f"{prefix}Saved policy-generated replay with champion{step}"
    if kind == "phase":
        spawn = event.get("spawn_ratio")
        target_start = event.get("start_ratio")
        target_end = event.get("end_ratio")
        section = (
            f" · spawn {_percent(spawn)}; target {_percent(target_start)}–{_percent(target_end)}"
            if spawn is not None else " · full track"
        )
        epsilon = event.get("initial_epsilon")
        explore = f" · epsilon {epsilon:.2f}" if epsilon is not None else ""
        return (
            f"{prefix}Phase {event['index']} · {event['mode']}{section} · "
            f"{event['steps']:,} planned steps{explore}"
        )
    if kind == "curriculum_reset":
        action = event.get("initial_previous_action", {})
        return (
            f"{prefix}Curriculum reset · {event['mode']} · spawn {_percent(event.get('spawn_ratio'))} · "
            f"speed {_number(event.get('initial_speed_mps'), ' m/s')} · "
            f"previous action steer {action.get('steer', 0):+g}, throttle {action.get('throttle', 0):g}, "
            f"brake {action.get('brake', 0):g} · actual steer "
            f"{_number(event.get('initial_actual_steering'))} · epsilon {_number(event.get('epsilon'))}"
        )
    if kind == "phase_summary":
        actions = ", ".join(str(value) for value in event.get("actions_seen", ())) or "none"
        return (
            f"{prefix}Phase {event['index']} complete · epsilon {_number(event.get('epsilon'))} · "
            f"actions tried {actions} · replay {event.get('replay_size', 0):,}"
        )
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
