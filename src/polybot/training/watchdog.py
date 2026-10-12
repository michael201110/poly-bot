"""Recover interrupted live training from a confirmed saved checkpoint."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from polybot.environment.curriculum import build_plan
from polybot.protocol import ProtocolViolation
from polybot.training.config import CurriculumConfig, CurriculumPhaseConfig, TrainingConfig
from polybot.training.runner import TrainingRunner


def remaining_config(config: TrainingConfig, completed: int) -> TrainingConfig | None:
    """Retain the original budget and the remaining curriculum after reconnecting."""
    if completed >= config.timesteps:
        return None
    phases = []
    for phase in build_plan(config.curriculum, config.timesteps).phases:
        if completed >= phase.steps:
            completed -= phase.steps
            continue
        phases.append(CurriculumPhaseConfig(
            mode=phase.mode, steps=phase.steps - completed,
            start_ratio=phase.start_ratio, end_ratio=phase.end_ratio,
            start_s=phase.start_s, end_s=phase.end_s, lead_in_ratio=phase.lead_in_ratio,
        ))
        completed = 0
    return replace(config, timesteps=sum(p.steps for p in phases),
                   curriculum=CurriculumConfig("custom", phases=tuple(phases)))


def run_with_watchdog(
    runner: TrainingRunner, *, resume: Path | None, fresh_replay: bool = False,
    rollback_to_champion: bool = False, pace_polish: bool = False,
    on_runner: Callable[[TrainingRunner], None] | None = None,
    retry_delay_s: float = 3.0, max_retries: int = 5,
) -> Path | None:
    """Restart an episode after transport failure; never resend an ambiguous step.

    Stop requests share one event across all attempts. Only the checkpoint
    explicitly saved by the failing attempt can be resumed after learning began.
    Protocol, CUDA, model and disk failures are not treated as reconnect events.
    """
    original = runner.config
    stop = runner.stop_requested
    initial_steps = runner.registry.read_metadata(resume).training_timesteps if resume else 0
    attempt = 0
    reconnecting = False
    while not stop.is_set():
        attempt_runner = runner
        attempt_start_step = runner.registry.read_metadata(resume).training_timesteps if resume else 0
        saved: Path | None = None
        status = runner.status

        def observe(event: dict) -> None:
            nonlocal saved
            if event.get("type") == "transport_disconnected":
                saved = Path(event["checkpoint"])
            if status:
                status(event)

        runner.status = observe
        try:
            return runner.run(resume=resume, fresh_replay=fresh_replay,
                              rollback_to_champion=rollback_to_champion, pace_polish=pace_polish,
                              preserve_resume_actor=reconnecting)
        except (ConnectionError, TimeoutError, ProtocolViolation) as exc:
            if original.backend != "websocket" or pace_polish or (
                isinstance(exc, ProtocolViolation) and not str(exc).startswith("stale_episode:")
            ):
                raise
            if stop.is_set():
                return saved
            if runner.model is not None and saved is None:
                # A failed disk write must not silently restore older learner state.
                raise
            if saved is not None:
                saved_steps = runner.registry.read_metadata(saved).training_timesteps
                # A healthy stretch of learning breaks the failure streak.
                # Five interruptions spread across a long session must not
                # consume a lifetime retry allowance and halt good training.
                if saved_steps - attempt_start_step >= 256:
                    attempt = 0
                resume = saved
                fresh_replay = False
                # Interrupted transferred-policy fidelity gates must be repeated.
                if original.grtqc and original.grtqc.training_origin != "scratch":
                    initialization = runner.registry.slot(
                        original.track_name, "grtqc", "initialization", track_slug=original.track_slug,
                    )
                    if runner.registry.read_metadata(initialization).evaluation is None:
                        resume, fresh_replay = initialization, True
            attempt += 1
            if attempt > max_retries:
                if status:
                    status({"type": "watchdog_exhausted", "error": str(exc),
                            "max_retries": max_retries, "resume": str(resume) if resume else None})
                raise
            completed = max(0, runner.registry.read_metadata(resume).training_timesteps - initial_steps) if resume else 0
            config = remaining_config(original, completed)
            if config is None:
                return resume
            if status:
                status({"type": "watchdog_reconnect", "attempt": attempt,
                        "max_retries": max_retries, "error": str(exc),
                        "resume": str(resume) if resume else None,
                        "remaining_steps": config.timesteps, "retry_delay_s": retry_delay_s})
            if stop.wait(retry_delay_s):
                return resume
            retry = TrainingRunner(config, status, runner.transport_factory)
            retry.stop_requested = stop
            retry.set_ai_overlay_settings(runner.ai_overlay_settings)
            runner = retry
            reconnecting = True
            if on_runner:
                on_runner(runner)
        finally:
            # Restore the original callback on the completed attempt.
            # Newly constructed retry runners already use that callback.
            attempt_runner.status = status
    return resume
