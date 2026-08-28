"""Headless PPO training service used by graphical and scripted front ends."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.training.anchored_ppo import TeacherAnchoredPPO
from polybot.training.config import TrainingConfig, policy_kwargs
from polybot.training.devices import DeviceInfo, resolve_device
from polybot.training.initialization import apply_forward_bias
from polybot.training.models import (
    IncompatibleModelError,
    ModelMetadata,
    ModelRegistry,
    git_commit,
)
from polybot.transport import WebSocketServerTransport

StatusCallback = Callable[[dict[str, Any]], None]


class RollingStepRate:
    """Measure recent environment throughput without counting resumed history."""

    def __init__(self, window_s: float = 5.0) -> None:
        if window_s <= 0:
            raise ValueError("rate window must be positive")
        self.window_s = window_s
        self.samples: deque[tuple[float, int]] = deque()

    def update(self, timesteps: int, now: float | None = None) -> float:
        timestamp = time.monotonic() if now is None else now
        self.samples.append((timestamp, timesteps))
        cutoff = timestamp - self.window_s
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        started_at, started_steps = self.samples[0]
        elapsed = timestamp - started_at
        if elapsed <= 0:
            return 0.0
        return max(0, timesteps - started_steps) / elapsed


class TrainingService:
    """One stoppable training run; it contains no UI dependencies."""

    def __init__(self, config: TrainingConfig, status: StatusCallback | None = None) -> None:
        self.config = config
        self.status = status or (lambda _event: None)
        self._stop = threading.Event()
        self.model: Any = None
        self.device: DeviceInfo | None = None

    def stop(self) -> None:
        self._stop.set()

    def save_model(self, name: str, *, best_lap_time_s: float | None = None) -> Path:
        """Persist the current in-memory policy and its compatibility metadata."""

        if self.model is None:
            raise RuntimeError("training model has not been initialised")
        if name not in {"latest", "best"}:
            raise ValueError("model name must be latest or best")
        cfg = self.config
        registry = ModelRegistry(cfg.output_root)
        output = registry.initialise_track(cfg.track_name) / name
        self.model.save(str(output))
        parameters = sum(p.numel() for p in self.model.policy.parameters())
        metadata = ModelMetadata(
            track_name=cfg.track_name,
            track_id=cfg.track_id,
            architecture=cfg.architecture,
            parameter_count=parameters,
            lookahead_count=cfg.lookahead_count,
            action_schema=(
                "pwm-multidiscrete-v1" if cfg.pwm_enabled else "digital-multidiscrete-v1"
            ),
            pwm_enabled=cfg.pwm_enabled,
            pwm_resolution=cfg.pwm_levels,
            frame_skip=cfg.frame_skip,
            training_timesteps=int(self.model.num_timesteps),
            best_lap_time_s=best_lap_time_s,
            seed=cfg.seed,
            reward_settings=asdict(cfg.rewards),
            ppo_hyperparameters={
                "learning_rate": cfg.learning_rate,
                "gamma": cfg.gamma,
                "gae_lambda": cfg.gae_lambda,
                "entropy_coefficient": cfg.entropy_coefficient,
                "rollout_steps": cfg.rollout_steps,
                "batch_size": cfg.batch_size,
                "ppo_epochs": cfg.ppo_epochs,
                "teacher_model": None if cfg.teacher_model is None else str(cfg.teacher_model),
                "teacher_kl_coefficient": cfg.teacher_kl_coefficient,
            },
            polybot_version="0.1.0",
            git_commit=git_commit(),
        )
        registry.write_metadata(metadata, name)
        return output.with_suffix(".zip")

    def save_latest(self) -> Path:
        """Persist the current in-memory policy and its resume metadata."""

        return self.save_model("latest")

    def run(self, *, resume: str | Path | None = None, transport: Any | None = None) -> Path:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback

        cfg = self.config
        self.device = resolve_device(cfg.device)
        if transport is None:
            transport = (
                MockSimulatorTransport()
                if cfg.backend == "mock"
                else WebSocketServerTransport(connect_timeout_s=300.0, request_timeout_s=300.0)
            )
        env = PolyTrackEnv(
            transport,
            track_id=cfg.track_id,
            lookahead_count=cfg.lookahead_count,
            frame_skip=cfg.frame_skip,
            max_episode_steps=2_000_000_000,
            max_episode_s=cfg.max_episode_seconds,
            reward_config=cfg.rewards,
            pwm_enabled=cfg.pwm_enabled,
            pwm_levels=cfg.pwm_levels,
            curriculum_start_ratio=cfg.curriculum.start_ratio,
            curriculum_end_ratio=cfg.curriculum.end_ratio,
            curriculum_start_s=cfg.curriculum.start_s,
            curriculum_end_s=cfg.curriculum.end_s,
            curriculum_random_quarters=cfg.curriculum.mode == "quarters-randomised",
        )
        registry = ModelRegistry(cfg.output_root)
        directory = registry.initialise_track(cfg.track_name)
        try:
            persisted_best_lap_s = registry.read_metadata(
                cfg.track_name, "best"
            ).best_lap_time_s
        except (FileNotFoundError, TypeError, ValueError):
            persisted_best_lap_s = None
        if not resume:
            archived = registry.archive_latest(cfg.track_name)
            if archived:
                self.status({"type": "archived", "path": str(archived)})
        service = self
        class Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.last_ui_update = 0.0
                self.episode = 1
                self.episode_reward = 0.0
                self.episode_reward_terms: dict[str, float] = {}
                self.episode_steps = 0
                self.max_progress = 0.0
                self.finishes = 0
                self.crashes = 0
                self.best_lap_s = persisted_best_lap_s
                self.step_rate = RollingStepRate(5.0)

            def _on_training_start(self) -> None:
                self.step_rate.update(self.num_timesteps)

            def _on_step(self) -> bool:
                infos = self.locals.get("infos")
                info = infos[-1] if infos is not None and len(infos) else {}
                rewards = self.locals.get("rewards")
                reward = float(rewards[-1]) if rewards is not None and len(rewards) else 0.0
                dones = self.locals.get("dones")
                done = bool(dones[-1]) if dones is not None and len(dones) else False
                self.episode_reward += reward
                for name, value in info.get("reward_terms", {}).items():
                    previous = self.episode_reward_terms.get(name, 0.0)
                    self.episode_reward_terms[name] = previous + float(value)
                self.episode_steps += 1
                track_length = max(1.0, float(info.get("track_length_m", 1.0)))
                progress = float(info.get("route_progress_m", 0.0)) / track_length
                self.max_progress = max(self.max_progress, progress)
                events = tuple(info.get("events", ()))
                elapsed_s = float(info.get("elapsed_s", 0.0))
                simulator_info = info.get("simulator_info", {})
                speed_kmh = float(simulator_info.get("speed_kmh", 0.0))
                now = time.monotonic()
                steps_per_second = self.step_rate.update(self.num_timesteps, now)
                if now - self.last_ui_update >= 0.25 or done:
                    service.status(
                        {
                            "type": "progress",
                            "timesteps": self.num_timesteps,
                            "steps_per_second": steps_per_second,
                            "episode": self.episode,
                            "episode_reward": self.episode_reward,
                            "episode_steps": self.episode_steps,
                            "max_progress": self.max_progress,
                            "elapsed_s": elapsed_s,
                            "speed_kmh": speed_kmh,
                            "finishes": self.finishes,
                            "crashes": self.crashes,
                            "best_lap_s": self.best_lap_s,
                            "quarter": info.get("curriculum_quarter"),
                        }
                    )
                    self.last_ui_update = now
                if done:
                    finished = "finish" in events
                    crashed = "crash" in events
                    self.finishes += int(finished)
                    self.crashes += int(crashed)
                    if finished and (self.best_lap_s is None or elapsed_s < self.best_lap_s):
                        self.best_lap_s = elapsed_s
                        best_path = service.save_model(
                            "best", best_lap_time_s=self.best_lap_s
                        )
                        service.status(
                            {
                                "type": "best_model",
                                "path": str(best_path),
                                "lap_s": self.best_lap_s,
                            }
                        )
                    result = "time_limit" if info.get("wrapper_time_limit") else next(
                        (
                            name
                            for name in (
                                "finish",
                                "crash",
                                "barrier_contact",
                                "off_track",
                                "stalled",
                                "time_limit",
                            )
                            if name in events
                        ),
                        events[-1] if events else "reset",
                    )
                    service.status(
                        {
                            "type": "episode",
                            "episode": self.episode,
                            "reward": self.episode_reward,
                            "reward_terms": dict(self.episode_reward_terms),
                            "steps": self.episode_steps,
                            "progress": self.max_progress,
                            "elapsed_s": elapsed_s,
                            "result": result,
                            "finishes": self.finishes,
                            "crashes": self.crashes,
                            "best_lap_s": self.best_lap_s,
                            "quarter": info.get("curriculum_quarter"),
                        }
                    )
                    self.episode += 1
                    self.episode_reward = 0.0
                    self.episode_reward_terms = {}
                    self.episode_steps = 0
                    self.max_progress = 0.0
                if cfg.checkpoint_interval and self.num_timesteps % cfg.checkpoint_interval == 0:
                    self.model.save(str(directory / "checkpoints" / f"step-{self.num_timesteps}"))
                    service.status({"type": "checkpoint", "timesteps": self.num_timesteps})
                return not service._stop.is_set()

        try:
            if resume:
                resume_path = Path(resume)
                metadata_name = resume_path.stem
                try:
                    resume_metadata = registry.read_metadata(cfg.track_name, metadata_name)
                except FileNotFoundError:
                    if cfg.pwm_enabled:
                        raise IncompatibleModelError(
                            "model has no compatibility metadata; select legacy digital mode "
                            "or add verified metadata before resuming"
                        ) from None
                else:
                    registry.assert_compatible(
                        resume_metadata,
                        track_name=cfg.track_name,
                        action_schema=(
                            "pwm-multidiscrete-v1"
                            if cfg.pwm_enabled
                            else "digital-multidiscrete-v1"
                        ),
                        architecture=cfg.architecture,
                    )
                self.model = TeacherAnchoredPPO.load(
                    str(resume),
                    env=env,
                    device=self.device.resolved,
                    custom_objects={
                        "n_steps": cfg.rollout_steps,
                        "batch_size": cfg.batch_size,
                        "n_epochs": cfg.ppo_epochs,
                        "learning_rate": cfg.learning_rate,
                        "gamma": cfg.gamma,
                        "gae_lambda": cfg.gae_lambda,
                        "ent_coef": cfg.entropy_coefficient,
                    },
                )
            else:
                self.model = TeacherAnchoredPPO(
                    "MlpPolicy",
                    env,
                    seed=cfg.seed,
                    device=self.device.resolved,
                    learning_rate=cfg.learning_rate,
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                    ent_coef=cfg.entropy_coefficient,
                    policy_kwargs=policy_kwargs(cfg.architecture),
                    n_steps=max(2, min(cfg.rollout_steps, cfg.timesteps)),
                    batch_size=max(2, min(cfg.batch_size, cfg.timesteps)),
                    n_epochs=cfg.ppo_epochs,
                    verbose=0,
                )
                apply_forward_bias(self.model)
            teacher = None
            if cfg.teacher_model is not None:
                if not cfg.teacher_model.is_file():
                    raise FileNotFoundError(f"teacher model not found: {cfg.teacher_model}")
                teacher = PPO.load(str(cfg.teacher_model), device=self.device.resolved)
            self.model.set_teacher(teacher, cfg.teacher_kl_coefficient)
            parameters = sum(p.numel() for p in self.model.policy.parameters())
            self.status(
                {
                    "type": "started",
                    "device": self.device.resolved,
                    "gpu_name": self.device.gpu_name,
                    "parameter_count": parameters,
                }
            )
            self.model.learn(
                cfg.timesteps, callback=Callback(), reset_num_timesteps=not bool(resume)
            )
            output = self.save_latest()
            self.status(
                {
                    "type": "stopped" if self._stop.is_set() else "completed",
                    "path": str(output),
                }
            )
            return output
        finally:
            env.close()
