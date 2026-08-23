"""Headless PPO training service used by graphical and scripted front ends."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.training.config import TrainingConfig, policy_kwargs
from polybot.training.devices import DeviceInfo, resolve_device
from polybot.training.models import (
    IncompatibleModelError,
    ModelMetadata,
    ModelRegistry,
    git_commit,
    load_ppo,
)
from polybot.transport import WebSocketServerTransport

StatusCallback = Callable[[dict[str, Any]], None]


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
            reward_config=cfg.rewards,
            pwm_enabled=cfg.pwm_enabled,
            pwm_levels=cfg.pwm_levels,
            curriculum_start_ratio=cfg.curriculum.start_ratio,
            curriculum_end_ratio=cfg.curriculum.end_ratio,
            curriculum_start_s=cfg.curriculum.start_s,
            curriculum_end_s=cfg.curriculum.end_s,
        )
        registry = ModelRegistry(cfg.output_root)
        directory = registry.initialise_track(cfg.track_name)
        output = directory / "latest"
        service = self
        started_at = time.monotonic()

        class Callback(BaseCallback):
            def _on_step(self) -> bool:
                info = (self.locals.get("infos") or [{}])[-1]
                rewards = self.locals.get("rewards") or [0.0]
                elapsed_real = max(time.monotonic() - started_at, 1e-9)
                service.status(
                    {
                        "type": "progress",
                        "track": cfg.track_name,
                        "model": cfg.model_name,
                        "architecture": cfg.architecture,
                        "timesteps": self.num_timesteps,
                        "steps_per_second": self.num_timesteps / elapsed_real,
                        "reward": float(rewards[-1]),
                        "model_path": str(output.with_suffix(".zip")),
                        "learning_rate": cfg.learning_rate,
                        **info,
                    }
                )
                if cfg.checkpoint_interval and self.num_timesteps % cfg.checkpoint_interval == 0:
                    self.model.save(str(directory / "checkpoints" / f"step-{self.num_timesteps}"))
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
                self.model = load_ppo(resume, env=env, device=self.device.resolved)
            else:
                self.model = PPO(
                    "MlpPolicy",
                    env,
                    seed=cfg.seed,
                    device=self.device.resolved,
                    learning_rate=cfg.learning_rate,
                    gamma=cfg.gamma,
                    gae_lambda=cfg.gae_lambda,
                    ent_coef=cfg.entropy_coefficient,
                    policy_kwargs=policy_kwargs(cfg.architecture),
                    n_steps=max(2, min(2048, cfg.timesteps)),
                    batch_size=max(2, min(64, cfg.timesteps)),
                    verbose=0,
                )
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
            self.model.save(str(output))
            metadata = ModelMetadata(
                track_name=cfg.track_name,
                track_id=cfg.track_id,
                architecture=cfg.architecture,
                parameter_count=parameters,
                lookahead_count=cfg.lookahead_count,
                action_schema="pwm-multidiscrete-v1"
                if cfg.pwm_enabled
                else "digital-multidiscrete-v1",
                pwm_enabled=cfg.pwm_enabled,
                pwm_resolution=cfg.pwm_levels,
                frame_skip=cfg.frame_skip,
                training_timesteps=int(self.model.num_timesteps),
                seed=cfg.seed,
                reward_settings=asdict(cfg.rewards),
                ppo_hyperparameters={
                    "learning_rate": cfg.learning_rate,
                    "gamma": cfg.gamma,
                    "gae_lambda": cfg.gae_lambda,
                    "entropy_coefficient": cfg.entropy_coefficient,
                },
                polybot_version="0.1.0",
                git_commit=git_commit(),
            )
            registry.write_metadata(metadata, "latest")
            self.status(
                {
                    "type": "stopped" if self._stop.is_set() else "completed",
                    "path": str(output.with_suffix(".zip")),
                }
            )
            return output.with_suffix(".zip")
        finally:
            env.close()
