"""Headless training service shared by PPO and TQC."""

from __future__ import annotations

import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.training.algorithms import action_schema, configure_model, create_model, load_model
from polybot.training.config import TrainingConfig
from polybot.training.devices import DeviceInfo, resolve_device
from polybot.training.models import (
    IncompatibleModelError,
    ModelMetadata,
    ModelRegistry,
    git_commit,
)
from polybot.transport import WebSocketServerTransport

StatusCallback = Callable[[dict[str, Any]], None]


class ScaledTrainingReward(gym.RewardWrapper):
    """Scale PPO targets while retaining raw game reward terms in info."""

    def __init__(self, env: gym.Env, scale: float) -> None:
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return float(reward) * self.scale


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


def tqc_policy_diagnostics(model: Any, observation: np.ndarray) -> dict[str, Any]:
    """Small current-state probe; never changes the policy or replay actions."""
    import torch

    with torch.no_grad():
        obs = torch.as_tensor(observation, dtype=torch.float32, device=model.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        mean, log_std, _ = model.policy.actor.get_action_dist_params(obs)
        deterministic = torch.tanh(mean)
        levels = (1.0, 0.5, 0.0, -0.5)
        actions = torch.tensor([[0.0, value] for value in levels], device=model.device)
        quantiles = model.critic(obs.expand(len(levels), -1), actions)
        estimates = quantiles.mean(dim=(1, 2)).tolist()
        critic_values = dict(zip((str(value) for value in levels), estimates, strict=True))
    return {
        "actor_longitudinal_mean": float(mean[0, 1].item()),
        "actor_longitudinal_log_std": float(log_std[0, 1].item()),
        "deterministic_longitudinal": float(deterministic[0, 1].item()),
        "critic_longitudinal_q": critic_values,
    }


class TrainingService:
    """One stoppable training run; it contains no UI dependencies."""

    def __init__(self, config: TrainingConfig, status: StatusCallback | None = None) -> None:
        self.config = config
        self.status = status or (lambda _event: None)
        self._stop = threading.Event()
        self.model: Any = None
        self.device: DeviceInfo | None = None
        self.simulator_ticks = 0
        self.episodes = 0
        self.finishes = 0
        self.crashes = 0
        self.max_progress = 0.0
        self.first_finish_timestep: int | None = None
        self.previous_wall_clock_seconds = 0.0
        self.started_at = time.monotonic()

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
        output = registry.initialise_track(cfg.track_name, cfg.algorithm) / name
        self.model.save(str(output))
        if cfg.algorithm == "tqc":
            self.model.save_replay_buffer(str(output.with_suffix(".replay.pkl")))
        parameters = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
        metadata = ModelMetadata(
            track_name=cfg.track_name,
            track_id=cfg.track_id,
            architecture=(cfg.architecture if cfg.algorithm == "ppo" else cfg.tqc.architecture),
            parameter_count=parameters,
            algorithm=cfg.algorithm.upper(),
            lookahead_count=cfg.lookahead_count,
            action_schema=action_schema(cfg),
            pwm_enabled=cfg.algorithm == "ppo" and cfg.pwm_enabled,
            pwm_resolution=cfg.pwm_levels,
            frame_skip=cfg.frame_skip,
            training_timesteps=int(self.model.num_timesteps),
            training_episodes=self.episodes,
            best_lap_time_s=best_lap_time_s,
            simulator_ticks=self.simulator_ticks,
            wall_clock_seconds=(
                self.previous_wall_clock_seconds + time.monotonic() - self.started_at
            ),
            finishes=self.finishes,
            crashes=self.crashes,
            reward_profile=cfg.reward_profile,
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
                "expert_imitation_coefficient": cfg.expert_imitation_coefficient,
                "reward_scale": cfg.reward_scale,
            } if cfg.algorithm == "ppo" else {},
            tqc_hyperparameters=asdict(cfg.tqc) if cfg.algorithm == "tqc" else {},
            polybot_version="0.1.0",
            git_commit=git_commit(),
        )
        registry.write_metadata(metadata, name, cfg.algorithm)
        return output.with_suffix(".zip")

    def save_latest(self) -> Path:
        """Persist the current in-memory policy and its resume metadata."""

        return self.save_model("latest")

    def run(self, *, resume: str | Path | None = None, transport: Any | None = None) -> Path:
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
            max_episode_steps=cfg.max_episode_steps,
            max_episode_s=cfg.max_episode_seconds,
            reward_config=cfg.rewards,
            pwm_enabled=cfg.algorithm == "ppo" and cfg.pwm_enabled,
            pwm_levels=cfg.pwm_levels,
            action_mode="continuous_pwm" if cfg.algorithm == "tqc" else None,
            curriculum_start_ratio=cfg.curriculum.start_ratio,
            curriculum_end_ratio=cfg.curriculum.end_ratio,
            curriculum_start_s=cfg.curriculum.start_s,
            curriculum_end_s=cfg.curriculum.end_s,
            curriculum_random_quarters=cfg.curriculum.mode == "quarters-randomised",
        )
        env = ScaledTrainingReward(env, cfg.reward_scale)
        registry = ModelRegistry(cfg.output_root)
        directory = registry.initialise_track(cfg.track_name, cfg.algorithm)
        persisted_best_lap_s = None
        if resume:
            try:
                persisted_best_lap_s = registry.read_metadata(
                    cfg.track_name, "best", cfg.algorithm
                ).best_lap_time_s
            except (FileNotFoundError, TypeError, ValueError):
                pass
        if not resume:
            archived = registry.archive_latest(cfg.track_name, cfg.algorithm)
            if archived:
                self.status({"type": "archived", "path": str(archived)})
        service = self
        class Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.last_ui_update = 0.0
                self.episode = service.episodes + 1
                self.episode_reward = 0.0
                self.episode_reward_terms: dict[str, float] = {}
                self.episode_steps = 0
                self.max_progress = 0.0
                self.finishes = service.finishes
                self.crashes = service.crashes
                self.best_lap_s = persisted_best_lap_s
                self.step_rate = RollingStepRate(5.0)
                self.tqc_actions: deque[tuple[float, float]] = deque(maxlen=256)
                self.tqc_probe: dict[str, Any] = {}
                self.last_tqc_probe_step = -1_000
                self.episode_observations: list[np.ndarray] = []
                self.episode_actions: list[np.ndarray] = []

            def _on_training_start(self) -> None:
                self.step_rate.update(self.num_timesteps)
                service.started_at = time.monotonic()

            def _on_step(self) -> bool:
                infos = self.locals.get("infos")
                info = infos[-1] if infos is not None and len(infos) else {}
                rewards = self.locals.get("rewards")
                reward = (
                    float(rewards[-1]) / cfg.reward_scale
                    if rewards is not None and len(rewards) else 0.0
                )
                dones = self.locals.get("dones")
                done = bool(dones[-1]) if dones is not None and len(dones) else False
                if cfg.algorithm == "tqc":
                    collector = self.locals["self"]
                    self.episode_observations.append(collector._last_obs[-1].copy())
                    self.episode_actions.append(self.locals["buffer_actions"][-1].copy())
                self.episode_reward += reward
                for name, value in info.get("reward_terms", {}).items():
                    previous = self.episode_reward_terms.get(name, 0.0)
                    self.episode_reward_terms[name] = previous + float(value)
                self.episode_steps += 1
                track_length = max(1.0, float(info.get("track_length_m", 1.0)))
                progress = float(info.get("route_progress_m", 0.0)) / track_length
                self.max_progress = max(self.max_progress, progress)
                service.max_progress = max(service.max_progress, progress)
                events = tuple(info.get("events", ()))
                elapsed_s = float(info.get("elapsed_s", 0.0))
                simulator_info = info.get("simulator_info", {})
                speed_kmh = float(simulator_info.get("speed_kmh", 0.0))
                service.simulator_ticks += int(info.get("ticks_advanced", cfg.frame_skip))
                actions = self.locals.get("actions")
                action = actions[-1] if actions is not None and len(actions) else None
                if action is None:
                    policy_steering = None
                    policy_throttle = None
                    policy_brake = None
                else:
                    if cfg.algorithm == "tqc":
                        policy_steering = float(action[0])
                        policy_throttle = float(action[1]) > 0
                        policy_brake = float(action[1]) < 0
                        self.tqc_actions.append((float(action[0]), float(action[1])))
                    else:
                        steering_value = int(action[0])
                        policy_steering = (
                            -1.0 + 2.0 * steering_value / (cfg.pwm_levels - 1)
                            if cfg.pwm_enabled else float(steering_value - 1)
                        )
                        policy_throttle = bool(action[1])
                        policy_brake = bool(action[2])
                now = time.monotonic()
                steps_per_second = self.step_rate.update(self.num_timesteps, now)
                if now - self.last_ui_update >= 0.25 or done:
                    replay = getattr(service.model, "replay_buffer", None)
                    training_values = getattr(service.model.logger, "name_to_value", {})
                    action_stats: dict[str, Any] = {}
                    if cfg.algorithm == "tqc" and self.tqc_actions:
                        recent = np.asarray(self.tqc_actions)
                        longitudinal = recent[:, 1]
                        action_stats = {
                            "longitudinal_mean": float(longitudinal.mean()),
                            "longitudinal_std": float(longitudinal.std()),
                            "longitudinal_positive_fraction": float((longitudinal > 0.1).mean()),
                            "longitudinal_near_zero_fraction": float(
                                (abs(longitudinal) <= 0.1).mean()
                            ),
                            "longitudinal_negative_fraction": float((longitudinal < -0.1).mean()),
                            "mean_absolute_steering": float(abs(recent[:, 0]).mean()),
                        }
                        new_obs = self.locals.get("new_obs")
                        if (new_obs is not None and
                                self.num_timesteps - self.last_tqc_probe_step >= 500):
                            self.tqc_probe = tqc_policy_diagnostics(service.model, new_obs[-1])
                            self.last_tqc_probe_step = self.num_timesteps
                        action_stats.update(self.tqc_probe)
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
                            "policy_steering": policy_steering,
                            "actual_steering": info.get("actual_steering"),
                            "expert_steering": info.get("expert_action", {}).get("steer"),
                            "policy_throttle": policy_throttle,
                            "policy_brake": policy_brake,
                            "finishes": self.finishes,
                            "crashes": self.crashes,
                            "best_lap_s": self.best_lap_s,
                            "quarter": info.get("curriculum_quarter"),
                            "simulator_ticks": service.simulator_ticks,
                            "wall_clock_seconds": (
                                service.previous_wall_clock_seconds + now - service.started_at
                            ),
                            "overall_max_progress": service.max_progress,
                            "replay_size": replay.size() if replay is not None else None,
                            "replay_capacity": cfg.tqc.buffer_size if replay is not None else None,
                            "updates": getattr(service.model, "_n_updates", None)
                            if cfg.algorithm == "tqc" else None,
                            "actor_loss": training_values.get("train/actor_loss")
                            if cfg.algorithm == "tqc" else None,
                            "critic_loss": training_values.get("train/critic_loss")
                            if cfg.algorithm == "tqc" else None,
                            "entropy_coefficient": training_values.get("train/ent_coef")
                            if cfg.algorithm == "tqc" else None,
                            **action_stats,
                        }
                    )
                    self.last_ui_update = now
                if done:
                    finished = "finish" in events
                    if finished and cfg.algorithm == "tqc":
                        service.model.remember_successful_trajectory(
                            np.asarray(self.episode_observations),
                            np.asarray(self.episode_actions),
                        )
                        service.status({
                            "type": "successful_trajectory_saved",
                            "steps": len(self.episode_actions),
                            "timesteps": self.num_timesteps,
                        })
                    self.episode_observations.clear()
                    self.episode_actions.clear()
                    crashed = "crash" in events
                    self.finishes += int(finished)
                    if finished and service.first_finish_timestep is None:
                        service.first_finish_timestep = self.num_timesteps
                    self.crashes += int(crashed)
                    service.episodes += 1
                    service.finishes += int(finished)
                    service.crashes += int(crashed)
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
                            "timesteps": self.num_timesteps,
                            "simulator_ticks": service.simulator_ticks,
                            "wall_clock_seconds": (
                                service.previous_wall_clock_seconds + now - service.started_at
                            ),
                            "overall_max_progress": service.max_progress,
                        }
                    )
                    self.episode += 1
                    self.episode_reward = 0.0
                    self.episode_reward_terms = {}
                    self.episode_steps = 0
                    self.max_progress = 0.0
                if cfg.checkpoint_interval and self.num_timesteps % cfg.checkpoint_interval == 0:
                    checkpoint = directory / "checkpoints" / f"step-{self.num_timesteps}"
                    latest = service.save_latest()
                    shutil.copy2(latest, checkpoint.with_suffix(".zip"))
                    shutil.copy2(
                        latest.with_suffix(".metadata.json"),
                        checkpoint.with_suffix(".metadata.json"),
                    )
                    if cfg.algorithm == "tqc":
                        shutil.copy2(
                            latest.with_suffix(".replay.pkl"),
                            checkpoint.with_suffix(".replay.pkl"),
                        )
                    service.status({"type": "checkpoint", "timesteps": self.num_timesteps})
                return not service._stop.is_set() and not (
                    cfg.max_episodes and service.episodes >= cfg.max_episodes
                )

        try:
            if resume:
                resume_path = Path(resume)
                try:
                    resume_metadata = registry.metadata_for_archive(resume_path)
                except FileNotFoundError:
                    if cfg.algorithm != "ppo" or cfg.pwm_enabled:
                        raise IncompatibleModelError(
                            "model has no compatibility metadata; select legacy digital mode "
                            "or add verified metadata before resuming"
                        ) from None
                else:
                    registry.assert_compatible(
                        resume_metadata,
                        track_name=cfg.track_name,
                        action_schema=action_schema(cfg),
                        algorithm=cfg.algorithm,
                        architecture=(cfg.architecture if cfg.algorithm == "ppo"
                                      else cfg.tqc.architecture),
                    )
                    if cfg.algorithm == "tqc" and any(
                        asdict(cfg.tqc).get(key) != value
                        for key, value in resume_metadata.tqc_hyperparameters.items()
                    ):
                        raise IncompatibleModelError(
                            "TQC resume settings differ from saved hyperparameters"
                        )
                    self.simulator_ticks = resume_metadata.simulator_ticks
                    self.episodes = resume_metadata.training_episodes or 0
                    self.finishes = resume_metadata.finishes
                    self.crashes = resume_metadata.crashes
                    self.previous_wall_clock_seconds = resume_metadata.wall_clock_seconds
                self.model = load_model(cfg, resume_path, env, self.device.resolved)
            else:
                self.model = create_model(cfg, env, self.device.resolved)
            configure_model(self.model, cfg, self.device.resolved, service.status)
            parameters = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
            actor_parameters = (
                sum(p.numel() for p in self.model.policy.actor.parameters())
                if cfg.algorithm == "tqc" else None
            )
            self.status(
                {
                    "type": "started",
                    "algorithm": cfg.algorithm,
                    "action_schema": action_schema(cfg),
                    "device": self.device.resolved,
                    "gpu_name": self.device.gpu_name,
                    "parameter_count": parameters,
                    "actor_parameter_count": actor_parameters,
                    "cuda_diagnostics": self.device.diagnostics,
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
