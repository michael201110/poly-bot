"""Standard TQC with a seeded, finite forward biased replay warmup."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch as th
from sb3_contrib import TQC
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import ConstantSchedule

from polybot.algorithms.base import AlgorithmBackend
from polybot.control.actions import ContinuousPwmActionAdapter
from polybot.training.config import ARCHITECTURES, TQCConfig

if TYPE_CHECKING:
    from polybot.training.config import TrainingConfig


class SeededWarmupTQC(TQC):
    """Execute and replay the same continuous action during warmup."""

    def __init__(
        self, *args: Any, warmup_forward_fraction: float = 0.8,
        warmup_steering_std: float = 0.35, **kwargs: Any,
    ) -> None:
        self.warmup_forward_fraction = warmup_forward_fraction
        self.warmup_steering_std = warmup_steering_std
        self._refill_replay_from_policy = False
        self._champion_actor: Any = None
        self._champion_observations: th.Tensor | None = None
        self._champion_action_drift_limit = 0.0
        self._anchor_action_drift: float | None = None
        self._warmup_rng = np.random.default_rng(kwargs.get("seed"))
        self.speed_bias_schedule: list[tuple[float, float, float]] = []
        super().__init__(*args, **kwargs)

    def predict(
        self, observation: np.ndarray | dict[str, np.ndarray], state: Any = None,
        episode_start: np.ndarray | None = None, deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        action, state = super().predict(observation, state, episode_start, deterministic)
        schedule = getattr(self, "speed_bias_schedule", ())
        if not schedule or isinstance(observation, dict):
            return action, state
        progress = np.asarray(observation)[..., 12]
        bias = np.zeros_like(progress, dtype=np.float32)
        for start, end, amount in schedule:
            # Taper each window over 2% of the track so action changes are smooth.
            fade = np.minimum((progress - start) / 0.02, (end - progress) / 0.02)
            bias += amount * np.clip(fade, 0.0, 1.0)
        adjusted = np.array(action, copy=True)
        adjusted[..., 1] = np.clip(adjusted[..., 1] + bias, -1.0, 1.0)
        return adjusted, state

    def _sample_action(
        self, learning_starts: int, action_noise: Any = None, n_envs: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps >= learning_starts:
            return super()._sample_action(learning_starts, action_noise, n_envs)
        if getattr(self, "_refill_replay_from_policy", False):
            return super()._sample_action(0, action_noise, n_envs)
        steering = np.clip(
            self._warmup_rng.normal(0, self.warmup_steering_std, n_envs), -1, 1
        )
        forward = self._warmup_rng.uniform(-1, 1, n_envs)
        forward_mask = self._warmup_rng.random(n_envs) < self.warmup_forward_fraction
        forward[forward_mask] = self._warmup_rng.uniform(0.35, 1, forward_mask.sum())
        action = np.stack((steering, forward), axis=1).astype(np.float32)
        # The action space is [-1, 1], so its replay representation is identical.
        return action, action.copy()

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "_champion_actor", "_champion_observations"]

    def anchor_to_current_policy(
        self, max_action_drift: float, observations: list[np.ndarray] | None = None
    ) -> None:
        """Keep a continued actor near the last proven policy on replay states."""
        self._champion_action_drift_limit = max_action_drift
        self._anchor_action_drift = None
        self._champion_actor = deepcopy(self.actor) if max_action_drift > 0 else None
        self._champion_observations = (
            th.as_tensor(np.asarray(observations, dtype=np.float32), device=self.device)
            if observations else None
        )
        if self._champion_observations is not None and self.replay_buffer is not None:
            count = min(2048, self.replay_buffer.size())
            if count:
                replay_observations = self.replay_buffer.sample(
                    count, env=self._vec_normalize_env
                ).observations
                self._champion_observations = th.cat(
                    (self._champion_observations, replay_observations), dim=0
                )
        if self._champion_actor is not None:
            self._champion_actor.eval()
            for parameter in self._champion_actor.parameters():
                parameter.requires_grad_(False)

    def _enforce_actor_anchor(self) -> None:
        if self._champion_actor is None or self.replay_buffer is None:
            return
        if self._champion_observations is None:
            count = min(2048, self.replay_buffer.size())
            if count == 0:
                return
            self._champion_observations = self.replay_buffer.sample(
                count, env=self._vec_normalize_env
            ).observations
        observations = self._champion_observations
        with th.no_grad():
            reference = self._champion_actor(observations, deterministic=True)
            for _ in range(8):
                current = self.actor(observations, deterministic=True)
                drift = (current - reference).abs().amax().item()
                self._anchor_action_drift = drift
                if drift <= self._champion_action_drift_limit:
                    break
                # Interpolate in weight space and re-check the actual actions;
                # the actor network itself is nonlinear.
                ratio = min(0.95, self._champion_action_drift_limit / drift)
                for parameter, anchor in zip(
                    self.actor.parameters(), self._champion_actor.parameters(), strict=True
                ):
                    parameter.lerp_(anchor, 1 - ratio)
            self._anchor_action_drift = (
                self.actor(observations, deterministic=True) - reference
            ).abs().amax().item()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self._champion_actor is None:
            super().train(gradient_steps, batch_size)
            return
        for _ in range(gradient_steps):
            super().train(1, batch_size)
            self._enforce_actor_anchor()


class TQCBackend(AlgorithmBackend):
    name = "tqc"

    def validate_config(self, config: TrainingConfig) -> None:
        if config.ppo is not None or config.dqn is not None:
            raise ValueError("TQC config cannot contain PPO or DQN settings")
        if config.tqc is None:
            config.tqc = TQCConfig()

    def action_adapter(self, config: TrainingConfig) -> ContinuousPwmActionAdapter:
        return ContinuousPwmActionAdapter()

    def architecture(self, config: TrainingConfig) -> str:
        assert config.tqc is not None
        return config.tqc.architecture

    def create_model(self, config: TrainingConfig, env: Any, device: str) -> Any:
        assert config.tqc is not None
        p = config.tqc
        layers = list(ARCHITECTURES[p.architecture])
        return SeededWarmupTQC(
            "MlpPolicy", env, seed=config.seed, device=device, verbose=0,
            learning_rate=p.learning_rate, buffer_size=p.replay_capacity,
            learning_starts=p.learning_starts, batch_size=p.batch_size,
            gamma=p.gamma, tau=p.tau, train_freq=p.train_frequency,
            gradient_steps=p.gradient_steps, ent_coef=p.entropy,
            warmup_forward_fraction=p.warmup_forward_fraction,
            warmup_steering_std=p.warmup_steering_std,
            policy_kwargs={"net_arch": {"pi": layers, "qf": layers}},
        )

    def load_model(
        self, path: Path, env: Any, device: str, *, resume: bool = False
    ) -> Any:
        model = SeededWarmupTQC.load(str(path), env=env, device=device)
        if resume:
            replay = path.with_name("replay.pkl")
            if not replay.is_file():
                raise FileNotFoundError(f"TQC resume requires replay state: {replay}")
            model.load_replay_buffer(str(replay))
        return model

    def save_model(self, model: Any, directory: Path, *, resume: bool = False) -> None:
        super().save_model(model, directory, resume=resume)
        if resume:
            model.save_replay_buffer(str(directory / "replay.pkl"))

    def configure_resume(
        self, model: Any, config: TrainingConfig, device: str, *, fresh_replay: bool = False
    ) -> None:
        assert config.tqc is not None
        p = config.tqc
        if model.replay_buffer is None:
            raise RuntimeError("TQC resume requires a replay buffer")
        model.learning_rate = p.learning_rate
        model.lr_schedule = ConstantSchedule(p.learning_rate)
        model.train_freq = TrainFreq(p.train_frequency, TrainFrequencyUnit.STEP)
        model.gradient_steps = p.gradient_steps
        model.batch_size = p.batch_size
        model.gamma = p.gamma
        model.tau = p.tau
        if fresh_replay:
            model.learning_starts = model.num_timesteps + max(p.learning_starts, p.batch_size)
            model._refill_replay_from_policy = True

    def parameter_counts(self, model: Any) -> dict[str, int]:
        actor = sum(p.numel() for p in model.actor.parameters() if p.requires_grad)
        critic = sum(p.numel() for p in model.critic.parameters() if p.requires_grad)
        entropy = 1 if getattr(model, "log_ent_coef", None) is not None else 0
        return {"actor": actor, "critic": critic, "total": actor + critic + entropy}

    def metrics(self, model: Any) -> dict[str, float | int | None]:
        values = model.logger.name_to_value
        return {
            "replay_size": model.replay_buffer.size() if model.replay_buffer else 0,
            "updates": model._n_updates,
            "actor_loss": values.get("train/actor_loss"),
            "critic_loss": values.get("train/critic_loss"),
            "entropy_coefficient": values.get("train/ent_coef"),
            "anchor_action_drift": model._anchor_action_drift,
        }
