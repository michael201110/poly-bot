"""Standard TQC with a seeded, finite forward biased replay warmup."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch as th
from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import ConstantSchedule, polyak_update

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
        self._adaptation_mode: str | None = None
        self._adaptation_noise = np.zeros(2, dtype=np.float32)
        self._adaptation_noise_probability = 1.0
        self._adaptation_sample_steps = 0
        self._adaptation_action_deviation: list[float] = []
        self.actor_lr = float(kwargs.get("learning_rate", 3e-4))
        self.critic_lr = float(kwargs.get("learning_rate", 3e-4))
        self._adaptation_diagnostics: dict[str, float] = {}
        self.speed_bias_schedule: list[tuple[float, float, float]] = []
        self.policy_overlays: list[dict[str, Any]] = []
        self._air_brake_active = False
        self._air_brake_base_action: np.ndarray | None = None
        super().__init__(*args, **kwargs)

    def _update_learning_rate(self, optimizers: Any) -> None:
        """Keep SB3 schedule updates separate for TQC's actor and critic."""
        values = []
        for optimizer in optimizers if isinstance(optimizers, (list, tuple)) else [optimizers]:
            if optimizer is self.actor.optimizer:
                rate = self.actor_lr
            elif optimizer is self.critic.optimizer:
                rate = self.critic_lr
            else:
                rate = self.lr_schedule(self._current_progress_remaining)
            for group in optimizer.param_groups:
                group["lr"] = rate
            values.append(rate)
        self.logger.record("train/learning_rate", max(values, default=0.0))

    def predict(
        self, observation: np.ndarray | dict[str, np.ndarray], state: Any = None,
        episode_start: np.ndarray | None = None, deterministic: bool = False,
    ) -> tuple[np.ndarray, Any]:
        action, state = super().predict(observation, state, episode_start, deterministic)
        self._air_brake_active = False
        schedule = getattr(self, "speed_bias_schedule", ())
        overlays = getattr(self, "policy_overlays", ())
        if isinstance(observation, dict):
            return action, state
        state_vector = np.asarray(observation)
        progress = state_vector[..., 12]
        bias = np.zeros_like(progress, dtype=np.float32)
        for start, end, amount in schedule:
            # Taper each window over 2% of the track so action changes are smooth.
            fade = np.minimum((progress - start) / 0.02, (end - progress) / 0.02)
            bias += amount * np.clip(fade, 0.0, 1.0)
        adjusted = np.array(action, copy=True)
        adjusted[..., 1] = np.clip(adjusted[..., 1] + bias, -1.0, 1.0)
        for layer in overlays:
            start, end = float(layer["start"]), float(layer["end"])
            taper = float(layer.get("taper", 0.01))
            width = max(end - start, 1e-9)
            taper = min(taper, width / 2)
            x = np.clip((progress - start) / max(taper, 1e-9), 0.0, 1.0)
            enter = x * x * (3.0 - 2.0 * x)
            x = np.clip((end - progress) / max(taper, 1e-9), 0.0, 1.0)
            leave = x * x * (3.0 - 2.0 * x)
            fade = np.minimum(enter, leave).astype(np.float32)
            kind = layer["kind"]
            amount = float(layer.get("amount", 0.0))
            if kind == "steer_bias":
                adjusted[..., 0] += amount * fade
            elif kind == "steer_gain":
                adjusted[..., 0] *= 1.0 + (amount - 1.0) * fade
            elif kind == "drive_bias":
                adjusted[..., 1] += amount * fade
            elif kind == "drive_gain":
                adjusted[..., 1] *= 1.0 + (amount - 1.0) * fade
            elif kind == "air_brake":
                contacts = state_vector[..., 17:21]
                airborne = np.all(contacts < 0.5, axis=-1)
                if np.any(airborne & (fade > 0)):
                    self._air_brake_active = True
                    self._air_brake_base_action = np.array(adjusted, copy=True)
                    adjusted[..., 1] = np.where(
                        airborne, -abs(float(layer.get("duty", 0.0))) * fade, adjusted[..., 1]
                    )
        adjusted = np.clip(adjusted, -1.0, 1.0)
        return adjusted, state

    def _sample_action(
        self, learning_starts: int, action_noise: Any = None, n_envs: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._adaptation_mode == "replay_expansion":
            self._adaptation_sample_steps += 1
            action, _ = self.predict(self._last_obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(n_envs, -1)
            interval = max(1, round(1.0 / self._adaptation_noise_probability))
            mask = self._warmup_rng.random(n_envs) < self._adaptation_noise_probability
            if self._adaptation_sample_steps % interval == 0:
                mask[0] = True
            noise = self._warmup_rng.normal(0.0, self._adaptation_noise, action.shape)
            noisy = action + noise * mask[:, None]
            noisy = np.clip(noisy, -1.0, 1.0).astype(np.float32)
            self._adaptation_action_deviation.extend(np.abs(noisy - action).mean(axis=1).tolist())
            return noisy, self.policy.scale_action(noisy)
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
        if self._adaptation_mode == "replay_expansion":
            return
        if self._adaptation_mode == "critic_only":
            self.train_critics(gradient_steps, batch_size)
            return
        if self._champion_actor is None:
            super().train(gradient_steps, batch_size)
            return
        for _ in range(gradient_steps):
            super().train(1, batch_size)
            self._enforce_actor_anchor()

    def train_critics(self, gradient_steps: int, batch_size: int) -> dict[str, float]:
        """Update only critic and target critic from replay; actor and entropy stay frozen."""
        if self.replay_buffer is None or self.replay_buffer.size() < batch_size:
            raise RuntimeError("critic adaptation requires at least one batch of replay")
        self.actor.set_training_mode(False)
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        losses: list[float] = []
        targets: list[float] = []
        target_stds: list[float] = []
        values: list[float] = []
        value_stds: list[float] = []
        disagreements: list[float] = []
        nearby_q_values: list[float] = []
        with th.no_grad():
            alpha = (
                self.log_ent_coef.detach().exp()
                if self.log_ent_coef is not None else self.ent_coef_tensor.detach().clone()
            )
        for step in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = batch.discounts if batch.discounts is not None else self.gamma
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(batch.next_observations)
                next_quantiles = self.critic_target(batch.next_observations, next_actions)
                quantiles_per_sample = self.critic.quantiles_total - (
                    self.top_quantiles_to_drop_per_net * self.critic.n_critics
                )
                next_quantiles = th.sort(next_quantiles.reshape(batch_size, -1), dim=1).values
                next_quantiles = next_quantiles[:, :quantiles_per_sample]
                target = batch.rewards + (1 - batch.dones) * discounts * (
                    next_quantiles - alpha * next_log_prob.reshape(-1, 1)
                )
                target = target.unsqueeze(1)
                target_stds.append(float(target.std(unbiased=False).cpu()))
            current = self.critic(batch.observations, batch.actions)
            loss = quantile_huber_loss(current, target, sum_over_quantiles=False)
            if not th.isfinite(loss):
                raise RuntimeError("non-finite TQC critic loss during critic adaptation")
            self.critic.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.critic.optimizer.step()
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
            losses.append(float(loss.detach().cpu()))
            targets.append(float(target.mean().cpu()))
            values.append(float(current.mean().detach().cpu()))
            value_stds.append(float(current.std(unbiased=False).detach().cpu()))
            per_critic = current.detach().mean(dim=2)
            disagreements.append(float(per_critic.std(dim=1, unbiased=False).mean().cpu()))
            with th.no_grad():
                nearby = th.clamp(batch.actions + th.randn_like(batch.actions) * 0.01, -1.0, 1.0)
                nearby_q_values.append(float(self.critic(batch.observations, nearby).mean().cpu()))
        self._n_updates += gradient_steps
        self._adaptation_diagnostics = {
            "critic_loss": float(np.mean(losses)),
            "target_q_mean": float(np.mean(targets)),
            "target_q_std": float(np.mean(target_stds)),
            "q_mean": float(np.mean(values)),
            "q_std": float(np.mean(value_stds)),
            "q_perturbed_action_mean": float(np.mean(nearby_q_values)),
            "critic_disagreement": float(np.mean(disagreements)),
        }
        if not all(np.isfinite(value) for value in self._adaptation_diagnostics.values()):
            raise RuntimeError("non-finite critic-adaptation diagnostics")
        if hasattr(self, "_logger"):
            for key, value in self._adaptation_diagnostics.items():
                self.logger.record(f"adaptation/{key}", value)
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        return self._adaptation_diagnostics


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
        model = SeededWarmupTQC(
            "MlpPolicy", env, seed=config.seed, device=device, verbose=0,
            learning_rate=p.learning_rate, buffer_size=p.replay_capacity,
            learning_starts=p.learning_starts, batch_size=p.batch_size,
            gamma=p.gamma, tau=p.tau, train_freq=p.train_frequency,
            gradient_steps=p.gradient_steps, ent_coef=p.entropy,
            warmup_forward_fraction=p.warmup_forward_fraction,
            warmup_steering_std=p.warmup_steering_std,
            policy_kwargs={"net_arch": {"pi": layers, "qf": layers}},
        )
        model.actor_lr = p.actor_learning_rate or p.learning_rate
        model.critic_lr = p.critic_learning_rate or p.learning_rate
        for group in model.actor.optimizer.param_groups:
            group["lr"] = model.actor_lr
        for group in model.critic.optimizer.param_groups:
            group["lr"] = model.critic_lr
        return model

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
        model.actor_lr = p.actor_learning_rate or p.learning_rate
        model.critic_lr = p.critic_learning_rate or p.learning_rate
        model.lr_schedule = ConstantSchedule(p.learning_rate)
        # Loading restores the optimizer state, including its old param-group LR.
        # Apply the new rate before the first resumed gradient step as well as
        # through the schedule used by subsequent SB3 updates.
        for optimizer, rate in (
            (model.actor.optimizer, model.actor_lr),
            (model.critic.optimizer, model.critic_lr),
            (getattr(model, "ent_coef_optimizer", None), p.learning_rate),
        ):
            if optimizer is not None:
                for group in optimizer.param_groups:
                    group["lr"] = rate
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
