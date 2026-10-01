"""Gated, disagreement-regularized TQC for continuous PolyTrack control.

The quantile target, truncation and entropy objective are inherited from TQC.
Gates modulate each hidden activation; the extra critic term is ensemble variance
at corresponding quantile indices, averaged across the batch and quantiles.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch as th
from sb3_contrib.common.utils import quantile_huber_loss
from sb3_contrib.tqc.policies import TQCPolicy
from stable_baselines3.common.utils import polyak_update
from torch import nn

from polybot.algorithms.tqc import SeededWarmupTQC, TQCBackend
from polybot.training.config import ARCHITECTURES, GRTQCConfig

if TYPE_CHECKING:
    from polybot.training.config import TrainingConfig


class GatedReLU(nn.Module):
    """ReLU features multiplied by an input-dependent sigmoid gate.

    The transferred actor uses 2*sigmoid with zero parameters: exactly one at
    initialization, but with a useful nonzero derivative. Fresh critics use a
    conventional sigmoid gate initialized to one half.
    """

    def __init__(self, width: int, *, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.gate = nn.Linear(width, width)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, values: th.Tensor) -> th.Tensor:
        return th.relu(values) * (self.scale * th.sigmoid(self.gate(values)))


def _gate_hidden_layers(module: nn.Sequential, *, scale: float) -> None:
    previous: nn.Linear | None = None
    for index, layer in enumerate(module):
        if isinstance(layer, nn.Linear):
            previous = layer
        elif isinstance(layer, nn.ReLU) and previous is not None:
            module[index] = GatedReLU(previous.out_features, scale=scale)


class GRTQCPolicy(TQCPolicy):
    """Keep original TQC linear parameter names for direct weight transfer."""

    def make_actor(self, features_extractor: Any = None) -> Any:
        actor = super().make_actor(features_extractor)
        _gate_hidden_layers(actor.latent_pi, scale=2.0)
        return actor

    def make_critic(self, features_extractor: Any = None) -> Any:
        critic = super().make_critic(features_extractor)
        for network in critic.q_networks:
            _gate_hidden_layers(network, scale=1.0)
        return critic


class GRTQC(SeededWarmupTQC):
    """TQC update with gated networks and critic disagreement regularization."""

    def __init__(
        self, *args: Any, disagreement_coefficient: float = 0.01,
        critic_warmup_updates: int = 10_000, critic_readiness_window: int = 200,
        critic_readiness_relative_change: float = 0.1,
        exploration_std: float = 0.0001, critic_collection_std: float = 0.001,
        actor_step_action_limit: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        self.disagreement_coefficient = disagreement_coefficient
        self.critic_warmup_updates = critic_warmup_updates
        self.critic_readiness_window = critic_readiness_window
        self.critic_readiness_relative_change = critic_readiness_relative_change
        self.exploration_std = exploration_std
        self.critic_collection_std = critic_collection_std
        self.actor_step_action_limit = actor_step_action_limit
        self.critic_updates_since_transfer = 0
        self.actor_unlocked = False
        self._critic_loss_history: deque[float] = deque(maxlen=critic_readiness_window)
        self._disagreement_history: deque[float] = deque(maxlen=critic_readiness_window)
        self._actor_reference_observations: th.Tensor | None = None
        super().__init__(*args, **kwargs)

    def set_actor_reference_observations(self, observations: np.ndarray, *, max_samples: int = 512) -> None:
        """Anchor each actor update at representative states from a full lap."""
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != int(np.prod(self.observation_space.shape)):
            raise ValueError("actor reference observations have an incompatible shape")
        if len(values) == 0:
            raise ValueError("actor reference observations cannot be empty")
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        if len(values) > max_samples:
            indices = np.linspace(0, len(values) - 1, max_samples, dtype=np.int64)
            values = values[indices]
        self._actor_reference_observations = th.as_tensor(
            values, dtype=th.float32, device=self.device,
        )

    def _sample_action(
        self, learning_starts: int, action_noise: Any = None, n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Stay near the verified transferred policy while learning values and
        # searching for pace. Critic/actor gradient paths remain stochastic.
        action, _ = self.predict(self._last_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(n_envs, -1)
        noise_std = self.exploration_std if self.actor_unlocked else self.critic_collection_std
        if noise_std:
            action = np.clip(
                action + self._warmup_rng.normal(0, noise_std, action.shape),
                -1.0, 1.0,
            ).astype(np.float32)
        return action, self.policy.scale_action(action)

    def _critic_ready(self) -> bool:
        if self.critic_updates_since_transfer < self.critic_warmup_updates:
            return False
        if len(self._critic_loss_history) < self.critic_readiness_window:
            return False
        for history in (self._critic_loss_history, self._disagreement_history):
            values = np.asarray(history, dtype=np.float64)
            if not np.isfinite(values).all():
                return False
            half = len(values) // 2
            first, second = values[:half].mean(), values[half:].mean()
            if second > first * (1 + self.critic_readiness_relative_change):
                return False
        return True

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.replay_buffer is None:
            raise RuntimeError("GRTQC needs replay before training")
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        for gradient_step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self.actor.action_log_prob(data.observations)
            actions_pi = self._apply_overlays_to_actions(actions_pi, data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef = (
                th.exp(self.log_ent_coef.detach())
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None
                else self.ent_coef_tensor
            )
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_actions = self._apply_overlays_to_actions(next_actions, data.next_observations)
                next_quantiles = self.critic_target(data.next_observations, next_actions)
                keep = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                next_quantiles = th.sort(next_quantiles.reshape(batch_size, -1), dim=1).values[:, :keep]
                targets = data.rewards + (1 - data.dones) * discounts * (
                    next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                )
                targets = targets.unsqueeze(1)
            current = self.critic(data.observations, data.actions)
            quantile_loss = quantile_huber_loss(current, targets, sum_over_quantiles=False)
            disagreement = current.var(dim=1, unbiased=False).mean()
            critic_loss = quantile_loss + self.disagreement_coefficient * disagreement
            if not th.isfinite(critic_loss):
                raise RuntimeError("non-finite GRTQC critic loss")
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            self.critic_updates_since_transfer += 1
            self._critic_loss_history.append(float(quantile_loss.item()))
            self._disagreement_history.append(float(disagreement.item()))
            if not self.actor_unlocked and self._critic_ready():
                self.actor_unlocked = True
            self.logger.record("train/critic_loss", float(critic_loss.item()))
            self.logger.record("train/quantile_loss", float(quantile_loss.item()))
            self.logger.record("train/disagreement", float(disagreement.item()))
            penalty = self.disagreement_coefficient * disagreement
            self.logger.record("train/disagreement_penalty", float(penalty.item()))
            self.logger.record("train/quantile_mean", float(current.detach().mean().item()))
            self.logger.record("train/quantile_std", float(current.detach().std(unbiased=False).item()))
            self.logger.record("train/target_mean", float(targets.mean().item()))
            self.logger.record("train/target_std", float(targets.std(unbiased=False).item()))
            self.logger.record("train/actor_unlocked", int(self.actor_unlocked))
            self.logger.record("train/critic_warmup_updates", self.critic_updates_since_transfer)
            if self.actor_unlocked:
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    self.ent_coef_optimizer.zero_grad()
                    ent_loss.backward()
                    self.ent_coef_optimizer.step()
                q_pi = self.critic(data.observations, actions_pi).mean(dim=(1, 2), keepdim=False).reshape(-1, 1)
                actor_loss = (ent_coef * log_prob - q_pi).mean()
                with th.no_grad():
                    before_actions = self.actor(data.observations, deterministic=True).detach()
                    reference_before = (
                        self.actor(self._actor_reference_observations, deterministic=True).detach()
                        if self._actor_reference_observations is not None else None
                    )
                    before_parameters = [parameter.detach().clone() for parameter in self.actor.parameters()]
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                with th.no_grad():
                    after_actions = self.actor(data.observations, deterministic=True)
                    reference_after = (
                        self.actor(self._actor_reference_observations, deterministic=True)
                        if self._actor_reference_observations is not None else None
                    )
                    local_proposed_drift = float((after_actions - before_actions).abs().max().item())
                    reference_proposed_drift = (
                        float((reference_after - reference_before).abs().max().item())
                        if reference_after is not None and reference_before is not None else 0.0
                    )
                    proposed_drift = max(local_proposed_drift, reference_proposed_drift)
                    if proposed_drift > self.actor_step_action_limit:
                        proposed_parameters = [parameter.detach().clone() for parameter in self.actor.parameters()]
                        low, high = 0.0, 1.0
                        for _ in range(12):
                            fraction = (low + high) / 2
                            for parameter, previous, proposed in zip(
                                self.actor.parameters(), before_parameters, proposed_parameters, strict=True,
                            ):
                                parameter.copy_(previous + fraction * (proposed - previous))
                            local_drift = float((
                                self.actor(data.observations, deterministic=True) - before_actions
                            ).abs().max().item())
                            reference_drift = (
                                float((
                                    self.actor(self._actor_reference_observations, deterministic=True)
                                    - reference_before
                                ).abs().max().item())
                                if reference_before is not None else 0.0
                            )
                            if max(local_drift, reference_drift) <= self.actor_step_action_limit:
                                low = fraction
                            else:
                                high = fraction
                        for parameter, previous, proposed in zip(
                            self.actor.parameters(), before_parameters, proposed_parameters, strict=True,
                        ):
                            parameter.copy_(previous + low * (proposed - previous))
                    executed_local_drift = float((
                        self.actor(data.observations, deterministic=True) - before_actions
                    ).abs().max().item())
                    executed_reference_drift = (
                        float((
                            self.actor(self._actor_reference_observations, deterministic=True)
                            - reference_before
                        ).abs().max().item())
                        if reference_before is not None else 0.0
                    )
                    executed_drift = max(executed_local_drift, executed_reference_drift)
                self.logger.record("train/actor_loss", float(actor_loss.item()))
                self.logger.record("train/actor_proposed_action_drift", proposed_drift)
                self.logger.record("train/actor_executed_action_drift", executed_drift)
                self.logger.record("train/actor_reference_action_drift", executed_reference_drift)
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(ent_coef.item()))


class GRTQCBackend(TQCBackend):
    name = "grtqc"

    def validate_config(self, config: TrainingConfig) -> None:
        if config.ppo is not None or config.tqc is not None:
            raise ValueError("GRTQC config cannot contain other algorithm settings")
        if config.grtqc is None:
            config.grtqc = GRTQCConfig()

    def architecture(self, config: TrainingConfig) -> str:
        assert config.grtqc is not None
        return config.grtqc.architecture

    def create_model(self, config: TrainingConfig, env: Any, device: str) -> GRTQC:
        assert config.grtqc is not None
        p = config.grtqc
        layers = list(ARCHITECTURES[p.architecture])
        model = GRTQC(
            GRTQCPolicy, env, seed=config.seed, device=device, verbose=0,
            learning_rate=p.learning_rate, buffer_size=p.replay_capacity,
            learning_starts=p.learning_starts, batch_size=p.batch_size,
            gamma=p.gamma, tau=p.tau, train_freq=p.train_frequency,
            gradient_steps=p.gradient_steps, ent_coef=p.entropy,
            warmup_forward_fraction=p.warmup_forward_fraction,
            warmup_steering_std=p.warmup_steering_std,
            disagreement_coefficient=p.disagreement_coefficient,
            critic_warmup_updates=p.critic_warmup_updates,
            critic_readiness_window=p.critic_readiness_window,
            critic_readiness_relative_change=p.critic_readiness_relative_change,
            exploration_std=p.exploration_std,
            critic_collection_std=p.critic_collection_std,
            actor_step_action_limit=p.actor_step_action_limit,
            policy_kwargs={"net_arch": {"pi": layers, "qf": layers}},
        )
        model.actor_lr = p.actor_learning_rate or p.learning_rate
        model.critic_lr = p.critic_learning_rate or p.learning_rate
        return model

    def load_model(self, path: Path, env: Any, device: str, *, resume: bool = False) -> GRTQC:
        model = GRTQC.load(str(path), env=env, device=device)
        if resume:
            replay = path.with_name("replay.pkl")
            if not replay.is_file():
                raise FileNotFoundError(f"GRTQC resume requires replay state: {replay}")
            model.load_replay_buffer(str(replay))
        return model

    def configure_resume(
        self, model: GRTQC, config: TrainingConfig, device: str, *, fresh_replay: bool = False,
    ) -> None:
        assert config.grtqc is not None
        original = config.tqc
        config.tqc = config.grtqc
        try:
            super().configure_resume(model, config, device, fresh_replay=fresh_replay)
        finally:
            config.tqc = original
        model.disagreement_coefficient = config.grtqc.disagreement_coefficient
        model.critic_warmup_updates = config.grtqc.critic_warmup_updates
        model.critic_readiness_window = config.grtqc.critic_readiness_window
        model.critic_readiness_relative_change = config.grtqc.critic_readiness_relative_change
        model.exploration_std = config.grtqc.exploration_std
        model.critic_collection_std = config.grtqc.critic_collection_std
        model.actor_step_action_limit = config.grtqc.actor_step_action_limit

    def metrics(self, model: GRTQC) -> dict[str, float | int | None]:
        metrics = super().metrics(model)
        values = model.logger.name_to_value
        metrics.update({
            "critic_disagreement": values.get("train/disagreement"),
            "disagreement_penalty": values.get("train/disagreement_penalty"),
            "quantile_mean": values.get("train/quantile_mean"),
            "target_mean": values.get("train/target_mean"),
            "critic_warmup_updates": model.critic_updates_since_transfer,
            "actor_unlocked": int(model.actor_unlocked),
            "actor_proposed_action_drift": values.get("train/actor_proposed_action_drift"),
            "actor_executed_action_drift": values.get("train/actor_executed_action_drift"),
            "actor_reference_action_drift": values.get("train/actor_reference_action_drift"),
        })
        return metrics
