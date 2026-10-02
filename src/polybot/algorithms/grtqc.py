"""Gated, disagreement-regularized TQC for continuous PolyTrack control.

The quantile target, truncation and entropy objective are inherited from TQC.
Gates modulate each hidden activation; the extra critic term is ensemble variance
at corresponding quantile indices, averaged across the batch and quantiles.
"""

from __future__ import annotations

from collections import deque
from math import log
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch as th
from sb3_contrib.common.utils import quantile_huber_loss
from sb3_contrib.tqc.policies import TQCPolicy
from stable_baselines3.common.buffers import NStepReplayBuffer, ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import polyak_update
from torch import nn

from polybot.algorithms.tqc import SeededWarmupTQC, TQCBackend
from polybot.control.actions import ContinuousActionAdapter
from polybot.environment.observations import extra_size
from polybot.environment.observations import size as observation_size
from polybot.training.config import ARCHITECTURES, GRTQCConfig
from polybot.training.critic_reference import on_policy_returns

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


class ActorPrefixExtractor(BaseFeaturesExtractor):
    """Leave privileged controller state to the critic, preserving source inputs."""

    def __init__(self, observation_space: Any, width: int) -> None:
        super().__init__(observation_space, features_dim=width)
        self.width = width

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return observations[..., :self.width].contiguous()


class ControllerStateLinear(nn.Linear):
    """Add learned PWM state without changing the inherited matrix multiply."""

    def __init__(self, original: nn.Linear) -> None:
        nn.Module.__init__(self)
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.controller_weight = nn.Parameter(original.weight.new_zeros((self.out_features, 4)))

    def forward(self, values: th.Tensor) -> th.Tensor:
        inherited = nn.functional.linear(values[..., :self.in_features].contiguous(), self.weight, self.bias)
        controller = nn.functional.linear(values[..., self.in_features:], self.controller_weight)
        return inherited + controller


def _enable_actor_controller_state(actor: Any, width: int) -> bool:
    first = actor.latent_pi[0]
    if isinstance(first, ControllerStateLinear):
        return False
    if not isinstance(first, nn.Linear) or first.in_features != width:
        raise ValueError("actor controller adapter requires the compatible inherited first layer")
    if int(np.prod(actor.observation_space.shape)) < width + 4:
        raise ValueError("actor controller adapter requires four real controller-state inputs")
    actor.latent_pi[0] = ControllerStateLinear(first)
    actor.features_extractor = ActorPrefixExtractor(actor.observation_space, width + 4)
    return True


def _configure_actor_trainability(actor: Any, adapter_only: bool) -> None:
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(not adapter_only or name.endswith(".controller_weight"))
        parameter.grad = None


class GRTQCPolicy(TQCPolicy):
    """Keep original TQC linear parameter names for direct weight transfer."""

    def __init__(
        self, *args: Any, actor_observation_size: int = 0,
        actor_controller_state: bool = False, controller_adapter_only: bool = False, **kwargs: Any,
    ) -> None:
        self.actor_observation_size = actor_observation_size
        self.actor_controller_state = actor_controller_state
        self.controller_adapter_only = controller_adapter_only
        if actor_controller_state and not actor_observation_size:
            raise ValueError("actor controller adapter requires the inherited observation width")
        if controller_adapter_only and not actor_controller_state:
            raise ValueError("controller-only adaptation requires the actor controller adapter")
        super().__init__(*args, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        parameters = super()._get_constructor_parameters()
        parameters["actor_observation_size"] = self.actor_observation_size
        parameters["actor_controller_state"] = self.actor_controller_state
        parameters["controller_adapter_only"] = self.controller_adapter_only
        return parameters

    def make_actor(self, features_extractor: Any = None) -> Any:
        if self.actor_observation_size:
            features_extractor = ActorPrefixExtractor(self.observation_space, self.actor_observation_size)
        actor = super().make_actor(features_extractor)
        _gate_hidden_layers(actor.latent_pi, scale=2.0)
        if self.actor_controller_state:
            _enable_actor_controller_state(actor, self.actor_observation_size)
        _configure_actor_trainability(actor, self.controller_adapter_only)
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
        exploration_correlation: float = 0.0,
        policy_std_limit: float = 0.0,
        critic_exploration_fraction: float = 1.0,
        actor_step_action_limit: float = 1e-5,
        actor_reference_drift_limit: float = 0.01,
        critic_mc_initialization_updates: int = 0, critic_mc_min_episodes: int = 5,
        critic_reference_error_limit: float = 0.2,
        **kwargs: Any,
    ) -> None:
        self.disagreement_coefficient = disagreement_coefficient
        self.critic_warmup_updates = critic_warmup_updates
        self.critic_readiness_window = critic_readiness_window
        self.critic_readiness_relative_change = critic_readiness_relative_change
        self.exploration_std = exploration_std
        self.critic_collection_std = critic_collection_std
        self.exploration_correlation = exploration_correlation
        self.policy_std_limit = policy_std_limit
        self.critic_exploration_fraction = critic_exploration_fraction
        self._critic_exploration_episodes: np.ndarray | None = None
        self._exploration_noise: np.ndarray | None = None
        self.actor_step_action_limit = actor_step_action_limit
        self.actor_reference_drift_limit = actor_reference_drift_limit
        self.critic_updates_since_transfer = 0
        self.actor_unlocked = False
        self._actor_evaluation_hold = False
        self._critic_loss_history: deque[float] = deque(maxlen=critic_readiness_window)
        self._disagreement_history: deque[float] = deque(maxlen=critic_readiness_window)
        self._actor_reference_observations: th.Tensor | None = None
        self._actor_reference_actions: th.Tensor | None = None
        self.critic_mc_initialization_updates = critic_mc_initialization_updates
        self.critic_mc_min_episodes = critic_mc_min_episodes
        self.critic_reference_error_limit = critic_reference_error_limit
        self.critic_mc_updates_done = 0
        self.invalidate_critic_reference()
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "_critic_reference_observations", "_critic_reference_actions", "_critic_reference_returns",
        ]

    def invalidate_critic_reference(self) -> None:
        self._critic_reference_observations: th.Tensor | None = None
        self._critic_reference_actions: th.Tensor | None = None
        self._critic_reference_returns: th.Tensor | None = None
        self._critic_reference_error: float | None = None
        self._critic_reference_probe_update = -1
        self._critic_reference_retry_update = 0
        self._critic_reference_episodes = 0

    def _initialize_critics_from_returns(self, batch_size: int) -> None:
        """Seed frozen-policy values from real complete episodes, then resume TD."""
        if not self.critic_mc_initialization_updates or self.actor_unlocked:
            return
        if self._critic_reference_observations is None:
            if self._n_updates < self._critic_reference_retry_update:
                return
            self._critic_reference_retry_update = self._n_updates + 200
            previous_air_brake = self._air_brake_active, self._air_brake_base_action
            try:
                samples = on_policy_returns(self.replay_buffer, self, self.gamma)
            except ValueError as error:
                if "no complete episodes" not in str(error):
                    raise
                return
            finally:
                self.policy.set_training_mode(True)
                self._air_brake_active, self._air_brake_base_action = previous_air_brake
            self._critic_reference_episodes = samples.episodes
            if samples.episodes < self.critic_mc_min_episodes:
                return
            self._critic_reference_observations = th.as_tensor(samples.observations, device=self.device)
            self._critic_reference_actions = th.as_tensor(samples.actions, device=self.device)
            self._critic_reference_returns = th.as_tensor(samples.returns, device=self.device)
            self._critic_reference_error = None
            self._critic_reference_probe_update = -1
            self.policy.set_training_mode(True)
        observations, actions, returns = (
            self._critic_reference_observations, self._critic_reference_actions, self._critic_reference_returns,
        )
        remaining = max(0, self.critic_mc_initialization_updates - self.critic_mc_updates_done)
        for _ in range(remaining):
            indices = th.randint(len(observations), (batch_size,), device=self.device)
            values = self.critic(observations[indices], actions[indices])
            loss = quantile_huber_loss(values, returns[indices].unsqueeze(1), sum_over_quantiles=False)
            loss = loss + self.disagreement_coefficient * values.var(dim=1, unbiased=False).mean()
            if not th.isfinite(loss):
                raise RuntimeError("non-finite complete-return critic initialization loss")
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            self.critic_mc_updates_done += 1
        if remaining:
            self._n_updates += remaining
            self.critic_updates_since_transfer += remaining
            self._critic_loss_history.clear()
            self._disagreement_history.clear()
            self._critic_reference_error = None

    def configure_actor_controller_state(self, enabled: bool, adapter_only: bool) -> bool:
        """Upgrade compatible phase-aware checkpoints without discarding critic/replay."""
        existing = bool(getattr(self.policy, "actor_controller_state", False))
        if existing and not enabled:
            raise ValueError("cannot disable controller inputs in an adapted actor")
        changed = False
        if enabled:
            width = self.policy.actor_observation_size
            changed = _enable_actor_controller_state(self.actor, width)
            if changed:
                # One parameter group must match reconstruction on save/load.
                # Start actor moments afresh; critic optimizer and replay stay intact.
                self.actor.optimizer = self.policy.optimizer_class(
                    self.actor.parameters(), lr=self.actor_lr, **self.policy.optimizer_kwargs,
                )
        if adapter_only != getattr(self.policy, "controller_adapter_only", False):
            self.actor.optimizer.state.clear()
        _configure_actor_trainability(self.actor, adapter_only)
        self.policy.actor_controller_state = enabled
        self.policy.controller_adapter_only = adapter_only
        self.policy_kwargs.update(actor_controller_state=enabled, controller_adapter_only=adapter_only)
        return changed

    def set_actor_reference_observations(
        self, observations: np.ndarray, *, reference_model: Any | None = None,
        max_samples: int = 512,
    ) -> None:
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
        reference_actor = reference_model.actor if reference_model is not None else self.actor
        with th.no_grad():
            self._actor_reference_actions = reference_actor(
                self._actor_reference_observations, deterministic=True,
            ).detach().clone()

    def _sample_action(
        self, learning_starts: int, action_noise: Any = None, n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Keep rollout exploration separate from the stochastic critic target
        # and smooth it over time rather than injecting independent PWM jitter.
        action, _ = self.predict(self._last_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(n_envs, -1)
        noise_std = self.exploration_std if self.actor_unlocked else self.critic_collection_std
        if not self.actor_unlocked and self.num_timesteps < learning_starts:
            noise_std = 0.0
        noise = self._rollout_noise(action.shape, noise_std)
        if not self.actor_unlocked and noise_std:
            if self._critic_exploration_episodes is None:
                self._critic_exploration_episodes = (
                    self._warmup_rng.random(n_envs) < self.critic_exploration_fraction
                )
            noise[~self._critic_exploration_episodes] = 0
            self._exploration_noise[~self._critic_exploration_episodes] = 0
        if noise_std:
            action = np.clip(action + noise, -1.0, 1.0).astype(np.float32)
        if self._air_brake_active and self.env is not None:
            observations = np.asarray(self._last_obs).reshape(n_envs, -1)
            base = np.asarray(self._air_brake_base_action, dtype=np.float32).reshape(n_envs, -1)
            for index in range(n_envs):
                airborne = np.all(observations[index, 17:21] < 0.5)
                if airborne and action[index, 1] != base[index, 1] + noise[index, 1]:
                    self.env.env_method(
                        "request_air_brake", np.clip(base[index] + noise[index], -1, 1), indices=index,
                    )
        return action, self.policy.scale_action(action)

    def _rollout_noise(self, shape: tuple[int, int], std: float) -> np.ndarray:
        if self._exploration_noise is None or self._exploration_noise.shape != shape:
            self._exploration_noise = np.zeros(shape, dtype=np.float32)
        if std == 0:
            self._exploration_noise.fill(0.0)
            return self._exploration_noise.copy()
        correlation = self.exploration_correlation
        innovation = self._warmup_rng.normal(0.0, std, shape).astype(np.float32)
        self._exploration_noise = (
            correlation * self._exploration_noise
            + np.sqrt(1.0 - correlation * correlation) * innovation
        ).astype(np.float32)
        return self._exploration_noise.copy()

    def _store_transition(
        self, replay_buffer: Any, buffer_action: np.ndarray, new_obs: Any,
        reward: np.ndarray, dones: np.ndarray, infos: list[dict[str, Any]],
    ) -> None:
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        # OffPolicyAlgorithm does not update _last_episode_starts. VecEnv has
        # already reset each finished environment, so reset its noise here.
        if self._exploration_noise is not None:
            self._exploration_noise[np.asarray(dones, dtype=bool)] = 0.0
        if self._critic_exploration_episodes is not None:
            ended = np.asarray(dones, dtype=bool)
            self._critic_exploration_episodes[ended] = (
                self._warmup_rng.random(int(ended.sum())) < self.critic_exploration_fraction
            )

    def set_env(self, env: Any, force_reset: bool = True) -> None:
        replay = self.replay_buffer
        if force_reset and replay is not None and replay.size():
            # Evaluations and phase switches reset the simulator. Preserve
            # bootstrap at that real observation, but stop multi-step returns
            # before rewards from the next reset episode.
            last = (replay.pos - 1) % replay.buffer_size
            unfinished = replay.dones[last] == 0
            replay.dones[last, unfinished] = 1
            replay.timeouts[last, unfinished] = 1
        self._exploration_noise = None
        self._critic_exploration_episodes = None
        super().set_env(env, force_reset=force_reset)

    def configure_replay_horizon(self, n_steps: int, gamma: float) -> bool:
        """Reuse raw transitions when changing the return horizon on resume.

        Both SB3 buffer classes have identical storage. Reuse their arrays to
        avoid a second large allocation; only sampling behavior changes.
        """
        replay = self.replay_buffer
        if replay is None or type(replay) not in {ReplayBuffer, NStepReplayBuffer}:
            raise ValueError("GRTQC return horizon requires standard continuous replay")
        desired = NStepReplayBuffer if n_steps > 1 else ReplayBuffer
        changed = (
            self.n_steps != n_steps or type(replay) is not desired
            or (n_steps > 1 and (replay.n_steps != n_steps or replay.gamma != gamma))
        )
        if changed:
            order = np.arange(replay.size())
            if replay.full:
                order = np.concatenate((order[replay.pos:], order[:replay.pos]))
            for offset in range(0, len(order) - 1, 1024):
                current = order[offset:min(offset + 1024, len(order) - 1)]
                following = order[offset + 1:min(offset + 1025, len(order))]
                reset = np.any(
                    np.abs(replay.next_observations[current] - replay.observations[following]) > 1e-6,
                    axis=2,
                ) & (replay.dones[current] == 0)
                rows, environments = np.nonzero(reset)
                replay.dones[current[rows], environments] = 1
                replay.timeouts[current[rows], environments] = 1
            converted = desired.__new__(desired)
            converted.__dict__.update(replay.__dict__)
            self.replay_buffer = replay = converted
            self.replay_buffer_class = desired
        self.n_steps = n_steps
        self.replay_buffer_kwargs.pop("n_steps", None)
        self.replay_buffer_kwargs.pop("gamma", None)
        if n_steps > 1:
            replay.n_steps, replay.gamma = n_steps, gamma
            self.replay_buffer_kwargs.update(n_steps=n_steps, gamma=gamma)
        return changed

    def _critic_ready(self) -> bool:
        if self.critic_mc_initialization_updates:
            if (
                self.critic_mc_updates_done < self.critic_mc_initialization_updates
                or self._critic_reference_observations is None
            ):
                return False
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
        if self.critic_mc_initialization_updates:
            if (
                self._critic_reference_error is None
                or self.critic_updates_since_transfer - self._critic_reference_probe_update >= 100
            ):
                indices = th.linspace(
                    0, len(self._critic_reference_observations) - 1, 512, device=self.device,
                ).long()
                with th.no_grad():
                    values = self.critic(
                        self._critic_reference_observations[indices], self._critic_reference_actions[indices],
                    ).mean(dim=(1, 2))[:, None]
                    returns = self._critic_reference_returns[indices]
                    self._critic_reference_error = float(
                        ((values - returns).abs().mean() / returns.abs().mean().clamp_min(1.0)).item()
                    )
                self._critic_reference_probe_update = self.critic_updates_since_transfer
            if (
                not np.isfinite(self._critic_reference_error)
                or self._critic_reference_error > self.critic_reference_error_limit
            ):
                return False
        return True

    def _training_actions_log_prob(self, observations: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Keep critic targets and actor sampling near collected actions.

        Smoothly bound the Gaussian standard deviation without changing the
        deterministic mean or discarding gradients through learned variance.
        The log probability uses this same bounded distribution.
        """
        if self.policy_std_limit == 0:
            return self.actor.action_log_prob(observations)
        mean, log_std, kwargs = self.actor.get_action_dist_params(observations)
        log_std = log_std - th.nn.functional.softplus(log_std - log(self.policy_std_limit))
        self.logger.record("train/policy_training_std_max", float(log_std.detach().exp().max().item()))
        return self.actor.action_dist.log_prob_from_params(mean, log_std, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.replay_buffer is None:
            raise RuntimeError("GRTQC needs replay before training")
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        self._initialize_critics_from_returns(batch_size)
        for gradient_step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = data.discounts if data.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions_pi, log_prob = self._training_actions_log_prob(data.observations)
            actions_pi = self._apply_overlays_to_actions(actions_pi, data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef = (
                th.exp(self.log_ent_coef.detach())
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None
                else self.ent_coef_tensor
            )
            with th.no_grad():
                next_actions, next_log_prob = self._training_actions_log_prob(data.next_observations)
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
            if self.actor_unlocked and not self._actor_evaluation_hold:
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
                    proposed_reference_cumulative_drift = (
                        float((reference_after - self._actor_reference_actions).abs().max().item())
                        if reference_after is not None and self._actor_reference_actions is not None else 0.0
                    )
                    proposed_drift = max(local_proposed_drift, reference_proposed_drift)
                    if (
                        proposed_drift > self.actor_step_action_limit
                        or (
                            self.actor_reference_drift_limit > 0
                            and proposed_reference_cumulative_drift > self.actor_reference_drift_limit
                        )
                    ):
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
                            reference_cumulative_drift = (
                                float((
                                    self.actor(self._actor_reference_observations, deterministic=True)
                                    - self._actor_reference_actions
                                ).abs().max().item())
                                if self._actor_reference_observations is not None
                                and self._actor_reference_actions is not None else 0.0
                            )
                            if (
                                max(local_drift, reference_drift) <= self.actor_step_action_limit
                                and (
                                    self.actor_reference_drift_limit == 0
                                    or reference_cumulative_drift <= self.actor_reference_drift_limit
                                )
                            ):
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
                    executed_reference_cumulative_drift = (
                        float((
                            self.actor(self._actor_reference_observations, deterministic=True)
                            - self._actor_reference_actions
                        ).abs().max().item())
                        if self._actor_reference_observations is not None
                        and self._actor_reference_actions is not None else 0.0
                    )
                    executed_drift = max(executed_local_drift, executed_reference_drift)
                self.logger.record("train/actor_loss", float(actor_loss.item()))
                self.logger.record("train/actor_proposed_action_drift", proposed_drift)
                self.logger.record("train/actor_executed_action_drift", executed_drift)
                self.logger.record("train/actor_reference_action_drift", executed_reference_drift)
                self.logger.record(
                    "train/actor_reference_cumulative_action_drift",
                    executed_reference_cumulative_drift,
                )
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", float(ent_coef.item()))


class GRTQCBackend(TQCBackend):
    name = "grtqc"

    def action_adapter(self, config: TrainingConfig) -> ContinuousActionAdapter:
        assert config.grtqc is not None
        return ContinuousActionAdapter(expose_controller_state=config.grtqc.critic_controller_state)

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
        width = observation_size(config.lookahead_count)
        if int(np.prod(env.observation_space.shape)) != width + extra_size(config):
            raise ValueError("GRTQC environment has an incompatible controller-state observation layout")
        model = GRTQC(
            GRTQCPolicy, env, seed=config.seed, device=device, verbose=0,
            learning_rate=p.learning_rate, buffer_size=p.replay_capacity,
            learning_starts=p.learning_starts, batch_size=p.batch_size,
            gamma=p.gamma, tau=p.tau, train_freq=p.train_frequency,
            gradient_steps=p.gradient_steps, ent_coef=p.entropy,
            n_steps=p.n_step_return,
            target_entropy=p.target_entropy,
            warmup_forward_fraction=p.warmup_forward_fraction,
            warmup_steering_std=p.warmup_steering_std,
            disagreement_coefficient=p.disagreement_coefficient,
            critic_warmup_updates=p.critic_warmup_updates,
            critic_readiness_window=p.critic_readiness_window,
            critic_readiness_relative_change=p.critic_readiness_relative_change,
            exploration_std=p.exploration_std,
            critic_collection_std=p.critic_collection_std,
            exploration_correlation=p.exploration_correlation,
            policy_std_limit=p.policy_std_limit,
            critic_exploration_fraction=p.critic_exploration_fraction,
            actor_step_action_limit=p.actor_step_action_limit,
            actor_reference_drift_limit=p.actor_reference_drift_limit,
            critic_mc_initialization_updates=p.critic_mc_initialization_updates,
            critic_mc_min_episodes=p.critic_mc_min_episodes,
            critic_reference_error_limit=p.critic_reference_error_limit,
            policy_kwargs={
                "net_arch": {"pi": layers, "qf": layers},
                "actor_observation_size": width if p.critic_controller_state else 0,
                "actor_controller_state": p.actor_controller_state,
                "controller_adapter_only": p.controller_adapter_only,
            },
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
        width = observation_size(config.lookahead_count)
        expected_width = width + extra_size(config)
        if int(np.prod(model.observation_space.shape)) != expected_width:
            raise ValueError("GRTQC resume has an incompatible controller-state observation layout")
        adapter_added = model.configure_actor_controller_state(
            config.grtqc.actor_controller_state, config.grtqc.controller_adapter_only,
        )
        target_entropy = (
            -float(np.prod(model.action_space.shape))
            if config.grtqc.target_entropy == "auto" else float(config.grtqc.target_entropy)
        )
        targets_changed = (
            model.gamma != config.grtqc.gamma
            or model.policy_std_limit != config.grtqc.policy_std_limit
            or model.target_entropy != target_entropy
        )
        original = config.tqc
        config.tqc = config.grtqc
        try:
            super().configure_resume(model, config, device, fresh_replay=fresh_replay)
        finally:
            config.tqc = original
        horizon_changed = model.configure_replay_horizon(config.grtqc.n_step_return, config.grtqc.gamma)
        initialization_pending = model.critic_mc_updates_done < config.grtqc.critic_mc_initialization_updates
        if horizon_changed or fresh_replay or targets_changed or adapter_added or initialization_pending:
            # Changed targets or recollected reward data must adapt the critics
            # before they can guide another update to the saved driving skill.
            model.actor_unlocked = False
            model.critic_updates_since_transfer = 0
            model._critic_loss_history.clear()
            model._disagreement_history.clear()
            if horizon_changed or fresh_replay or targets_changed or adapter_added:
                model.critic_mc_updates_done = 0
            model.invalidate_critic_reference()
        model.critic_mc_initialization_updates = config.grtqc.critic_mc_initialization_updates
        model.critic_mc_min_episodes = config.grtqc.critic_mc_min_episodes
        model.critic_reference_error_limit = config.grtqc.critic_reference_error_limit
        model.disagreement_coefficient = config.grtqc.disagreement_coefficient
        model.critic_warmup_updates = config.grtqc.critic_warmup_updates
        model.critic_readiness_window = config.grtqc.critic_readiness_window
        model.critic_readiness_relative_change = config.grtqc.critic_readiness_relative_change
        model.exploration_std = config.grtqc.exploration_std
        model.critic_collection_std = config.grtqc.critic_collection_std
        model.exploration_correlation = config.grtqc.exploration_correlation
        model.policy_std_limit = config.grtqc.policy_std_limit
        model.critic_exploration_fraction = config.grtqc.critic_exploration_fraction
        model._critic_exploration_episodes = None
        if model.ent_coef != config.grtqc.entropy:
            # An explicit changed initialization must take effect on resume;
            # otherwise preserve the coefficient learned by an unchanged run.
            initial = float(config.grtqc.entropy[5:]) if config.grtqc.entropy.startswith("auto_") else 1.0
            with th.no_grad():
                model.log_ent_coef.fill_(log(initial))
            if model.ent_coef_optimizer is not None:
                model.ent_coef_optimizer.state.clear()
            model.ent_coef = config.grtqc.entropy
        model.target_entropy = target_entropy
        model._exploration_noise = None
        model._actor_evaluation_hold = False
        model.actor_step_action_limit = config.grtqc.actor_step_action_limit
        model.actor_reference_drift_limit = config.grtqc.actor_reference_drift_limit

    @staticmethod
    def restore_actor_weights(model: GRTQC, verified: GRTQC) -> None:
        """A legacy phase-aware fallback initializes only the new adapter to zero."""
        incoming = verified.actor.state_dict()
        destination = set(model.actor.state_dict())
        missing = destination - set(incoming)
        if missing - {"latent_pi.0.controller_weight"}:
            raise ValueError("verified actor has incompatible inherited weights")
        if set(incoming) - destination:
            raise ValueError("verified actor has incompatible controller inputs")
        result = model.actor.load_state_dict(incoming, strict=False)
        if result.unexpected_keys:
            raise ValueError("verified actor has incompatible controller inputs")
        if missing:
            with th.no_grad():
                model.actor.latent_pi[0].controller_weight.zero_()

    def parameter_counts(self, model: GRTQC) -> dict[str, int]:
        counts = super().parameter_counts(model)
        counts["actor_trainable"] = counts["actor"]
        counts["actor"] = sum(parameter.numel() for parameter in model.actor.parameters())
        return counts

    def metrics(self, model: GRTQC) -> dict[str, float | int | None]:
        metrics = super().metrics(model)
        values = model.logger.name_to_value
        metrics.update({
            "critic_disagreement": values.get("train/disagreement"),
            "disagreement_penalty": values.get("train/disagreement_penalty"),
            "quantile_mean": values.get("train/quantile_mean"),
            "target_mean": values.get("train/target_mean"),
            "critic_warmup_updates": model.critic_updates_since_transfer,
            "policy_training_std_max": values.get("train/policy_training_std_max"),
            "actor_unlocked": int(model.actor_unlocked),
            "actor_evaluation_pending": int(model._actor_evaluation_hold),
            "critic_mc_updates": model.critic_mc_updates_done,
            "critic_mc_initialization_updates": model.critic_mc_initialization_updates,
            "critic_reference_episodes": model._critic_reference_episodes,
            "critic_reference_relative_error": model._critic_reference_error,
            "actor_controller_state": int(getattr(model.policy, "actor_controller_state", False)),
            "controller_adapter_only": int(getattr(model.policy, "controller_adapter_only", False)),
            "actor_total_parameters": sum(parameter.numel() for parameter in model.actor.parameters()),
            "actor_trainable_parameters": sum(
                parameter.numel() for parameter in model.actor.parameters() if parameter.requires_grad
            ),
            "actor_proposed_action_drift": values.get("train/actor_proposed_action_drift"),
            "actor_executed_action_drift": values.get("train/actor_executed_action_drift"),
            "actor_reference_action_drift": values.get("train/actor_reference_action_drift"),
            "actor_reference_cumulative_action_drift": values.get(
                "train/actor_reference_cumulative_action_drift"
            ),
        })
        return metrics
