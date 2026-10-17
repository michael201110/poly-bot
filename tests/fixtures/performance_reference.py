"""Frozen numerical oracle from 7f815a8, before the performance pass.

Keep the original math/order here so equivalence tests do not compare an
optimized function against itself. Not imported by production code.
"""
import math

import numpy as np
import torch as th
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.utils import polyak_update

from polybot.algorithms.grtqc import CRITIC_MC_UPDATES_PER_TRAIN_CALL


def train_reference(self, gradient_steps: int, batch_size: int = 64) -> None:
    if self.replay_buffer is None:
        raise RuntimeError("GRTQC needs replay before training")
    self.policy.set_training_mode(True)
    optimizers = [self.actor.optimizer, self.critic.optimizer]
    if self.ent_coef_optimizer is not None:
        optimizers.append(self.ent_coef_optimizer)
    self._update_learning_rate(optimizers)
    self._initialize_critics_from_returns(
        batch_size, max_updates=max(1, gradient_steps) * CRITIC_MC_UPDATES_PER_TRAIN_CALL,
    )
    for gradient_step in range(gradient_steps):
        data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        discounts = data.discounts if data.discounts is not None else self.gamma
        if self.use_sde:
            self.actor.reset_noise()
        actor_observations = data.observations
        if self.actor_verified_state_sampling:
            if self._actor_reference_observations is None:
                raise RuntimeError("verified-state actor sampling requires a verified full-lap reference")
            indices = th.randint(len(self._actor_reference_observations), (batch_size,), device=self.device)
            actor_observations = self._actor_reference_observations[indices]
        actions_pi, log_prob = self._training_actions_log_prob(actor_observations)
        actions_pi = self._critic_actions(actions_pi, actor_observations)
        log_prob = log_prob.reshape(-1, 1)
        ent_coef = (
            th.exp(self.log_ent_coef.detach())
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None
            else self.ent_coef_tensor
        )
        with th.no_grad():
            next_actions, next_log_prob = self._training_actions_log_prob(data.next_observations)
            next_actions = self._critic_actions(next_actions, data.next_observations)
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
        self._record("train/critic_gradient_norm", self._gradient_norm(self.critic))
        critic_before = [p.detach().clone() for p in self.critic.parameters()]
        critic_gradients = {name: float(p.grad.detach().norm())
                            for name, p in self.critic.named_parameters() if p.grad is not None}
        self.critic.optimizer.step()
        critic_changes = {name: float((p.detach() - old).norm())
                          for (name, p), old in zip(self.critic.named_parameters(), critic_before, strict=True)}
        self._record("train/critic_layer_gradient_norms", critic_gradients)
        self._record("train/critic_layer_update_norms", critic_changes)
        self._record("train/critic_update_norm", float(np.linalg.norm(list(critic_changes.values()))))
        self.critic_updates_since_transfer += 1
        self.critic_recovery_updates_remaining = max(0, self.critic_recovery_updates_remaining - 1)
        self._critic_loss_history.append(float(quantile_loss.item()))
        self._disagreement_history.append(float(disagreement.item()))
        if not self.actor_unlocked and self._critic_ready():
            self.actor_unlocked = True
        self._record("train/critic_loss", float(critic_loss.item()))
        self._record("train/quantile_loss", float(quantile_loss.item()))
        self._record("train/disagreement", float(disagreement.item()))
        penalty = self.disagreement_coefficient * disagreement
        self._record("train/disagreement_penalty", float(penalty.item()))
        self._record("train/quantile_mean", float(current.detach().mean().item()))
        self._record("train/quantile_std", float(current.detach().std(unbiased=False).item()))
        self._record("train/target_mean", float(targets.mean().item()))
        self._record("train/target_std", float(targets.std(unbiased=False).item()))
        self._record("train/td_absolute_mean", float((
            targets.mean(dim=(1, 2)) - current.detach().mean(dim=(1, 2))
        ).abs().mean()))
        self._record("train/actor_unlocked", int(self.actor_unlocked))
        self._record("train/critic_warmup_updates", self.critic_updates_since_transfer)
        self._record("train/critic_recovery_updates_remaining", self.critic_recovery_updates_remaining)
        if (
            self.actor_unlocked and not self._actor_evaluation_hold
            and (self._n_updates + gradient_step + 1) % self.actor_update_interval == 0
        ):
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad()
                ent_loss.backward()
                self.ent_coef_optimizer.step()
            q_pi, mean_q_pi, q_uncertainty = self._lower_confidence_actor_value(
                self.critic(actor_observations, actions_pi),
                getattr(self, "actor_uncertainty_coefficient", 0.0),
            )
            uncertainty_penalty = getattr(self, "actor_uncertainty_coefficient", 0.0) * q_uncertainty
            actor_loss = (ent_coef * log_prob - q_pi).mean()
            self._record("train/actor_q_mean", float(mean_q_pi.detach().mean().item()))
            self._record("train/actor_q_uncertainty", float(q_uncertainty.detach().mean().item()))
            self._record("train/actor_uncertainty_penalty", float(uncertainty_penalty.detach().mean().item()))
            with th.no_grad():
                before_actions = self.actor(data.observations, deterministic=True).detach()
                reference_before = (
                    self.actor(self._actor_reference_observations, deterministic=True).detach()
                    if self._actor_reference_observations is not None else None
                )
                before_parameters = [parameter.detach().clone() for parameter in self.actor.parameters()]
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self._record("train/actor_gradient_norm", self._gradient_norm(self.actor))
            layer_gradients = {name: float(p.grad.detach().norm())
                               for name, p in self.actor.named_parameters() if p.grad is not None}
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
            changes = {name: float((p.detach() - old).norm())
                       for (name, p), old in zip(self.actor.named_parameters(), before_parameters, strict=True)}
            actor_update_norm = float(np.linalg.norm(list(changes.values())))
            reference_limit_tolerance = max(1e-9, self.actor_reference_drift_limit * 1e-7)
            reference_cap_saturated = (
                self.actor_reference_drift_limit > 0
                and proposed_reference_cumulative_drift > self.actor_reference_drift_limit
                and executed_reference_cumulative_drift >= (
                    self.actor_reference_drift_limit - reference_limit_tolerance
                )
                and actor_update_norm <= 1e-12
                and executed_drift <= 1e-10
            )
            if reference_cap_saturated:
                self._actor_reference_update_saturated = True
            elif actor_update_norm > 1e-12:
                self._actor_reference_update_saturated = False
            self._record("train/actor_update_norm", actor_update_norm)
            self._record("train/actor_reference_update_saturated", int(reference_cap_saturated))
            self._record("train/actor_layer_gradient_norms", layer_gradients)
            self._record("train/actor_layer_update_norms", changes)
            self._record("train/actor_adam_steps", max(
                (float(v["step"]) for v in self.actor.optimizer.state.values() if "step" in v), default=0))
            self._record("train/actor_loss", float(actor_loss.item()))
            self._record("train/actor_proposed_action_drift", proposed_drift)
            self._record("train/actor_executed_action_drift", executed_drift)
            self._record("train/actor_reference_action_drift", executed_reference_drift)
            self._record(
                "train/actor_reference_cumulative_action_drift",
                executed_reference_cumulative_drift,
            )
        if gradient_step % self.target_update_interval == 0:
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
    self._n_updates += gradient_steps
    self._record("train/n_updates", self._n_updates, exclude="tensorboard")
    self._record("train/ent_coef", float(ent_coef.item()))


def observation_reference(self) -> np.ndarray:
    """Return a normalized, track-invariant feature vector for the policy."""

    width = max(self.track_half_width_m, 0.01)
    base = [
        *(np.clip(self.local_velocity_mps, -200.0, 200.0) / 100.0),
        *(np.clip(self.local_acceleration_mps2, -100.0, 100.0) / 50.0),
        *(np.clip(self.angular_velocity_radps, -50.0, 50.0) / 10.0),
        *np.clip(self.up_vector, -1.0, 1.0),
        float(np.clip(self.route_progress_m / self.track_length_m, -1.0, 2.0)),
        float(np.clip(self.lateral_offset_m / width, -5.0, 5.0)),
        float(np.clip(self.heading_error_rad / math.pi, -1.0, 1.0)),
        float(np.clip(self.pitch_rad / math.pi, -1.0, 1.0)),
        float(np.clip(self.roll_rad / math.pi, -1.0, 1.0)),
        *(np.asarray(self.wheel_contacts, dtype=np.float64)),
        *(np.clip(self.suspension_lengths_m, -2.0, 2.0) / 2.0),
        *(np.clip(self.suspension_velocities_mps, -20.0, 20.0) / 10.0),
        *(np.clip(self.wheel_skids, -100.0, 100.0) / 50.0),
        float(np.clip(self.actual_steering, -1.0, 1.0)),
        float(np.clip(self.ghost_relative_position_m[0] / 50.0, -5.0, 5.0)),
        float(np.clip(self.ghost_relative_position_m[1] / 50.0, -5.0, 5.0)),
        float(np.clip(self.ghost_relative_position_m[2] / 100.0, -5.0, 5.0)),
        float(np.clip(self.ghost_heading_error_rad / math.pi, -1.0, 1.0)),
        float(np.clip(self.ghost_target_speed_mps / 100.0, 0.0, 2.0)),
        float(self.expert_action.steer),
        float(self.expert_action.throttle),
        float(self.expert_action.brake),
        float(self.previous_action.steer),
        float(self.previous_action.throttle),
        float(self.previous_action.brake),
    ]
    future: list[float] = []
    for forward_m, right_m, up_m, curvature_inv_m in self.lookahead:
        future.extend(
            (
                float(np.clip(forward_m / 100.0, -5.0, 5.0)),
                float(np.clip(right_m / 50.0, -5.0, 5.0)),
                float(np.clip(up_m / 50.0, -5.0, 5.0)),
                float(np.clip(curvature_inv_m / 0.2, -5.0, 5.0)),
            )
        )
    vector = np.asarray([*base, *future, *self.lookahead_mask], dtype=np.float32)
    expected = self.vector_size(len(self.lookahead))
    if vector.shape != (expected,):  # Defensive check for schema edits.
        raise AssertionError(
            f"feature schema produced {vector.size} values, expected {expected}"
        )
    return vector


