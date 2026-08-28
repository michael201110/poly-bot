"""PPO with an optional fixed teacher-policy KL anchor."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


def policy_logits(policy: Any, observations: th.Tensor) -> th.Tensor:
    """Return the concatenated MultiDiscrete action logits for a policy."""

    features = policy.extract_features(observations)
    latent_policy = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_policy)


def categorical_teacher_kl(
    student_logits: th.Tensor,
    teacher_logits: th.Tensor,
    action_widths: tuple[int, ...],
) -> th.Tensor:
    """Mean KL(teacher || student) across MultiDiscrete action heads."""

    losses = []
    offset = 0
    for width in action_widths:
        student = student_logits[:, offset : offset + width]
        teacher = teacher_logits[:, offset : offset + width]
        losses.append(
            F.kl_div(
                F.log_softmax(student, dim=1),
                F.softmax(teacher, dim=1),
                reduction="batchmean",
            )
        )
        offset += width
    return th.stack(losses).mean()


def expert_action_loss(
    logits: th.Tensor,
    observations: th.Tensor,
    action_widths: tuple[int, ...],
    *,
    expert_action_offset: int = 39,
) -> th.Tensor:
    """Cross-entropy loss for the protocol-v2 ghost controls in observations."""

    if len(action_widths) != 3:
        raise ValueError("ghost action imitation requires steer, throttle, and brake heads")
    expert = observations[:, expert_action_offset : expert_action_offset + 3]
    if expert.shape[1] != 3:
        raise ValueError("observations do not contain protocol-v2 expert actions")
    steer_width = action_widths[0]
    targets = (
        ((expert[:, 0].clamp(-1, 1) + 1) * (steer_width - 1) / 2).round().long(),
        expert[:, 1].round().clamp(0, action_widths[1] - 1).long(),
        expert[:, 2].round().clamp(0, action_widths[2] - 1).long(),
    )
    losses = []
    offset = 0
    for width, target in zip(action_widths, targets, strict=True):
        losses.append(F.cross_entropy(logits[:, offset : offset + width], target))
        offset += width
    return th.stack(losses).mean()


class TeacherAnchoredPPO(PPO):
    """PPO whose actor is softly constrained to a frozen teacher policy."""

    teacher_policy: Any | None = None
    teacher_kl_coefficient: float = 0.0
    expert_imitation_coefficient: float = 0.0

    def set_teacher(self, teacher: PPO | None, coefficient: float = 0.0) -> None:
        if coefficient < 0:
            raise ValueError("teacher KL coefficient cannot be negative")
        self.teacher_policy = None if teacher is None else teacher.policy
        self.teacher_kl_coefficient = coefficient
        if self.teacher_policy is not None:
            self.teacher_policy.set_training_mode(False)
            for parameter in self.teacher_policy.parameters():
                parameter.requires_grad_(False)

    def set_expert_imitation(self, coefficient: float = 0.0) -> None:
        if coefficient < 0:
            raise ValueError("expert imitation coefficient cannot be negative")
        if coefficient and not isinstance(self.action_space, spaces.MultiDiscrete):
            raise TypeError("expert imitation requires a MultiDiscrete action space")
        self.expert_imitation_coefficient = coefficient

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "teacher_policy"]

    def _teacher_kl(self, observations: th.Tensor) -> th.Tensor:
        if self.teacher_policy is None or self.teacher_kl_coefficient <= 0:
            return th.zeros((), device=observations.device)
        if not isinstance(self.action_space, spaces.MultiDiscrete):
            raise TypeError("teacher anchoring currently requires a MultiDiscrete action space")
        with th.no_grad():
            teacher_logits = policy_logits(self.teacher_policy, observations)
        student_logits = policy_logits(self.policy, observations)
        widths = tuple(int(width) for width in self.action_space.nvec)
        return categorical_teacher_kl(student_logits, teacher_logits, widths)

    def train(self) -> None:
        """Run the normal PPO update with teacher KL included in every minibatch."""

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        teacher_kls: list[float] = []
        expert_losses: list[float] = []
        approx_kl_divs: list[float] = []
        continue_training = True
        loss = th.zeros((), device=self.device)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(
                    th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                entropy_loss = (
                    -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                )
                entropy_losses.append(entropy_loss.item())
                teacher_kl = self._teacher_kl(rollout_data.observations)
                teacher_kls.append(teacher_kl.item())
                expert_loss = th.zeros((), device=self.device)
                if self.expert_imitation_coefficient > 0:
                    student_logits = policy_logits(self.policy, rollout_data.observations)
                    widths = tuple(int(width) for width in self.action_space.nvec)
                    expert_loss = expert_action_loss(
                        student_logits, rollout_data.observations, widths
                    )
                expert_losses.append(expert_loss.item())
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.teacher_kl_coefficient * teacher_kl
                    + self.expert_imitation_coefficient * expert_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(float(approx_kl_div))
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/teacher_kl", np.mean(teacher_kls))
        self.logger.record("train/expert_action_loss", np.mean(expert_losses))
        self.logger.record("train/loss", loss.item())
        self.logger.record(
            "train/explained_variance",
            explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()),
        )
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
