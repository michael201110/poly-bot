"""Continuous-control PPO backend with optional fixed PPO teacher."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch as th

from polybot.algorithms.base import AlgorithmBackend
from polybot.algorithms.ppo_initialization import apply_forward_bias
from polybot.algorithms.ppo_teacher import TeacherAnchoredPPO
from polybot.control.actions import ContinuousActionAdapter
from polybot.training.config import ARCHITECTURES, PPOConfig

if TYPE_CHECKING:
    from polybot.training.config import TrainingConfig


class PPOBackend(AlgorithmBackend):
    name = "ppo"

    def validate_config(self, config: TrainingConfig) -> None:
        if config.tqc is not None or config.dqn is not None:
            raise ValueError("PPO config cannot contain DQN or TQC settings")
        if config.ppo is None:
            config.ppo = PPOConfig()

    def action_adapter(self, config: TrainingConfig) -> ContinuousActionAdapter:
        return ContinuousActionAdapter()

    def architecture(self, config: TrainingConfig) -> str:
        assert config.ppo is not None
        return config.ppo.architecture

    def create_model(self, config: TrainingConfig, env: Any, device: str) -> Any:
        assert config.ppo is not None
        p = config.ppo
        layers = list(ARCHITECTURES[p.architecture])
        model = TeacherAnchoredPPO(
            "MlpPolicy", env, seed=config.seed, device=device, verbose=0,
            learning_rate=p.learning_rate, gamma=p.gamma, gae_lambda=p.gae_lambda,
            ent_coef=p.entropy_coefficient, n_steps=p.rollout_steps,
            batch_size=p.batch_size, n_epochs=p.epochs,
            target_kl=p.target_kl,
            policy_kwargs={"net_arch": {"pi": layers, "vf": layers}},
        )
        apply_forward_bias(
            model, p.initial_forward_bias,
            steering_strength=p.initial_steering_bias,
        )
        self._configure_action_std(model, p.action_std)
        if p.teacher_model:
            teacher = TeacherAnchoredPPO.load(p.teacher_model, device=device)
            self._configure_action_std(teacher, p.action_std)
            model.set_teacher(teacher, p.teacher_kl_coefficient)
        model.set_expert_imitation(p.imitation_coefficient)
        return model

    def load_model(
        self, path: Path, env: Any, device: str, *, resume: bool = False
    ) -> Any:
        return TeacherAnchoredPPO.load(str(path), env=env, device=device)

    def configure_resume(
        self, model: Any, config: TrainingConfig, device: str, *, fresh_replay: bool = False
    ) -> None:
        del fresh_replay
        assert config.ppo is not None
        p = config.ppo
        if model.n_steps != p.rollout_steps:
            raise ValueError(
                "PPO rollout steps cannot change while resuming; start a fresh model "
                "to change the rollout buffer size"
            )
        model.learning_rate = p.learning_rate
        model._setup_lr_schedule()
        for group in model.policy.optimizer.param_groups:
            group["lr"] = p.learning_rate
        model.ent_coef = p.entropy_coefficient
        model.n_epochs = p.epochs
        model.batch_size = p.batch_size
        model.gamma = p.gamma
        model.gae_lambda = p.gae_lambda
        model.target_kl = p.target_kl
        self._configure_action_std(model, p.action_std)
        if p.teacher_model:
            teacher = TeacherAnchoredPPO.load(p.teacher_model, device=device)
            self._configure_action_std(teacher, p.action_std)
            model.set_teacher(teacher, p.teacher_kl_coefficient)
        else:
            model.set_teacher(None, 0.0)
        model.set_expert_imitation(p.imitation_coefficient)

    @staticmethod
    def _configure_action_std(model: Any, action_std: float | None) -> None:
        if action_std is None:
            return
        with th.no_grad():
            model.policy.log_std.fill_(math.log(action_std))
        model.policy.log_std.requires_grad_(False)

    def parameter_counts(self, model: Any) -> dict[str, int]:
        total = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
        actor = sum(
            p.numel() for module in (
                model.policy.mlp_extractor.policy_net, model.policy.action_net
            ) for p in module.parameters() if p.requires_grad
        )
        critic = sum(
            p.numel() for module in (
                model.policy.mlp_extractor.value_net, model.policy.value_net
            ) for p in module.parameters() if p.requires_grad
        )
        return {"actor": actor, "critic": critic, "total": total}

    def metrics(self, model: Any) -> dict[str, float | int | None]:
        values = model.logger.name_to_value
        return {
            "policy_loss": values.get("train/policy_gradient_loss"),
            "value_loss": values.get("train/value_loss"),
            "entropy": values.get("train/entropy_loss"),
            "explained_variance": values.get("train/explained_variance"),
            "kl": values.get("train/approx_kl"),
            "teacher_kl": values.get("train/teacher_kl"),
            "clip_fraction": values.get("train/clip_fraction"),
        }
