"""PPO backend with discrete PWM actions and optional fixed teacher."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from polybot.algorithms.base import AlgorithmBackend
from polybot.algorithms.ppo_initialization import apply_forward_bias
from polybot.algorithms.ppo_teacher import TeacherAnchoredPPO
from polybot.control.actions import DiscretePwmActionAdapter
from polybot.training.config import ARCHITECTURES, PPOConfig

if TYPE_CHECKING:
    from polybot.training.config import TrainingConfig


class PPOBackend(AlgorithmBackend):
    name = "ppo"

    def validate_config(self, config: TrainingConfig) -> None:
        if config.tqc is not None:
            raise ValueError("PPO config cannot contain TQC settings")
        if config.ppo is None:
            config.ppo = PPOConfig()

    def action_adapter(self, config: TrainingConfig) -> DiscretePwmActionAdapter:
        assert config.ppo is not None
        return DiscretePwmActionAdapter(config.ppo.pwm_levels)

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
            policy_kwargs={"net_arch": {"pi": layers, "vf": layers}},
        )
        apply_forward_bias(
            model, p.initial_forward_bias,
            steering_strength=p.initial_steering_bias,
        )
        if p.teacher_model:
            teacher = TeacherAnchoredPPO.load(p.teacher_model, device=device)
            model.set_teacher(teacher, p.teacher_kl_coefficient)
        model.set_expert_imitation(p.imitation_coefficient)
        return model

    def load_model(
        self, path: Path, env: Any, device: str, *, resume: bool = False
    ) -> Any:
        return TeacherAnchoredPPO.load(str(path), env=env, device=device)

    def configure_resume(self, model: Any, config: TrainingConfig, device: str) -> None:
        assert config.ppo is not None
        p = config.ppo
        if p.teacher_model:
            teacher = TeacherAnchoredPPO.load(p.teacher_model, device=device)
            model.set_teacher(teacher, p.teacher_kl_coefficient)
        model.set_expert_imitation(p.imitation_coefficient)

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
            "clip_fraction": values.get("train/clip_fraction"),
        }
