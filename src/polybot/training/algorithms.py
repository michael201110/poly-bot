"""Algorithm-specific policy construction and loading for the shared trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from polybot.training.config import TQC_ARCHITECTURES, TrainingConfig, policy_kwargs
from polybot.training.initialization import apply_forward_bias


def action_schema(config: TrainingConfig) -> str:
    if config.algorithm == "tqc":
        return "continuous-pwm-v1"
    return "pwm-multidiscrete-v1" if config.pwm_enabled else "digital-multidiscrete-v1"


def model_class(algorithm: str) -> type:
    if algorithm == "ppo":
        from polybot.training.anchored_ppo import TeacherAnchoredPPO

        return TeacherAnchoredPPO
    if algorithm == "tqc":
        from polybot.training.forward_tqc import ForwardWarmupTQC

        return ForwardWarmupTQC
    raise ValueError(f"unsupported algorithm: {algorithm}")


def create_model(config: TrainingConfig, env: Any, device: str) -> Any:
    if config.algorithm == "ppo":
        model = model_class("ppo")(
            "MlpPolicy", env, seed=config.seed, device=device,
            learning_rate=config.learning_rate, gamma=config.gamma,
            gae_lambda=config.gae_lambda, ent_coef=config.entropy_coefficient,
            policy_kwargs=policy_kwargs(config.architecture),
            n_steps=max(2, min(config.rollout_steps, config.timesteps)),
            batch_size=max(2, min(config.batch_size, config.timesteps)),
            n_epochs=config.ppo_epochs, verbose=0,
        )
        apply_forward_bias(model, strength=1.5, steering_strength=1.0)
        return model
    settings = config.tqc
    model = model_class("tqc")(
        "MlpPolicy", env, seed=config.seed, device=device,
        learning_rate=settings.learning_rate, buffer_size=settings.buffer_size,
        learning_starts=settings.learning_starts, batch_size=settings.batch_size,
        gamma=settings.gamma, tau=settings.tau, train_freq=settings.train_freq,
        gradient_steps=settings.gradient_steps, ent_coef=settings.ent_coef,
        forward_warmup_fraction=settings.forward_warmup_fraction,
        forward_warmup_steering_std=settings.forward_warmup_steering_std,
        forward_prior_initial=settings.forward_prior_initial,
        forward_prior_steps=settings.forward_prior_steps,
        policy_kwargs={"net_arch": {
            "pi": list(TQC_ARCHITECTURES[settings.architecture]),
            "qf": list(TQC_ARCHITECTURES[settings.architecture]),
        }}, verbose=0,
    )
    import torch

    with torch.no_grad():
        model.policy.actor.mu.bias[1] += settings.initial_throttle_bias
    if settings.success_demo_path:
        import numpy as np

        with np.load(settings.success_demo_path) as demo:
            model.remember_successful_trajectory(
                demo["observations"], demo["actions"]
            )
        for _ in range(500):
            model._rehearse_success(settings.batch_size)
    return model


def load_model(config: TrainingConfig, path: Path, env: Any, device: str) -> Any:
    if config.algorithm == "ppo":
        model = model_class("ppo").load(
            str(path), env=env, device=device,
            custom_objects={
                "n_steps": config.rollout_steps,
                "batch_size": config.batch_size,
                "n_epochs": config.ppo_epochs,
                "learning_rate": config.learning_rate,
                "gamma": config.gamma,
                "gae_lambda": config.gae_lambda,
                "ent_coef": config.entropy_coefficient,
            },
        )
    else:
        model = model_class("tqc").load(str(path), env=env, device=device)
        replay_path = path.with_suffix(".replay.pkl")
        if not replay_path.is_file():
            raise FileNotFoundError(
                f"TQC resume requires its replay buffer: {replay_path}"
            )
        model.load_replay_buffer(str(replay_path))
    return model


def configure_model(model: Any, config: TrainingConfig, device: str, status: Any) -> None:
    if config.algorithm != "ppo":
        return
    from stable_baselines3 import PPO

    teacher = None
    if config.teacher_model is not None:
        if not config.teacher_model.is_file():
            raise FileNotFoundError(f"teacher model not found: {config.teacher_model}")
        teacher = PPO.load(str(config.teacher_model), device=device)
    model.set_teacher(teacher, config.teacher_kl_coefficient)
    model.set_expert_imitation(
        config.expert_imitation_coefficient
        if config.rewards.expert_action_bonus_per_s > 0 else 0.0
    )
    model.training_status = status
