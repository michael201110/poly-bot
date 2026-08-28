"""Transfer a trained PPO policy into a smaller network by policy distillation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.training.config import policy_kwargs
from polybot.training.devices import resolve_device
from polybot.training.models import ModelMetadata, ModelRegistry, git_commit
from polybot.transport import WebSocketServerTransport


@dataclass(frozen=True, slots=True)
class DistillationMetrics:
    initial_loss: float
    final_loss: float
    action_agreement: float
    observations: int


def _policy_tensors(policy: Any, observations: Any) -> tuple[Any, Any]:
    features = policy.extract_features(observations)
    latent_policy, latent_value = policy.mlp_extractor(features)
    return policy.action_net(latent_policy), policy.value_net(latent_value).flatten()


def _distillation_loss(
    teacher_logits: Any,
    teacher_values: Any,
    student_logits: Any,
    student_values: Any,
    *,
    action_dims: tuple[int, ...],
) -> Any:
    import torch
    import torch.nn.functional as functional

    action_loss = torch.zeros((), device=student_logits.device)
    offset = 0
    for width in action_dims:
        teacher_chunk = teacher_logits[:, offset : offset + width]
        student_chunk = student_logits[:, offset : offset + width]
        action_loss = action_loss + functional.kl_div(
            functional.log_softmax(student_chunk, dim=1),
            functional.softmax(teacher_chunk, dim=1),
            reduction="batchmean",
        )
        offset += width
    value_scale = teacher_values.detach().std().clamp_min(1.0)
    value_loss = functional.mse_loss(
        student_values / value_scale, teacher_values / value_scale
    )
    return action_loss + 0.1 * value_loss


def distill_policy(
    teacher: Any,
    student: Any,
    observations: np.ndarray,
    *,
    action_dims: tuple[int, ...],
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
) -> DistillationMetrics:
    """Fit ``student`` to the teacher's categorical policy and value function."""

    import torch

    if observations.ndim != 2 or len(observations) < 1:
        raise ValueError("observations must be a non-empty two-dimensional array")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    device = next(student.policy.parameters()).device
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    teacher.policy.set_training_mode(False)
    student.policy.set_training_mode(True)
    optimizer = torch.optim.Adam(student.policy.parameters(), lr=learning_rate)

    with torch.no_grad():
        teacher_logits, teacher_values = _policy_tensors(teacher.policy, tensor)
        initial_student_logits, initial_student_values = _policy_tensors(student.policy, tensor)
        initial_loss = float(
            _distillation_loss(
                teacher_logits,
                teacher_values,
                initial_student_logits,
                initial_student_values,
                action_dims=action_dims,
            ).item()
        )

    for _epoch in range(epochs):
        for indices in torch.randperm(len(tensor), device=device).split(batch_size):
            batch = tensor[indices]
            with torch.no_grad():
                teacher_batch_logits, teacher_batch_values = _policy_tensors(
                    teacher.policy, batch
                )
            student_batch_logits, student_batch_values = _policy_tensors(student.policy, batch)
            loss = _distillation_loss(
                teacher_batch_logits,
                teacher_batch_values,
                student_batch_logits,
                student_batch_values,
                action_dims=action_dims,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.policy.parameters(), 1.0)
            optimizer.step()

    student.policy.set_training_mode(False)
    with torch.no_grad():
        teacher_logits, teacher_values = _policy_tensors(teacher.policy, tensor)
        student_logits, student_values = _policy_tensors(student.policy, tensor)
        final_loss = float(
            _distillation_loss(
                teacher_logits,
                teacher_values,
                student_logits,
                student_values,
                action_dims=action_dims,
            ).item()
        )
        agreements = []
        offset = 0
        for width in action_dims:
            agreements.append(
                teacher_logits[:, offset : offset + width].argmax(dim=1)
                == student_logits[:, offset : offset + width].argmax(dim=1)
            )
            offset += width
        agreement = float(torch.stack(agreements, dim=1).all(dim=1).float().mean().item())
    return DistillationMetrics(initial_loss, final_loss, agreement, len(observations))


def collect_teacher_observations(
    teacher: Any,
    env: PolyTrackEnv,
    *,
    samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Collect deterministic teacher states from randomized track quarters."""

    if samples < 1:
        raise ValueError("samples must be positive")
    collected: list[np.ndarray] = []
    episode = 0
    observation, _ = env.reset(seed=seed)
    while len(collected) < samples:
        collected.append(np.asarray(observation, dtype=np.float32).copy())
        action, _ = teacher.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            episode += 1
            observation, _ = env.reset(seed=seed + episode)
        if len(collected) % 2_000 == 0:
            print(f"Collected {len(collected):,}/{samples:,} teacher observations", flush=True)
    return np.stack(collected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Distill an XL PolyBot policy into medium")
    parser.add_argument("teacher", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("models"))
    parser.add_argument("--track-name", default="Winter 4 Medium")
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--pwm-levels", type=int, default=41)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)

    from stable_baselines3 import PPO

    selected_device = resolve_device(args.device)
    teacher = PPO.load(str(args.teacher), device=selected_device.resolved)
    transport = WebSocketServerTransport(connect_timeout_s=300.0, request_timeout_s=300.0)
    env = PolyTrackEnv(
        transport,
        track_id="current",
        lookahead_count=12,
        frame_skip=args.frame_skip,
        max_episode_steps=2_000_000_000,
        max_episode_s=60.0,
        pwm_enabled=True,
        pwm_levels=args.pwm_levels,
        curriculum_random_quarters=True,
    )
    try:
        observations = collect_teacher_observations(teacher, env, samples=args.samples)
    finally:
        env.close()

    dummy_env = PolyTrackEnv(
        MockSimulatorTransport(),
        track_id="mock/gentle-s",
        lookahead_count=12,
        frame_skip=args.frame_skip,
        pwm_enabled=True,
        pwm_levels=args.pwm_levels,
    )
    try:
        student = PPO(
            "MlpPolicy",
            dummy_env,
            device=selected_device.resolved,
            policy_kwargs=policy_kwargs("medium"),
            learning_rate=1e-4,
            gamma=0.9995,
            gae_lambda=0.995,
            ent_coef=0.0005,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
            verbose=0,
        )
        metrics = distill_policy(
            teacher,
            student,
            observations,
            action_dims=(args.pwm_levels, 2, 2),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        registry = ModelRegistry(args.output_root)
        output = registry.initialise_track(args.track_name) / "latest"
        student.save(str(output))
        parameters = sum(parameter.numel() for parameter in student.policy.parameters())
        registry.write_metadata(
            ModelMetadata(
                track_name=args.track_name,
                track_id="current",
                architecture="medium",
                parameter_count=parameters,
                pwm_enabled=True,
                pwm_resolution=args.pwm_levels,
                frame_skip=args.frame_skip,
                reward_settings={},
                ppo_hyperparameters={
                    "learning_rate": 1e-4,
                    "gamma": 0.9995,
                    "gae_lambda": 0.995,
                    "entropy_coefficient": 0.0005,
                    "rollout_steps": 2048,
                    "batch_size": 256,
                    "ppo_epochs": 5,
                    "distillation": asdict(metrics),
                    "teacher": str(args.teacher),
                },
                git_commit=git_commit(),
            ),
            "latest",
        )
    finally:
        dummy_env.close()
    print(
        f"Saved {parameters:,}-parameter medium student to {output.with_suffix('.zip')}\n"
        f"Distillation loss {metrics.initial_loss:.5f} -> {metrics.final_loss:.5f}; "
        f"exact action agreement {metrics.action_agreement:.1%}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
