"""Standard TQC with a seeded, finite forward biased replay warmup."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sb3_contrib import TQC

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
        self._warmup_rng = np.random.default_rng(kwargs.get("seed"))
        super().__init__(*args, **kwargs)

    def _sample_action(
        self, learning_starts: int, action_noise: Any = None, n_envs: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps >= learning_starts:
            return super()._sample_action(learning_starts, action_noise, n_envs)
        steering = np.clip(
            self._warmup_rng.normal(0, self.warmup_steering_std, n_envs), -1, 1
        )
        forward = self._warmup_rng.uniform(-1, 1, n_envs)
        forward_mask = self._warmup_rng.random(n_envs) < self.warmup_forward_fraction
        forward[forward_mask] = self._warmup_rng.uniform(0.35, 1, forward_mask.sum())
        action = np.stack((steering, forward), axis=1).astype(np.float32)
        # The action space is [-1, 1], so its replay representation is identical.
        return action, action.copy()


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
        }
