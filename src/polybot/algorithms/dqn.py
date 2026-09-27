"""Standard Stable-Baselines3 DQN with native nine-way digital controls."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from stable_baselines3 import DQN

from polybot.algorithms.base import AlgorithmBackend
from polybot.control.native_digital import NativeDigitalActionAdapter
from polybot.training.config import ARCHITECTURES, DQNConfig

if TYPE_CHECKING:
    from polybot.training.config import TrainingConfig


class DQNBackend(AlgorithmBackend):
    name = "dqn"

    def validate_config(self, config: TrainingConfig) -> None:
        if config.ppo is not None or config.tqc is not None:
            raise ValueError("DQN config cannot contain PPO or TQC settings")
        if config.dqn is None:
            config.dqn = DQNConfig()

    def action_adapter(self, config: TrainingConfig) -> NativeDigitalActionAdapter:
        return NativeDigitalActionAdapter()

    def architecture(self, config: TrainingConfig) -> str:
        assert config.dqn is not None
        return config.dqn.architecture

    def create_model(self, config: TrainingConfig, env: Any, device: str) -> DQN:
        assert config.dqn is not None
        p = config.dqn
        return DQN(
            "MlpPolicy", env, seed=config.seed, device=device, verbose=0,
            learning_rate=p.learning_rate, buffer_size=p.replay_capacity,
            learning_starts=p.learning_starts, batch_size=p.batch_size,
            gamma=p.gamma, train_freq=p.train_frequency,
            gradient_steps=p.gradient_steps,
            target_update_interval=p.target_update_interval,
            exploration_fraction=p.exploration_fraction,
            exploration_initial_eps=p.exploration_initial_eps,
            exploration_final_eps=p.exploration_final_eps,
            policy_kwargs={"net_arch": list(ARCHITECTURES[p.architecture])},
        )

    def load_model(
        self, path: Path, env: Any, device: str, *, resume: bool = False
    ) -> DQN:
        model = DQN.load(str(path), env=env, device=device)
        if resume:
            replay = path.with_name("replay.pkl")
            if not replay.is_file():
                raise FileNotFoundError(f"DQN resume requires replay state: {replay}")
            model.load_replay_buffer(str(replay))
        return model

    def save_model(self, model: DQN, directory: Path, *, resume: bool = False) -> None:
        super().save_model(model, directory, resume=resume)
        if resume:
            model.save_replay_buffer(str(directory / "replay.pkl"))

    def configure_resume(self, model: DQN, config: TrainingConfig, device: str) -> None:
        if model.replay_buffer is None:
            raise RuntimeError("DQN resume requires a loaded replay buffer")

    def parameter_counts(self, model: DQN) -> dict[str, int]:
        q_network = sum(p.numel() for p in model.policy.q_net.parameters() if p.requires_grad)
        return {"actor": 0, "critic": q_network, "total": q_network}

    def metrics(self, model: DQN) -> dict[str, float | int | None]:
        values = getattr(getattr(model, "_logger", None), "name_to_value", {})
        return {
            "replay_size": model.replay_buffer.size() if model.replay_buffer else 0,
            "updates": model._n_updates,
            "loss": values.get("train/loss"),
            "exploration_rate": model.exploration_rate,
        }
