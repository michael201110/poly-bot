"""QR-DQN with native digital controls and legacy DQN inference support."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from sb3_contrib import QRDQN
from stable_baselines3 import DQN

from polybot.algorithms.base import AlgorithmBackend
from polybot.control.native_digital import NativeDigitalActionAdapter
from polybot.training.config import DQN_ARCHITECTURES, DQNConfig

if TYPE_CHECKING:
    from polybot.training.config import TrainingConfig


class PhaseExplorationSchedule:
    """Decay DQN epsilon against one curriculum phase's own step count."""

    def __init__(self, config: DQNConfig, phase_steps: int) -> None:
        self.initial = config.exploration_initial_eps
        self.final = config.exploration_final_eps
        self.decay_steps = max(1, round(phase_steps * config.exploration_fraction))
        self.steps = 0
        self.value = self.initial

    def __call__(self, _progress_remaining: float) -> float:
        return self.value

    def advance(self, completed_steps: int) -> float:
        self.steps = max(self.steps, completed_steps)
        fraction = min(1.0, self.steps / self.decay_steps)
        self.value = self.initial + fraction * (self.final - self.initial)
        return self.value


class DQNBackend(AlgorithmBackend):
    name = "dqn"

    def validate_config(self, config: TrainingConfig) -> None:
        if config.ppo is not None or config.tqc is not None:
            raise ValueError("DQN config cannot contain PPO or TQC settings")
        if config.dqn is None:
            config.dqn = DQNConfig()

    def action_adapter(self, config: TrainingConfig) -> NativeDigitalActionAdapter:
        assert config.dqn is not None
        return NativeDigitalActionAdapter(brake_enabled=config.dqn.action_set == "full")

    def architecture(self, config: TrainingConfig) -> str:
        assert config.dqn is not None
        return config.dqn.architecture

    def create_model(self, config: TrainingConfig, env: Any, device: str) -> QRDQN:
        assert config.dqn is not None
        p = config.dqn
        return QRDQN(
            "MlpPolicy", env, seed=config.seed, device=device, verbose=0,
            learning_rate=p.learning_rate, buffer_size=p.replay_capacity,
            learning_starts=p.learning_starts, batch_size=p.batch_size,
            gamma=p.gamma, train_freq=p.train_frequency,
            gradient_steps=p.gradient_steps,
            target_update_interval=p.target_update_interval,
            exploration_fraction=p.exploration_fraction,
            exploration_initial_eps=p.exploration_initial_eps,
            exploration_final_eps=p.exploration_final_eps,
            policy_kwargs={
                "net_arch": list(DQN_ARCHITECTURES[p.architecture]),
                "n_quantiles": p.n_quantiles,
            },
        )

    def load_model(
        self, path: Path, env: Any, device: str, *, resume: bool = False
    ) -> QRDQN | DQN:
        from polybot.models.registry import ModelRegistry

        metadata = (
            ModelRegistry().read_metadata(path.parent)
            if (path.parent / "metadata.json").is_file() else None
        )
        if metadata is not None and metadata.implementation != "qr_dqn":
            if resume:
                raise ValueError("Legacy DQN cannot resume as QR-DQN; migrate its checkpoint first")
            return DQN.load(str(path), env=env, device=device)
        model = QRDQN.load(str(path), env=env, device=device)
        if resume:
            replay = path.with_name("replay.pkl")
            if not replay.is_file():
                raise FileNotFoundError(f"DQN resume requires replay state: {replay}")
            model.load_replay_buffer(str(replay))
        return model

    def save_model(self, model: QRDQN, directory: Path, *, resume: bool = False) -> None:
        super().save_model(model, directory, resume=resume)
        if resume:
            model.save_replay_buffer(str(directory / "replay.pkl"))

    def configure_resume(
        self, model: QRDQN, config: TrainingConfig, device: str, *, fresh_replay: bool = False
    ) -> None:
        if model.replay_buffer is None:
            raise RuntimeError("DQN resume requires a loaded replay buffer")
        if fresh_replay:
            assert config.dqn is not None
            model.learning_starts = model.num_timesteps + max(
                config.dqn.learning_starts, config.dqn.batch_size
            )

    def begin_phase(self, model: QRDQN, config: TrainingConfig, phase_steps: int) -> None:
        """Reheat epsilon once per phase while keeping replay and optimizer intact."""
        assert config.dqn is not None
        schedule = PhaseExplorationSchedule(config.dqn, phase_steps)
        model.exploration_schedule = schedule
        model.exploration_rate = schedule.initial

    def advance_phase(self, model: QRDQN, completed_steps: int) -> None:
        schedule = model.exploration_schedule
        if not isinstance(schedule, PhaseExplorationSchedule):
            raise RuntimeError("QR-DQN phase exploration was not initialized")
        model.exploration_rate = schedule.advance(completed_steps)

    def parameter_counts(self, model: QRDQN | DQN) -> dict[str, int]:
        network = model.policy.quantile_net if isinstance(model, QRDQN) else model.policy.q_net
        q_network = sum(p.numel() for p in network.parameters() if p.requires_grad)
        return {"actor": 0, "critic": q_network, "total": q_network}

    def metrics(self, model: QRDQN | DQN) -> dict[str, float | int | None]:
        values = getattr(getattr(model, "_logger", None), "name_to_value", {})
        return {
            "replay_size": model.replay_buffer.size() if model.replay_buffer else 0,
            "updates": model._n_updates,
            "loss": values.get("train/loss"),
            "exploration_rate": model.exploration_rate,
        }
