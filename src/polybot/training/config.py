"""Serializable configuration shared by the CLI, GUI, and training manager."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from polybot.env import RewardConfig, summer_1_reward_config

ARCHITECTURE_PRESETS: dict[str, tuple[int, ...]] = {
    "legacy": (64, 64),
    "small": (128, 128),
    "medium": (256, 256, 256),
    "large": (512, 512, 512),
    "xl": (1024, 1024, 512),
}


def architecture(name: str) -> tuple[int, ...]:
    try:
        return ARCHITECTURE_PRESETS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown architecture preset: {name}") from exc


def policy_kwargs(preset: str) -> dict[str, Any]:
    layers = list(architecture(preset))
    return {"net_arch": {"pi": layers, "vf": layers}}


def estimate_ppo_parameters(
    observation_size: int, action_dims: tuple[int, ...], preset: str
) -> int:
    """Exact count for SB3's default MlpExtractor and MultiCategorical heads."""
    layers = architecture(preset)
    per_branch = (observation_size + 1) * layers[0]
    per_branch += sum((a + 1) * b for a, b in zip(layers, layers[1:], strict=False))
    heads = (layers[-1] + 1) * (sum(action_dims) + 1)
    return 2 * per_branch + heads


@dataclass(slots=True)
class CurriculumConfig:
    mode: str = "full"
    start_ratio: float | None = None
    end_ratio: float | None = None
    start_s: float | None = None
    end_s: float | None = None


@dataclass(slots=True)
class TrainingConfig:
    backend: str = "websocket"
    track_name: str = "Summer 1"
    track_id: str = "current"
    model_name: str = "default"
    architecture: str = "xl"
    device: str = "auto"
    pwm_enabled: bool = True
    pwm_levels: int = 41
    frame_skip: int = 30
    max_episode_seconds: float = 60.0
    timesteps: int = 100_000
    max_episodes: int | None = None
    learning_rate: float = 1e-4
    gamma: float = 0.9995
    gae_lambda: float = 0.995
    entropy_coefficient: float = 0.001
    rollout_steps: int = 8192
    batch_size: int = 1024
    ppo_epochs: int = 3
    teacher_model: Path | None = None
    teacher_kl_coefficient: float = 0.0
    checkpoint_interval: int = 10_000
    output_root: Path = Path("models")
    seed: int = 0
    lookahead_count: int = 12
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    rewards: RewardConfig = field(default_factory=summer_1_reward_config)

    def __post_init__(self) -> None:
        if self.backend not in {"websocket", "mock"}:
            raise ValueError("backend must be websocket or mock")
        architecture(self.architecture)
        self.device = self.device.lower()
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.pwm_levels < 3 or self.pwm_levels % 2 == 0:
            raise ValueError("pwm_levels must be an odd integer >= 3")
        if self.frame_skip < 1 or self.timesteps < 1:
            raise ValueError("frame_skip and timesteps must be positive")
        if self.max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be positive")
        if self.rollout_steps < 2:
            raise ValueError("rollout_steps must be at least 2")
        if not 2 <= self.batch_size <= self.rollout_steps:
            raise ValueError("batch_size must be between 2 and rollout_steps")
        if self.rollout_steps % self.batch_size:
            raise ValueError("batch_size must divide rollout_steps evenly")
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be positive")
        if self.teacher_kl_coefficient < 0:
            raise ValueError("teacher KL coefficient cannot be negative")
        if self.teacher_kl_coefficient and self.teacher_model is None:
            raise ValueError("teacher_model is required when teacher KL is enabled")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_root"] = str(self.output_root)
        result["teacher_model"] = (
            None if self.teacher_model is None else str(self.teacher_model)
        )
        return result
