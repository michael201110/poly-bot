"""Serializable configuration shared by the CLI, GUI, and training manager."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from polybot.env import RewardConfig, summer_1_reward_config

ARCHITECTURE_PRESETS: dict[str, tuple[int, ...]] = {
    "legacy": (64, 64),
    "compact": (104, 104),
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


TQC_ARCHITECTURES: dict[str, tuple[int, ...]] = {
    "tiny": (64, 64),
    "standard": (256, 256),
    "compact": (128, 128),
}


def estimate_tqc_actor_parameters(observation_size: int, action_size: int, preset: str) -> int:
    """Actor used for action selection, including mean and log-standard-deviation heads."""
    layers = TQC_ARCHITECTURES[preset]
    actor = (observation_size + 1) * layers[0]
    actor += sum((a + 1) * b for a, b in zip(layers, layers[1:], strict=False))
    actor += 2 * (layers[-1] + 1) * action_size
    return actor


def estimate_tqc_parameters(observation_size: int, action_size: int, preset: str) -> int:
    """Actor, two critics and their target copies (SB3-contrib defaults)."""
    layers = TQC_ARCHITECTURES[preset]
    critic = (observation_size + action_size + 1) * layers[0]
    critic += sum((a + 1) * b for a, b in zip(layers, layers[1:], strict=False))
    critic += (layers[-1] + 1) * 25
    return estimate_tqc_actor_parameters(observation_size, action_size, preset) + 4 * critic


@dataclass(slots=True)
class TqcConfig:
    architecture: str = "standard"
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 10_000
    batch_size: int = 256
    gamma: float = 0.999
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto_0.01"
    forward_warmup_fraction: float = 0.8
    forward_warmup_steering_std: float = 0.45
    initial_throttle_bias: float = 1.0
    forward_prior_initial: float = 0.0
    forward_prior_steps: int = 0
    success_demo_path: str = ""

    def __post_init__(self) -> None:
        if self.architecture not in TQC_ARCHITECTURES:
            raise ValueError("unknown TQC architecture")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("TQC learning rate must be finite and positive")
        if self.buffer_size < 1 or self.learning_starts < 0:
            raise ValueError("invalid TQC learning or replay settings")
        if self.batch_size < 1 or self.train_freq < 1 or self.gradient_steps < 1:
            raise ValueError("TQC batch size, train frequency and gradient steps must be positive")
        if not (math.isfinite(self.gamma) and 0 < self.gamma <= 1):
            raise ValueError("TQC gamma must be finite and in (0, 1]")
        if not (math.isfinite(self.tau) and 0 < self.tau <= 1):
            raise ValueError("TQC gamma and tau must be in (0, 1]")
        if self.ent_coef != "auto" and not self.ent_coef.startswith("auto_"):
            try:
                value = float(self.ent_coef)
                if not math.isfinite(value) or value < 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("TQC entropy must be 'auto' or a nonnegative number") from exc
        elif self.ent_coef.startswith("auto_"):
            try:
                initial = float(self.ent_coef.removeprefix("auto_"))
                if not math.isfinite(initial) or initial <= 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("TQC automatic entropy initial value must be positive") from exc
        if not math.isfinite(self.forward_warmup_fraction) or not (
            0 <= self.forward_warmup_fraction <= 1
        ):
            raise ValueError("TQC forward warmup fraction must be in [0, 1]")
        if not math.isfinite(self.forward_warmup_steering_std) or (
            self.forward_warmup_steering_std < 0
        ):
            raise ValueError("TQC warmup steering standard deviation must be nonnegative")
        if not math.isfinite(self.initial_throttle_bias) or self.initial_throttle_bias < 0:
            raise ValueError("TQC initial throttle bias must be nonnegative")
        if not math.isfinite(self.forward_prior_initial) or not (
            0 <= self.forward_prior_initial <= 1
        ):
            raise ValueError("TQC forward prior must be in [0, 1]")
        if self.forward_prior_steps < 0:
            raise ValueError("TQC forward prior steps must be nonnegative")


@dataclass(slots=True)
class CurriculumConfig:
    mode: str = "full"
    start_ratio: float | None = None
    end_ratio: float | None = None
    start_s: float | None = None
    end_s: float | None = None


@dataclass(slots=True)
class TrainingConfig:
    algorithm: str = "ppo"
    backend: str = "websocket"
    track_name: str = "Summer 1"
    track_id: str = "current"
    model_name: str = "default"
    reward_profile: str | None = None
    architecture: str = "xl"
    device: str = "auto"
    pwm_enabled: bool = True
    pwm_levels: int = 41
    frame_skip: int = 30
    max_episode_seconds: float = 60.0
    max_episode_steps: int = 2_000_000_000
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
    expert_imitation_coefficient: float | None = None
    reward_scale: float = 0.01
    checkpoint_interval: int = 10_000
    output_root: Path = Path("models")
    seed: int = 0
    lookahead_count: int = 12
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    rewards: RewardConfig = field(default_factory=summer_1_reward_config)
    tqc: TqcConfig = field(default_factory=TqcConfig)

    def __post_init__(self) -> None:
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        if self.algorithm not in {"ppo", "tqc"}:
            raise ValueError("algorithm must be ppo or tqc")
        if self.expert_imitation_coefficient is None:
            self.expert_imitation_coefficient = 1.0 if self.algorithm == "ppo" else 0.0
        if self.algorithm == "tqc" and (self.teacher_model or self.teacher_kl_coefficient or
                                         self.expert_imitation_coefficient != 0):
            raise ValueError("teacher and PPO expert imitation settings are unsupported for TQC")
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
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
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
        if self.expert_imitation_coefficient < 0:
            raise ValueError("expert imitation coefficient cannot be negative")
        if self.teacher_kl_coefficient and self.teacher_model is None:
            raise ValueError("teacher_model is required when teacher KL is enabled")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_root"] = str(self.output_root)
        result["teacher_model"] = (
            None if self.teacher_model is None else str(self.teacher_model)
        )
        return result
