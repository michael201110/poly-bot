"""Versioned, typed configuration shared by CLI, GUI and saved runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from polybot.environment.rewards import RewardConfig, summer_1_reward_config

CONFIG_SCHEMA = "polybot.config.v2"
ARCHITECTURES = {"tiny": (64, 64), "compact": (128, 128), "standard": (256, 256)}
DQN_ARCHITECTURES = {**ARCHITECTURES, "yosh_2020": (64, 16)}


@dataclass(slots=True)
class PPOConfig:
    architecture: str = "compact"
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.005
    rollout_steps: int = 512
    batch_size: int = 128
    epochs: int = 5
    pwm_levels: int = 41
    teacher_model: str | None = None
    teacher_kl_coefficient: float = 0.0
    imitation_coefficient: float = 0.0
    initial_forward_bias: float = 1.0
    initial_steering_bias: float = 0.5

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError("unknown PPO architecture")
        if self.pwm_levels < 3 or self.pwm_levels % 2 != 1:
            raise ValueError("PPO PWM levels must be odd and >= 3")
        if self.rollout_steps < 2 or not 2 <= self.batch_size <= self.rollout_steps:
            raise ValueError("invalid PPO rollout or batch size")
        if self.rollout_steps % self.batch_size or self.epochs < 1:
            raise ValueError("PPO batch must divide rollout; epochs must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("invalid PPO discount settings")
        if self.learning_rate <= 0 or self.entropy_coefficient < 0:
            raise ValueError("invalid PPO learning settings")
        if self.teacher_kl_coefficient < 0 or (
            self.teacher_kl_coefficient and not self.teacher_model
        ):
            raise ValueError("teacher KL needs a teacher model and nonnegative coefficient")
        if self.imitation_coefficient < 0:
            raise ValueError("PPO imitation coefficient cannot be negative")
        if min(self.initial_forward_bias, self.initial_steering_bias) < 0:
            raise ValueError("PPO initial action biases cannot be negative")


@dataclass(slots=True)
class DQNConfig:
    architecture: str = "compact"
    action_set: str = "full"
    n_quantiles: int = 32
    learning_rate: float = 1e-4
    replay_capacity: int = 250_000
    learning_starts: int = 5_000
    batch_size: int = 128
    gamma: float = 0.995
    train_frequency: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.20
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05

    def __post_init__(self) -> None:
        if self.architecture not in DQN_ARCHITECTURES:
            raise ValueError("unknown DQN architecture")
        if self.action_set not in {"full", "no_brake"}:
            raise ValueError("DQN action set must be full or no_brake")
        if not 2 <= self.n_quantiles <= 200:
            raise ValueError("DQN quantiles must be between 2 and 200")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("DQN learning rate must be positive and finite")
        if self.replay_capacity < 1 or self.learning_starts < 0:
            raise ValueError("DQN replay capacity must be positive; learning starts must be nonnegative")
        if min(self.batch_size, self.train_frequency, self.gradient_steps, self.target_update_interval) < 1:
            raise ValueError("DQN batch size, train frequency, gradient steps and target interval must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("DQN gamma must be in (0, 1]")
        if not 0 <= self.exploration_fraction <= 1:
            raise ValueError("DQN exploration fraction must be in [0, 1]")
        if not (0 <= self.exploration_final_eps <= self.exploration_initial_eps <= 1):
            raise ValueError("DQN epsilon values must be in [0, 1], with final <= initial")


@dataclass(slots=True)
class TQCConfig:
    architecture: str = "compact"
    learning_rate: float = 3e-4
    replay_capacity: int = 250_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 0.995
    tau: float = 0.005
    train_frequency: int = 2
    gradient_steps: int = 1
    entropy: str = "auto_0.1"
    warmup_forward_fraction: float = 0.8
    warmup_steering_std: float = 0.35

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError("unknown TQC architecture")
        if self.replay_capacity < 1 or self.learning_starts < 0:
            raise ValueError("invalid TQC replay settings")
        if min(self.batch_size, self.train_frequency, self.gradient_steps) < 1:
            raise ValueError("invalid TQC batch or update frequency")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("invalid TQC learning settings")
        if not 0 <= self.warmup_forward_fraction <= 1 or self.warmup_steering_std < 0:
            raise ValueError("invalid TQC warmup settings")
        if self.entropy != "auto" and not self.entropy.startswith("auto_"):
            raise ValueError("TQC entropy must be auto or auto_<positive initial value>")
        if self.entropy.startswith("auto_") and float(self.entropy[5:]) <= 0:
            raise ValueError("TQC initial entropy coefficient must be positive")


@dataclass(slots=True)
class CurriculumPhaseConfig:
    mode: str
    steps: int
    start_ratio: float | None = None
    end_ratio: float | None = None
    start_s: float | None = None
    end_s: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"full", "section", "quarters-randomised", "timed"}:
            raise ValueError("invalid custom curriculum phase mode")
        if self.steps < 1:
            raise ValueError("custom curriculum phase steps must be positive")
        if self.mode == "section" and not (
            self.start_ratio is not None and self.end_ratio is not None
            and 0 <= self.start_ratio < self.end_ratio <= 1
        ):
            raise ValueError("custom section requires 0 <= start < end <= 1")
        if self.mode == "timed" and not (
            self.start_s is not None and self.end_s is not None
            and 0 <= self.start_s < self.end_s
        ):
            raise ValueError("custom timed section requires 0 <= start < end")


@dataclass(slots=True)
class CurriculumConfig:
    mode: str = "full"
    start_ratio: float | None = None
    end_ratio: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    phases: tuple[CurriculumPhaseConfig, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"full", "section", "quarters", "quarters-randomised", "q4-full", "timed", "custom"}:
            raise ValueError("unknown curriculum mode")
        if self.mode == "custom" and not self.phases:
            raise ValueError("custom curriculum requires at least one phase")
        if self.mode != "custom" and self.phases:
            raise ValueError("custom phases require custom curriculum mode")
        if self.mode == "section" and not (
            self.start_ratio is not None and self.end_ratio is not None
            and 0 <= self.start_ratio < self.end_ratio <= 1
        ):
            raise ValueError("section requires 0 <= start < end <= 1")
        if self.mode == "timed" and not (
            self.start_s is not None and self.end_s is not None
            and 0 <= self.start_s < self.end_s
        ):
            raise ValueError("timed section requires 0 <= start < end")


@dataclass(slots=True)
class EvaluationConfig:
    interval_steps: int = 10_000
    episodes: int = 3

    def __post_init__(self) -> None:
        if self.interval_steps < 1 or self.episodes < 1:
            raise ValueError("evaluation interval and episodes must be positive")


@dataclass(slots=True)
class TrainingConfig:
    algorithm: str = "tqc"
    backend: str = "mock"
    track_name: str = "Mock straight"
    track_id: str = "mock/straight"
    device: str = "auto"
    seed: int = 0
    frame_skip: int = 4
    timesteps: int = 100_000
    max_episode_seconds: float = 60.0
    max_episode_steps: int = 30_000
    lookahead_count: int = 12
    reward_profile: str | None = None
    reward_scale: float = 0.01
    checkpoint_interval: int = 10_000
    output_root: Path = Path("models")
    log_root: Path = Path("logs")
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    rewards: RewardConfig = field(default_factory=summer_1_reward_config)
    ppo: PPOConfig | None = None
    dqn: DQNConfig | None = None
    tqc: TQCConfig | None = None

    def __post_init__(self) -> None:
        from polybot.algorithms.registry import backend_for

        backend_for(self.algorithm).validate_config(self)
        if self.backend not in {"mock", "websocket"} or self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("invalid simulator backend or device")
        if self.frame_skip < 1 or self.timesteps < 1 or self.lookahead_count < 1:
            raise ValueError("frame skip, budget and lookahead must be positive")
        if self.max_episode_seconds <= 0 or self.max_episode_steps < 1:
            raise ValueError("invalid episode limit")
        if self.checkpoint_interval < 0 or not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("invalid checkpoint interval or reward scale")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = CONFIG_SCHEMA
        value["curriculum"]["phases"] = [asdict(phase) for phase in self.curriculum.phases]
        value["output_root"] = str(self.output_root)
        value["log_root"] = str(self.log_root)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingConfig:
        value = value.copy()
        if value.pop("schema") != CONFIG_SCHEMA:
            raise ValueError("only v2 configuration is supported")
        value["output_root"] = Path(value["output_root"])
        value["log_root"] = Path(value["log_root"])
        curriculum = value["curriculum"].copy()
        curriculum["phases"] = tuple(CurriculumPhaseConfig(**phase) for phase in curriculum.get("phases", ()))
        value["curriculum"] = CurriculumConfig(**curriculum)
        value["evaluation"] = EvaluationConfig(**value["evaluation"])
        value["rewards"] = RewardConfig(**value["rewards"])
        for algorithm, config_type in (("ppo", PPOConfig), ("dqn", DQNConfig), ("tqc", TQCConfig)):
            if value.get(algorithm) is not None:
                value[algorithm] = config_type(**value[algorithm])
        return cls(**value)
