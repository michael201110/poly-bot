"""Versioned, typed configuration shared by CLI, GUI and saved runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from polybot.environment.rewards import RewardConfig, summer_1_reward_config

CONFIG_SCHEMA = "polybot.config.v2"
ARCHITECTURES = {
    "tiny": (64, 64), "compact": (128, 128), "standard": (256, 256),
    # Match the current TQC actor: two 128-wide ReLU layers and tanh-squashed actions.
    "tqc_compatible": (128, 128),
    # Freeze those TQC layers during PPO and learn a linear output residual.
    "tqc_residual": (128, 128),
}


@dataclass(slots=True)
class PPOConfig:
    architecture: str = "compact"
    residual_action_limit: float = 0.1
    residual_progress_start: float = 0.0
    residual_progress_end: float = 1.0
    learning_rate: float = 3e-4
    action_std: float | None = None
    gamma: float = 0.995
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.005
    rollout_steps: int = 512
    batch_size: int = 128
    epochs: int = 5
    teacher_model: str | None = None
    teacher_kl_coefficient: float = 0.0
    imitation_coefficient: float = 0.0
    initial_forward_bias: float = 1.0
    initial_steering_bias: float = 0.5
    target_lap_s: float = 22.0
    target_kl: float = 0.01

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError("unknown PPO architecture")
        if not math.isfinite(self.residual_action_limit) or not 0 <= self.residual_action_limit <= 1:
            raise ValueError("PPO residual action limit must be in [0, 1]")
        if not (
            math.isfinite(self.residual_progress_start)
            and math.isfinite(self.residual_progress_end)
            and 0 <= self.residual_progress_start <= self.residual_progress_end <= 1
        ):
            raise ValueError("PPO residual progress window must be within [0, 1]")
        if self.rollout_steps < 2 or not 2 <= self.batch_size <= self.rollout_steps:
            raise ValueError("invalid PPO rollout or batch size")
        if self.rollout_steps % self.batch_size or self.epochs < 1:
            raise ValueError("PPO batch must divide rollout; epochs must be positive")
        if not 0 < self.gamma <= 1 or not 0 < self.gae_lambda <= 1:
            raise ValueError("invalid PPO discount settings")
        if self.learning_rate <= 0 or self.entropy_coefficient < 0:
            raise ValueError("invalid PPO learning settings")
        if self.action_std is not None and (
            not math.isfinite(self.action_std) or self.action_std <= 0
        ):
            raise ValueError("PPO action standard deviation must be positive and finite")
        if self.teacher_kl_coefficient < 0 or (
            self.teacher_kl_coefficient and not self.teacher_model
        ):
            raise ValueError("teacher KL needs a teacher model and nonnegative coefficient")
        if self.imitation_coefficient < 0:
            raise ValueError("PPO imitation coefficient cannot be negative")
        if min(self.initial_forward_bias, self.initial_steering_bias) < 0:
            raise ValueError("PPO initial action biases cannot be negative")
        if self.target_lap_s < 0:
            raise ValueError("PPO target lap must be nonnegative")
        if self.target_kl <= 0:
            raise ValueError("PPO target KL must be positive")


@dataclass(slots=True)
class TQCConfig:
    architecture: str = "compact"
    learning_rate: float = 3e-4
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
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
    champion_action_drift_limit: float = 0.0
    champion_lap_tolerance_s: float = 0.0
    adaptation_replay_steps: int = 25_000
    adaptation_steering_noise_std: float = 0.01
    adaptation_longitudinal_noise_std: float = 0.01
    adaptation_noise_probability: float = 0.0001
    critic_adaptation_updates: int = 5_000
    actor_polish_block_steps: int = 1_000
    adaptation_max_position_deviation_m: float = 5.0
    adaptation_max_heading_deviation_rad: float = 0.75
    adaptation_max_action_disagreement: float = 0.05

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError("unknown TQC architecture")
        if self.replay_capacity < 1 or self.learning_starts < 0:
            raise ValueError("invalid TQC replay settings")
        if min(self.batch_size, self.train_frequency, self.gradient_steps) < 1:
            raise ValueError("invalid TQC batch or update frequency")
        if self.learning_rate <= 0 or not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("invalid TQC learning settings")
        if any(
            value is not None and (not math.isfinite(value) or value <= 0)
            for value in (self.actor_learning_rate, self.critic_learning_rate)
        ):
            raise ValueError("TQC actor and critic learning rates must be positive and finite")
        if not 0 <= self.warmup_forward_fraction <= 1 or self.warmup_steering_std < 0:
            raise ValueError("invalid TQC warmup settings")
        if not 0 <= self.champion_action_drift_limit <= 2:
            raise ValueError("TQC champion action drift limit must be in [0, 2]")
        if not 0 <= self.champion_lap_tolerance_s <= 10:
            raise ValueError("TQC champion lap tolerance must be in [0, 10] seconds")
        if self.entropy != "auto" and not self.entropy.startswith("auto_"):
            raise ValueError("TQC entropy must be auto or auto_<positive initial value>")
        if self.entropy.startswith("auto_") and float(self.entropy[5:]) <= 0:
            raise ValueError("TQC initial entropy coefficient must be positive")
        if min(self.adaptation_replay_steps, self.critic_adaptation_updates,
               self.actor_polish_block_steps) < 1:
            raise ValueError("TQC adaptation budgets must be positive")
        if not 0 <= self.adaptation_steering_noise_std <= 0.1 or not (
            0 <= self.adaptation_longitudinal_noise_std <= 0.1
        ):
            raise ValueError("TQC local action noise must be in [0, 0.1]")
        if not 0 < self.adaptation_noise_probability <= 1:
            raise ValueError("TQC local action-noise probability must be in (0, 1]")
        if self.adaptation_max_position_deviation_m <= 0 or (
            self.adaptation_max_heading_deviation_rad <= 0
        ):
            raise ValueError("TQC closed-loop deviation limits must be positive")
        if not 0 < self.adaptation_max_action_disagreement <= 2:
            raise ValueError("TQC action-disagreement limit must be in (0, 2]")


@dataclass(slots=True)
class GRTQCConfig(TQCConfig):
    """Gated TQC with an ensemble-quantile disagreement penalty."""

    disagreement_coefficient: float = 0.01
    critic_warmup_updates: int = 10_000
    critic_readiness_window: int = 200
    critic_readiness_relative_change: float = 0.1
    target_lap_s: float = 22.0
    reference_lap_s: float = 24.263
    contact_candidate_lap_tolerance_s: float = 0.5
    exploration_std: float = 0.0001
    critic_collection_std: float = 0.001
    exploration_correlation: float = 0.0
    policy_std_limit: float = 0.0
    critic_exploration_fraction: float = 1.0
    n_step_return: int = 1
    target_entropy: float | str = "auto"
    actor_step_action_limit: float = 1e-5
    actor_reference_drift_limit: float = 0.01
    recovery_critic_cooldown_updates: int = 1000
    actor_evaluation_interval_steps: int = 1000
    recovery_weak_evaluations: int = 3
    screen_actor_evaluations: bool = False
    finish_episode_before_actor_eval: bool = False
    critic_controller_state: bool = False
    critic_environment_state: bool = False
    critic_raw_actions: bool = False
    pace_only_actor_acceptance: bool = False
    actor_verified_state_sampling: bool = False
    learn_from_actor_evaluations: bool = False
    actor_controller_state: bool = False
    controller_adapter_only: bool = False
    critic_mc_initialization_updates: int = 0
    critic_mc_min_episodes: int = 5
    critic_reference_error_limit: float = 0.2

    def __post_init__(self) -> None:
        TQCConfig.__post_init__(self)
        if self.critic_environment_state and not self.critic_controller_state:
            raise ValueError("GRTQC environment state requires controller-state observations")
        if self.actor_controller_state and not self.critic_controller_state:
            raise ValueError("GRTQC actor controller inputs require controller-state observations")
        if self.controller_adapter_only and not self.actor_controller_state:
            raise ValueError("GRTQC controller-only adaptation requires actor controller inputs")
        if self.critic_mc_initialization_updates < 0 or self.critic_mc_min_episodes < 1:
            raise ValueError("GRTQC complete-return initialization needs nonnegative updates and positive episodes")
        if not 0 < self.critic_reference_error_limit <= 1:
            raise ValueError("GRTQC critic reference error limit must be in (0, 1]")
        if not math.isfinite(self.disagreement_coefficient) or self.disagreement_coefficient < 0:
            raise ValueError("GRTQC disagreement coefficient must be nonnegative and finite")
        if self.critic_warmup_updates < 1 or self.critic_readiness_window < 2:
            raise ValueError("GRTQC critic warmup and readiness window must be positive")
        if not 0 < self.critic_readiness_relative_change < 1:
            raise ValueError("GRTQC readiness relative change must be in (0, 1)")
        if self.target_lap_s <= 0:
            raise ValueError("GRTQC target lap must be positive")
        if self.reference_lap_s <= 0:
            raise ValueError("GRTQC reference lap must be positive")
        if not 0 <= self.contact_candidate_lap_tolerance_s <= 10:
            raise ValueError("GRTQC contact-candidate pace tolerance must be in [0, 10] seconds")
        if not 0 <= self.exploration_std <= 0.1:
            raise ValueError("GRTQC exploration standard deviation must be in [0, 0.1]")
        if not 0 <= self.critic_collection_std <= 0.1:
            raise ValueError("GRTQC critic collection standard deviation must be in [0, 0.1]")
        if not 0 <= self.exploration_correlation < 1:
            raise ValueError("GRTQC exploration correlation must be in [0, 1)")
        if not 0 <= self.policy_std_limit <= 1:
            raise ValueError("GRTQC policy standard deviation limit must be in [0, 1]; 0 disables it")
        if not 0 <= self.critic_exploration_fraction <= 1:
            raise ValueError("GRTQC critic exploration fraction must be in [0, 1]")
        if not isinstance(self.n_step_return, int) or not 1 <= self.n_step_return <= self.replay_capacity:
            raise ValueError("GRTQC return horizon must be a positive integer within replay capacity")
        if self.target_entropy != "auto" and not math.isfinite(float(self.target_entropy)):
            raise ValueError("GRTQC target entropy must be auto or finite")
        if self.recovery_critic_cooldown_updates < 1:
            raise ValueError("GRTQC recovery critic cooldown must be positive")
        if self.actor_evaluation_interval_steps < 1 or self.recovery_weak_evaluations < 1:
            raise ValueError("GRTQC actor evaluation interval and recovery count must be positive")
        if not 0 < self.actor_step_action_limit <= 0.1:
            raise ValueError("GRTQC actor step action limit must be in (0, 0.1]")
        if not 0 <= self.actor_reference_drift_limit <= 0.1:
            raise ValueError("GRTQC cumulative reference action limit must be in [0, 0.1]; 0 disables it")


@dataclass(slots=True)
class CurriculumPhaseConfig:
    mode: str
    steps: int
    start_ratio: float | None = None
    end_ratio: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    lead_in_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.mode not in {"full", "section", "quarters-randomised", "timed"}:
            raise ValueError("invalid custom curriculum phase mode")
        if self.steps < 1:
            raise ValueError("custom curriculum phase steps must be positive")
        if not 0 <= self.lead_in_ratio < 1:
            raise ValueError("curriculum lead-in ratio must be in [0, 1)")
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
    lead_in_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.mode not in {"full", "section", "quarters", "quarters-randomised", "q4-full", "timed", "custom"}:
            raise ValueError("unknown curriculum mode")
        if self.mode == "custom" and not self.phases:
            raise ValueError("custom curriculum requires at least one phase")
        if self.mode != "custom" and self.phases:
            raise ValueError("custom phases require custom curriculum mode")
        if not 0 <= self.lead_in_ratio < 1:
            raise ValueError("curriculum lead-in ratio must be in [0, 1)")
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
    algorithm: str = "grtqc"
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
    evaluation: EvaluationConfig = field(default_factory=lambda: EvaluationConfig(episodes=5))
    rewards: RewardConfig = field(default_factory=summer_1_reward_config)
    ppo: PPOConfig | None = None
    tqc: TQCConfig | None = None
    grtqc: GRTQCConfig | None = None

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
        if self.algorithm == "grtqc" and self.evaluation.episodes < 5:
            raise ValueError("GRTQC requires at least five deterministic evaluation laps")

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
        if value.pop("dqn", None) is not None:
            raise ValueError("unsupported algorithm in saved configuration")
        value["output_root"] = Path(value["output_root"])
        value["log_root"] = Path(value["log_root"])
        curriculum = value["curriculum"].copy()
        curriculum["phases"] = tuple(CurriculumPhaseConfig(**phase) for phase in curriculum.get("phases", ()))
        value["curriculum"] = CurriculumConfig(**curriculum)
        value["evaluation"] = EvaluationConfig(**value["evaluation"])
        value["rewards"] = RewardConfig(**value["rewards"])
        for algorithm, config_type in (("ppo", PPOConfig), ("tqc", TQCConfig), ("grtqc", GRTQCConfig)):
            if value.get(algorithm) is not None:
                if algorithm == "ppo" and "pwm_levels" in value[algorithm]:
                    raise ValueError(
                        "legacy PPO pwm_levels is unsupported: PPO now uses continuous Box(2) actions; "
                        "remove pwm_levels and start a fresh PPO model"
                    )
                value[algorithm] = config_type(**value[algorithm])
        return cls(**value)
