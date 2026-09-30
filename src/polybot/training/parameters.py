"""Plain language help shared by GUI, CLI and parameter documentation."""

from __future__ import annotations

from dataclasses import dataclass, fields

from polybot.environment.rewards import RewardConfig
from polybot.training.config import CurriculumConfig, DQNConfig, EvaluationConfig, PPOConfig, TQCConfig


@dataclass(frozen=True, slots=True)
class ParameterInfo:
    label: str
    description: str
    category: str
    advanced: bool = False
    unit: str = ""
    algorithm: str | None = None


def _info(category: str, values: dict[str, str], *, algorithm: str | None = None) -> dict[str, ParameterInfo]:
    return {
        key: ParameterInfo(key.replace("_", " ").capitalize(), description, category,
                           algorithm=algorithm)
        for key, description in values.items()
    }


GENERAL_INFO = _info("General", {
    "track_name": "Name used to group model files. Choose a separate name for each track.",
    "track_id": "Simulator track identifier. 'current' uses the track open in PolyTrack.",
    "backend": "Mock is a fast local test track; WebSocket connects to PolyTrack in your browser.",
    "algorithm": (
        "PPO uses fresh continuous-control rollouts; DQN uses QR-DQN with native digital actions and replay; "
        "TQC learns continuous controls from replay."
    ),
    "device": "Auto tries CUDA and explains a CPU fallback. PPO often runs well on CPU.",
    "seed": "Starting number for repeatable exploration and simulator resets.",
    "frame_skip": "Physics ticks per policy decision. More ticks improve throughput but slow reactions.",
    "timesteps": "Total environment decisions to train across all curriculum phases.",
    "max_episode_seconds": "End an attempt after this many simulated seconds.",
    "max_episode_steps": "Safety cap on decisions in one attempt, even if the time limit is longer.",
    "lookahead_count": "Number of future road samples in each observation. Changes model shape.",
    "reward_profile": "Named reward recipe. The exact resolved coefficients remain available below.",
    "reward_scale": "Multiplies rewards before neural-network training; raw diagnostics stay unchanged.",
    "checkpoint_interval": "Save a resumable snapshot after this many training decisions. Zero disables it.",
    "output_root": "Folder for v2 latest, champion and checkpoint models.",
    "log_root": "Folder for structured JSONL training events.",
})

PPO_INFO = _info("PPO", {
    "architecture": (
        "Network size. tqc_compatible matches the 128×128 TQC actor; tqc_residual freezes that "
        "actor and trains a linear correction head, keeping the TQC response smooth and stable. "
        "so a TQC actor can be copied exactly into PPO."
    ),
    "action_std": "Fixed exploration noise during PPO fine-tuning; blank keeps the learned standard deviation.",
    "residual_action_limit": (
        "Maximum correction added to the frozen TQC mean for tqc_residual. A smaller limit "
        "protects teacher behavior; increase it gradually to search for pace."
    ),
    "learning_rate": "Size of each network update. Around 0.0001–0.0003 is a common starting range.",
    "gamma": "How much future reward matters. Higher values look farther ahead.",
    "gae_lambda": "Balances smooth long-term advantage estimates against short-term accuracy.",
    "entropy_coefficient": "Rewards exploration. Higher values keep actions more random for longer.",
    "rollout_steps": "Fresh decisions PPO gathers before an update; it discards old rollout data.",
    "batch_size": "Decisions per optimizer minibatch. Must divide rollout steps exactly.",
    "epochs": "Number of passes over each fresh rollout. More passes cost time and may overfit.",
    "teacher_model": "Optional fixed PPO teacher checkpoint for a continuous Gaussian policy anchor.",
    "teacher_kl_coefficient": "How strongly PPO stays near the teacher's action distribution.",
    "imitation_coefficient": "Weight on matching ghost actions when the car is near its position.",
    "initial_forward_bias": "Starting PPO preference for throttle and no brake; learning can override it.",
    "initial_steering_bias": "Reduces initial steering exploration noise; PPO still outputs continuous steering.",
    "target_lap_s": "Stop training after evaluation confirms a lap faster than this time; zero disables the target.",
    "target_kl": "Stop a PPO update when its approximate KL drift exceeds 1.5 times this limit.",
}, algorithm="ppo")

DQN_INFO = _info("DQN", {
    "architecture": (
        "Q-network width. yosh_2020 uses 64 then 16 hidden units from Yosh's older "
        "Trackmania model; PolyBot inputs differ."
    ),
    "action_set": (
        "Full has nine digital actions; no_brake has six coast/throttle actions "
        "for an initial DQN learning stage."
    ),
    "n_quantiles": "Number of return quantiles QR-DQN predicts per action. More can help but slow updates.",
    "learning_rate": "Size of Q-network weight updates. Around 0.0001 is a cautious starting point.",
    "replay_capacity": "Maximum past decisions kept for reuse; larger history costs more memory.",
    "learning_starts": "Number of digital driving decisions collected before Q-network updates begin.",
    "batch_size": "Stored transitions sampled per Q-network update. Larger batches cost more compute.",
    "gamma": "How much future reward contributes to each action's Q-value.",
    "train_frequency": (
        "Environment decisions collected before a training round; higher values reduce update frequency."
    ),
    "gradient_steps": "Q-network updates per training round. More updates use more compute and can overfit replay.",
    "target_update_interval": (
        "Environment steps between copies to the target Q-network; "
        "very short intervals can destabilize targets."
    ),
    "exploration_fraction": (
        "Fraction of the training budget spent reducing random-action probability "
        "from initial to final epsilon."
    ),
    "exploration_initial_eps": "Probability of a random digital action at the start of training; 1 means fully random.",
    "exploration_final_eps": "Minimum random-action probability after the exploration schedule ends.",
}, algorithm="dqn")

TQC_INFO = _info("TQC", {
    "architecture": "Actor and critic network width. Standard 256×256 may train slowly on a T500.",
    "learning_rate": "Size of each actor and critic update. Around 0.0001–0.0003 is common.",
    "actor_learning_rate": (
        "Optional independent actor LR; null uses learning_rate. Tuned-champion polish should be microscopic."
    ),
    "critic_learning_rate": "Optional independent critic LR; null uses learning_rate. Used for TQC critic adaptation.",
    "replay_capacity": "Past decisions stored for reuse. More history costs RAM and can retain old behaviour.",
    "learning_starts": "Collect this many experiences before gradient updates begin.",
    "batch_size": "Past experiences per update. 128–512 is a useful range; larger uses more compute.",
    "gamma": "How much future reward matters. Very high values make distant outcomes influential.",
    "tau": "How quickly target critics follow the main critics; smaller is smoother.",
    "train_frequency": "Environment decisions collected per update. 2–4 can improve TPS on slower GPUs.",
    "gradient_steps": "Updates per training event. More costs compute and can destabilize critics.",
    "entropy": "Automatic exploration coefficient, such as auto_0.1. Higher initial values explore more.",
    "warmup_forward_fraction": "Fraction of seeded warmup actions biased toward forward throttle; expires completely.",
    "warmup_steering_std": "Steering variation during warmup. Larger values explore wider turns.",
    "champion_action_drift_limit": (
        "On best-model continuation, cap deterministic action drift from the starting champion "
        "on saved replay states. 0 disables the cap; 0.03 is conservative."
    ),
    "champion_lap_tolerance_s": (
        "Continue learning after a fully completed evaluation lap this many seconds slower "
        "than champion. The champion stays saved; larger regressions still roll back."
    ),
    "adaptation_replay_steps": (
        "Fresh local transitions collected around the deterministic champion before updating critics."
    ),
    "adaptation_steering_noise_std": (
        "Gaussian steering perturbation in normalized action units; champion remains the center."
    ),
    "adaptation_longitudinal_noise_std": (
        "Gaussian throttle/brake perturbation in normalized action units during replay expansion."
    ),
    "adaptation_noise_probability": (
        "Chance of local action noise at each decision; the tuned profile uses 0.01% "
        "plus a forced sample every 10,000 decisions."
    ),
    "critic_adaptation_updates": "Number of critic-only replay updates in the explicit critic adaptation stage.",
    "actor_polish_block_steps": "Short environment-step budget for optional experimental actor-gradient polishing.",
    "adaptation_max_position_deviation_m": (
        "Maximum candidate-to-reference position deviation before actor polish is rejected."
    ),
    "adaptation_max_heading_deviation_rad": (
        "Maximum candidate-to-reference heading deviation before actor polish is rejected."
    ),
    "adaptation_max_action_disagreement": (
        "Maximum closed-loop action difference per control dimension against the champion."
    ),
}, algorithm="tqc")

CURRICULUM_INFO = _info("Curriculum", {
    "mode": (
        "Full track, one section, sequential quarters, random quarters, Q4 then full, "
        "time window, or custom phases."
    ),
    "start_ratio": "Start position as a fraction of the lap; 0.75 means three quarters in.",
    "end_ratio": "Stop position as a fraction of the lap; reaching it awards the section bonus.",
    "lead_in_ratio": "Spawn this far before a target section so the policy takes over in a moving state.",
    "start_s": "Start time in the reference lap for a timed section.",
    "end_s": "End time in the reference lap for a timed section.",
    "phases": (
        "For custom mode, enter a JSON list of phases with mode, positive steps and any section bounds. "
        "Phase steps must sum to the total budget."
    ),
})
EVALUATION_INFO = _info("Evaluation", {
    "interval_steps": "Pause training after this many decisions to test a frozen policy deterministically.",
    "episodes": "Number of identical seeded attempts per evaluation; more reduces lucky results.",
})

REWARD_DESCRIPTIONS = {
    "progress_per_m": "Points added for each new metre of forward route progress.",
    "elapsed_cost_per_s": "Points per simulated second; a more negative value urges faster laps.",
    "on_track_speed_per_m": "Points per metre travelled while centered and facing forward.",
    "speed_pace_reward_per_m_per_mps": "Extra points per metre and metre per second of safe forward speed.",
    "speed_pace_limit_mps": "Maximum speed counted in the pace term, in metres per second.",
    "airborne_speed_per_m": "Points per metre of forward flight, reduced when the car tilts.",
    "airborne_brake_bonus_per_s": (
        "Rewards applied brake duty only while all four wheels are airborne. "
        "PolyTrack can gain lap time from braking during jumps."
    ),
    "ground_brake_penalty_per_s": (
        "Penalty for applied brake duty when any wheel touches the ground; "
        "it never applies to four-wheel air braking."
    ),
    "takeoff_target_speed_mps": "Takeoff speed used as the zero point for the jump speed bonus.",
    "takeoff_speed_reward_per_mps": "Points for each metre per second above or below target takeoff speed.",
    "takeoff_speed_reward_limit": "Maximum absolute takeoff speed reward for one jump.",
    "imitation_bonus_per_s": "Maximum points per second for staying near the ghost position and orientation.",
    "imitation_position_scale_m": "Metres of position error that noticeably reduce ghost imitation credit.",
    "imitation_rotation_scale_rad": "Radians of angle error that noticeably reduce ghost imitation credit.",
    "expert_action_bonus_per_s": "Maximum points per second for matching ghost steering and pedal demand.",
    "ghost_speed_bonus_per_s": "Maximum points per second for matching the ghost's forward speed.",
    "ghost_speed_scale_mps": "Metres per second of speed error that noticeably reduce ghost speed credit.",
    "guidance_reward_scale": "Multiplier on all ghost guidance terms; zero disables their reward.",
    "guidance_min_forward_speed_mps": "Minimum forward speed before ghost guidance is rewarded.",
    "guidance_min_on_track_factor": "Minimum centered and aligned score before ghost guidance is rewarded.",
    "low_speed_penalty_per_s": "Points per slow second after the grace period; negative discourages crawling.",
    "low_speed_grace_s": "Seconds of low speed allowed before its penalty starts.",
    "unsafe_speed_penalty_per_m": "Points per fast metre away from a safe centered line; negative punishes it.",
    "barrier_contact_penalty": "Points applied once when a verified barrier impact ends an attempt.",
    "barrier_early_penalty": "Extra points weighted toward an early barrier impact.",
    "barrier_collision_impulse_threshold": (
        "Minimum untyped impact impulse treated as a barrier contact; landing can also cause impulse."
    ),
    "failure_progress_clawback_per_m": "Points per metre previously earned in an attempt that ends in failure.",
    "failure_early_penalty": "Extra failure points weighted toward early route progress.",
    "off_track_landing_penalty": "Points when the car lands outside the configured corridor.",
    "airborne_spin_penalty_per_rad": "Points per radian of excessive rotation while all wheels are airborne.",
    "airborne_spin_deadzone_radps": "Small yaw and roll rotation rates ignored during a jump.",
    "airborne_pitch_deadzone_radps": "Pitch rate ignored before jump spin cost begins.",
    "airborne_tilt_penalty_per_s": "Points per second of steep airborne pitch beyond tolerance.",
    "airborne_roll_penalty_per_s": "Points per second of airborne roll away from upright.",
    "airborne_pitch_tolerance_rad": "Pitch angle tolerated before airborne tilt costs points.",
    "airborne_roll_limit_rad": "Roll angle considered unsafe when sustained in flight.",
    "airborne_roll_timeout_s": "Seconds of unsafe airborne roll before the attempt ends.",
    "airborne_roll_failure_penalty": "Points added once when sustained airborne roll ends an attempt.",
    "ground_slip_tolerance_rad": "Tyre slip angle allowed with four wheels grounded before cost begins.",
    "ground_slip_penalty_per_rad_s": (
        "Points per extra radian of ground slip each second; more negative punishes sliding."
    ),
    "ground_spin_deadzone_radps": (
        "Ground yaw rate ignored before spin cost begins, in radians per second."
    ),
    "ground_spin_penalty_per_rad_s": (
        "Points per excess grounded yaw radian per second; negative discourages wall spins."
    ),
    "ground_spin_min_grounded_wheels": (
        "Minimum wheel contacts required before grounded spin cost applies; avoids flight penalties."
    ),
    "checkpoint_bonus": "Points awarded for each checkpoint reached.",
    "checkpoint_fast_bonus": "Extra checkpoint points when the split is faster than target.",
    "checkpoint_target_s": "Target seconds per checkpoint used to calculate the fast bonus.",
    "checkpoint_speed_bonus_per_mps": (
        "Extra checkpoint points per metre per second of arrival speed."
    ),
    "checkpoint_speed_bonus_limit_mps": "Highest arrival speed counted for a checkpoint bonus.",
    "finish_bonus": "Points awarded for completing the whole track.",
    "finish_fast_bonus": "Additional points for a quick complete lap.",
    "finish_target_s": "Lap time at or below which the full fast finish bonus applies.",
    "finish_pace_decay_per_s": "How quickly fast finish credit falls per second beyond the target.",
    "curriculum_section_bonus": "Points for reaching the end of a curriculum section.",
    "crash_penalty": "Points applied once when the simulator reports a crash.",
    "stall_penalty": "Points applied once when low speed lasts beyond the stall timeout.",
    "off_track_penalty": "Points applied once for a sustained off-track position.",
    "early_off_track_penalty": "Points for an off-track exit early in an attempt.",
    "action_change_penalty": "Points per unit change in steering and pedal demand between decisions.",
    "max_forward_progress_per_step_m": (
        "Maximum forward metres credited in one decision despite route projection jumps."
    ),
    "max_reverse_progress_per_step_m": (
        "Maximum backward metres charged in one decision despite route projection jumps."
    ),
    "stall_speed_threshold_mps": "Speed below which the stall timer runs, in metres per second.",
    "stall_timeout_s": "Seconds of sustained low speed before ending the attempt.",
    "reference_corridor_scale": "Multiplier on reference half-width used by center and off-track checks.",
    "off_track_lateral_ratio": "Lateral offset divided by corridor width that counts as off-track evidence.",
    "off_track_heading_ratio": "Lateral ratio that can count as off-track when heading error is large.",
    "off_track_heading_rad": "Heading error needed with the heading-based off-track ratio.",
    "off_track_wall_ride_roll_rad": "Bank angle above which lateral distance is ignored for a wall ride.",
    "off_track_wall_ride_min_grounded_wheels": "Minimum grounded wheels for the wall ride bank exception.",
    "off_track_timeout_s": "Seconds of sustained off-track evidence before ending the attempt.",
    "off_track_min_grounded_wheels": "Minimum wheel contacts needed before off-track evidence counts.",
    "landing_grace_s": "Seconds after takeoff or landing when stall and off-track timers pause.",
    "early_run_s": "Simulated seconds considered early for extra failure penalties.",
}


def _reward_category(name: str) -> str:
    if name.startswith(("ghost", "expert", "imitation", "guidance")):
        return "Guidance"
    if name.startswith(("airborne", "takeoff", "off_track_landing", "landing")):
        return "Airborne behaviour"
    if name.startswith(("checkpoint", "finish", "curriculum")):
        return "Milestones"
    if name.startswith(("crash", "stall", "barrier", "failure", "off_track", "early_run")):
        return "Failure"
    if name.startswith(("ground", "action", "unsafe", "low_speed", "reference")):
        return "Driving quality"
    return "Progress"


REWARD_INFO = {
    name: ParameterInfo(
        name.replace("_", " ").capitalize(), description, _reward_category(name),
        advanced=True,
    )
    for name, description in REWARD_DESCRIPTIONS.items()
}

METRIC_INFO = _info("Status", {
    "steps_per_second": "Environment decisions per wall-clock second, including learner updates.",
    "progress": "Fraction of the track reached in the current attempt; a single attempt can vary.",
    "run_max_progress": "Farthest fraction reached in any attempt in this training session.",
    "simulator_ticks": "Total fixed physics updates executed in the simulator.",
    "finishes": "Number of completed training attempts; champion still depends on evaluation.",
    "crashes": "Number of training attempts ending in a crash or barrier impact.",
    "replay_size": "Past DQN or TQC decisions available for reuse. It fills during early learning.",
    "updates": "Number of gradient update rounds applied to the selected replay-based learner.",
    "loss": "DQN error between predicted and bootstrapped Q-values; lower does not necessarily mean better driving.",
    "exploration_rate": "DQN epsilon: probability of a random action instead of the highest-Q action.",
    "entropy_coefficient": "TQC exploration weight, also called alpha; auto mode adjusts it over time.",
    "actor_loss": "Change to TQC's action policy. Lower is not always better driving.",
    "anchor_action_drift": (
        "Largest deterministic control change from the proven champion on its saved driving path."
    ),
    "critic_loss": "Change to TQC's value estimates; spikes can signal instability.",
    "policy_loss": "PPO policy update signal; compare trends with deterministic evaluation.",
    "value_loss": "PPO critic prediction error; it depends strongly on reward scale.",
    "entropy": "PPO action randomness; more randomness usually means more exploration.",
    "explained_variance": "How much PPO's critic explains observed returns; near 1 is better fit.",
    "kl": "How far PPO's updated policy moved from the rollout policy.",
    "clip_fraction": "Share of PPO policy updates limited by its safety clipping range.",
})


def validate_metadata() -> None:
    for config_type, info in (
        (PPOConfig, PPO_INFO), (DQNConfig, DQN_INFO), (TQCConfig, TQC_INFO),
        (CurriculumConfig, CURRICULUM_INFO), (EvaluationConfig, EVALUATION_INFO),
        (RewardConfig, REWARD_INFO),
    ):
        missing = {field.name for field in fields(config_type)} - info.keys()
        if missing:
            raise AssertionError(f"missing {config_type.__name__} help: {sorted(missing)}")
