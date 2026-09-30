"""Algorithm neutral training, evaluation, and checkpoint orchestration."""

from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from polybot.algorithms.registry import backend_for
from polybot.environment.curriculum import CurriculumPhase, build_plan
from polybot.environment.env import AirBrakeActionWrapper, PolyTrackEnv
from polybot.environment.observations import SCHEMA as OBSERVATION_SCHEMA
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import (
    PPO_ACTION_SEMANTICS,
    REWARD_SEMANTICS,
    ModelMetadata,
    ModelRegistry,
    track_slug,
)
from polybot.training.config import TrainingConfig
from polybot.training.devices import resolve_device
from polybot.training.evaluation import EvaluationResult, evaluate_model
from polybot.training.metrics import EventSink
from polybot.training.pace_history import append_pace_history
from polybot.training.promotion import promote_directory
from polybot.transport import WebSocketServerTransport


def _evaluation_for_current_checkpoint(
    evaluation: EvaluationResult | None, *, current_steps: int, evaluated_steps: int,
) -> EvaluationResult | None:
    """Keep latest metadata honest when a run stops between policy evaluations."""
    return evaluation if evaluation is not None and current_steps == evaluated_steps else None


class ScaledTrainingReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env, scale: float) -> None:
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return float(reward) * self.scale


class TrainingRunner:
    def __init__(
        self, config: TrainingConfig,
        status: Callable[[dict[str, Any]], None] | None = None,
        transport_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config
        self.backend = backend_for(config.algorithm)
        self.registry = ModelRegistry(config.output_root)
        self.status = status
        self.transport_factory = transport_factory
        self.stop_requested = threading.Event()
        self.model: Any = None
        self.ticks = 0
        self.episodes = 0
        self.finishes = 0
        self.crashes = 0
        self.max_progress = 0.0
        self.max_section_progress = 0.0
        self.phase_index = 0
        self.phase_reset_logged = False
        self.started = time.monotonic()
        self.previous_wall_seconds = 0.0
        self.device = None
        self.last_evaluation: EvaluationResult | None = None
        self._champion_refill_source: Path | None = None
        self._champion_refill_updates: int | None = None
        self._champion_path_observations: list[Any] | None = None
        self._consecutive_rollbacks = 0
        self._last_rollback_severe = False
        self.sink: EventSink | None = None
        self._ppo_air_brake_overlays: list[dict[str, Any]] = []
        self._ppo_speed_bias_schedule: list[list[float]] = []

    def stop(self) -> None:
        self.stop_requested.set()

    def _transport(self) -> Any:
        if self.transport_factory is not None:
            return self.transport_factory()
        if self.config.backend == "mock":
            return MockSimulatorTransport()
        return WebSocketServerTransport(connect_timeout_s=300, request_timeout_s=60)

    def _environment(self, phase: CurriculumPhase | None = None) -> PolyTrackEnv:
        cfg = self.config
        env = PolyTrackEnv(
            self._transport(), track_id=cfg.track_id, lookahead_count=cfg.lookahead_count,
            frame_skip=cfg.frame_skip, max_episode_steps=cfg.max_episode_steps,
            max_episode_s=cfg.max_episode_seconds, reward_config=cfg.rewards,
            # A cold PML worker may replay the reference ghost for an entire lap
            # before answering reset; keep that initialization from hitting the
            # short per-step timeout used by local mock environments.
            request_timeout_s=180.0 if cfg.backend == "websocket" else 10.0,
            action_adapter=self.backend.action_adapter(cfg),
            **(phase.env_kwargs() if phase is not None else {}),
        )
        if cfg.algorithm == "ppo" and (
            self._ppo_air_brake_overlays or self._ppo_speed_bias_schedule
        ):
            return AirBrakeActionWrapper(
                env, self._ppo_air_brake_overlays, self._ppo_speed_bias_schedule,
            )
        return env

    def _emit(self, event: dict[str, Any]) -> None:
        assert self.sink is not None
        self.sink.emit(event)

    def _metadata(self, evaluation: EvaluationResult | None = None) -> ModelMetadata:
        cfg = self.config
        counts = self.backend.parameter_counts(self.model)
        assert self.device is not None
        overlays = (
            self._ppo_air_brake_overlays if cfg.algorithm == "ppo"
            else list(getattr(self.model, "policy_overlays", []))
        )
        speed_bias_schedule = (
            self._ppo_speed_bias_schedule if cfg.algorithm == "ppo"
            else list(getattr(self.model, "speed_bias_schedule", []))
        )
        return ModelMetadata(
            algorithm=cfg.algorithm, architecture=self.backend.architecture(cfg),
            actor_parameters=counts["actor"], critic_parameters=counts["critic"],
            total_trainable_parameters=counts["total"],
            observation_schema=OBSERVATION_SCHEMA,
            action_schema=self.backend.action_adapter(cfg).schema,
            track_name=cfg.track_name, track_id=cfg.track_id,
            lookahead_count=cfg.lookahead_count, reward_profile=cfg.reward_profile,
            curriculum=asdict(cfg.curriculum), training_config=cfg.to_dict(),
            training_timesteps=int(self.model.num_timesteps), simulator_ticks=self.ticks,
            wall_seconds=self.previous_wall_seconds + time.monotonic() - self.started,
            seed=cfg.seed, device=self.device.resolved,
            finishes=self.finishes, crashes=self.crashes,
            evaluation=evaluation.to_dict() if evaluation is not None else None,
            implementation="qr_dqn" if cfg.algorithm == "dqn" else None,
            reward_semantics=REWARD_SEMANTICS,
            critic_adaptation_required=bool(getattr(self.model, "critic_adaptation_required", False)),
            adaptation_stage=getattr(self.model, "adaptation_stage", None),
            adaptation_rollback_count=int(getattr(self.model, "adaptation_rollback_count", 0)),
            policy_overlays=list(overlays),
            speed_bias_schedule=list(speed_bias_schedule),
            action_semantics=(
                PPO_ACTION_SEMANTICS if cfg.algorithm in {"ppo", "tqc"}
                else "native_digital_v2"
            ),
        )

    def _save(self, name: str, evaluation: EvaluationResult | None = None) -> Path:
        cfg = self.config
        directory = self.registry.slot(cfg.track_name, cfg.algorithm, name)
        staging = (
            directory.parent / f".{directory.name}-staging-{uuid4().hex}"
            if name == "champion" else directory
        )
        self.backend.save_model(self.model, staging, resume=True)
        self.registry.write_metadata(staging, self._metadata(evaluation))
        if name == "champion":
            search_metadata = directory / "speed-search.json"
            if search_metadata.is_file():
                shutil.copy2(search_metadata, staging / search_metadata.name)
            promote_directory(staging, directory, require_replay=cfg.algorithm in {"tqc", "dqn"})
        return directory

    def _evaluate(self) -> EvaluationResult:
        cfg = self.config
        observations: list[Any] | None = (
            [] if cfg.algorithm == "tqc" and self.model._champion_actor is not None else None
        )
        result = evaluate_model(
            self.model, self._environment, episodes=cfg.evaluation.episodes,
            seed=cfg.seed + 1_000_000,
            observation_sink=observations,
        )
        self.last_evaluation = result
        self._emit({"type": "evaluation", "timesteps": self.model.num_timesteps, **result.to_dict()})
        champion_dir = self.registry.slot(cfg.track_name, cfg.algorithm, "champion")
        champion = None
        champion_reward_mismatch = False
        if (champion_dir / "metadata.json").is_file():
            champion_metadata = self.registry.read_metadata(champion_dir)
            previous = champion_metadata.evaluation
            champion_reward_mismatch = (
                champion_metadata.training_config["rewards"] != cfg.to_dict()["rewards"]
                or (cfg.algorithm == "tqc" and champion_metadata.reward_semantics != REWARD_SEMANTICS)
            )
            if previous is not None:
                champion = EvaluationResult(**previous)
        # Old champions have no replay file. A resumed champion can collect a
        # policy-generated buffer before its first update; keep that buffer with
        # the proven policy so later rollbacks never train from failed attempts.
        if (
            cfg.algorithm == "tqc" and champion is not None
            and self._champion_refill_source == champion_dir
            and (champion_reward_mismatch or not (champion_dir / "replay.pkl").is_file())
            and self.model.num_timesteps >= self.model.learning_starts
            and self.model._n_updates == self._champion_refill_updates
        ):
            if champion_reward_mismatch:
                self._save("champion", champion)
            else:
                self.model.save_replay_buffer(str(champion_dir / "replay.pkl"))
            self._emit({"type": "champion_replay", "timesteps": self.model.num_timesteps})
        refill_in_progress = (
            cfg.algorithm == "tqc"
            and self._champion_refill_source == champion_dir
            and self.model._n_updates == self._champion_refill_updates
            and self.model.num_timesteps < self.model.learning_starts
        )
        if champion is None or (result.rank() > champion.rank() and not refill_in_progress):
            path = self._save("champion", result)
            self._emit({"type": "champion", "path": str(path), "timesteps": self.model.num_timesteps})
            if cfg.algorithm == "tqc":
                append_pace_history(
                    path, source="rl_finetune", evaluation=result, model=self.model,
                    reward_profile=cfg.reward_profile,
                    air_brake_bonus_per_s=cfg.rewards.airborne_brake_bonus_per_s,
                    learning_rate=cfg.tqc.learning_rate if cfg.tqc else None,
                )
            if cfg.algorithm == "tqc" and self.model._champion_actor is not None:
                assert cfg.tqc is not None
                self._champion_path_observations = observations
                self.model.anchor_to_current_policy(
                    cfg.tqc.champion_action_drift_limit, observations
                )
        return result

    def _restore_ppo_champion_if_worse(
        self, result: EvaluationResult, env: Any, *,
        progress_tolerance: float = 0.0, lap_tolerance_s: float = 0.0,
    ) -> bool:
        cfg = self.config
        champion_dir = self.registry.slot(cfg.track_name, "ppo", "champion")
        metadata_path = champion_dir / "metadata.json"
        if not metadata_path.is_file():
            return False
        payload = self.registry.read_metadata(champion_dir)
        if not payload.evaluation:
            return False
        champion_result = EvaluationResult(**{
            name: payload.evaluation[name]
            for name in EvaluationResult.__dataclass_fields__
            if name in payload.evaluation
        })
        if result.rank() >= champion_result.rank():
            return False
        if result.finish_rate >= champion_result.finish_rate:
            progress_regression = (
                result.median_progress < champion_result.median_progress - progress_tolerance
            )
            lap_regression = (
                result.finish_rate > 0
                and result.median_lap_s is not None
                and champion_result.median_lap_s is not None
                and result.median_lap_s > champion_result.median_lap_s + lap_tolerance_s
            )
            if not progress_regression and not lap_regression:
                return False
        current_steps = int(self.model.num_timesteps)
        self.model = self.backend.load_model(
            champion_dir / "policy.zip", env, self.device.resolved, resume=True
        )
        self.model.num_timesteps = current_steps
        self.backend.configure_resume(self.model, cfg, self.device.resolved)
        rng_seed = None
        if cfg.algorithm == "ppo":
            rng_seed = (int(cfg.seed) + current_steps) % (2**32 - 1)
            self.model.set_random_seed(rng_seed)
        self.last_evaluation = champion_result
        self._last_rollback_severe = (
            result.finish_rate < champion_result.finish_rate
            or result.median_progress < champion_result.median_progress - 0.05
        )
        self._emit({
            "type": "ppo_champion_restore", "timesteps": self.model.num_timesteps,
            "candidate_median_lap_s": result.median_lap_s,
            "champion_median_lap_s": champion_result.median_lap_s,
            "candidate_finish_rate": result.finish_rate,
            "champion_finish_rate": champion_result.finish_rate,
            "rng_seed": rng_seed,
        })
        return True

    def _restore_champion_if_worse(
        self, result: EvaluationResult, training_env: Any,
        *, phase_start: int, phase_steps: int,
    ) -> bool:
        cfg = self.config
        champion_dir = self.registry.slot(cfg.track_name, cfg.algorithm, "champion")
        champion_meta = self.registry.read_metadata(champion_dir)
        if champion_meta.evaluation is None:
            return False
        champion = EvaluationResult(**champion_meta.evaluation)
        if result.rank() >= champion.rank():
            return False
        if (
            cfg.algorithm == "tqc" and cfg.tqc is not None
            and cfg.tqc.champion_lap_tolerance_s > 0
            and result.finish_rate == champion.finish_rate == 1.0
            and result.median_progress == champion.median_progress == 1.0
            and result.median_lap_s is not None
            and champion.median_lap_s is not None
            and result.median_lap_s <= champion.median_lap_s + cfg.tqc.champion_lap_tolerance_s
        ):
            self._emit({
                "type": "lap_tolerance", "timesteps": self.model.num_timesteps,
                "evaluated_lap_s": result.median_lap_s,
                "champion_lap_s": champion.median_lap_s,
                "tolerance_s": cfg.tqc.champion_lap_tolerance_s,
            })
            return False
        if (
            cfg.algorithm == "tqc"
            and self._champion_refill_source == champion_dir
            and self.model._n_updates == self._champion_refill_updates
        ):
            # The policy is still byte-for-byte the champion during refill.
            # Rewinding it would only discard clean new-reward replay.
            return False
        self.registry.validate(champion_meta, cfg, self.backend.action_adapter(cfg).schema)
        if champion_meta.architecture != self.backend.architecture(cfg):
            raise ValueError("champion architecture differs from current model")
        reward_mismatch = (
            champion_meta.training_config["rewards"] != cfg.to_dict()["rewards"]
            or (cfg.algorithm == "tqc" and champion_meta.reward_semantics != REWARD_SEMANTICS)
        )

        current_steps = int(self.model.num_timesteps)
        previous_drift = getattr(self.model, "_anchor_action_drift", None)
        replay_file = champion_dir / "replay.pkl"
        refill_replay = cfg.algorithm in {"dqn", "tqc"} and (
            reward_mismatch or not replay_file.is_file()
        )
        if reward_mismatch and not refill_replay:
            raise ValueError("champion reward settings differ from current replay rewards")
        restored = self.backend.load_model(
            champion_dir / "policy.zip", training_env, self.device.resolved,
            resume=not refill_replay,
        )
        if cfg.algorithm == "tqc":
            restored.policy_overlays = list(champion_meta.policy_overlays)
        restored.num_timesteps = current_steps
        self.backend.configure_resume(
            restored, cfg, self.device.resolved, fresh_replay=refill_replay
        )
        if cfg.algorithm == "tqc":
            assert cfg.tqc is not None
            restored.anchor_to_current_policy(
                cfg.tqc.champion_action_drift_limit, self._champion_path_observations
            )
        if cfg.algorithm == "dqn":
            self.backend.begin_phase(restored, cfg, phase_steps)
            self.backend.advance_phase(restored, current_steps - phase_start)
        self.model = restored
        # The poor result belongs to the discarded policy. The restored policy
        # has not been evaluated at this step, so latest must not claim its score.
        self.last_evaluation = None
        self._last_rollback_severe = (
            result.finish_rate < champion.finish_rate
            or result.median_progress < champion.median_progress - 0.05
        )
        self._emit({
            "type": "rollback", "timesteps": current_steps,
            "champion_timesteps": champion_meta.training_timesteps,
            "evaluated_progress": result.median_progress,
            "champion_progress": champion.median_progress,
            "evaluated_finish_rate": result.finish_rate,
            "champion_finish_rate": champion.finish_rate,
            "evaluated_lap_s": result.median_lap_s,
            "champion_lap_s": champion.median_lap_s,
            "severe": self._last_rollback_severe,
            "replay_source": "fresh" if refill_replay else "champion",
            "learning_rate": cfg.tqc.learning_rate if cfg.tqc else None,
            "actor_drift": previous_drift,
            "rollback_count": self._consecutive_rollbacks + 1,
        })
        return True

    def run(
        self, *, resume: Path | None = None, fresh_replay: bool = False,
        rollback_to_champion: bool = False, pace_polish: bool = False,
        freeze_ppo_actor: bool = False,
        ppo_rollback_progress_tolerance: float = 0.0,
        ppo_rollback_lap_tolerance_s: float = 0.0,
        allow_ppo_reward_change: bool = False,
    ) -> Path:
        cfg = self.config
        if freeze_ppo_actor and cfg.algorithm != "ppo":
            raise ValueError("actor-frozen value warmup is only supported for PPO")
        if ppo_rollback_progress_tolerance < 0 or ppo_rollback_lap_tolerance_s < 0:
            raise ValueError("PPO rollback tolerances cannot be negative")
        if resume is not None and cfg.algorithm == "tqc":
            resume_metadata = self.registry.read_metadata(resume)
            if resume_metadata.critic_adaptation_required:
                raise ValueError(
                    "this tuned TQC checkpoint requires critic adaptation before normal continuation"
                )
        if pace_polish:
            if cfg.algorithm != "tqc" or cfg.curriculum.mode != "full":
                raise ValueError("pace polish requires TQC and full-track training")
            champion = self.registry.slot(cfg.track_name, "tqc", "champion")
            if resume is not None and resume.resolve() != champion.resolve():
                raise ValueError("pace polish must resume the champion, not latest")
            resume = champion
            if not (champion / "metadata.json").is_file():
                raise FileNotFoundError("pace polish requires an evaluated champion")
            if self.registry.read_metadata(champion).critic_adaptation_required:
                raise ValueError("tuned champion must complete critic adaptation before pace polish")
            rollback_to_champion = True
        self.device = resolve_device(cfg.device, algorithm=cfg.algorithm)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        log_path = cfg.log_root / f"{track_slug(cfg.track_name)}-{cfg.algorithm}-{stamp}.jsonl"
        self.sink = EventSink(log_path, self.status)
        plan = build_plan(cfg.curriculum, cfg.timesteps)
        self._emit({"type": "plan", "total_steps": plan.total_steps,
                    "phases": [asdict(phase) for phase in plan.phases]})
        first_env = self._environment(plan.phases[0])
        training_env: Any = ScaledTrainingReward(first_env, cfg.reward_scale)
        resume_rng_seed: int | None = None
        try:
            if resume is None:
                if fresh_replay:
                    raise ValueError("fresh replay requires a saved model")
                self.model = self.backend.create_model(cfg, training_env, self.device.resolved)
            else:
                metadata = self.registry.read_metadata(resume)
                self.registry.validate(metadata, cfg, self.backend.action_adapter(cfg).schema)
                if metadata.architecture != self.backend.architecture(cfg):
                    raise ValueError("resume architecture differs from saved model")
                reward_changed = (
                    metadata.training_config["rewards"] != cfg.to_dict()["rewards"]
                    or (cfg.algorithm == "tqc" and metadata.reward_semantics != REWARD_SEMANTICS)
                )
                if reward_changed and not fresh_replay and not (
                    allow_ppo_reward_change and cfg.algorithm == "ppo"
                ):
                    raise ValueError("resume reward settings differ from saved replay rewards")
                if fresh_replay and cfg.algorithm not in {"dqn", "tqc"}:
                    raise ValueError("fresh replay applies only to DQN and TQC")
                self.model = self.backend.load_model(
                    resume / "policy.zip", training_env, self.device.resolved,
                    resume=not fresh_replay,
                )
                if cfg.algorithm == "tqc":
                    self.model.policy_overlays = list(metadata.policy_overlays)
                elif cfg.algorithm == "ppo":
                    self._ppo_air_brake_overlays = list(metadata.policy_overlays)
                    self._ppo_speed_bias_schedule = list(metadata.speed_bias_schedule)
                    self.model.policy_overlays = list(self._ppo_air_brake_overlays)
                    self.model.speed_bias_schedule = list(self._ppo_speed_bias_schedule)
                    if self._ppo_air_brake_overlays or self._ppo_speed_bias_schedule:
                        training_env.close()
                        training_env = ScaledTrainingReward(
                            self._environment(plan.phases[0]), cfg.reward_scale
                        )
                        self.model.set_env(training_env)
                self.backend.configure_resume(
                    self.model, cfg, self.device.resolved, fresh_replay=fresh_replay
                )
                if reward_changed and allow_ppo_reward_change and cfg.algorithm == "ppo":
                    # PPO has no replay buffer. Discard stale Adam moments after a
                    # deliberate reward-shaping change while keeping the policy/value weights.
                    self.model.policy.optimizer.state.clear()
                    self._emit({
                        "type": "ppo_reward_change_resume",
                        "source_reward_profile": metadata.training_config.get("reward_profile"),
                        "target_reward_profile": cfg.reward_profile,
                        "optimizer_state_reset": True,
                    })
                if cfg.algorithm == "ppo":
                    resume_rng_seed = (
                        int(cfg.seed) + int(self.model.num_timesteps)
                    ) % (2**32 - 1)
                    self.model.set_random_seed(resume_rng_seed)
                if rollback_to_champion and cfg.algorithm == "tqc":
                    assert cfg.tqc is not None
                    self.model.anchor_to_current_policy(cfg.tqc.champion_action_drift_limit)
                if fresh_replay and cfg.algorithm == "tqc" and resume.name == "champion":
                    self._champion_refill_source = resume
                    self._champion_refill_updates = self.model._n_updates
                self.ticks = metadata.simulator_ticks
                self.finishes = metadata.finishes
                self.crashes = metadata.crashes
                self.previous_wall_seconds = metadata.wall_seconds
                self.started = time.monotonic()
            self._emit({
                "type": "started", "algorithm": cfg.algorithm, "device": self.device.resolved,
                "device_reason": self.device.diagnostics.get("selection_reason"),
                "gpu_name": self.device.gpu_name,
                "parameters": self.backend.parameter_counts(self.model),
                "log": str(log_path),
                "resume_source": str(resume) if resume is not None else None,
                "resume_rng_seed": resume_rng_seed,
                "fresh_replay": fresh_replay,
                "rollback_on_regression": rollback_to_champion,
                "mode": "pace_polish" if pace_polish else "training",
                "learning_rate": cfg.tqc.learning_rate if cfg.tqc else None,
                "replay_size": self.model.replay_buffer.size()
                if getattr(self.model, "replay_buffer", None) else None,
            })
            if freeze_ppo_actor:
                actor_modules = (
                    self.model.policy.mlp_extractor.policy_net,
                    self.model.policy.action_net,
                )
                for module in actor_modules:
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
                self.model.policy.log_std.requires_grad_(False)
                self._emit({"type": "value_warmup", "actor_frozen": True})
            if (
                resume is not None and rollback_to_champion and cfg.algorithm == "tqc"
                and cfg.tqc is not None and cfg.tqc.champion_action_drift_limit > 0
            ):
                training_env.close()
                observations = []
                baseline = evaluate_model(
                    self.model, self._environment, episodes=1,
                    seed=cfg.seed + 1_000_000, observation_sink=observations,
                )
                self._champion_path_observations = observations
                self._emit({
                    "type": "anchor_baseline", "timesteps": self.model.num_timesteps,
                    "finish_rate": baseline.finish_rate,
                    "median_progress": baseline.median_progress,
                    "observations": len(observations),
                })
                training_env = ScaledTrainingReward(self._environment(plan.phases[0]), cfg.reward_scale)
                self.model.set_env(training_env)
                self.model.anchor_to_current_policy(
                    cfg.tqc.champion_action_drift_limit, observations
                )
            start_steps = self.model.num_timesteps
            # A fast champion can diverge after very few PWM decisions even
            # under a tight continuous-action anchor. Check the first 1,000
            # environment decisions before settling into the configured interval.
            next_eval = min(1_000, cfg.evaluation.interval_steps) if pace_polish else cfg.evaluation.interval_steps
            next_checkpoint = cfg.checkpoint_interval if cfg.checkpoint_interval else cfg.timesteps + 1
            last_evaluated_steps = -1
            runner = self

            class Callback(BaseCallback):
                def __init__(self, stop_at: int | None = None) -> None:
                    super().__init__()
                    self.stop_at = stop_at
                    self.last_status = 0.0
                    self.episode_reward = 0.0
                    self.episode_progress = 0.0
                    self.last_info: dict[str, Any] = {}

                def _on_step(self) -> bool:
                    info = self.locals.get("infos", [{}])[-1]
                    self.last_info = info
                    self.episode_reward += float(self.locals.get("rewards", [0])[-1]) / cfg.reward_scale
                    self.episode_progress = max(
                        self.episode_progress,
                        float(info.get("route_progress_m", 0))
                        / max(1.0, float(info.get("track_length_m", 1))),
                    )
                    runner.max_progress = max(runner.max_progress, self.episode_progress)
                    section_progress = info.get("section_progress")
                    if section_progress is not None:
                        runner.max_section_progress = max(
                            runner.max_section_progress, float(section_progress)
                        )
                    runner.ticks += int(info.get("ticks_advanced", 0))
                    reset_diagnostics = info.get("curriculum_reset_diagnostics")
                    if reset_diagnostics and not runner.phase_reset_logged:
                        runner._emit({
                            "type": "curriculum_reset", "phase": runner.phase_index,
                            "mode": phase.mode, "spawn_ratio": phase.spawn_ratio,
                            "start_ratio": phase.start_ratio, "end_ratio": phase.end_ratio,
                            "action_set": cfg.dqn.action_set if cfg.dqn else None,
                            "epsilon": getattr(runner.model, "exploration_rate", None),
                            "replay_size": (
                                runner.model.replay_buffer.size()
                                if cfg.algorithm == "dqn" else None
                            ),
                            **reset_diagnostics,
                        })
                        runner.phase_reset_logged = True
                    actions = self.locals.get("actions")
                    if cfg.algorithm == "dqn" and actions is not None:
                        try:
                            runner.phase_actions_seen.add(int(actions[0]))
                        except (TypeError, ValueError, IndexError):
                            pass
                    if cfg.algorithm == "dqn":
                        runner.backend.advance_phase(
                            runner.model, self.num_timesteps - phase_start
                        )
                    now = time.monotonic()
                    if now - self.last_status > 0.5:
                        runner._emit({
                            "type": "progress", "timesteps": self.num_timesteps,
                            "episode": runner.episodes + 1,
                            "progress": self.episode_progress,
                            "run_max_progress": runner.max_progress,
                            "section_progress": section_progress,
                            "run_max_section_progress": runner.max_section_progress,
                            "curriculum_stage": info.get("curriculum_stage", "full track"),
                            "reward": self.episode_reward,
                            "simulator_ticks": runner.ticks,
                            "finishes": runner.finishes,
                            "crashes": runner.crashes,
                            "wall_seconds": runner.previous_wall_seconds + now - runner.started,
                            "steps_per_second": (self.num_timesteps - start_steps)
                            / max(0.001, now - runner.started),
                            "device": runner.device.resolved,
                            "learning_rate": cfg.tqc.learning_rate if cfg.tqc else None,
                            "air_brake_bonus_per_s": cfg.rewards.airborne_brake_bonus_per_s,
                            "speed_bias_schedule": getattr(runner.model, "speed_bias_schedule", []),
                            "rollback_count": runner._consecutive_rollbacks,
                            **runner.backend.metrics(runner.model),
                        })
                        self.last_status = now
                    if bool(self.locals.get("dones", [False])[-1]):
                        events = set(info.get("events", ()))
                        runner.episodes += 1
                        runner.finishes += int("finish" in events)
                        runner.crashes += int("crash" in events or "barrier_contact" in events)
                        runner._emit({
                            "type": "episode", "episode": runner.episodes,
                            "timesteps": self.num_timesteps, "reward": self.episode_reward,
                            "progress": self.episode_progress, "events": sorted(events),
                            "section_progress": info.get("section_progress"),
                            "elapsed_s": info.get("elapsed_s"),
                            "reward_terms": info.get("reward_terms", {}),
                            "air_brake_summary": info.get("air_brake_summary", {}),
                        })
                        self.episode_reward = 0.0
                        self.episode_progress = 0.0
                    return not runner.stop_requested.is_set() and (
                        self.stop_at is None or self.num_timesteps < self.stop_at
                    )

            for index, phase in enumerate(plan.phases):
                if self.stop_requested.is_set():
                    break
                if index:
                    training_env.close()
                    training_env = ScaledTrainingReward(self._environment(phase), cfg.reward_scale)
                    self.model.set_env(training_env)
                phase_start = self.model.num_timesteps
                self.phase_index = index + 1
                self.phase_reset_logged = False
                self.phase_actions_seen: set[int] = set()
                self.max_section_progress = 0.0
                if cfg.algorithm == "dqn":
                    self.backend.begin_phase(self.model, cfg, phase.steps)
                phase_event = {"type": "phase", "index": index + 1, **asdict(phase)}
                if cfg.algorithm == "dqn":
                    phase_event.update({
                        "initial_epsilon": self.model.exploration_rate,
                        "replay_size": self.model.replay_buffer.size(),
                        "action_set": cfg.dqn.action_set if cfg.dqn else None,
                    })
                self._emit(phase_event)
                while self.model.num_timesteps - phase_start < phase.steps:
                    if self.stop_requested.is_set():
                        break
                    consumed = self.model.num_timesteps - start_steps
                    remaining = phase.steps - (self.model.num_timesteps - phase_start)
                    interval = min(next_eval - consumed, next_checkpoint - consumed)
                    chunk = max(1, min(remaining, interval))
                    before = self.model.num_timesteps
                    if cfg.algorithm == "dqn":
                        # Keep SB3's global progress horizon stable across eval/checkpoint
                        # chunks. PhaseExplorationSchedule advances independently per step.
                        self.model.learn(
                            cfg.timesteps - consumed,
                            callback=Callback(stop_at=before + chunk),
                            reset_num_timesteps=False,
                        )
                    else:
                        self.model.learn(chunk, callback=Callback(), reset_num_timesteps=False)
                    if self.model.num_timesteps == before:
                        break
                    consumed = self.model.num_timesteps - start_steps
                    if consumed >= next_checkpoint:
                        path = self._save(f"checkpoints/step-{self.model.num_timesteps}")
                        self._emit({"type": "checkpoint", "path": str(path),
                                    "timesteps": self.model.num_timesteps})
                        next_checkpoint = consumed + cfg.checkpoint_interval
                    if consumed >= next_eval and not self.stop_requested.is_set():
                        training_env.close()
                        result = self._evaluate()
                        last_evaluated_steps = self.model.num_timesteps
                        next_eval = consumed + cfg.evaluation.interval_steps
                        training_env = ScaledTrainingReward(self._environment(phase), cfg.reward_scale)
                        ppo_restored = (
                            rollback_to_champion and cfg.algorithm == "ppo"
                            and self._restore_ppo_champion_if_worse(
                                result, training_env,
                                progress_tolerance=ppo_rollback_progress_tolerance,
                                lap_tolerance_s=ppo_rollback_lap_tolerance_s,
                            )
                        )
                        restored = rollback_to_champion and self._restore_champion_if_worse(
                            result, training_env, phase_start=phase_start,
                            phase_steps=phase.steps,
                        ) if cfg.algorithm == "tqc" else ppo_restored
                        if restored:
                            self._consecutive_rollbacks = (
                                self._consecutive_rollbacks + 1
                                if self._last_rollback_severe else 0
                            )
                            if self._consecutive_rollbacks >= (1 if pace_polish else 3):
                                self._emit({
                                    "type": "regression_stop",
                                    "timesteps": self.model.num_timesteps,
                                    "consecutive_rollbacks": self._consecutive_rollbacks,
                                })
                                self.stop_requested.set()
                        else:
                            self._consecutive_rollbacks = 0
                        if not restored:
                            self.model.set_env(training_env)
                        self._save("latest", self.last_evaluation)
                        if (
                            cfg.algorithm == "ppo" and cfg.ppo is not None
                            and result.confirms_target_lap(cfg.ppo.target_lap_s)
                        ):
                            self._emit({
                                "type": "target_reached", "algorithm": "ppo",
                                "target_lap_s": cfg.ppo.target_lap_s,
                                "confirmed_lap_s": result.best_lap_s,
                                "timesteps": self.model.num_timesteps,
                            })
                            self.stop_requested.set()
                if cfg.algorithm == "dqn":
                    self._emit({
                        "type": "phase_summary", "index": index + 1,
                        "epsilon": self.model.exploration_rate,
                        "actions_seen": sorted(getattr(self, "phase_actions_seen", set())),
                        "replay_size": self.model.replay_buffer.size(),
                        "timesteps": self.model.num_timesteps,
                    })
            training_env.close()
            if not self.stop_requested.is_set() and self.model.num_timesteps != last_evaluated_steps:
                result = self._evaluate()
                last_evaluated_steps = self.model.num_timesteps
                if rollback_to_champion:
                    if cfg.algorithm == "tqc":
                        self._restore_champion_if_worse(
                            result, training_env, phase_start=phase_start,
                            phase_steps=phase.steps,
                        )
                    elif cfg.algorithm == "ppo":
                        self._restore_ppo_champion_if_worse(
                            result, training_env,
                            progress_tolerance=ppo_rollback_progress_tolerance,
                            lap_tolerance_s=ppo_rollback_lap_tolerance_s,
                        )
                if (
                    cfg.algorithm == "ppo" and cfg.ppo is not None
                    and result.confirms_target_lap(cfg.ppo.target_lap_s)
                ):
                    self._emit({
                        "type": "target_reached", "algorithm": "ppo",
                        "target_lap_s": cfg.ppo.target_lap_s,
                        "confirmed_lap_s": result.best_lap_s,
                        "timesteps": self.model.num_timesteps,
                    })
            latest_evaluation = _evaluation_for_current_checkpoint(
                self.last_evaluation,
                current_steps=int(self.model.num_timesteps),
                evaluated_steps=last_evaluated_steps,
            )
            latest = self._save("latest", latest_evaluation)
            self._emit({"type": "stopped" if self.stop_requested.is_set() else "completed",
                        "path": str(latest), "timesteps": self.model.num_timesteps})
            return latest
        finally:
            training_env.close()
            self.sink.close()
