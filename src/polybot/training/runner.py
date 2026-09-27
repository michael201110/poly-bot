"""Algorithm neutral training, evaluation, and checkpoint orchestration."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback

from polybot.algorithms.registry import backend_for
from polybot.environment.curriculum import CurriculumPhase, build_plan
from polybot.environment.env import PolyTrackEnv
from polybot.environment.observations import SCHEMA as OBSERVATION_SCHEMA
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import ModelMetadata, ModelRegistry, track_slug
from polybot.training.config import TrainingConfig
from polybot.training.devices import resolve_device
from polybot.training.evaluation import EvaluationResult, evaluate_model
from polybot.training.metrics import EventSink
from polybot.transport import WebSocketServerTransport


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
        self.started = time.monotonic()
        self.previous_wall_seconds = 0.0
        self.device = None
        self.last_evaluation: EvaluationResult | None = None
        self.sink: EventSink | None = None

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
        return PolyTrackEnv(
            self._transport(), track_id=cfg.track_id, lookahead_count=cfg.lookahead_count,
            frame_skip=cfg.frame_skip, max_episode_steps=cfg.max_episode_steps,
            max_episode_s=cfg.max_episode_seconds, reward_config=cfg.rewards,
            action_adapter=self.backend.action_adapter(cfg),
            **(phase.env_kwargs() if phase is not None else {}),
        )

    def _emit(self, event: dict[str, Any]) -> None:
        assert self.sink is not None
        self.sink.emit(event)

    def _metadata(self, evaluation: EvaluationResult | None = None) -> ModelMetadata:
        cfg = self.config
        counts = self.backend.parameter_counts(self.model)
        assert self.device is not None
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
        )

    def _save(self, name: str, evaluation: EvaluationResult | None = None) -> Path:
        cfg = self.config
        directory = self.registry.slot(cfg.track_name, cfg.algorithm, name)
        self.backend.save_model(self.model, directory, resume=name != "champion")
        self.registry.write_metadata(directory, self._metadata(evaluation))
        return directory

    def _evaluate(self) -> EvaluationResult:
        cfg = self.config
        result = evaluate_model(
            self.model, self._environment, episodes=cfg.evaluation.episodes,
            seed=cfg.seed + 1_000_000,
        )
        self.last_evaluation = result
        self._emit({"type": "evaluation", "timesteps": self.model.num_timesteps, **result.to_dict()})
        champion_dir = self.registry.slot(cfg.track_name, cfg.algorithm, "champion")
        champion = None
        if (champion_dir / "metadata.json").is_file():
            previous = self.registry.read_metadata(champion_dir).evaluation
            if previous is not None:
                champion = EvaluationResult(**previous)
        if champion is None or result.rank() > champion.rank():
            path = self._save("champion", result)
            self._emit({"type": "champion", "path": str(path), "timesteps": self.model.num_timesteps})
        return result

    def run(self, *, resume: Path | None = None) -> Path:
        cfg = self.config
        self.device = resolve_device(cfg.device, algorithm=cfg.algorithm)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        log_path = cfg.log_root / f"{track_slug(cfg.track_name)}-{cfg.algorithm}-{stamp}.jsonl"
        self.sink = EventSink(log_path, self.status)
        plan = build_plan(cfg.curriculum, cfg.timesteps)
        self._emit({"type": "plan", "total_steps": plan.total_steps,
                    "phases": [asdict(phase) for phase in plan.phases]})
        first_env = self._environment(plan.phases[0])
        training_env: Any = ScaledTrainingReward(first_env, cfg.reward_scale)
        try:
            if resume is None:
                self.model = self.backend.create_model(cfg, training_env, self.device.resolved)
            else:
                metadata = self.registry.read_metadata(resume)
                self.registry.validate(metadata, cfg, self.backend.action_adapter(cfg).schema)
                if metadata.architecture != self.backend.architecture(cfg):
                    raise ValueError("resume architecture differs from saved model")
                if metadata.training_config["rewards"] != cfg.to_dict()["rewards"]:
                    raise ValueError("resume reward settings differ from saved replay rewards")
                self.model = self.backend.load_model(
                    resume / "policy.zip", training_env, self.device.resolved, resume=True
                )
                self.backend.configure_resume(self.model, cfg, self.device.resolved)
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
            })
            start_steps = self.model.num_timesteps
            next_eval = cfg.evaluation.interval_steps
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
                    runner.ticks += int(info.get("ticks_advanced", 0))
                    now = time.monotonic()
                    if now - self.last_status > 0.5:
                        runner._emit({
                            "type": "progress", "timesteps": self.num_timesteps,
                            "episode": runner.episodes + 1,
                            "progress": self.episode_progress,
                            "run_max_progress": runner.max_progress,
                            "reward": self.episode_reward,
                            "simulator_ticks": runner.ticks,
                            "finishes": runner.finishes,
                            "crashes": runner.crashes,
                            "wall_seconds": runner.previous_wall_seconds + now - runner.started,
                            "steps_per_second": (self.num_timesteps - start_steps)
                            / max(0.001, now - runner.started),
                            "device": runner.device.resolved,
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
                            "elapsed_s": info.get("elapsed_s"),
                            "reward_terms": info.get("reward_terms", {}),
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
                self._emit({"type": "phase", "index": index + 1, **asdict(phase)})
                while self.model.num_timesteps - phase_start < phase.steps:
                    if self.stop_requested.is_set():
                        break
                    consumed = self.model.num_timesteps - start_steps
                    remaining = phase.steps - (self.model.num_timesteps - phase_start)
                    interval = min(next_eval - consumed, next_checkpoint - consumed)
                    chunk = max(1, min(remaining, interval))
                    before = self.model.num_timesteps
                    if cfg.algorithm == "dqn":
                        # DQN's epsilon schedule uses learn()'s total_timesteps. Passing
                        # only the next evaluation chunk exhausts exploration early.
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
                        self._evaluate()
                        last_evaluated_steps = self.model.num_timesteps
                        self._save("latest", self.last_evaluation)
                        next_eval = consumed + cfg.evaluation.interval_steps
                        training_env = ScaledTrainingReward(self._environment(phase), cfg.reward_scale)
                        self.model.set_env(training_env)
            training_env.close()
            if not self.stop_requested.is_set() and self.model.num_timesteps != last_evaluated_steps:
                self._evaluate()
            latest = self._save("latest", self.last_evaluation)
            self._emit({"type": "stopped" if self.stop_requested.is_set() else "completed",
                        "path": str(latest), "timesteps": self.model.num_timesteps})
            return latest
        finally:
            training_env.close()
            self.sink.close()
