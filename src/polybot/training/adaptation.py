"""Explicit, checkpointed TQC adaptation for manually or speed-search tuned champions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from polybot.training.config import TrainingConfig
from polybot.training.devices import resolve_device
from polybot.training.evaluation import EvaluationResult, evaluate_model
from polybot.training.promotion import promote_directory
from polybot.training.runner import ScaledTrainingReward, TrainingRunner


def run_adaptation(
    config: TrainingConfig, stage: str = "full",
    status: Any = None,
) -> Path:
    """Run collect, critics, polish, full, or rollback against the saved champion."""
    if config.algorithm != "tqc" or config.curriculum.mode != "full" or config.tqc is None:
        raise ValueError("champion adaptation requires full-track TQC")
    if stage not in {"collect", "validate", "critics", "polish", "full", "rollback"}:
        raise ValueError(f"unknown adaptation stage: {stage}")
    runner = TrainingRunner(config, status=status)
    runner.device = resolve_device(config.device, algorithm="tqc")
    champion = runner.registry.slot(config.track_name, "tqc", "champion")
    champion_metadata = runner.registry.read_metadata(champion)
    if champion_metadata.evaluation is not None:
        runner.last_evaluation = EvaluationResult(**champion_metadata.evaluation)
    work = champion.parent / "adaptation" / "working"
    snapshot = champion.parent / "adaptation" / "polish-base"
    work.parent.mkdir(parents=True, exist_ok=True)
    cfg = config.tqc
    env = ScaledTrainingReward(runner._environment(), config.reward_scale)

    def emit(event: dict[str, Any]) -> None:
        if status:
            status(event)

    def reopen_training_env() -> None:
        nonlocal env
        env = ScaledTrainingReward(runner._environment(), config.reward_scale)

    def load(path: Path, replay: bool) -> Any:
        model = runner.backend.load_model(path / "policy.zip", env, runner.device.resolved, resume=replay)
        saved_metadata = runner.registry.read_metadata(path)
        model.critic_adaptation_required = saved_metadata.critic_adaptation_required
        model.adaptation_stage = saved_metadata.adaptation_stage
        model.adaptation_rollback_count = saved_metadata.adaptation_rollback_count
        model.policy_overlays = list(saved_metadata.policy_overlays)
        if replay:
            runner.backend.configure_resume(model, config, runner.device.resolved)
        else:
            model.actor_lr = cfg.actor_learning_rate or cfg.learning_rate
            model.critic_lr = cfg.critic_learning_rate or cfg.learning_rate
            for optimizer, rate in ((model.actor.optimizer, model.actor_lr),
                                    (model.critic.optimizer, model.critic_lr)):
                for group in optimizer.param_groups:
                    group["lr"] = rate
        return model

    def save(path: Path, required: bool, state: str) -> None:
        runner.model.critic_adaptation_required = required
        runner.model.adaptation_stage = state
        staging = path.parent / f".{path.name}-staging-{uuid4().hex}"
        runner.backend.save_model(runner.model, staging, resume=True)
        runner.registry.write_metadata(staging, runner._metadata(runner.last_evaluation))
        if path == champion:
            old_search = champion / "speed-search.json"
            if old_search.is_file():
                shutil.copy2(old_search, staging / old_search.name)
        if path.exists():
            shutil.rmtree(path)
        promote_directory(staging, path, require_replay=True)

    def collect() -> None:
        metadata = runner.registry.read_metadata(champion)
        runner.model = load(champion, replay=False)
        runner.model.critic_adaptation_required = True
        runner.model.adaptation_stage = "replay_expansion"
        runner.model._adaptation_mode = "replay_expansion"
        runner.model._adaptation_noise = np.asarray((
            cfg.adaptation_steering_noise_std, cfg.adaptation_longitudinal_noise_std
        ), dtype=np.float32)
        runner.model._adaptation_noise_probability = cfg.adaptation_noise_probability
        runner.model._adaptation_sample_steps = 0
        runner.model._adaptation_action_deviation = []
        runner.model.learn(cfg.adaptation_replay_steps, reset_num_timesteps=False, progress_bar=False)
        runner.model._adaptation_mode = None
        mean_action_deviation = float(np.mean(runner.model._adaptation_action_deviation))
        if mean_action_deviation <= 0.0:
            raise RuntimeError("replay expansion collected no local action perturbations")
        # The trainer owns a live websocket listener; close it before opening the
        # independent deterministic evaluation listener on the same local port.
        save(work, required=True, state="replay_expanded")
        env.close()
        evaluation = evaluate_model(
            runner.model, runner._environment, episodes=config.evaluation.episodes,
            seed=config.seed + 2_000_000,
        )
        runner.last_evaluation = evaluation
        save(work, required=True, state="replay_expanded")
        emit({"type": "adaptation_stage", "stage": "replay_expansion",
              "transitions_collected": cfg.adaptation_replay_steps,
              "replay_size": runner.model.replay_buffer.size(),
              "action_noise_std": runner.model._adaptation_noise.tolist(),
              "action_noise_probability": cfg.adaptation_noise_probability,
              "mean_action_deviation": mean_action_deviation,
              **evaluation.to_dict()})
        if evaluation.finish_rate < 1.0:
            raise RuntimeError("replay-expanded champion did not pass deterministic lap validation")
        emit({"type": "adaptation_source", "champion_metadata": metadata.adaptation_stage})
        reopen_training_env()

    def critics() -> None:
        if not (work / "metadata.json").is_file():
            raise FileNotFoundError("collect local champion replay before critic adaptation")
        runner.model = load(work, replay=True)
        runner.model.critic_adaptation_required = True
        runner.model._adaptation_mode = "critic_only"
        for parameter in runner.model.actor.parameters():
            parameter.requires_grad_(False)
        interval = max(1, min(500, cfg.critic_adaptation_updates // 10))
        for index in range(cfg.critic_adaptation_updates):
            stats = runner.model.train_critics(1, cfg.batch_size)
            if (index + 1) % interval == 0:
                emit({"type": "adaptation_critic_progress", "updates": index + 1,
                      "total_updates": cfg.critic_adaptation_updates, **stats})
        for parameter in runner.model.actor.parameters():
            parameter.requires_grad_(True)
        runner.model._adaptation_mode = None
        runner.model.critic_adaptation_required = False
        runner.model.adaptation_stage = "critics_adapted"
        save(work, required=False, state="critics_adapted")
        env.close()
        evaluation = evaluate_model(
            runner.model, runner._environment, episodes=config.evaluation.episodes,
            seed=config.seed + 2_000_000,
        )
        runner.last_evaluation = evaluation
        emit({"type": "adaptation_critic_validation", **evaluation.to_dict()})
        if evaluation.finish_rate < 1.0 or evaluation.median_lap_s is None:
            raise RuntimeError("critic-adapted checkpoint failed deterministic validation; champion retained")
        save(champion, required=False, state="critics_adapted")
        emit({"type": "adaptation_stage", "stage": "critic_adaptation",
              "updates": cfg.critic_adaptation_updates,
              **runner.model._adaptation_diagnostics})
        reopen_training_env()

    def validate_replay() -> None:
        if not (work / "metadata.json").is_file():
            raise FileNotFoundError("collect local champion replay before validation")
        reference = load(champion, replay=False)
        runner.model = load(work, replay=True)
        env.close()
        noisy = evaluate_model(
            runner.model, runner._environment, episodes=config.evaluation.episodes,
            seed=config.seed + 2_500_000, reference_model=reference,
            action_noise_std=(cfg.adaptation_steering_noise_std,
                              cfg.adaptation_longitudinal_noise_std),
            action_noise_probability=cfg.adaptation_noise_probability,
        )
        emit({"type": "adaptation_noisy_validation", **noisy.to_dict()})
        if noisy.finish_rate < 1 or noisy.crash_rate or noisy.off_track_rate or noisy.stall_rate:
            raise RuntimeError("local replay perturbation failed closed-loop lap validation")

    def polish() -> None:
        if cfg.actor_polish_block_steps < 1:
            raise ValueError(
                "actor-gradient polishing is experimental and disabled in this profile; "
                "set actor_polish_block_steps explicitly to enable it"
            )
        metadata = runner.registry.read_metadata(champion)
        if metadata.critic_adaptation_required:
            raise RuntimeError("critic adaptation must finish before actor polish")
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(champion, snapshot)
        reference = load(snapshot, replay=False)
        runner.model = load(champion, replay=True)
        emit({"type": "adaptation_polish_started",
              "actor_learning_rate": runner.model.actor_lr,
              "critic_learning_rate": runner.model.critic_lr,
              "actor_update_block_steps": cfg.actor_polish_block_steps,
              "rollback_count": 0})
        runner.model.learn(cfg.actor_polish_block_steps, reset_num_timesteps=False, progress_bar=False)
        env.close()
        result = evaluate_model(
            runner.model, runner._environment, episodes=config.evaluation.episodes,
            seed=config.seed + 3_000_000, reference_model=reference,
        )
        runner.last_evaluation = result
        safe = (
            result.finish_rate == 1.0 and result.median_progress == 1.0
            and result.crash_rate == 0 and result.off_track_rate == 0 and result.stall_rate == 0
            and result.max_position_deviation_m <= cfg.adaptation_max_position_deviation_m
            and result.max_heading_deviation_rad <= cfg.adaptation_max_heading_deviation_rad
            and result.max_steering_disagreement <= cfg.adaptation_max_action_disagreement
            and result.max_longitudinal_disagreement <= cfg.adaptation_max_action_disagreement
            and result.median_lap_s is not None and metadata.evaluation is not None
            and metadata.evaluation.get("median_lap_s") is not None
            and result.median_lap_s < float(metadata.evaluation["median_lap_s"])
        )
        if safe:
            save(champion, required=False, state="actor_polished")
            emit({"type": "adaptation_polish_accepted", **result.to_dict()})
        else:
            runner.model = load(snapshot, replay=True)
            runner.model.adaptation_rollback_count = metadata.adaptation_rollback_count + 1
            if metadata.evaluation is not None:
                runner.last_evaluation = EvaluationResult(**metadata.evaluation)
            save(champion, required=False, state=metadata.adaptation_stage or "critics_adapted")
            emit({"type": "adaptation_polish_rejected", "reason": "unsafe or no confirmed speed gain",
                  "rollback_count": runner.model.adaptation_rollback_count,
                  **result.to_dict()})

    try:
        if stage == "rollback":
            if not (snapshot / "metadata.json").is_file():
                raise FileNotFoundError("no complete polish-base snapshot exists")
            staging = champion.parent / f".champion-rollback-{uuid4().hex}"
            shutil.copytree(snapshot, staging)
            promote_directory(staging, champion, require_replay=True)
            emit({"type": "adaptation_rollback", "path": str(champion)})
            return champion
        if stage in {"collect", "full"}:
            collect()
        if stage == "validate":
            validate_replay()
        if stage in {"critics", "full"}:
            critics()
        if stage == "polish":
            polish()
        return champion
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("collect", "validate", "critics", "polish", "full", "rollback"),
        default="full",
    )
    args = parser.parse_args()
    config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
    run_adaptation(config, args.stage, lambda event: print(json.dumps(event), flush=True))


if __name__ == "__main__":
    main()
