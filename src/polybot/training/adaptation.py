"""Explicit, checkpointed TQC adaptation for manually or speed-search tuned champions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from polybot.training.config import TrainingConfig
from polybot.training.devices import resolve_device
from polybot.training.evaluation import EvaluationResult, evaluate_model
from polybot.training.pace_history import append_pace_history
from polybot.training.promotion import promote_directory
from polybot.training.runner import ScaledTrainingReward, TrainingRunner


def _policy_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    with (directory / "policy.zip").open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    # Policy overlays are part of the effective champion even though they are
    # stored in metadata rather than baked into policy.zip.
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8-sig"))
    overlays = json.dumps(metadata.get("policy_overlays", []), sort_keys=True, separators=(",", ":"))
    digest.update(overlays.encode("utf-8"))
    return digest.hexdigest()


def candidate_diagnostics_pass(
    candidate: EvaluationResult, reference: EvaluationResult, config: TrainingConfig,
) -> tuple[bool, dict[str, Any]]:
    """Gate adaptation candidates on reliability, closed-loop drift, and lap time."""
    assert config.tqc is not None
    # Prefer same-seed closed-loop lap deltas. Comparing the candidate's laps
    # with champion metadata can mistake ordinary timing variation for policy
    # regression (or improvement), since that metadata may come from another run.
    lap_delta = candidate.lap_time_delta_s
    if lap_delta is None and candidate.median_lap_s is not None and reference.median_lap_s is not None:
        lap_delta = candidate.median_lap_s - reference.median_lap_s
    finite_metrics = (
        candidate.finish_rate, candidate.median_progress, candidate.mean_progress,
        candidate.crash_rate, candidate.off_track_rate, candidate.stall_rate,
        candidate.max_position_deviation_m, candidate.max_progress_deviation_m,
        candidate.max_heading_deviation_rad, candidate.max_steering_disagreement,
        candidate.max_longitudinal_disagreement,
        reference.finish_rate, reference.median_progress, reference.mean_progress,
        reference.crash_rate, reference.off_track_rate, reference.stall_rate,
        reference.max_position_deviation_m, reference.max_progress_deviation_m,
        reference.max_heading_deviation_rad, reference.max_steering_disagreement,
        reference.max_longitudinal_disagreement,
    )
    finite_metrics = finite_metrics + tuple(
        value for value in (candidate.median_lap_s, reference.median_lap_s,
                            candidate.lap_time_delta_s, lap_delta)
        if value is not None
    )
    allowed_lap_delta = min(max(config.tqc.champion_lap_tolerance_s, 0.001), 0.01)
    limits = {
        "position_deviation_limit_m": config.tqc.adaptation_max_position_deviation_m,
        "progress_deviation_limit_m": config.tqc.adaptation_max_position_deviation_m,
        "heading_deviation_limit_rad": config.tqc.adaptation_max_heading_deviation_rad,
        "action_disagreement_limit": config.tqc.adaptation_max_action_disagreement,
        "lap_delta_limit_s": allowed_lap_delta,
    }
    failures = []
    if not all(math.isfinite(value) for value in finite_metrics):
        failures.append("non-finite evaluation metrics")
    if candidate.finish_rate != 1.0 or candidate.median_progress != 1.0:
        failures.append("incomplete laps")
    if candidate.crash_rate or candidate.off_track_rate or candidate.stall_rate:
        failures.append("crash, off-track, or stall")
    if candidate.max_position_deviation_m > limits["position_deviation_limit_m"]:
        failures.append("position drift")
    if candidate.max_progress_deviation_m > limits["progress_deviation_limit_m"]:
        failures.append("progress drift")
    if candidate.max_heading_deviation_rad > limits["heading_deviation_limit_rad"]:
        failures.append("heading drift")
    if max(candidate.max_steering_disagreement, candidate.max_longitudinal_disagreement) > (
        limits["action_disagreement_limit"]
    ):
        failures.append("action disagreement")
    if lap_delta is None or lap_delta > allowed_lap_delta:
        failures.append("lap-time regression")
    diagnostics = {
        **limits,
        "lap_delta_s": lap_delta,
        "max_position_deviation_m": candidate.max_position_deviation_m,
        "max_progress_deviation_m": candidate.max_progress_deviation_m,
        "max_heading_deviation_rad": candidate.max_heading_deviation_rad,
        "max_steering_disagreement": candidate.max_steering_disagreement,
        "max_longitudinal_disagreement": candidate.max_longitudinal_disagreement,
        "passed": not failures,
        "rejection_reasons": failures,
    }
    return not failures, diagnostics


def run_adaptation(
    config: TrainingConfig, stage: str = "full",
    status: Any = None,
) -> Path:
    """Run collect, critics, polish, full, or rollback against the saved champion."""
    if config.algorithm != "tqc" or config.curriculum.mode != "full" or config.tqc is None:
        raise ValueError("champion adaptation requires full-track TQC")
    if stage not in {"collect", "validate", "critics", "promote", "polish", "full", "rollback"}:
        raise ValueError(f"unknown adaptation stage: {stage}")
    if stage in {"collect", "full"} and config.tqc.adaptation_replay_steps < config.tqc.batch_size:
        raise ValueError(
            "local replay expansion must collect at least one critic batch of transitions"
        )
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

    def save(
        path: Path, required: bool, state: str,
        extra_files: dict[str, dict[str, Any]] | None = None,
    ) -> Path | None:
        runner.model.critic_adaptation_required = required
        runner.model.adaptation_stage = state
        staging = path.parent / f".{path.name}-staging-{uuid4().hex}"
        runner.backend.save_model(runner.model, staging, resume=True)
        runner.registry.write_metadata(staging, runner._metadata(runner.last_evaluation))
        for name, value in (extra_files or {}).items():
            (staging / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        if path == champion:
            old_search = champion / "speed-search.json"
            if old_search.is_file():
                shutil.copy2(old_search, staging / old_search.name)
        return promote_directory(staging, path, require_replay=True)

    def collect() -> None:
        metadata = runner.registry.read_metadata(champion)
        if metadata.evaluation is None:
            raise ValueError("local replay expansion requires a fully evaluated champion")
        source_reference = EvaluationResult(**metadata.evaluation)
        source_digest = _policy_digest(champion)
        reference = load(champion, replay=False)
        # Keep the champion's experience and add local transitions to that replay.
        # A replay-free candidate would throw away useful history and leave critics
        # trained on only the short expansion window.
        runner.model = load(champion, replay=True)
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
        env.close()
        evaluation = evaluate_model(
            runner.model, runner._environment, episodes=config.evaluation.episodes,
            seed=config.seed + 2_000_000, reference_model=reference,
        )
        runner.last_evaluation = evaluation
        safe, diagnostics = candidate_diagnostics_pass(evaluation, source_reference, config)
        # Candidate diagnostics compare closed-loop behavior to a fresh deterministic
        # run of the source champion, not to critic or replay-derived estimates.
        if not safe:
            emit({"type": "adaptation_replay_rejected", **diagnostics,
                  **evaluation.to_dict()})
            raise RuntimeError("replay-expanded checkpoint failed closed-loop validation")
        save(work, required=True, state="replay_expanded", extra_files={"working-source.json": {
            "source_policy_sha256": source_digest,
            "source_saved_at": metadata.saved_at,
            "replay_expansion_evaluation": evaluation.to_dict(),
        }})
        emit({"type": "adaptation_stage", "stage": "replay_expansion",
              "transitions_collected": cfg.adaptation_replay_steps,
              "replay_size": runner.model.replay_buffer.size(),
              "action_noise_std": runner.model._adaptation_noise.tolist(),
              "action_noise_probability": cfg.adaptation_noise_probability,
              "mean_action_deviation": mean_action_deviation,
              **evaluation.to_dict()})
        emit({"type": "adaptation_source", "champion_metadata": metadata.adaptation_stage})
        reopen_training_env()

    def critics() -> None:
        if not (work / "metadata.json").is_file():
            raise FileNotFoundError("collect local champion replay before critic adaptation")
        work_metadata = runner.registry.read_metadata(work)
        source_path = work / "working-source.json"
        if work_metadata.adaptation_stage != "replay_expanded" or not source_path.is_file():
            raise RuntimeError("critic adaptation requires a validated replay-expansion checkpoint")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("source_policy_sha256") != _policy_digest(champion):
            raise RuntimeError("champion changed since replay expansion; collect fresh local replay")
        reference_result = EvaluationResult(**source["replay_expansion_evaluation"])
        reference_model = load(work, replay=False)
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
        env.close()
        evaluation = evaluate_model(
            runner.model, runner._environment, episodes=config.evaluation.episodes,
            seed=config.seed + 2_000_000, reference_model=reference_model,
        )
        runner.last_evaluation = evaluation
        safe, diagnostics = candidate_diagnostics_pass(evaluation, reference_result, config)
        emit({"type": "adaptation_critic_validation", **diagnostics, **evaluation.to_dict()})
        if not safe:
            emit({"type": "adaptation_critic_rejected", **diagnostics})
            raise RuntimeError("critic-adapted candidate rejected; champion and replay-expansion checkpoint retained")
        source["critic_adaptation_evaluation"] = evaluation.to_dict()
        source["critic_adaptation_diagnostics"] = diagnostics
        save(work, required=True, state="critics_validated",
             extra_files={"working-source.json": source})
        emit({"type": "adaptation_stage", "stage": "critic_adaptation",
              "updates": cfg.critic_adaptation_updates, "candidate_saved": str(work),
              **runner.model._adaptation_diagnostics})
        reopen_training_env()

    def promote_validated_candidate() -> None:
        if not (work / "metadata.json").is_file():
            raise FileNotFoundError("critic adaptation candidate is not saved")
        work_metadata = runner.registry.read_metadata(work)
        source_path = work / "working-source.json"
        if work_metadata.adaptation_stage != "critics_validated" or not source_path.is_file():
            raise RuntimeError("promotion requires an independently validated critic candidate")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if source.get("source_policy_sha256") != _policy_digest(champion):
            raise RuntimeError("champion changed before candidate promotion; recollect local replay")
        metadata = runner.registry.read_metadata(champion)
        if metadata.evaluation is None:
            raise ValueError("candidate promotion requires a source champion evaluation")
        reference_result = EvaluationResult(**metadata.evaluation)
        reference_model = load(champion, replay=False)
        runner.model = load(work, replay=True)
        runner.model.critic_adaptation_required = True
        runner.model._adaptation_mode = None
        env.close()
        evaluation = evaluate_model(
            runner.model, runner._environment, episodes=config.evaluation.episodes,
            seed=config.seed + 3_000_000, reference_model=reference_model,
        )
        safe, diagnostics = candidate_diagnostics_pass(evaluation, reference_result, config)
        emit({"type": "adaptation_promotion_validation", **diagnostics, **evaluation.to_dict()})
        if not safe:
            emit({"type": "adaptation_promotion_rejected", **diagnostics})
            raise RuntimeError("final closed-loop gate rejected candidate; source champion remains active")
        runner.last_evaluation = evaluation
        runner.model.critic_adaptation_required = False
        runner.model.adaptation_stage = "critics_adapted"
        backup = save(champion, required=False, state="critics_adapted")
        append_pace_history(
            champion, source="critic_adaptation", evaluation=evaluation, model=runner.model,
            reward_profile=config.reward_profile,
            air_brake_bonus_per_s=config.rewards.airborne_brake_bonus_per_s,
            learning_rate=cfg.critic_learning_rate or cfg.learning_rate,
        )
        source_path.unlink(missing_ok=True)
        emit({"type": "adaptation_stage", "stage": "candidate_promotion",
              "champion_backup": str(backup) if backup else None,
              "evaluation": evaluation.to_dict(), "diagnostics": diagnostics})
        reopen_training_env()

    def validate_replay() -> None:
        if not (work / "metadata.json").is_file():
            raise FileNotFoundError("collect local champion replay before validation")
        metadata = runner.registry.read_metadata(champion)
        if metadata.evaluation is None:
            raise ValueError("local replay validation requires a champion evaluation")
        reference_result = EvaluationResult(**metadata.evaluation)
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
        safe, diagnostics = candidate_diagnostics_pass(noisy, reference_result, config)
        emit({"type": "adaptation_noisy_validation", **diagnostics, **noisy.to_dict()})
        if not safe:
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
        if metadata.evaluation is None:
            raise ValueError("actor candidate diagnostics require a champion evaluation")
        reference_result = EvaluationResult(**metadata.evaluation)
        safe, diagnostics = candidate_diagnostics_pass(result, reference_result, config)
        safe = (
            safe and result.median_lap_s is not None and reference_result.median_lap_s is not None
            and result.median_lap_s < reference_result.median_lap_s
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
                  **diagnostics,
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
        if stage == "promote" or stage == "full":
            promote_validated_candidate()
        if stage == "polish":
            polish()
        return champion
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("collect", "validate", "critics", "promote", "polish", "full", "rollback"),
        default="full",
    )
    args = parser.parse_args()
    config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
    run_adaptation(config, args.stage, lambda event: print(json.dumps(event), flush=True))


if __name__ == "__main__":
    main()
