"""Transfer the immutable Summer 1 TQC champion into a gated GRTQC checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch as th

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.algorithms.tqc import TQCBackend
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import REWARD_SEMANTICS, ModelRegistry
from polybot.training.config import EvaluationConfig, GRTQCConfig, TrainingConfig


def initialize(
    source: Path, destination: Path, *, device: str = "cpu",
    training_config: TrainingConfig | None = None,
    allow_frame_skip_change: bool = False,
) -> dict[str, float | int | str]:
    if destination.exists():
        raise FileExistsError(f"initialization checkpoint already exists: {destination}")
    source_metadata = ModelRegistry(source.parents[2]).read_metadata(source)
    if source_metadata.algorithm != "tqc":
        raise ValueError("source must be a TQC checkpoint")
    source_config = TrainingConfig.from_dict(source_metadata.training_config)
    assert source_config.tqc is not None
    settings = asdict(source_config.tqc)
    settings.update({
        "learning_rate": 3e-5,
        "actor_learning_rate": 1e-7,
        "critic_learning_rate": 5e-5,
        "learning_starts": 3_000,
        "train_frequency": 2,
        "gradient_steps": 1,
        "disagreement_coefficient": 0.01,
        "critic_warmup_updates": 10_000,
        "critic_readiness_window": 200,
        "critic_readiness_relative_change": 0.1,
        "target_lap_s": 22.0,
        "exploration_std": 0.0001,
        "critic_collection_std": 0.001,
        "actor_step_action_limit": 1e-5,
    })
    default_config = replace(
        source_config, algorithm="grtqc", tqc=None,
        grtqc=GRTQCConfig(**settings),
        output_root=destination.parents[2],
        timesteps=2_000_000,
        evaluation=EvaluationConfig(interval_steps=5_000, episodes=5),
    )
    config = training_config or default_config
    if config.algorithm != "grtqc" or config.grtqc is None:
        raise ValueError("initialization config must use GRTQC")
    if config.output_root.resolve() != destination.parents[2].resolve():
        raise ValueError("initialization destination must be under the config output root")
    for key in ("track_id", "track_name", "lookahead_count"):
        if getattr(config, key) != getattr(source_config, key):
            raise ValueError(f"transfer config differs from source in {key}")
    if config.frame_skip != source_config.frame_skip and not allow_frame_skip_change:
        raise ValueError("frame skip differs; explicitly allow it after measuring the source lap")
    if (
        config.frame_skip != source_config.frame_skip
        and config.grtqc.reference_lap_s == 24.263
    ):
        raise ValueError("set the measured source lap for the changed frame skip")
    source_model = TQCBackend().load_model(source / "policy.zip", None, device, resume=False)
    source_model.policy_overlays = list(source_metadata.policy_overlays)
    # Older champions persisted this schedule in policy.zip but left metadata
    # empty. Keep the archive's schedule unless metadata explicitly supplies it.
    if source_metadata.speed_bias_schedule:
        source_model.speed_bias_schedule = list(source_metadata.speed_bias_schedule)
    target_backend = GRTQCBackend()
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id=config.track_id,
        lookahead_count=config.lookahead_count, frame_skip=config.frame_skip,
        action_adapter=target_backend.action_adapter(config),
    )
    try:
        target = target_backend.create_model(config, env, device)
    finally:
        env.close()
    source_actor = source_model.actor.state_dict()
    transfer = target.actor.load_state_dict(source_actor, strict=False)
    if transfer.unexpected_keys or any(".gate." not in key for key in transfer.missing_keys):
        raise RuntimeError(f"actor transfer mismatch: {transfer}")
    if not source_actor or len(transfer.missing_keys) != 4:
        raise RuntimeError("unexpected TQC actor architecture")
    target.policy_overlays = list(source_metadata.policy_overlays)
    target.speed_bias_schedule = list(source_model.speed_bias_schedule)
    # The source replay supplies real, saved Summer 1 observations for fidelity
    # checks; its old transitions/rewards are not copied into the new learner.
    source_model.load_replay_buffer(str(source / "replay.pkl"))
    replay = source_model.replay_buffer
    if replay is None or replay.size() < 2048:
        raise RuntimeError("source champion has too few replay observations")
    indices = np.linspace(0, replay.size() - 1, 2048, dtype=np.int64)
    observations = np.asarray(replay.observations[indices, 0], dtype=np.float32)
    raw_difference = []
    executed_difference = []
    for batch in np.array_split(observations, 16):
        with th.no_grad():
            tensor = th.as_tensor(batch, device=device)
            raw_source = source_model.actor(tensor, deterministic=True).cpu().numpy()
            raw_target = target.actor(tensor, deterministic=True).cpu().numpy()
        raw_difference.append(np.abs(raw_source - raw_target))
        executed_source, _ = source_model.predict(batch, deterministic=True)
        executed_target, _ = target.predict(batch, deterministic=True)
        executed_difference.append(np.abs(executed_source - executed_target))
    raw_max = float(np.concatenate(raw_difference).max())
    executed_max = float(np.concatenate(executed_difference).max())
    if raw_max > 1e-4 or executed_max > 1e-4:
        raise RuntimeError(f"actor transfer drift too large: raw={raw_max}, executed={executed_max}")
    destination.mkdir(parents=True)
    target_backend.save_model(target, destination, resume=False)
    counts = target_backend.parameter_counts(target)
    metadata = replace(
        source_metadata, algorithm="grtqc", architecture=config.grtqc.architecture,
        actor_parameters=counts["actor"], critic_parameters=counts["critic"],
        total_trainable_parameters=counts["total"], training_config=config.to_dict(),
        reward_profile=config.reward_profile,
        reward_semantics=REWARD_SEMANTICS,
        training_timesteps=0, simulator_ticks=0, wall_seconds=0.0,
        finishes=0, crashes=0, evaluation=None, implementation="grtqc-gated-variance-v1",
        speed_bias_schedule=list(source_model.speed_bias_schedule),
    )
    ModelRegistry(config.output_root).write_metadata(destination, metadata)
    report: dict[str, float | int | str] = {
        "source": str(source),
        "source_policy_sha256": hashlib.sha256((source / "policy.zip").read_bytes()).hexdigest(),
        "sampled_observations": len(observations),
        "transferred_actor_tensors": len(source_actor),
        "new_gate_tensors": len(transfer.missing_keys),
        "raw_action_max_abs_error": raw_max,
        "executed_action_max_abs_error": executed_max,
        "expected_reference_lap_s": config.grtqc.reference_lap_s,
        "live_validation": "pending",
    }
    (destination / "transfer.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion"
    ))
    parser.add_argument("--destination", type=Path, default=Path(
        "models/summer-1/grtqc/initialization"
    ))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path, help="GRTQC training profile for this transfer")
    parser.add_argument("--allow-frame-skip-change", action="store_true")
    args = parser.parse_args()
    config = (
        TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        if args.config else None
    )
    print(json.dumps(initialize(
        args.source, args.destination, device=args.device, training_config=config,
        allow_frame_skip_change=args.allow_frame_skip_change,
    ), indent=2))


if __name__ == "__main__":
    main()
