"""CLI entry points over the same v2 configuration and runner used by the GUI."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from polybot.algorithms.registry import ALGORITHMS, backend_for
from polybot.controller import CenterlineController
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import ModelRegistry
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    EvaluationConfig,
    GRTQCConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.devices import resolve_device
from polybot.training.evaluation import evaluate_model
from polybot.training.reward_profiles import RewardProfileStore
from polybot.training.runner import TrainingRunner


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--backend", choices=("mock", "websocket"), default="mock")
    parser.add_argument("--track-name")
    parser.add_argument("--track-id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episode-seconds", type=float, default=60)
    parser.add_argument("--lookahead", type=int, default=12)
    parser.add_argument("--reward-profile")
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--curriculum", choices=(
        "full", "section", "quarters", "quarters-randomised", "q4-full", "timed", "custom"
    ), default="full")
    parser.add_argument("--custom-phases", type=Path, help="JSON list of custom phases whose steps sum to --timesteps")
    parser.add_argument("--section-start", type=float)
    parser.add_argument("--section-end", type=float)
    parser.add_argument("--time-start", type=float)
    parser.add_argument("--time-end", type=float)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--output-root", type=Path, default=Path("models"))
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument(
        "--visual-replay", action=argparse.BooleanOptionalAction, default=None,
        help="record visual replays (default: on for WebSocket, off for mock)",
    )
    parser.add_argument("--visual-replay-sample-hz", type=float, default=20.0)
    parser.add_argument(
        "--visual-replay-observations", action="store_true",
        help="also save each policy observation and requested action",
    )


def _algorithm_options(parser: argparse.ArgumentParser) -> None:
    ppo = parser.add_argument_group("PPO")
    ppo.add_argument(
        "--ppo-architecture",
        choices=("tiny", "compact", "standard", "tqc_compatible", "tqc_residual"),
    )
    ppo.add_argument("--ppo-residual-action-limit", type=float)
    ppo.add_argument("--ppo-residual-progress-start", type=float)
    ppo.add_argument("--ppo-residual-progress-end", type=float)
    ppo.add_argument("--ppo-lr", type=float)
    ppo.add_argument("--ppo-rollout", type=int)
    ppo.add_argument("--ppo-batch", type=int)
    ppo.add_argument("--ppo-epochs", type=int)
    ppo.add_argument("--ppo-gamma", type=float)
    ppo.add_argument("--ppo-gae-lambda", type=float)
    ppo.add_argument("--ppo-entropy", type=float)
    ppo.add_argument("--teacher-model")
    ppo.add_argument("--teacher-kl", type=float)
    ppo.add_argument("--ppo-imitation", type=float)
    ppo.add_argument("--ppo-initial-forward-bias", type=float)
    ppo.add_argument("--ppo-initial-steering-bias", type=float)
    tqc = parser.add_argument_group("TQC")
    tqc.add_argument("--tqc-architecture", choices=("tiny", "compact", "standard"))
    tqc.add_argument("--tqc-lr", type=float)
    tqc.add_argument("--tqc-replay", type=int)
    tqc.add_argument("--tqc-learning-starts", type=int)
    tqc.add_argument("--tqc-batch", type=int)
    tqc.add_argument("--tqc-gamma", type=float)
    tqc.add_argument("--tqc-tau", type=float)
    tqc.add_argument("--tqc-train-frequency", type=int)
    tqc.add_argument("--tqc-gradient-steps", type=int)
    tqc.add_argument("--tqc-entropy")
    tqc.add_argument("--tqc-warmup-forward", type=float)
    tqc.add_argument("--tqc-warmup-steering-std", type=float)
    grtqc = parser.add_argument_group("GRTQC")
    grtqc.add_argument("--grtqc-architecture", choices=("tiny", "compact", "standard"))
    grtqc.add_argument("--grtqc-lr", type=float)
    grtqc.add_argument("--grtqc-replay", type=int)
    grtqc.add_argument("--grtqc-learning-starts", type=int)
    grtqc.add_argument("--grtqc-batch", type=int)
    grtqc.add_argument("--grtqc-gamma", type=float)
    grtqc.add_argument("--grtqc-tau", type=float)
    grtqc.add_argument("--grtqc-train-frequency", type=int)
    grtqc.add_argument("--grtqc-gradient-steps", type=int)
    grtqc.add_argument("--grtqc-entropy")
    grtqc.add_argument("--grtqc-disagreement", type=float)
    grtqc.add_argument("--grtqc-critic-warmup-updates", type=int)
    grtqc.add_argument("--grtqc-target-lap", type=float)


def _config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> TrainingConfig:
    if args.algorithm is None:
        parser.error("--algorithm is required when --config is not provided")
    values = vars(args)
    for prefix in ("ppo_", "tqc_", "grtqc_"):
        if prefix != f"{args.algorithm}_" and any(
            value is not None for key, value in values.items() if key.startswith(prefix)
        ):
            parser.error(f"{prefix.removesuffix('_').upper()} settings do not apply to {args.algorithm.upper()}")
    if args.algorithm != "ppo" and (args.teacher_model or args.teacher_kl is not None):
        parser.error("teacher settings only apply to PPO")
    shared = {
        "architecture": values[f"{args.algorithm}_architecture"],
        "learning_rate": values[f"{args.algorithm}_lr"],
        "batch_size": values[f"{args.algorithm}_batch"],
        "gamma": values[f"{args.algorithm}_gamma"],
    }
    if args.algorithm == "ppo":
        mapping = {
            "rollout_steps": args.ppo_rollout, "epochs": args.ppo_epochs,
            "gae_lambda": args.ppo_gae_lambda,
            "entropy_coefficient": args.ppo_entropy,
            "teacher_model": args.teacher_model,
            "teacher_kl_coefficient": args.teacher_kl,
            "imitation_coefficient": args.ppo_imitation,
            "initial_forward_bias": args.ppo_initial_forward_bias,
            "initial_steering_bias": args.ppo_initial_steering_bias,
            "residual_action_limit": args.ppo_residual_action_limit,
            "residual_progress_start": args.ppo_residual_progress_start,
            "residual_progress_end": args.ppo_residual_progress_end,
        }
        mapping.update(shared)
        specific: dict[str, Any] = {"ppo": PPOConfig(**{
            key: value for key, value in mapping.items() if value is not None
        })}
    else:
        mapping = {
            "replay_capacity": values[f"{args.algorithm}_replay"],
            "learning_starts": values[f"{args.algorithm}_learning_starts"],
            "tau": values[f"{args.algorithm}_tau"],
            "train_frequency": values[f"{args.algorithm}_train_frequency"],
            "gradient_steps": values[f"{args.algorithm}_gradient_steps"],
            "entropy": values[f"{args.algorithm}_entropy"],
        }
        if args.algorithm == "tqc":
            mapping.update({
                "warmup_forward_fraction": args.tqc_warmup_forward,
                "warmup_steering_std": args.tqc_warmup_steering_std,
            })
        else:
            mapping.update({
                "disagreement_coefficient": args.grtqc_disagreement,
                "critic_warmup_updates": args.grtqc_critic_warmup_updates,
                "target_lap_s": args.grtqc_target_lap,
            })
        mapping.update(shared)
        config_type = TQCConfig if args.algorithm == "tqc" else GRTQCConfig
        specific = {args.algorithm: config_type(**{
            key: value for key, value in mapping.items() if value is not None
        })}
    backend = args.backend
    track_name = args.track_name or ("Summer 1" if backend == "websocket" else "Mock straight")
    track_id = args.track_id or ("current" if backend == "websocket" else "mock/straight")
    profile = args.reward_profile or "Balanced"
    rewards = RewardProfileStore().load(profile)
    return TrainingConfig(
        algorithm=args.algorithm, backend=backend, track_name=track_name, track_id=track_id,
        device=args.device, seed=args.seed,
        frame_skip=args.frame_skip or (30 if backend == "websocket" else 4),
        timesteps=args.timesteps, max_episode_seconds=args.episode_seconds,
        lookahead_count=args.lookahead, reward_profile=profile,
        reward_scale=args.reward_scale, checkpoint_interval=args.checkpoint_interval,
        output_root=args.output_root, log_root=args.log_root,
        visual_replay_enabled=args.visual_replay,
        visual_replay_sample_hz=args.visual_replay_sample_hz,
        visual_replay_observations=args.visual_replay_observations,
        curriculum=CurriculumConfig(
            args.curriculum, args.section_start, args.section_end,
            args.time_start, args.time_end,
            tuple(CurriculumPhaseConfig(**phase) for phase in json.loads(
                args.custom_phases.read_text(encoding="utf-8")
            )) if args.custom_phases else (),
        ),
        evaluation=EvaluationConfig(args.eval_interval, args.eval_episodes),
        **({"rewards": rewards} if rewards is not None else {}), **specific,
    )


def _event(event: dict[str, Any]) -> None:
    if event["type"] in {"started", "phase", "episode", "evaluation", "champion", "completed", "stopped"}:
        print(json.dumps(event, allow_nan=False), flush=True)


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a v2 GRTQC, TQC or legacy PPO policy")
    parser.add_argument("--parameter-help", action="store_true",
                        help="print the central plain-language parameter reference")
    parser.add_argument("--config", type=Path, help="v2 JSON config shared with the GUI")
    parser.add_argument("--resume", nargs="?", const="latest")
    _common(parser)
    _algorithm_options(parser)
    args = parser.parse_args(argv)
    if args.parameter_help:
        from polybot.training.parameters import (
            CURRICULUM_INFO,
            EVALUATION_INFO,
            GENERAL_INFO,
            GRTQC_INFO,
            PPO_INFO,
            REWARD_INFO,
            TQC_INFO,
        )

        for title, mapping in (
            ("General", GENERAL_INFO), ("PPO", PPO_INFO), ("GRTQC", GRTQC_INFO), ("TQC", TQC_INFO),
            ("Curriculum", CURRICULUM_INFO), ("Evaluation", EVALUATION_INFO),
            ("Reward coefficients", REWARD_INFO),
        ):
            print(f"\n{title}")
            for name, detail in mapping.items():
                print(f"  {name}: {detail.description}")
        return 0
    try:
        cfg = (
            TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
            if args.config else _config_from_args(args, parser)
        )
        registry = ModelRegistry(cfg.output_root)
        resume = None
        if args.resume is not None:
            resume = (
                registry.slot(cfg.track_name, cfg.algorithm, "latest")
                if args.resume == "latest" else Path(args.resume)
            )
        elif cfg.algorithm == "grtqc" and cfg.grtqc.training_origin != "scratch":
            resume = registry.algorithm_dir(cfg.track_name, "grtqc") / "initialization"
        runner = TrainingRunner(cfg, _event)
        print(f"planned training steps: {cfg.timesteps}", flush=True)
        runner.run(resume=resume)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


def _configure_saved_overlays(
    runner: TrainingRunner, model: Any, metadata: Any,
) -> None:
    """Restore checkpoint overlays for standalone evaluation and playback."""
    if runner.config.algorithm == "ppo":
        runner._ppo_air_brake_overlays = list(metadata.policy_overlays)
        runner._ppo_speed_bias_schedule = [
            list(row) for row in metadata.speed_bias_schedule
        ]
        if model is not None:
            model.policy_overlays = list(metadata.policy_overlays)
            model.speed_bias_schedule = [
                list(row) for row in metadata.speed_bias_schedule
            ]
    elif runner.config.algorithm in {"tqc", "grtqc"}:
        if model is not None:
            model.policy_overlays = list(metadata.policy_overlays)
            # Older TQC saves kept this schedule in the policy archive before
            # it was added to metadata. An empty metadata default must not
            # erase that serialized runtime state.
            if metadata.speed_bias_schedule:
                model.speed_bias_schedule = [
                    list(row) for row in metadata.speed_bias_schedule
                ]


def _saved_model(args: argparse.Namespace) -> tuple[TrainingConfig, Any, Any, Path, Any]:
    registry = ModelRegistry(args.output_root)
    directory = registry.slot(args.track_name, args.algorithm, args.slot)
    metadata = registry.read_metadata(directory)
    cfg = TrainingConfig.from_dict(metadata.training_config)
    cfg.backend = args.backend or cfg.backend
    cfg.device = args.device or cfg.device
    backend = backend_for(cfg.algorithm)
    registry.validate(metadata, cfg, backend.action_adapter(cfg).schema)
    runner = TrainingRunner(cfg)
    _configure_saved_overlays(runner, None, metadata)
    env = runner._environment()
    try:
        device = resolve_device(cfg.device, algorithm=cfg.algorithm)
        model = backend.load_model(directory / "policy.zip", env, device.resolved)
        _configure_saved_overlays(runner, model, metadata)
    finally:
        env.close()
    return cfg, backend, model, directory, metadata


def _model_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--track-name", required=True)
    parser.add_argument("--slot", choices=("initialization", "latest", "champion"), default="champion")
    parser.add_argument("--output-root", type=Path, default=Path("models"))
    parser.add_argument("--backend", choices=("mock", "websocket"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = _model_parser("Evaluate a saved v2 policy deterministically")
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args(argv)
    try:
        cfg, _, model, _, metadata = _saved_model(args)
        runner = TrainingRunner(cfg)
        _configure_saved_overlays(runner, model, metadata)
        result = evaluate_model(
            model, runner._environment, episodes=args.episodes or cfg.evaluation.episodes,
            seed=cfg.seed + 1_000_000,
        )
        print(json.dumps(result.to_dict(), indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


def drive_main(argv: Sequence[str] | None = None) -> int:
    parser = _model_parser("Play one deterministic lap from a v2 policy")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg, _, model, _, metadata = _saved_model(args)
        runner = TrainingRunner(cfg)
        _configure_saved_overlays(runner, model, metadata)
        env = runner._environment()
        try:
            observation, _ = env.reset(seed=cfg.seed)
            while True:
                started = time.monotonic()
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(json.dumps({"events": info["events"],
                                      "progress_m": info["route_progress_m"],
                                      "elapsed_s": info["elapsed_s"]}))
                    break
                if args.realtime:
                    target = info["ticks_advanced"] * env.simulator_capabilities["fixed_dt_s"]
                    time.sleep(max(0.0, target - (time.monotonic() - started)))
        finally:
            env.close()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


def smoke_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the protocol and environment mock")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args(argv)
    from polybot.environment.env import PolyTrackEnv

    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight")
    controller = CenterlineController()
    try:
        for index in range(args.episodes):
            env.reset(seed=index)
            while True:
                action = controller.policy_action(env.latest_telemetry)
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(json.dumps({"events": info["events"],
                                      "progress_m": info["route_progress_m"]}))
                    break
    finally:
        env.close()
    return 0
