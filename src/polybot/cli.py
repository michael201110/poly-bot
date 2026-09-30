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
    DQNConfig,
    EvaluationConfig,
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
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    parser.add_argument("--output-root", type=Path, default=Path("models"))
    parser.add_argument("--log-root", type=Path, default=Path("logs"))


def _algorithm_options(parser: argparse.ArgumentParser) -> None:
    ppo = parser.add_argument_group("PPO")
    ppo.add_argument("--ppo-architecture", choices=("tiny", "compact", "standard"))
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
    dqn = parser.add_argument_group("DQN")
    dqn.add_argument("--dqn-architecture", choices=("tiny", "compact", "standard", "yosh_2020"))
    dqn.add_argument("--dqn-action-set", choices=("full", "no_brake"))
    dqn.add_argument("--dqn-quantiles", type=int)
    dqn.add_argument("--dqn-lr", type=float)
    dqn.add_argument("--dqn-replay", type=int)
    dqn.add_argument("--dqn-learning-starts", type=int)
    dqn.add_argument("--dqn-batch", type=int)
    dqn.add_argument("--dqn-gamma", type=float)
    dqn.add_argument("--dqn-train-frequency", type=int)
    dqn.add_argument("--dqn-gradient-steps", type=int)
    dqn.add_argument("--dqn-target-update-interval", type=int)
    dqn.add_argument("--dqn-exploration-fraction", type=float)
    dqn.add_argument("--dqn-initial-eps", type=float)
    dqn.add_argument("--dqn-final-eps", type=float)
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


def _config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> TrainingConfig:
    if args.algorithm is None:
        parser.error("--algorithm is required when --config is not provided")
    values = vars(args)
    for prefix in ("ppo_", "dqn_", "tqc_"):
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
        }
        mapping.update(shared)
        specific: dict[str, Any] = {"ppo": PPOConfig(**{
            key: value for key, value in mapping.items() if value is not None
        })}
    elif args.algorithm == "dqn":
        mapping = {
            "action_set": args.dqn_action_set,
            "n_quantiles": args.dqn_quantiles,
            "replay_capacity": args.dqn_replay,
            "learning_starts": args.dqn_learning_starts,
            "train_frequency": args.dqn_train_frequency,
            "gradient_steps": args.dqn_gradient_steps,
            "target_update_interval": args.dqn_target_update_interval,
            "exploration_fraction": args.dqn_exploration_fraction,
            "exploration_initial_eps": args.dqn_initial_eps,
            "exploration_final_eps": args.dqn_final_eps,
        }
        mapping.update(shared)
        specific = {"dqn": DQNConfig(**{
            key: value for key, value in mapping.items() if value is not None
        })}
    else:
        mapping = {
            "replay_capacity": args.tqc_replay,
            "learning_starts": args.tqc_learning_starts,
            "tau": args.tqc_tau,
            "train_frequency": args.tqc_train_frequency,
            "gradient_steps": args.tqc_gradient_steps,
            "entropy": args.tqc_entropy,
            "warmup_forward_fraction": args.tqc_warmup_forward,
            "warmup_steering_std": args.tqc_warmup_steering_std,
        }
        mapping.update(shared)
        specific = {"tqc": TQCConfig(**{
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
    parser = argparse.ArgumentParser(description="Train a v2 PPO, DQN or TQC policy")
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
            DQN_INFO,
            EVALUATION_INFO,
            GENERAL_INFO,
            PPO_INFO,
            REWARD_INFO,
            TQC_INFO,
        )

        for title, mapping in (
            ("General", GENERAL_INFO), ("PPO", PPO_INFO), ("DQN", DQN_INFO), ("TQC", TQC_INFO),
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
        runner = TrainingRunner(cfg, _event)
        print(f"planned training steps: {cfg.timesteps}", flush=True)
        runner.run(resume=resume)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 0


def _saved_model(args: argparse.Namespace) -> tuple[TrainingConfig, Any, Any, Path]:
    registry = ModelRegistry(args.output_root)
    directory = registry.slot(args.track_name, args.algorithm, args.slot)
    metadata = registry.read_metadata(directory)
    cfg = TrainingConfig.from_dict(metadata.training_config)
    cfg.backend = args.backend or cfg.backend
    cfg.device = args.device or cfg.device
    backend = backend_for(cfg.algorithm)
    registry.validate(metadata, cfg, backend.action_adapter(cfg).schema)
    runner = TrainingRunner(cfg)
    env = runner._environment()
    try:
        device = resolve_device(cfg.device, algorithm=cfg.algorithm)
        model = backend.load_model(directory / "policy.zip", env, device.resolved)
        if cfg.algorithm == "tqc":
            model.policy_overlays = list(metadata.policy_overlays)
    finally:
        env.close()
    return cfg, backend, model, directory


def _model_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--track-name", required=True)
    parser.add_argument("--slot", choices=("latest", "champion"), default="champion")
    parser.add_argument("--output-root", type=Path, default=Path("models"))
    parser.add_argument("--backend", choices=("mock", "websocket"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    return parser


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = _model_parser("Evaluate a saved v2 policy deterministically")
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args(argv)
    try:
        cfg, _, model, _ = _saved_model(args)
        runner = TrainingRunner(cfg)
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
        cfg, _, model, _ = _saved_model(args)
        env = TrainingRunner(cfg)._environment()
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
