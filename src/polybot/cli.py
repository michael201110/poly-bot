"""Command-line entry points for smoke testing and training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from polybot.controller import CenterlineController
from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.transport import WebSocketServerTransport


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--track", default="mock/gentle-s", help="simulator track identifier")
    parser.add_argument("--seed", type=int, default=0, help="episode/trainer seed")
    parser.add_argument("--lookahead", type=int, default=12, help="future track sample count")
    parser.add_argument("--frame-skip", type=int, default=4, help="physics ticks held per action")
    parser.add_argument("--max-steps", type=int, default=2_000, help="Gymnasium episode limit")


def smoke_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the baseline against the mock simulator")
    _common_arguments(parser)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    env = PolyTrackEnv(
        MockSimulatorTransport(),
        track_id=args.track,
        lookahead_count=args.lookahead,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps,
    )
    controller = CenterlineController()
    summaries: list[dict[str, object]] = []
    try:
        for episode in range(args.episodes):
            _, info = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                telemetry = env.latest_telemetry
                assert telemetry is not None
                _, reward, terminated, truncated, info = env.step(
                    controller.policy_action(telemetry)
                )
                total_reward += reward
            summaries.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "finished": "finish" in info["events"],
                    "crashed": "crash" in info["events"],
                    "steps": info["tick"] // args.frame_skip,
                    "elapsed_s": round(float(info["elapsed_s"]), 3),
                    "progress_m": round(float(info["route_progress_m"]), 3),
                    "reward": round(total_reward, 3),
                }
            )
    finally:
        env.close()

    print(json.dumps(summaries, indent=2))
    return 0 if all(bool(summary["finished"]) for summary in summaries) else 1


def train_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO against a PolyBot simulator")
    _common_arguments(parser)
    parser.add_argument("--backend", choices=("mock", "websocket"), default="mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-out", type=Path, default=Path("models/polybot-ppo"))
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=None,
        help="optional TensorBoard output directory (requires tensorboard)",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="show Stable-Baselines progress UI (requires tqdm and rich)",
    )
    args = parser.parse_args(argv)

    if args.timesteps < 1:
        parser.error("--timesteps must be positive")

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        parser.error("training dependencies are missing; install with: pip install -e '.[train]'")
        raise AssertionError("unreachable") from exc

    if args.backend == "mock":
        transport = MockSimulatorTransport()
    else:
        transport = WebSocketServerTransport(args.host, args.port)
        print(f"Waiting for the local PolyTrack bridge at {transport.endpoint} ...", flush=True)

    env = PolyTrackEnv(
        transport,
        track_id=args.track,
        lookahead_count=args.lookahead,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps,
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=1,
            tensorboard_log=(
                str(args.tensorboard_log) if args.tensorboard_log is not None else None
            ),
        )
        model.learn(total_timesteps=args.timesteps, progress_bar=args.progress_bar)
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.model_out))
        print(f"Saved model to {args.model_out}.zip")
    finally:
        env.close()
    return 0


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved PPO policy")
    _common_arguments(parser)
    parser.add_argument("model", type=Path, help="Stable-Baselines PPO model (.zip is optional)")
    parser.add_argument("--backend", choices=("mock", "websocket"), default="mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        parser.error("training dependencies are missing; install with: pip install -e '.[train]'")
        raise AssertionError("unreachable") from exc

    if args.backend == "mock":
        transport = MockSimulatorTransport()
    else:
        transport = WebSocketServerTransport(args.host, args.port)
        print(f"Waiting for the local PolyTrack bridge at {transport.endpoint} ...", flush=True)
    env = PolyTrackEnv(
        transport,
        track_id=args.track,
        lookahead_count=args.lookahead,
        frame_skip=args.frame_skip,
        max_episode_steps=args.max_steps,
    )
    model = PPO.load(str(args.model), env=env)
    summaries: list[dict[str, object]] = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            info: dict[str, object] = {}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            summaries.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "finished": "finish" in info["events"],
                    "crashed": "crash" in info["events"],
                    "elapsed_s": round(float(info["elapsed_s"]), 3),
                    "progress_m": round(float(info["route_progress_m"]), 3),
                    "reward": round(total_reward, 3),
                }
            )
    finally:
        env.close()
    print(json.dumps(summaries, indent=2))
    return 0 if all(bool(summary["finished"]) for summary in summaries) else 1
