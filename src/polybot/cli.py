"""Command-line entry points for smoke testing, training, and driving."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from polybot.controller import CenterlineController
from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.transport import SimulatorTransport, WebSocketServerTransport


def _common_arguments(
    parser: argparse.ArgumentParser,
    *,
    track_default: str | None = "mock/gentle-s",
    frame_skip_default: int | None = 4,
    max_steps_default: int | None = 2_000,
) -> None:
    track_help = "simulator track identifier"
    frame_skip_help = "physics ticks held per action"
    if track_default is None:
        track_help += " (default: mock/gentle-s for mock, current for WebSocket)"
    if frame_skip_default is None:
        frame_skip_help += " (default: 4 for mock, 10 for WebSocket)"
    max_steps_help = "Gymnasium episode limit"
    if max_steps_default is None:
        max_steps_help += " (default: 2000 for mock, 30000 for WebSocket)"
    parser.add_argument("--track", default=track_default, help=track_help)
    parser.add_argument("--seed", type=int, default=0, help="episode/trainer seed")
    parser.add_argument("--lookahead", type=int, default=12, help="future track sample count")
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=frame_skip_default,
        help=frame_skip_help,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=max_steps_default,
        help=max_steps_help,
    )


def _websocket_arguments(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(host="127.0.0.1", port=8765)
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for the PolyTrack mod to connect",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="seconds to wait for each simulator response",
    )


def _apply_backend_defaults(args: argparse.Namespace) -> None:
    if args.track is None:
        args.track = "current" if args.backend == "websocket" else "mock/gentle-s"
    if args.frame_skip is None:
        args.frame_skip = 10 if args.backend == "websocket" else 4
    if args.max_steps is None:
        args.max_steps = 30_000 if args.backend == "websocket" else 2_000


def _validate_websocket_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.connect_timeout <= 0:
        parser.error("--connect-timeout must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.host != "127.0.0.1" or args.port != 8765:
        parser.error("the PolyTrack mod currently requires --host 127.0.0.1 --port 8765")


def _make_transport(args: argparse.Namespace) -> SimulatorTransport:
    if getattr(args, "backend", "websocket") == "mock":
        return MockSimulatorTransport()

    transport = WebSocketServerTransport(
        args.host,
        args.port,
        connect_timeout_s=args.connect_timeout,
        request_timeout_s=args.request_timeout,
    )
    print(f"Waiting for the local PolyTrack mod at {transport.endpoint} ...", flush=True)
    return transport


def _episode_summary(
    episode: int,
    seed: int,
    info: dict[str, object],
    total_reward: float,
) -> dict[str, object]:
    return {
        "episode": episode,
        "seed": seed,
        "finished": "finish" in info["events"],
        "crashed": "crash" in info["events"],
        "elapsed_s": round(float(info["elapsed_s"]), 3),
        "progress_m": round(float(info["route_progress_m"]), 3),
        "reward": round(total_reward, 3),
    }


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
    _common_arguments(
        parser,
        track_default=None,
        frame_skip_default=None,
        max_steps_default=None,
    )
    parser.add_argument("--backend", choices=("mock", "websocket"), default="mock")
    _websocket_arguments(parser)
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
    _apply_backend_defaults(args)

    if args.timesteps < 1:
        parser.error("--timesteps must be positive")
    _validate_websocket_arguments(parser, args)

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        parser.error("training dependencies are missing; install with: pip install -e '.[train]'")
        raise AssertionError("unreachable") from exc

    transport = _make_transport(args)

    try:
        env = PolyTrackEnv(
            transport,
            track_id=args.track,
            lookahead_count=args.lookahead,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_steps,
            request_timeout_s=args.request_timeout,
        )
    except BaseException:
        transport.close()
        raise
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
    _common_arguments(
        parser,
        track_default=None,
        frame_skip_default=None,
        max_steps_default=None,
    )
    parser.add_argument("model", type=Path, help="Stable-Baselines PPO model (.zip is optional)")
    parser.add_argument("--backend", choices=("mock", "websocket"), default="mock")
    _websocket_arguments(parser)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args(argv)
    _apply_backend_defaults(args)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    _validate_websocket_arguments(parser, args)
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        parser.error("training dependencies are missing; install with: pip install -e '.[train]'")
        raise AssertionError("unreachable") from exc

    transport = _make_transport(args)
    try:
        env = PolyTrackEnv(
            transport,
            track_id=args.track,
            lookahead_count=args.lookahead,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_steps,
            request_timeout_s=args.request_timeout,
        )
    except BaseException:
        transport.close()
        raise
    summaries: list[dict[str, object]] = []
    try:
        model = PPO.load(str(args.model), env=env)
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


def drive_main(argv: Sequence[str] | None = None) -> int:
    """Drive the currently selected local PolyTrack race with a policy."""

    parser = argparse.ArgumentParser(
        description="Drive the current PolyTrack race through the local-only AI bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _common_arguments(
        parser,
        track_default="current",
        frame_skip_default=10,
        max_steps_default=30_000,
    )
    _websocket_arguments(parser)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--model",
        type=Path,
        help="saved Stable-Baselines PPO model; omit to use the centreline controller",
    )
    policy.add_argument(
        "--centerline",
        action="store_true",
        help="explicitly use the built-in centreline controller (the default)",
    )
    parser.add_argument("--episodes", type=int, default=1, help="number of races to drive")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample PPO actions instead of using deterministic predictions",
    )
    args = parser.parse_args(argv)

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.stochastic and args.model is None:
        parser.error("--stochastic requires --model")
    _validate_websocket_arguments(parser, args)

    ppo_type = None
    if args.model is not None:
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            parser.error("PPO dependencies are missing; install with: pip install -e '.[train]'")
            raise AssertionError("unreachable") from exc
        ppo_type = PPO

    transport = _make_transport(args)
    try:
        env = PolyTrackEnv(
            transport,
            track_id=args.track,
            lookahead_count=args.lookahead,
            frame_skip=args.frame_skip,
            max_episode_steps=args.max_steps,
            request_timeout_s=args.request_timeout,
        )
    except BaseException:
        transport.close()
        raise
    controller = CenterlineController()
    summaries: list[dict[str, object]] = []
    try:
        model = ppo_type.load(str(args.model), env=env) if ppo_type is not None else None
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            observation, _ = env.reset(seed=episode_seed)
            total_reward = 0.0
            terminated = truncated = False
            info: dict[str, object] = {}
            while not (terminated or truncated):
                if model is not None:
                    action, _ = model.predict(
                        observation,
                        deterministic=not args.stochastic,
                    )
                else:
                    telemetry = env.latest_telemetry
                    assert telemetry is not None
                    action = controller.policy_action(telemetry)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            summaries.append(_episode_summary(episode, episode_seed, info, total_reward))
    finally:
        env.close()

    print(json.dumps(summaries, indent=2))
    return 0
