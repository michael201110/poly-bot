"""Play complete saved attempts within the existing swarm transport limits.

The sample filter chooses whole recordings; it never truncates or resamples one.
No simulator/training process is started. The bridge must already be installed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polybot.replay_swarm import (
    ColorScale,
    ReplayPlaybackOptions,
    ReplaySelection,
    SelectedReplayPayload,
    evenly_sample_replays,
    replay_camera_rank,
    resolve_replay_directories,
    send_replay_swarm,
)
from polybot.training.visual_replays import load_replay_episode, load_replay_index
from polybot.transport import WebSocketServerTransport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cars", type=int, default=500)
    parser.add_argument("--min-samples", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=250)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cars <= 500 or not 2 <= args.min_samples <= args.max_samples:
        parser.error("cars must be 1..500 and 2 <= min-samples <= max-samples")
    candidates = [
        ReplaySelection(directory, entry)
        for directory in resolve_replay_directories(args.run)
        for entry in load_replay_index(directory)
        if args.min_samples <= entry["sample_count"] <= args.max_samples
    ]
    if len(candidates) < args.cars:
        parser.error(f"only {len(candidates)} complete recordings match the sample filter")
    selected = evenly_sample_replays(candidates, args.cars)
    scale = ColorScale(maximum_step=max(1, selected[-1].training_step))
    leader = min(selected, key=replay_camera_rank)
    payloads = [
        SelectedReplayPayload(
            item,
            load_replay_episode(
                item.replay_directory, item.metadata,
                include_optional=False, include_hud=item is leader,
            ),
            scale.hex_color(item.training_step),
        )
        for item in selected
    ]
    summary = {
        "cars": len(payloads),
        "complete_samples": sum(len(item.payload.samples) for item in payloads),
        "colour_max_step": scale.maximum_step,
        "source": str(args.run),
        "selection": [
            {"episode": item.episode_key, "step": item.selection.training_step,
             "samples": len(item.payload.samples), "colour": item.color}
            for item in payloads
        ],
    }
    transport = WebSocketServerTransport(
        port=args.port, connect_timeout_s=60, request_timeout_s=120,
    )
    try:
        result = send_replay_swarm(
            transport, action="play", payloads=payloads,
            options=ReplayPlaybackOptions(speed=1, opacity=1, end_behavior="disappear"),
        )
        summary["result"] = result
    finally:
        transport.close()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "selection"}, indent=2))


if __name__ == "__main__":
    main()
