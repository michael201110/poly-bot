"""Profile native renderer cars in an already installed PML bridge. No training.

Duplicates one complete saved attempt with all recorded wheel samples intact;
only IDs/colors differ. Default source must have <=500 samples for 500 cars.
The bridge's rolling metrics measure real frames, not a mocked Three.js scene.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from polybot.replay_swarm import ReplayPlaybackOptions, ReplaySelection, SelectedReplayPayload, send_replay_swarm
from polybot.training.visual_replays import ReplayPayload, load_replay_episode
from polybot.transport import WebSocketServerTransport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 50, 100, 250, 500])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--settle-seconds", type=float, default=3)
    args = parser.parse_args()
    if any(size < 1 or size > 500 for size in args.sizes) or args.settle_seconds < 0:
        parser.error("sizes must be 1..500 and settle seconds nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads((args.run / "index.json").read_text(encoding="utf-8"))["episodes"]
    entry = next(e for e in entries if e["episode_id"] == args.episode)
    # Match normal swarm loading: all wheel/transform samples for every car,
    # plus the selected camera run's full HUD/controls stream.
    payload = load_replay_episode(args.run, entry, include_optional=False, include_hud=True)
    options = ReplayPlaybackOptions(speed=.1, opacity=1.)
    report = {"source": str(args.run), "episode": args.episode, "samples_per_car": len(payload.samples),
              "leader_hud_frames": len(payload.hud_frames or []), "settle_seconds": args.settle_seconds, "cases": []}
    transport = WebSocketServerTransport(port=args.port, connect_timeout_s=60, request_timeout_s=120)
    try:
        for count in args.sizes:
            if count * len(payload.samples) > 250000:
                raise ValueError("whole attempt exceeds bridge sample budget; choose a shorter saved attempt")
            payloads = []
            for index in range(count):
                metadata = {**entry, "episode_id": f"profile-{index:06d}", "training_step_start": index * 4000}
                selection = ReplaySelection(args.run, metadata)
                recording = ReplayPayload(
                    metadata, payload.samples, hud_frames=payload.hud_frames if index == 0 else None,
                )
                payloads.append(SelectedReplayPayload(selection, recording, f"#{255-index%256:02x}{index%256:02x}00"))
            send_replay_swarm(transport, action="play", payloads=payloads, options=options)
            time.sleep(args.settle_seconds)
            samples = []
            for _ in range(3):
                time.sleep(1)
                samples.append(send_replay_swarm(transport, action="status", payloads=[], options=options))
            if samples[-1]["playback_seconds"] <= samples[0]["playback_seconds"]:
                raise RuntimeError(
                    "native playback stopped advancing; discard timing results and inspect browser errors"
                )
            report["cases"].append({"cars": count, "samples": samples})
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Measured {count} native cars", flush=True)
    finally:
        try:
            send_replay_swarm(transport, action="clear", payloads=[], options=options)
        except Exception as exc:
            print(f"Replay cleanup failed: {exc}", file=sys.stderr)
        transport.close()


if __name__ == "__main__":
    main()
