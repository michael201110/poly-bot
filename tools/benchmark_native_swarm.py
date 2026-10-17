"""Profile native renderer cars in an already installed PML bridge. No training.

Duplicates one complete saved attempt with all recorded wheel samples intact;
only IDs/colors differ. Default source must have <=500 samples for 500 cars.
The bridge's rolling metrics measure real frames, not a mocked Three.js scene.
"""
from __future__ import annotations

import argparse
import json
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
    args = parser.parse_args()
    entries = json.loads((args.run / "index.json").read_text(encoding="utf-8"))["episodes"]
    entry = next(e for e in entries if e["episode_id"] == args.episode)
    payload = load_replay_episode(args.run, entry, include_optional=False, include_hud=False)
    options = ReplayPlaybackOptions(speed=.1, opacity=1.)
    report = {"source": str(args.run), "episode": args.episode, "samples_per_car": len(payload.samples), "cases": []}
    transport = WebSocketServerTransport(port=args.port, connect_timeout_s=60, request_timeout_s=120)
    try:
        for count in args.sizes:
            if count * len(payload.samples) > 250000:
                raise ValueError("whole attempt exceeds bridge sample budget; choose a shorter saved attempt")
            payloads = []
            for index in range(count):
                metadata = {**entry, "episode_id": f"profile-{index:06d}", "training_step_start": index * 4000}
                selection = ReplaySelection(args.run, metadata)
                payloads.append(SelectedReplayPayload(selection, ReplayPayload(metadata, payload.samples),
                                                       f"#{255-index%256:02x}{index%256:02x}00"))
            send_replay_swarm(transport, action="play", payloads=payloads, options=options)
            time.sleep(3)
            samples = []
            for _ in range(3):
                time.sleep(1)
                samples.append(send_replay_swarm(transport, action="status", payloads=[], options=options))
            report["cases"].append({"cars": count, "samples": samples})
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Measured {count} native cars", flush=True)
        send_replay_swarm(transport, action="clear", payloads=[], options=options)
    finally:
        transport.close()


if __name__ == "__main__":
    main()
