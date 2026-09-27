# Running PolyBot in PolyTrack

The real-game path targets PolyTrack 0.6.3 and retains 0.6.2 compatibility through PolyModLoader. It is for local training and
demonstrations, not leaderboard or multiplayer automation.

## One-time setup

1. Install Python 3.11 or newer and install this repository into its virtual environment:

   ```text
   python -m venv .venv
   # Activate: .venv\Scripts\Activate.ps1 (PowerShell), source .venv/bin/activate (bash)
   python -m pip install -e ".[dev,train,gui]"
   ```

2. Open the [PolyModLoader web build](https://web.polymodloader.com/). If your browser blocks its
   connection to localhost, follow the [PML user guide](https://github-wiki-see.page/m/polytrackmods/PolyModLoader/wiki/For-Users)
   to install the desktop app. Use a loader running PolyTrack 0.6.3 (or the supported 0.6.2 build).
   See the pinned source revisions and limits of the compatibility checks below.

3. Open **Mods**, choose **Add URL**, paste the PolyBot mod URL, select `latest`, then click
   **Load** and **Apply**:

   ```text
   https://cdn.polymodloader.com/gh/michael201110/poly-bot/main/pml-mod
   ```

The repository ships only the source-level mixin under `pml-mod/`. It does not redistribute the
game's JavaScript bundle, WASM binary, or assets.

## Train or play in the game

Load a track and a ghost lap in PolyTrack, then enter the race. The ghost defines the reference route and lookahead; without it, reset reports `missing_reference`. If the mod was enabled after entering a race, restart that race.

Open the v2 GUI and keep **WebSocket** selected, or start an explicit algorithm from the CLI:

```powershell
polybot-gui
polybot-train --algorithm tqc --backend websocket --track-name "Summer 1" --track-id current --frame-skip 30 --timesteps 100000 --reward-profile Balanced
```

The Python listener waits at `ws://127.0.0.1:8765` for the mod. The GUI can train PPO, DQN, or TQC, stop cleanly, evaluate, and play latest or champion. A model is only champion after deterministic full-track evaluation.

```powershell
polybot-eval --algorithm tqc --track-name "Summer 1" --slot champion --backend websocket --episodes 5
polybot-drive --algorithm tqc --track-name "Summer 1" --slot champion --backend websocket --realtime
```

The training command defaults to `--track-id current` and frame skip 30 for WebSocket, and `mock/straight` and 4 for the mock. The simulator follows Python's fixed-step requests rather than render timing. Only one training, evaluation, or playback listener can use the fixed local port at once.

## Troubleshooting

If the mod does not connect, confirm PolyModLoader is enabled in the active race, the loaded track has a ghost reference, and no other PolyBot process owns port 8765. Run `polybot-doctor --smoke tqc` to check the selected compute device independently of the game. Run `python tools/validate_pml_mod.py` for manifest validation; include raw pinned game bundles for the stronger source check described below.

## Integration details

The mod uses PolyModLoader's `registerSimWorkerMixin` extension point. It connects the simulation
worker directly to the Python WebSocket server and implements the versioned `hello`, `reset`, and
`step` operations in [`protocol.md`](protocol.md). Driving uses worker messages; native Backspace keyboard events synchronize the
main-thread recorder on finish and aborted-run resets.

Read-only leaderboard access remains available solely to load a reference ghost. The mod rejects
leaderboard/profile writes, verification calls, multiplayer sockets, and ICE-server requests, and
allows local finish feedback before a native restart.

One native `updateCarModel` call advances one millisecond of physics. A policy action is held for
the requested number of ticks; 10 ticks gives a 100 Hz control rate. A true episode reset uses the
worker's original delete/create/start path rather than the game's checkpoint-respawn control.

The authoritative 0.6.2/0.6.3 state packet supplies transform, speed, checkpoint/finish state, wheel
contacts, suspension values and velocities, wheel skid, steering, and applied controls. Linear and
angular velocities and acceleration are derived from consecutive transforms. The route reference
supplies progress, lateral/heading error, policy lookahead points, position-aligned ghost pose, target
speed, and the recorded expert controls.

The token-based mixin depends on exact source anchors. Updating to another PolyTrack version requires
checking the worker tokens, state decoder, one-tick helper, and reset path before changing the
manifest target. Useful upstream references are:

- [worker initialization and version check](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/simulation_worker.bundle.js#L19227-L19235)
- [native update and worker loops](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/simulation_worker.bundle.js#L19571-L19676)
- [authoritative state decoder](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/main.bundle.js#L13530-L13647)
- [`registerSimWorkerMixin` type](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/PolyTypes.d.ts#L136-L179)

Before treating a new adapter as training-ready, verify that it can reset a simple track
repeatably, advance an exact tick count, report ordered progress and checkpoints, replay the same
action transcript without meaningful divergence, reject stale episode IDs, and keep public writes
and multiplayer disabled.

## Bundle validation

The September 2026 maintenance check uses raw upstream bundles from these immutable revisions:

| PolyTrack | PolyModLoader source revision | Mod | Imported worker runtime |
| --- | --- | --- | --- |
| 0.6.2 | [`c46423b`](https://github.com/polytrackmods/PolyModLoader/tree/c46423b1774939b97302c9f23ff4e3d86179156e) | 0.1.29 | 0.1.28 |
| 0.6.3 | [`6ba4f09`](https://git.polymodloader.com/polytrackmods/PolyModLoader/src/commit/6ba4f099a7b9b11ba88c7152d2246240ba04a6d7) | 0.1.29 | 0.1.28 |

The 0.1.29 entry point deliberately reuses the 0.1.28 worker at the immutable PolyBot commit
`020ea536816934f307904b79fc51d2edb16cf789`. Its bundled worker copy is identical. The 0.6.2
PolyTypes import is also intentional; it supplies the loader API used by both targets.
The worker learns the actual game version from the game's initialization message.

[Kodub's 0.6.3 release notes](https://kodub.itch.io/polytrack/devlog/1665945/polytrack-063-track-of-the-week)
describe compatibility with earlier 0.6 releases. Source comparison additionally confirms that the
worker message handler and physics stepping code are unchanged from the checked 0.6.2 revision;
the worker adds a finish-detector helper and updates its version check. This is source validation,
not a completed in-game driving or determinism test.

Download `main.bundle.js` and `simulation_worker.bundle.js` from the appropriate revision above
without changing their bytes or line endings, then run:

```text
python tools/validate_pml_mod.py --game-version 0.6.3 --worker /path/to/simulation_worker.bundle.js --main /path/to/main.bundle.js
```

The default game version is 0.6.3. Use `--game-version 0.6.2` for the retained target.
The validator checks SHA-256 hashes of the raw files and exact source anchors, accounting for
PolyModLoader's built-in worker URL replacement before our mixin runs. `--anchors-only` bypasses
hash checks for investigation; matching anchors alone do not establish compatibility with a new
game version. Running without bundle paths checks only the release manifests.

After a game or loader update, run `polybot-drive --algorithm tqc --track-name "Summer 1" --slot champion --backend websocket` with a reference ghost on a
simple track. Check initial connection, reset, checkpoint progress, local finish feedback and
restart, then perform the deterministic transcript checks described above. Inspect network traffic
to verify that public writes and multiplayer remain blocked. These interactive checks require a
running game and are not covered by the Python test suite.
