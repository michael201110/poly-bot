# Running PolyBot in PolyTrack

The real-game path targets PolyTrack 0.6.2 through PolyModLoader. It is for local training and
demonstrations, not leaderboard or multiplayer automation.

## One-time setup

1. Install Python 3.11 or newer and install this repository into its virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install -e ".[dev,train]"
   ```

2. Open the [PolyModLoader web build](https://web.polymodloader.com/). If your browser blocks its
   connection to localhost, follow the [PML user guide](https://github-wiki-see.page/m/polytrackmods/PolyModLoader/wiki/For-Users)
   to install the desktop app. The adapter is checked against the exact
   [v0.6.2-2 source](https://github.com/polytrackmods/PolyModLoader/tree/v0.6.2-2) and requires a
   loader running PolyTrack 0.6.2.

3. Open **Mods**, choose **Add URL**, paste the PolyBot mod URL, select `latest`, then click
   **Load** and **Apply**:

   ```text
   https://cdn.polymodloader.com/gh/michael201110/poly-bot/main/pml-mod
   ```

The repository ships only the source-level mixin under `pml-mod/`. It does not redistribute the
game's JavaScript bundle, WASM binary, or assets.

## Let the built-in controller drive

Start the Python listener:

```powershell
cd C:\Users\ASUS\Documents\GitHub\poly-bot
.venv\Scripts\Activate.ps1
polybot-drive --centerline
```

The `Waiting for the local PolyTrack mod at ws://127.0.0.1:8765 ...` line is expected. In PolyTrack,
choose a track, load a ghost lap, and enter the race. The game worker connects to Python and the
controller's actions are applied to the visible car. If a race was already open when you enabled the
mod, restart that race.

The adapter uses the selected ghost trajectory as its route reference. Without one, reset
fails with `missing_reference`. The centreline controller is mainly an end-to-end compatibility
check; it has not learned the track and may fail on difficult sections.

## Drive with a trained PPO policy

Pass the model created by `polybot-train`; the `.zip` suffix is optional:

```powershell
polybot-drive --model models/polybot-real
```

Predictions are deterministic by default. Add `--stochastic` only when deliberately sampling PPO's
action distribution. A model must use the same observation layout, especially `--lookahead`, that
was used for training.

The drive command defaults to the current game track, 12 lookahead samples, a 10-tick action repeat,
a 30,000-step (300 simulated second) episode limit, one episode, and the fixed localhost endpoint:

```powershell
polybot-drive --track current --lookahead 12 --frame-skip 10
```

## Train against the real worker

Start training before launching the race, just as for `polybot-drive`:

```powershell
polybot-train --backend websocket --timesteps 1000000 --model-out models/polybot-real
```

For the WebSocket backend, `polybot-train` automatically defaults to `--track current` and
`--frame-skip 10`; these remain `mock/gentle-s` and `4` for the mock backend. Resetting an episode
recreates the car at the race start. The game remains visible, but fixed-step training is controlled
by Python rather than render timing.

You can evaluate a model against the currently selected race without changing it:

```powershell
polybot-eval models/polybot-real --backend websocket --episodes 5
```

## Troubleshooting

- **`polybot-drive` is not recognized:** activate `.venv`, then run
  `python -m pip install -e ".[dev,train]"` again. As a direct fallback, run
  `.venv\Scripts\polybot-drive.exe --centerline`.
- **It stays on `Waiting ...`:** the listener is working but the mod has not connected. Confirm the
  PolyBot mod is enabled, launch through PolyModLoader, and enter a race. Check that both sides use
  port 8765.
- **The browser blocks localhost access:** allow local-network access when prompted, or use the
  compatible PolyModLoader desktop build.
- **`missing_reference`:** load a ghost for the selected track and restart the race.
- **Version or mixin-token error:** the local game worker is not the supported 0.6.2 build. The mod
  deliberately refuses to patch an unknown bundle.
- **Model observation-space error:** pass the same `--lookahead` value used during training.
- **Reset or step timeout:** let the current operation finish, or increase
  `--request-timeout 120`. The connection wait can similarly be changed with
  `--connect-timeout`.

## Integration details

The mod uses PolyModLoader's `registerSimWorkerMixin` extension point. It connects the simulation
worker directly to the Python WebSocket server and implements the versioned `hello`, `reset`, and
`step` operations in [`protocol.md`](protocol.md). No DOM scraping or synthetic keyboard events are
involved.

Read-only leaderboard access remains available solely to load a reference ghost. The mod rejects
leaderboard/profile writes, verification calls, multiplayer sockets, and ICE-server requests, and
masks the AI car's finish state before the game UI receives it.

One native `updateCarModel` call advances one millisecond of physics. A policy action is held for
the requested number of ticks; 10 ticks gives a 100 Hz control rate. A true episode reset uses the
worker's original delete/create/start path rather than the game's checkpoint-respawn control.

The authoritative 0.6.2 state packet supplies transform, speed, checkpoint/finish state, wheel
contacts, suspension values, steering, and applied controls. Linear and angular velocities are
derived from consecutive transforms. The route reference supplies progress, lateral/heading error,
and policy lookahead points.

The token-based mixin is intentionally fail-closed. Updating to another PolyTrack version requires
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
