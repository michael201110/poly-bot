# Connecting a local PolyTrack build

The integration target is the downloadable desktop build, running locally and offline. The hosted
game is not suitable for high-throughput training and must not be automated against public services.

## Confirmed PolyTrack 0.6.2 seam

The community PolyModLoader project has a public PolyTrack 0.6.2 build. Pin commit
[`c46423b`](https://github.com/polytrackmods/PolyModLoader/commit/c46423b1774939b97302c9f23ff4e3d86179156e)
rather than assuming its default branch targets the same game version. The worker performs its own
literal version check, so a mismatch should fail closed.

The build identifies these important files inside the desktop app:

- `main.bundle.js` — gameplay and UI code;
- `simulation_worker.bundle.js` — simulation worker logic;
- `electron/main.js` and `electron/preload.js` — Electron process bridge.

The most useful supported extension point is PolyModLoader's `registerSimWorkerMixin`. It transforms
the worker source before creating its Blob URL. These transformations are token-based and therefore
brittle: every mixin must assert how many tokens it replaced and refuse to run on an unknown build.

Inside the worker, the existing native `updateCarModel` call advances one physics tick and writes an
encoded car-state packet. One call represents **1 ms**, not one rendered frame. The normal realtime
loop accumulates wall time; the existing fast loop consumes prerecorded ghost controls. Neither loop
should drive RL. Add a third, manually stepped mode around the existing one-tick helper.

Relevant upstream locations:

- [worker initialization and version check](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/simulation_worker.bundle.js#L19227-L19235)
- [native update and realtime loop](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/simulation_worker.bundle.js#L19571-L19640)
- [existing fast replay loop](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/simulation_worker.bundle.js#L19642-L19676)
- [`registerSimWorkerMixin` type](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/PolyTypes.d.ts#L136-L179)

Do not commit or redistribute a downloaded/decompiled game bundle, WASM binary, or copyrighted assets
to this repository. Ship only the compatibility-checked mixin and require a local game installation.

## Worker patch design

Vanilla worker message IDs occupy `0` through `10`. Reserve custom IDs outside that range, initially:

- `100` — `AiStep { requestId, carId, repeat, controls }`
- `101` — `AiStepResult { requestId, framesAdvanced, stateBuffer }`

The mixin should:

1. Add an AI-manual mode that suppresses realtime scheduling and the fast ghost loop.
2. Accept only one in-flight step for one car per worker initially.
3. On `AiStep`, hold the supplied five control booleans and invoke the existing native one-tick helper
   `repeat` times synchronously.
4. Increment the game's frame counter on each call and stop early at finish or the game's frame limit.
5. Return the final encoded state packet, request ID, and actual number of frames advanced.

A policy repeat of 10–20 native ticks gives a 100–50 Hz control rate. The Python default of four is
appropriate for the slower mock simulator; pass `--frame-skip 10` or higher for the 1 kHz real worker.

Use `DeleteCar`, followed by the original cached `CreateCar` payload and `StartCar`, for a true episode
reset. The normal reset control is a checkpoint respawn and does not guarantee a clean initial state.

The encoded packet is variable-length with a 227-byte maximum. Parse it using the game's decoder;
do not hash or compare the unused tail, which may contain stale bytes. The packet includes transform,
speed, checkpoint/finish state, contacts, suspension/wheel values, steering, and applied controls. It
does **not** include linear or angular velocity vectors; derive those from consecutive transforms with
`dt = framesAdvanced * 0.001`. Continuous route progress and lookahead must come from decoded track
geometry, gated by the native ordered checkpoint index.

Reference: [authoritative 0.6.2 state decoder](https://github.com/polytrackmods/PolyModLoader/blob/c46423b1774939b97302c9f23ff4e3d86179156e/main.bundle.js#L13530-L13647)

## Adapter contract

Implement `PolyTrackTrainingGameApi` using
[`bridge/polytrack_game_api.stub.js`](../bridge/polytrack_game_api.stub.js). The transport bridge calls
five methods:

1. `ensureOffline()` blocks leaderboard, multiplayer, analytics, and remote game API calls. It must
   return exactly `true` before the trainer is allowed to connect.
2. `describe()` returns the fixed physics timestep, game version, and maximum ticks per action.
3. `reset(...)` loads a track, restores a clean initial state, and returns telemetry at tick zero.
4. `step(...)` holds the supplied controls while synchronously advancing an exact number of physics
   ticks, then returns telemetry and events.
5. `close()` releases controls and restores reversible hooks.

Load the game API implementation before `polytrack_training_bridge.js`, then connect it:

```js
const gameApi = new PolyTrackTrainingGameApi();
const bridge = new PolyTrackTrainingBridge({
  gameApi,
  url: "ws://127.0.0.1:8765",
});
await bridge.start();
```

## Recommended discovery order

1. Locate the worker message that supplies driving inputs.
2. Add the custom manual-step messages around the existing 1 ms update call.
3. Cache the original create payload and implement delete/create/start episode reset.
4. Reuse the authoritative state decoder in `main.bundle.js` rather than guessing packet offsets.
5. Derive velocity from consecutive decoded transforms.
6. Decode the active track into ordered centreline samples and checkpoint-gated progress.
7. Run PolyTrack's existing determinism test, then replay the same action transcript at least 20 times.

Avoid starting with DOM scraping. Speed text and timer labels are insufficient for stable control,
and synthetic keyboard events do not provide deterministic stepping.

## First acceptance test

The real adapter is ready for initial training when it can:

- reset one simple track to byte-equivalent or tolerance-equivalent telemetry;
- hold throttle for exactly 240 ticks and report increasing ordered progress;
- replay an action transcript repeatedly without meaningful divergence;
- reject a stale `episode_id` after reset;
- complete the hand-written centreline-controller test;
- operate with public network requests disabled.

PolyTrack's terms prohibit fraudulent leaderboard records and disruptive automated server access.
Keep all training and evaluation local, and clearly label any recorded AI demonstration.
