# Visual replay recording (Stage 2)

Training can save a compact, render-oriented transform history for each attempt.
Recording observes the environment result only; it does not change actions,
rewards, observations supplied to the learner, or simulator stepping.

## Configuration

`visual_replay_enabled` accepts `true`, `false`, or `null` (automatic). Automatic
is the default: it records WebSocket/PolyTrack runs and skips mock runs. In the
GUI, choose **automatic**, **enabled**, or **disabled** under General settings.
The CLI accepts:

```powershell
polybot-train --algorithm tqc --backend websocket --visual-replay --visual-replay-sample-hz 20
polybot-train --algorithm tqc --backend websocket --no-visual-replay
```

`visual_replay_sample_hz` is a maximum transform sampling rate; its default is
20 Hz. Samples are selected from telemetry already returned to Python and the
episode's initial and final transforms are retained. It cannot increase the
available telemetry rate. `visual_replay_observations` defaults to `false`;
enabling it stores the policy observation and requested action separately at
every policy decision.

## Files and representation

Each training invocation gets a unique run directory under the existing model
algorithm hierarchy:

```text
<output-root>/<track-slug>/<algorithm>/visual_replays/<run-id>/
    index.json
    episode-000001.npz
    episode-000002.npz
    ...
```

The JSON index contains searchable episode metadata and is updated atomically
after an NPZ is committed. It lets a future replay tool filter by training
step, track, status, duration, and progress without opening every payload. A
run with no `visual_replays` directory or no `index.json` simply has no
recorded replays.

Each compressed NPZ contains arrays:

* `ticks` (`int64`) and `elapsed_s` (`float64`)
* `position_m` (`float32`, N×3)
* `quaternion_xyzw` (`float32`, N×4)

When observation recording is enabled, the payload additionally contains
`decision_ticks`, `decision_elapsed_s`, `observations`, and `actions`. These
decision-rate arrays are separate from the visual transform samples. No model
weights or neural-network replay-buffer entries are included.

Writes use a bounded background queue and temporary files followed by atomic
replacement. If the queue is full or persistence fails, PolyBot emits a
warning and drops the affected visual replay; training continues.

## Sampling resolution and approximate size

The stored schema accepts arbitrary timestamped transforms, so a future native
recorder can supply per-tick samples. **At present Python only receives the
final transform for each policy/environment action.** With frame skip 30, this
means roughly one available transform per 30 physics ticks, not a reconstructed
20 Hz trajectory. The configured rate only decimates available samples; it
cannot interpolate missing physics states.

With the game's documented 1 ms physics tick, frame skip 30, a 20 Hz maximum,
and all requested ticks advancing, one million policy decisions correspond to
at most about 600,000 visual samples. The four required arrays occupy about
26 MB before compression at that sample count. A typical compressed run should
be approximately **15–30 MB**, depending on track motion, episode count, and
ZIP/index overhead. This is an estimate, not a fixed size guarantee. Optional
observation logging can make storage substantially larger.

The current renderer interpolates from recorded samples. It does not yet add a
native per-physics-tick recorder or wheel/suspension animation.

## Inspecting a swarm selection (Stage 3)

`polybot-replay-swarm --dry-run` prints a selection report without connecting
to PolyTrack or opening NPZ payloads. Point `--run` at a replay run
directory containing `index.json`, at its `visual_replays` parent, or at a
parent containing `visual_replays/<run-id>` directories:

```powershell
polybot-replay-swarm --run models/summer-1/tqc/visual_replays/run-id `
  --steps 0:25000 --max-cars 200 --color-min-step 0 `
  --color-max-step 1000000 --dry-run
```

Step and episode bounds are inclusive. Episode ranges use the numeric suffix
of IDs such as `episode-000418`. `--finished-only` and `--failed-only` are
mutually exclusive; failed-only includes `failed` and `timeout` statuses.
When selection exceeds `--max-cars`, a seeded equal-rank stratified sample
spans the sorted training-age distribution. Set `--sample-seed` to reproduce
the selection. Colors interpolate through red, orange, yellow, lime, and green
and clamp at the configured minimum/maximum steps. Repeat `--color-stop
STEP:COLOR` to replace the default gradient with custom stops, for example:

```powershell
polybot-replay-swarm --run <replay-run> --steps 0:1000000 --max-cars 250 `
  --color-stop 0:#ff0000 --color-stop 250000:#ff8000 `
  --color-stop 500000:#ffff00 --color-stop 750000:#80ff00 `
  --color-stop 1000000:#00ff00 --dry-run
```

## Playing a replay swarm in PolyTrack

The CLI reuses the existing `polybot.sim` version-2 WebSocket connection. It
loads all selected trajectories once in bounded chunks; a single shared clock
then drives renderer-only native cars on the game main thread. It creates no
simulation-worker cars and starts no additional server. The active trainer or
evaluator must be stopped while controlling the bridge on that same port.
PML releases 0.1.37 and 0.1.38 target PolyTrack 0.6.2 and 0.6.3 respectively.

### One, 50, and 250 cars

```powershell
polybot-replay-swarm --run models/summer-1/tqc/visual_replays/run-id `
  --steps 0:25000 --max-cars 1 --color-min-step 0 `
  --color-max-step 1000000 --action play

polybot-replay-swarm --run models/summer-1/tqc/visual_replays/run-id `
  --steps 0:25000 --max-cars 50 --action play

polybot-replay-swarm --run models/summer-1/tqc/visual_replays/run-id `
  --steps 0:1000000 --max-cars 250 --color-min-step 0 `
  --color-max-step 1000000 --opacity 0.45 --end-behavior fade --action play
```

For a custom gradient:

```powershell
polybot-replay-swarm --run <run> --max-cars 200 `
  --color-stop 0:#ff0000 --color-stop 250000:#ff8000 `
  --color-stop 500000:#ffff00 --color-stop 750000:#80ff00 `
  --color-stop 1000000:#00ff00 --action play
```

Each ghost receives its own color from `training_step_start`; step range
selection, seeded stratified sampling, and the existing color-scale code are
shared with dry-run inspection. All ghosts start at replay time zero and share
one playback clock. A short episode may disappear, freeze, or fade at its own
recorded endpoint while longer episodes continue.
The loader skips optional observation/action arrays when preparing visual
playback, even if those were recorded for training analysis.

`load` loads the selected swarm paused. Controls address the loaded swarm:

```powershell
polybot-replay-swarm --run <run> --action status
polybot-replay-swarm --run <run> --action pause
polybot-replay-swarm --run <run> --action resume --speed 2
polybot-replay-swarm --run <run> --action restart
polybot-replay-swarm --run <run> --action seek --seek-seconds 8.5
polybot-replay-swarm --run <run> --action configure --opacity 0.4
polybot-replay-swarm --run <run> --action clear
```

### Replay Swarm GUI

The **Replay Swarm** tab provides the same index filtering, deterministic
stratified selection, training-age colours, and bridge actions as the CLI.
Choose a replay run directory (or its parent), set the inclusive episode-start
step range, and optionally set episode IDs or a finish/failure filter. Custom
colour stops use `STEP:#RRGGBB` entries separated by commas or newlines.
**Inspect selection** reads only `index.json` files and prints the selection
summary without connecting to PolyTrack; **Set full run range** fills in the
observed training-step bounds.

**Load swarm** loads ghosts paused and **Play swarm** loads and starts them.
Pause, resume, restart, seek, settings, clear, and bridge-status actions work
without a replay path once a swarm is loaded. These operations run in a
background thread so index reads, compressed-payload loading, and bridge
requests do not block the interface. Stop any active trainer/evaluator before
using the GUI to control its local bridge.

`status` reports loaded/visible ghosts, shared playback position and duration,
training-step range, and average replay-render update time (not GPU frame time).
`--max-cars` supports 1–500, but actual comfortable counts depend on the game
and hardware. Requests are rejected above 500 ghosts, 250,000 aggregate
samples, 500,000 samples per episode, or 32 MiB of encoded trajectory data.
No live GPU/FPS measurement has been made. The renderer currently constructs
one native renderer car per selected episode; native geometry/material sharing
and shadow costs have not been established from the public mod API.

Repeat the synthetic renderer benchmark with
`node --expose-gc tools/benchmark_replay_swarm.mjs`. It exercises the renderer
state block from bridge 0.1.38 using mock cars and 120 samples per ghost. One run
measured about 0.091/0.139/0.412/0.525 ms per measured frame for 50/100/250/500
ghosts. Heap deltas were 1.14/1.84/4.29/4.04 MiB in that run; V8 heap values
fluctuate. These figures exclude native car construction, game rendering,
geometry, shadows, and GPU work; they are not PolyTrack FPS estimates.

### Exact manual live-test checklist

1. Train briefly with visual replay recording enabled, then stop training.
2. Find the output under `<output-root>/<track-slug>/<algorithm>/visual_replays/<run-id>/`.
3. Launch PolyTrack 0.6.3 and load Summer 1.
4. Confirm PML has loaded bridge 0.1.38; use bridge 0.1.37 for PolyTrack 0.6.2.
5. Run the one-car command above and verify its per-step color and recorded path.
6. Clear the ghost, then run the 50-car command; note FPS and check age colors.
7. Clear, then load 100 cars and note FPS.
8. Clear, then load 250 cars and note FPS; stop if the game becomes unstable.
9. If performance permits, repeat with 500 cars (the configured hard ceiling).
10. Check the selected step range and that early and later attempts have distinct tints.
11. Exercise pause and resume; verify all ghosts stop and move together.
12. Exercise restart; verify all ghosts return to their own first sample together.
13. Seek to a mid-run time and verify each ghost's position and orientation.
14. Confirm early-ended replays fade while longer replays continue.
15. Change `--end-behavior disappear` and `freeze` and verify both endpoints.
16. Clear and verify every ghost disappears; reload and clear again to check for leaks.
17. Check the real car does not collide with ghosts and remains controllable.
18. Confirm ghosts do not activate checkpoints, affect results, or add leaderboard entries.
19. Confirm the game has only the real player as a physics participant.
20. Stop PolyTrack control, resume ordinary training, and confirm training still runs normally.
21. Record FPS and `--action status` average render-update time at each tested count.

Python currently records only the final transform returned for each
policy/environment action. Rendering interpolation makes those samples move
smoothly, but does not recover the missing intermediate physics-tick
transforms (for frame skip 30, at most one recorded point per 30 ticks).
High-frequency fidelity requires the future native/PML per-tick recorder.
