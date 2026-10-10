# Visual replay recording (Stage 2)

Training can save a compact, render-oriented transform history for each attempt.
Recording observes the environment result only; it does not change actions,
rewards, observations supplied to the learner, or simulator stepping.

During visual playback, the native in-game timer follows the fixed camera run's
replay clock, including pause and seek. It holds that run's finish time while
the car coasts away. Clearing playback restores the normal player timer.
Only the fixed camera run produces car audio during visual playback; other
swarm cars, the hidden player and native ghosts are muted. Pausing mutes it,
and its sound fades while it coasts away at the end. Native SFX volume applies.

## Configuration

`visual_replay_enabled` accepts `true`, `false`, or `null` (automatic). Automatic
is the default: it records WebSocket/PolyTrack runs and skips mock runs. In the
GUI Replay tab, choose **Automatic for live training**, **Always record**, or
**Do not record**. The same setting remains available under General settings.
The CLI accepts:

```powershell
polybot-train --algorithm tqc --backend websocket --visual-replay --visual-replay-sample-hz 20
polybot-train --algorithm tqc --backend websocket --no-visual-replay
```

`visual_replay_sample_hz` is a maximum transform sampling rate; its default is
20 Hz. Samples are selected from telemetry already returned to Python and the
episode's initial and final transforms are retained. It cannot increase the
available telemetry rate. `visual_replay_observations` defaults to `true`;
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

The track slug scopes this path, and the Replay Swarm GUI lists runs only for
the selected track and algorithm. See [Track workspaces](tracks.md) for
registry and migration details.

The JSON index contains searchable episode metadata and is updated atomically
after an NPZ is committed. It lets a future replay tool filter by training
step, track, status, duration, and progress without opening every payload. A
run with no `visual_replays` directory or no `index.json` simply has no
recorded replays.

Each compressed NPZ contains arrays:

* `ticks` (`int64`) and `elapsed_s` (`float64`)
* `position_m` (`float32`, N×3)
* `quaternion_xyzw` (`float32`, N×4)
* optional `wheel_state` (`float32`, N×42), captured by bridge 0.1.41: for
  each wheel, contact flag, contact position XYZ, contact normal XYZ,
  suspension length, rotation delta, and skid value; followed by steering and
  the brake-light flag.

When observation recording is enabled, the payload additionally contains
`decision_ticks`, `decision_elapsed_s`, `observations`, and `actions`. These
decision-rate arrays are separate from the visual transform samples. No model
weights or neural-network replay-buffer entries are included.

Observation recording also embeds labelled features in the replay HUD, independently
of which live HUD panels are visible. Features are captured **before** the action
and match its saved policy input exactly, including controller/training-state
features. `observation_tick` and `observation_elapsed_s` identify that input;
the frame's simulator tick/time identify the action's resulting state and reward.
`observation_source: recorded` distinguishes these values from any estimates.
The HUD also saves raw simulator telemetry, speed, checkpoint/line alignment,
reference information and delta estimates, plus the protected best time and
model context when available. Native wheel state remains in the separate
transform-rate array. Recording does not modify policy inputs or train the model.

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
wheel telemetry adds 168 bytes per sample before compression (about 101 MB at
600,000 samples); optional observation logging can also increase storage.

The current renderer interpolates from recorded samples, including recorded
wheel and suspension state when available. Native skid marks use that wheel
state and respect the game's skid-mark setting. Older transform-only replays
remain supported: wheel contacts are estimated on the actual track surface,
and visual slip is estimated from lateral motion. Those older skid marks are
an approximation, not recovered historical tyre telemetry. Neither method
changes replay trajectories or physics. A native per-physics-tick recorder is
not yet available.

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
shared with dry-run inspection. The default scale is red at step 0, orange at
500,000, yellow at 1,000,000, yellow-green at 1,500,000, and green at 2,000,000.
Adjust **Green at step** beside the colour legend (CLI:
`--color-max-step`); the intermediate stops spread evenly across that range.
Later steps remain green. Choosing another run does not change this setting.
Attempts from the same training step have the same colour, including the
100-attempt collection recorded from the frozen champion.
**Apply colours** recolours the currently loaded cars without restarting them.
All ghosts start at replay time zero and share one playback clock. A short
episode may disappear, freeze, or fade at its own recorded endpoint while
longer episodes continue.
Cars default to 100% opacity; appearance controls can reduce it. Seeking or
restarting clears skid trails so marks never bridge a time jump.
Replay cars share one stock body, wheel, and exhaust style. Their frame and rim
colors stay charcoal and light gray, while both body paint colors match the
training-step gradient.
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

### Replay GUI

The Replay tab gives the usual workflow first: choose a saved run, highlight an
attempt, and press **Watch selected attempt**. The run list shows when the run
was recorded, how many attempts it contains, and its best finished lap. Use
**Refresh runs** after training to find newly saved attempts. Finished times,
failure status, and progress appear beside each attempt. Click a row to watch
that attempt; tick its checkbox to include it in a comparison. Hovering shows
the recorded sample count and training step.

To compare attempts, tick them in the second list and press **Compare selected
attempts**. Leaving all attempts unticked compares all matching attempts in the
selected run. Pause, resume, restart, clear, and jump-to-time controls stay
visible while advanced filters, external folders, appearance, and bridge
settings are tucked under **Advanced filters and playback settings**.
Enable **Play alongside loaded ghosts** to keep the game's loaded ghosts
visible during either a single replay or a swarm. They follow the replay
opacity setting, and changing the option or opacity applies to a replay that is
already loaded.

Loading replay data never resumes training or changes model weights. Stop any
active trainer/evaluator before using the GUI to control its local bridge.

`status` reports loaded/visible ghosts, shared playback position and duration,
training-step range, and average replay-render update time (not GPU frame time).
`--max-cars` supports 1–500, but actual comfortable counts depend on the game
and hardware. Requests are rejected above 500 ghosts, 250,000 aggregate
samples, 500,000 samples per episode, or 64 MiB of encoded trajectory/HUD data.
Live profiling on the T500 identified rendering and GPU memory pressure as the
main swarm limit; see [swarm performance findings](swarm-performance.md).
The renderer constructs one native car per episode. Status also provides a
rolling 60-frame `performance` block: frame interval, replay update CPU time,
draw calls, triangles, and time inside native renderer calls. Renderer time
includes driver/GPU waits; it is not a GPU timestamp measurement.

Repeat the synthetic renderer benchmark with
`node --expose-gc tools/benchmark_replay_swarm.mjs`. It exercises the renderer
state block from bridge 0.1.38 using mock cars and 120 samples per ghost. One run
measured about 0.091/0.139/0.412/0.525 ms per measured frame for 50/100/250/500
ghosts. Heap deltas were 1.14/1.84/4.29/4.04 MiB in that run; V8 heap values
fluctuate. These figures exclude native car construction, game rendering,
geometry, shadows, and GPU work; they are not PolyTrack FPS estimates.

### Exact manual live-test checklist

1. Leave **Save replays** on Automatic (or select Always record), train briefly, then stop training.
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
