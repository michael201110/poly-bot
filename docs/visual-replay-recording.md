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

Stage 2 only records and stores data. Stage 3 adds dry-run replay selection and
trajectory/color utilities; in-game ghost rendering remains a later stage.

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

## Playing one replay in PolyTrack (Stage 4)

Stage 4 displays **one** selected episode as a non-physical native renderer car.
It sends chunked transforms through the existing `polybot.sim` version-2
WebSocket connection; it does not create a second server or change simulation
steps. The player must be in a loaded race with the PolyBot mod enabled. Start
no trainer/evaluator on the same port while controlling the ghost.

```powershell
polybot-replay-swarm --run models/summer-1/tqc/visual_replays/run-id `
  --steps 0:25000 --max-cars 1 --color-min-step 0 `
  --color-max-step 1000000 --action play
```

`play` loads and starts the selected replay. `load` loads it paused; `resume`,
`pause`, `restart`, and `clear` control the already loaded ghost. `seek` requires
`--seek-seconds` measured from the first sample; optional `--speed` (0.1–8),
`--opacity` (0–1), `--color`, and `--end-behavior` (disappear/freeze/fade)
configure playback (the default opacity is 0.5). Use `--action configure` to change settings without
changing whether the ghost is playing:

```powershell
polybot-replay-swarm --run <replay-run> --max-cars 1 --action pause
polybot-replay-swarm --run <replay-run> --max-cars 1 --action resume --speed 2
polybot-replay-swarm --run <replay-run> --max-cars 1 --action seek --seek-seconds 8.5
polybot-replay-swarm --run <replay-run> --max-cars 1 --action configure --opacity 0.4 --color "#44cc88"
polybot-replay-swarm --run <replay-run> --max-cars 1 --action clear
```

Manual integration checklist: load a track, enter its race, issue the play
command, then verify the replay moves without moving/colliding with the player
car; exercise pause/resume, seek, restart, color, opacity, end behavior, and
clear. Repeat with PolyTrack 0.6.2 and 0.6.3. No live game session was available
for this implementation, so rendering/FPS and CPU/GPU impact remain unmeasured.

Python currently records only the final transform returned for each
policy/environment action. Rendering interpolation makes those samples move
smoothly, but does not recover the missing intermediate physics-tick
transforms (for frame skip 30, at most one recorded point per 30 ticks).
High-frequency fidelity requires the future native/PML per-tick recorder.
Stage 4 is single-car only; it does not yet implement the multi-episode swarm.
