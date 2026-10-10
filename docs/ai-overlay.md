# Live AI HUD

PolyBot can display the policy's current inputs and outputs inside PolyTrack.
The overlay is informational only: it does not write controls, change reward
calculations, alter training, or participate in physics.

## Enable and configure

The PolyBot bridge adds a **Show AI HUD / Hide AI HUD** button directly to the
PolyTrack page. The button remains available whether or not Python is connected;
the panel shows a waiting message until telemetry arrives. Python supplies the
live policy telemetry and does not control whether the panel is shown. The GUI's
**AI HUD** tab configures the display layout and persists those settings to
`config/ai-overlay.json`. Settings can be saved while a run is active and apply
to subsequent policy decisions. Mock runs do not create or transmit HUD frames.

Install the latest PolyBot bridge from the repository's PML URL and select
`latest`. Bridge **0.1.41** adds HUD support for PolyTrack **0.6.3**. The
existing 0.1.37 release for PolyTrack 0.6.2 and 0.1.38 release for 0.6.3 are
left unchanged; they do not advertise HUD support. See
[Running PolyBot in PolyTrack](game-integration.md) for bridge setup.

## HUD modes

Switching modes fades the current panel out and the next panel in over 280 ms
total. OFF fades out; returning fades in. Reduced-motion preferences disable
these animations. Live telemetry continues updating during the transition.

Every mode has a movable, resizable window. Drag the top grip to move it and
the bottom-right grip to resize it; focused grips also accept arrow keys
(Shift for larger changes). Double-click the top grip to reset that mode's
layout. Sizes and positions are saved locally per mode and constrained to the
current viewport. Narrow windows simplify the row layout while retaining all
values. Scrollable panels use thin translucent scrollbars.

When no model or replay is active, the HUD shows “No model or replay loaded”
and placeholder values. Model disconnects and clearing a replay remove the
previous telemetry; paused replays retain it.

Choose a mode in **In-game display > HUD mode**, then save. The in-game mode
selector can switch instantly during live driving or replay playback. OFF hides
the PolyBot panel; the selector remains available to turn it back on. The game's
own UI is separate. Show/Hide AI HUD still toggles the panel.

| Mode | Display |
| --- | --- |
| OFF | No PolyBot HUD panel |
| CONTROLS | Enlarged applied steering, throttle and brake |
| NEURAL_NET | Animated policy schematic with named input/output nodes, live values and controls |
| RL_DEBUG | Inputs, output transformations, applied controls, reward components and episode state |
| REWARD | Large accumulated reward, accumulated components and controls |
| OBSERVATIONS | Curated inputs, car-local upcoming reference-line diagram and controls |
| TRAINING | Compact model, stage, decision count and run budget |
| TRAINING_GRAPH | Model/stage info and completed lap times from this viewing session |
| GHOST_RACE | Timer, checkpoint/reference split, estimated live delta and controls |
| WR_CHASE | Timer, configured WR target, available WR delta and controls |
| COMPARISON | AI/reference speeds, reference split, delta and controls |
| RACING_LINE_ANALYSIS | Checkpoint-section entry/exit speeds, section delta and line offset/heading |
| DETAILED_CONTROLS | Enlarged applied controls with four decimal places |
| CHAMPION | Timer, protected verified best, reference delta and controls |
| MINIMAL_RACE | Timer, reference delta and small controls |

Input modes rank driving signals and show approximately **60%** of the full
input vector, prioritizing speed, alignment, curvature/lookahead, contacts and
skid. This changes only the display. The policy still receives every input.
Full layout uses a wider grid. Lookahead points controls the diagram length.
Scale, labels and visibility options remain available.

Panels use translucent glass with backdrop blur, soft highlights and rounded
edges. Network input and output nodes have names and normalized live values;
the hidden nodes remain illustrative.

Reward displays show accumulated totals **since the start of the current run**,
including accumulated component/group contributions. They reset with the episode.
Raw per-step rewards remain in recorded telemetry for analysis, but are not the
HUD's headline values. Existing replay totals are prepared from the complete
recording before playback, so seeking, looping and switching HUD modes do not
double-count rewards. Learner-scaled and unscaled totals are labelled separately.

The network's hidden layer is an animated **schematic**, not measured hidden
activations. Its input and output values are actual telemetry. The training graph
records completed lap times encountered while viewing; it is not a historical
training log or verified evaluation graph. Checkpoint analysis uses actual
checkpoint changes, not hand-coded corners.

Reference delta matches the nearest saved racing-line position within the current
checkpoint and is labelled as an estimate. Reference speeds are the policy's
reference target speeds at that position. In **WR Chase**, enter a verified world
record time in seconds; zero means unknown. A target time alone cannot establish
a live same-position delta: a matching saved reference trace is required. Missing
values and traces display as unavailable, never invented race results.

Replays use today's saved HUD preferences without modifying their recorded
telemetry. Older recordings may lack newer race/reference/model fields and show
unavailable for those values. Live GUI changes apply at the next policy decision; for
an already loaded replay, switch modes in-game or reload to apply saved GUI
preferences.

## Values and timing

The observation panel is described from the exact `polybot.observation.v2`
vector and any controller/training-state values appended to it. Each row shows
the normalized number passed to the policy and its corresponding source value
and units. If a future observation schema is not explicitly supported by the
HUD, the panel reports that schema instead of assigning potentially incorrect
labels.

The controls panel distinguishes the policy's raw output, any policy
transformation, the adapter's requested control demand, and the average
controls actually applied across the simulator ticks in that action. Reward
totals, terms, groups, and event labels come from the existing Python
environment result. The learner-scaled reward is shown separately from the
unscaled environment reward.

`policy decisions` and `simulator tick` are separate counters. The former is
the global training decision count (`model.num_timesteps`); it is not a
physics-tick count. Simulation time and tick are read from simulator telemetry.
Python currently receives state and rewards at policy-action boundaries, so
the HUD updates at that cadence (with a 30 Hz upper bound). A large frame skip
therefore produces fewer HUD updates; the overlay does not invent intermediate
motion or claim per-tick measurements.

Frames are sent as bounded one-way notifications over the existing WebSocket
only when the bridge advertises `ai_overlay_hud`; no response is requested and
there is no extra request/response cycle. Frames are capped at 30 Hz, and
failed HUD notifications only log a warning. With an older bridge, PolyBot
logs a warning and continues training without the overlay.
