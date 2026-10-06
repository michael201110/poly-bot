# Live AI HUD

PolyBot can display the policy's current inputs and outputs inside PolyTrack.
The overlay is informational only: it does not write controls, change reward
calculations, alter training, or participate in physics.

## Enable and configure

The GUI's **AI HUD** tab controls the overlay and persists its settings to
`config/ai-overlay.json`. Settings can be saved while a run is active; they
apply to subsequent policy decisions. The overlay is enabled by default for
the WebSocket game backend. Mock runs do not create or transmit HUD frames.

Install the latest PolyBot bridge from the repository's PML URL and select
`latest`. Bridge **0.1.39** adds HUD support for PolyTrack **0.6.3**. The
existing 0.1.37 release for PolyTrack 0.6.2 and 0.1.38 release for 0.6.3 are
left unchanged; they do not advertise HUD support. See
[Running PolyBot in PolyTrack](game-integration.md) for bridge setup.

The GUI can select a compact or full layout, scale it, hide policy inputs,
controls, reward details, episode status, labels, or event popups, and choose
how many lookahead points to display. The overlay does not intercept mouse or
keyboard input.

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
