# Training and model management

Install Python 3.11+ on Windows or Linux, create and activate a normal virtual environment, then:

```text
python -m pip install -e ".[train,gui]"
polybot-gui
```

The GUI runs training in a worker thread and uses the same `TrainingService` and
`TrainingManager` available to headless callers. Stop requests finish the current environment/PPO
step and save `latest.zip`. The Python manager implements resume, checkpoint, curriculum phase,
repeat, recovery, and graceful-stop behavior without shell-specific executables or paths.

## Devices and network presets

New models default to separate actor and critic networks of `1024, 1024, 512` (`xl`). With the
81-value observation and the default 41 PWM steering levels this is 3,340,334 trainable
parameters. `legacy`, `medium`, and `large` remain available.

`auto` selects CUDA when `torch.cuda.is_available()` and otherwise CPU. `cpu` always forces CPU.
`cuda` fails early with a useful message when unavailable. SB3 receives the resolved device for
both creation and loading, so archives remain portable between CPU and CUDA installations. Install
a PyTorch build appropriate to the host driver using the official PyTorch instructions; PolyBot
does not pin a platform-specific wheel.

## Track registry and compatibility

Models live under `models/<track-slug>/`. `latest.zip` and `checkpoints/` are disposable local
artifacts. A published track has `best.zip`, `metadata.json`, and a README. Metadata binds the
track ID, observation/action schema, architecture, PWM settings, hyperparameters, reward settings,
seed, training counters, version, and commit.

Legacy archives without metadata are never assumed to use PWM. Treat them as
`legacy`/`digital-multidiscrete-v1`; either supply matching metadata or use the old digital action
mode. Registry compatibility checks reject track, observation, action, and architecture mismatches
unless a track override is explicit.

## PWM

The policy selects one of 41 evenly spaced steering duties from -1 through +1. A deterministic
accumulator spreads pulses across every 1 ms physics tick within `frame_skip`; throttle and brake
remain digital. Each simulator request still contains only steering -1, 0, or +1. Changing steering
direction resets accumulated error, and resetting an episode resets the scheduler. Digital mode
retains the original `MultiDiscrete([3,2,2])` action schema.

## Evaluation and promotion

Evaluate a candidate over multiple laps and retain lap times, finishes, crashes, mean, median, and
best time. Promotion is deliberate: `ModelRegistry.promote(candidate, metadata)` copies the chosen
archive to the track's `best.zip` and updates metadata. A newer checkpoint never replaces best by
itself, and the trainer never pushes to GitHub. If best archives exceed normal GitHub limits, track
`models/**/best.zip` with Git LFS.
