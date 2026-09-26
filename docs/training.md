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
105-value protocol-v2 observation (12 lookahead samples) and the default 41 PWM steering levels
this is 3,389,486 trainable parameters. `legacy`, `small` (66,094 parameters), `medium`, and `large`
remain available. Older protocol-v1 policies used 81 observations; they require migration before
use with the current observation layout.

For a lightweight policy, `compact` uses two 104-unit layers per branch: 48,718 parameters
with 12 lookahead samples and 41 PWM steering levels.

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

## Reward profiles

The GUI reward table is generated from `RewardConfig` and exposes every coefficient and threshold.
Selecting a profile loads all values into the editable table. Type a new name (or select an existing
custom name), edit values, and choose **Save reward profile**. Custom profiles are stored as readable
JSON files under `profiles/rewards/` and can be edited, copied, or version-controlled.

## Ghost control guidance

PPO uses a training reward scale of `0.01` by default; episode scores and reward
breakdowns remain in their original units. Actor and critic gradients are clipped
separately. Each PPO update writes value loss, explained variance, policy statistics,
and critic saturation to the session JSONL log. The GUI shows a short update summary.
When changing the reward scale, start a fresh model: an older critic predicts values
in the previous reward units.

Ghost control imitation and its reward fade with position, heading, and speed error
relative to the reference. This prevents full-strength copying when the learner
needs a different action to recover. The falloff uses Gaussian scales of 2 metres,
0.35 radians, and 10 m/s. A coefficient of `0.2` is the conservative fresh-run setting;
the controls remain digital demonstration targets, so PWM intermediate levels must
still be learned through PPO.

With a reward profile that supplies an expert-action bonus, the trainer can also fit the
ghost's recorded steering, throttle, and brake directly. The GUI's **Expert imitation
coefficient** controls this extra policy loss: `1.0` is the default and `0.0` disables
it. This setting is separate from **Teacher KL coefficient**, which requires a fixed
teacher model archive.

To compare guidance fairly, first stop training cleanly and copy `latest.zip` and
`latest.metadata.json` to a named checkpoint. Resume two runs from that same checkpoint,
using the same reward profile, track, seed, and training budget. Set the expert imitation
coefficient to `1.0` for one run and `0.0` for the other, and use separate model output
locations so their `latest.zip` files do not overwrite each other. Compare evaluation
laps rather than training reward alone.
