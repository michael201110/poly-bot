# Training and model management

Install Python 3.11+ on Windows or Linux, create and activate a normal virtual environment, then:

```text
python -m pip install -e ".[train,gui]"
polybot-gui
```

The GUI runs PPO or TQC training in a worker thread using `TrainingService` and
`TrainingManager`. A stop request finishes the current environment step and saves `latest.zip`.
The manager implements resume, checkpoints, curriculum, and graceful stop.

## Algorithms and actions

PPO remains the default. Its action space is `MultiDiscrete([41, 2, 2])` with 41 steering duties,
digital throttle, and digital brake. The existing `TeacherAnchoredPPO`, teacher KL, and expert
imitation features remain PPO-only. PPO collects on-policy rollouts.

TQC uses `sb3-contrib` and the same telemetry observations and reward profile as PPO. Its action
space is `Box(low=[-1, -1], high=[1, 1], dtype=float32)`: the first component is signed steering
duty, the second is signed longitudinal duty (positive throttle, negative brake). Separate
accumulators convert these to digital left/right/throttle/brake pulses at 1 ms physics ticks.
Throttle and brake cannot be pressed together. The fractional pulse error carries across
environment steps and resets at the next episode. The wire protocol is unchanged. TQC learns
off-policy from a replay buffer with automatic entropy tuning by default.

TQC is offered for benchmarking and sample-efficiency experiments; it is not assumed to be
better than PPO. For a fair comparison, choose the same track, Summer 1 reward profile, seed,
frame skip, and environment-step budget. The model metadata records the algorithm, seed,
action schema, timesteps, simulator ticks, elapsed wall time, reward profile, best lap,
finish/crash counts, and algorithm-specific hyperparameters. Episode rewards in the GUI and logs
are raw game reward; the default `0.01` reward scale only changes values sent to the learner.

Example GUI launches:

```powershell
polybot-gui --algorithm ppo --fresh
polybot-gui --algorithm tqc --tqc-architecture standard --fresh
```

Example headless runs on the local mock simulator:

```powershell
polybot-train --algorithm ppo --backend mock --track mock/gentle-s --seed 7 --timesteps 100000
polybot-train --algorithm tqc --backend mock --track mock/gentle-s --seed 7 --timesteps 100000
```

For TQC, `--output-root PATH` changes the root of the algorithm-scoped registry.
The older PPO command keeps `--model-out PATH` for its original archive layout.

For real PolyTrack training, select `--backend websocket --track current` after loading the mod
and a ghost reference. The GUI offers the complete editable reward profile. The TQC command can
also load a saved profile with `--reward-profile summer-1-balanced`.

TQC defaults: actor and critics each use two 256-unit hidden layers (`standard`); learning rate
`0.0003`; replay buffer `1,000,000`; learning starts `10,000`; batch size `256`; gamma `0.999`;
tau `0.005`; train frequency `1`; gradient steps `1`; entropy coefficient `auto`. The
`compact` preset uses two 128-unit layers. The library defaults apply to TQC quantile count,
quantiles dropped, and critic count. These settings are separate from PPO settings in config,
metadata, and the GUI.

## Devices and network presets

New models default to separate actor and critic networks of `1024, 1024, 512` (`xl`). With the
105-value protocol-v2 observation (12 lookahead samples) and the default 41 PWM steering levels
this is 3,389,486 trainable parameters. `legacy`, `small` (66,094 parameters), `medium`, and `large`
remain available. Older protocol-v1 policies used 81 observations; they require migration before
use with the current observation layout.

For a lightweight policy, `compact` uses two 104-unit layers per branch: 48,718 parameters
with 12 lookahead samples and 41 PWM steering levels.

`auto` selects CUDA only when PyTorch can initialize a GPU and identify it; otherwise it uses CPU
and logs why. `cpu` forces CPU. Explicit `cuda` fails before training when unusable. Run
`polybot-doctor` for Python, PyTorch build, NVIDIA GPU, cuDNN, availability, device count, and
selected-device diagnostics. `polybot-doctor --device cuda --smoke-tqc` also creates a small TQC
policy and checks that its parameters are actually on CUDA.

### Windows NVIDIA installation

Create a fresh virtual environment and install the CPU-compatible training dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[train,gui,dev]"
.\.venv\Scripts\polybot-doctor.exe
```

For NVIDIA acceleration, select the current stable Windows/Pip/CUDA command on the
[official PyTorch installer](https://pytorch.org/get-started/locally/). In the same virtual
environment, replace a CPU-only wheel with the chosen CUDA-enabled distribution. For example,
if the installer currently offers the CUDA 12.6 channel:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\polybot-doctor.exe --device cuda --smoke-tqc
```

Use the index URL shown by the installer if it changes; this example is not a permanent version
pin. The PyTorch wheel supplies its CUDA runtime. You do not need the full CUDA Toolkit just to
train with the wheel. A compatible NVIDIA driver is still required. `.[train]` installs
`stable-baselines3` and matching `sb3-contrib` versions, which pull in PyTorch; it cannot
guarantee that the selected PyTorch wheel has CUDA support. If `nvidia-smi` sees a GPU but
`polybot-doctor` reports a CPU-only build, install the CUDA wheel. If the build has CUDA but
initialization fails, check the driver and Windows GPU availability rather than reinstalling
the CUDA Toolkit.

## Track registry and compatibility

New models live under `models/<track-slug>/ppo/` or `models/<track-slug>/tqc/`. Each algorithm
has its own `latest.zip`, `best.zip`, metadata, and checkpoints. TQC also saves matching
`*.replay.pkl` buffers; keep the buffer alongside the archive to resume TQC. Existing PPO
archives directly under `models/<track-slug>/` are recognized without moving or rewriting them.
Metadata binds the
track ID, observation/action schema, architecture, PWM settings, hyperparameters, reward settings,
seed, training counters, version, and commit.

Legacy archives without metadata are never assumed to use PWM. To evaluate or play one, pass
`--algorithm ppo` explicitly and select the matching digital action mode. Registry resume checks
reject algorithm, track, observation, action, and architecture mismatches. Fresh training only
archives the selected algorithm's own latest model and never moves legacy PPO archives.

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

Expert control imitation fades with position and heading error relative to the ghost,
using Gaussian scales of 2 metres and 0.35 radians. This lets the policy learn
acceleration even when it starts slower than the ghost. Guidance rewards require
forward, on-track progress; the recovery profile also penalizes sustained low speed.
The controls remain digital demonstration targets, so PPO must still learn PWM
intermediate levels through exploration.

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
