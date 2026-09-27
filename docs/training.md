# Training with PolyBot v2

PolyBot uses one typed `TrainingConfig` in the GUI, CLI, and saved model metadata. Choose PPO, DQN, or TQC explicitly in the CLI. In the GUI, each algorithm has its own parameter panel; the other panels are hidden. Select a preset first, then expose advanced settings if needed. Saving a configuration writes the exact resolved JSON accepted by `polybot-train --config path.json`.

## The three learners

PPO is **on-policy**: it collects a fresh rollout, updates on it for several epochs, and discards that rollout. Its action is `MultiDiscrete([PWM levels, 2, 2])`: steering is a pulse duty, and throttle and brake are binary. The adapter resolves an overlapping throttle/brake request in favour of braking, so both are never sent together. A configurable initial bias helps a fresh PPO policy drive forward; learning can override it. An optional fixed PPO teacher and ghost-action imitation remain available.

DQN is **off-policy**: standard Stable-Baselines3 DQN stores experiences in replay and learns one Q-value per native digital action. It normally chooses the highest-value action but sometimes chooses randomly according to its decaying epsilon. It has no actor network, teacher, forward guard, or custom exploration rule. **DQN does not use PWM**: it holds exactly the selected keys throughout `frame_skip` physics ticks. Its default action schema is `digital-discrete-9-v2`:

| Index | Steering | Pedal |
| ---: | --- | --- |
| 0 | straight | coast |
| 1 | straight | throttle |
| 2 | straight | brake |
| 3 | left | coast |
| 4 | left | throttle |
| 5 | left | brake |
| 6 | right | coast |
| 7 | right | throttle |
| 8 | right | brake |

Set `dqn.action_set` to `no_brake` (CLI: `--dqn-action-set no_brake`) for a six-action first stage: straight/left/right, each with coast or throttle. This has its own `digital-discrete-6-no-brake-v2` schema and cannot be resumed as a nine-action model directly. `expand_no_brake_checkpoint` copies the six learned Q outputs into the corresponding nine-action outputs, initializes brake outputs below coast, remaps replay actions, and writes a resumable nine-action checkpoint. `tools/start_staged_dqn.py` runs a fresh six-action Summer 1 model in the GUI and automatically performs this transfer after a deterministic evaluation reaches 15% progress (no earlier than 100,000 steps), or at 600,000 steps at the latest. The second stage uses the remaining budget up to two million total steps. A user-requested early stop does not trigger transfer.

TQC is **off-policy**: it reuses past decisions from a replay buffer. Its `Box(2)` action contains steering in [-1, 1] and signed longitudinal demand in [-1, 1]. Positive longitudinal demand means throttle duty; negative means brake duty. Deterministic pulse scheduling turns that demand into digital physics-tick actions. The learner's replay action is the exact continuous action executed through this adapter. TQC uses automatic entropy tuning and a seeded, forward-biased replay warmup until `learning_starts`. The mature policy has no forward guard, actor recovery snapshot, rehearsal, or track-specific rule.

The `tiny`, `compact`, and `standard` architecture choices use two hidden layers of 64, 128, or 256 units. DQN also offers `yosh_2020` with 64 then 16 hidden units, matching the architecture [Yosh published for an earlier Trackmania model](https://www.youtube.com/watch?v=_oNK08LvZ-g); it is not a verified specification for his later DQN. PolyBot's observation inputs differ. The GUI shows actual network and total parameter counts when training starts. For DQN, metadata records `actor_parameters=0` and puts the Q-network count in `critic_parameters`; the GUI calls it a Q-network. PPO often has higher decisions per second; replay-based DQN and TQC can reuse experiences. Hardware, architecture, and update frequency determine speed.

## What a training step means

One environment step is one policy decision. `frame_skip` is the number of fixed physics ticks for that decision. PPO/TQC may schedule different digital pulses within it; DQN holds one digital control unchanged. If the simulator limits packet length, PolyBot splits that hold into identical smaller packets. Increasing frame skip often improves throughput but leaves the policy less time to correct a mistake. TPS is environment steps per wall-clock second, including neural-network updates and simulator communication.

`timesteps` is the **total** planned budget across curriculum phases. Full track and fixed/timed sections use one phase; sequential quarters uses five phases (Q1–Q4 and full); Q4 then full uses two. Random quarters samples a seeded quarter each episode. Custom mode accepts an explicit JSON list of full, section, timed, or random-quarter phases. Their positive `steps` must sum exactly to `timesteps`. The GUI's Curriculum tab shows the resolved plan; CLI users can pass `--curriculum custom --custom-phases phases.json` or a full v2 config file. The section completion bonus is a reward coefficient.

## Rewards

Choose **Balanced**, **Learning**, or **Pace** without editing 69 numbers. Existing named track profiles remain available. The Rewards tab offers six simple controls and shows their affected parameters; Advanced reveals every numerical coefficient with an explanation of its unit and effect. Save, duplicate, reset, or compare profiles. The exact resolved values are serialized with every model. Training never silently changes them.

The reward system calculates named components for Progress, Guidance, Driving quality, Airborne behaviour, Milestones, and Failure. Each step logs every raw term plus category totals; the totals must sum to the raw reward. All three algorithms use normalized `ControlDemand` for reward accounting. For example, brake duty 0.05 incurs 5% of a per-second ground-brake penalty; PPO and DQN binary brake remain full duty. The `reward_scale` multiplies the score sent to the learner, while event diagnostics retain raw points. DQN and TQC store those rewards in replay, so changing reward definitions mid-run would leave stale values and is rejected on resume.

The reference ghost defines route progress and lookahead. Optional ghost guidance rewards are separately controlled by the profile. A high reward for touching a barrier or a false landing penalty can teach the wrong driving line; inspect the per-term diagnostics and deterministic evaluation before deciding a profile helps.

## Evaluation and models

At `evaluation.interval_steps`, the runner releases the training simulator and drives a frozen policy deterministically on full-track seeded episodes. It reports finish rate, median/mean progress, best and median finished lap, and crash/off-track/stall rates. Champion ranking uses those results. The latest policy can regress without replacing champion. A stochastic training finish alone never promotes a champion.

```text
models/<track>/<algorithm>/
  latest/policy.zip, metadata.json, replay.pkl (DQN or TQC)
  champion/policy.zip, metadata.json
  checkpoints/step-<N>/policy.zip, metadata.json, replay.pkl (DQN or TQC)
```

Metadata includes the app version and v2 schema versions, architecture, parameter counts, observation and action schemas, track, reward profile, curriculum, complete configuration, training steps, physics ticks, elapsed wall time, seed, device, finish/crash totals, evaluation, and Git commit. Resume requires compatible track, action and observation schemas, architecture, and rewards. DQN/TQC resume also requires replay; champion inference does not. Existing v2 PPO/TQC configs and models remain loadable. No v1 model is loadable through this path.

## Observation and protocol

All three algorithms receive the same normalized `polybot.observation.v2` vector: 45 state features followed by four values and one validity mask for each lookahead point (`45 + 5N`, so 105 values at N=12). State order is local velocity (3), acceleration (3), angular velocity (3), up vector (3), route progress, lateral and heading errors, pitch, roll, wheel contacts (4), suspension lengths (4), suspension velocities (4), wheel skids (4), actual steering, ghost relative position (3), ghost heading, ghost target speed, ghost action (3), and previous digital action (3). Lookahead points contain forward, right, up and curvature values. See `Telemetry.to_vector()` for the exact scaling and clipping. No algorithm-specific observation slice is masked.

The simulator protocol remains `polybot.sim` version 2. The wire action is always digital steering {-1,0,1} and binary throttle/brake. The mock implements the same handshake/reset/step contract as the PolyModLoader adapter. See [protocol](protocol.md).

## Devices and diagnosis

`device=auto` prefers CPU for PPO's small MLP updates and uses CUDA for DQN/TQC if PyTorch can initialize it. The selection reason is logged. Explicit `cuda` fails when unavailable and overrides PPO's CPU preference. `polybot-doctor` reports PyTorch version, CUDA build, availability, GPU count/name, and NVIDIA driver visibility. `polybot-doctor --smoke ppo|dqn|tqc` builds a tiny policy; the DQN smoke also trains, saves, and reloads replay. Small DQN networks may run faster on CPU than CUDA because GPU launch overhead can dominate.

For a fresh run, start with Balanced rewards, compact architecture, 2–4 TQC train frequency on modest hardware, and an evaluation interval long enough to avoid spending most of the run on testing. The GUI warns about unusual combinations but never silently changes them. For all exact meanings, hover over a field; the descriptions come from `training/parameters.py` and are checked by tests.

The same central help is available without the GUI through `polybot-train --parameter-help`.
