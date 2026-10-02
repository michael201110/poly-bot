# Independent Summer 1 GRTQC scratch experiment

Success requires a deterministic five-finish evaluation with median below 23.000 seconds. The project target remains below 22.000 seconds. Initialization, completion and critic calibration are milestones, not success. No result is established yet.

## Isolation and origin

Profile: `profiles/training/summer-1-grtqc-scratch-30.json`. Output: `models/experiments/grtqc-scratch-20261002/`. Logs: `logs/grtqc-scratch-20261002/`. Seed: 20261002. Simulator: loopback port 8766, separate from the transferred branch's 8765. The original TQC source and all earlier experiment/replay/checkpoint directories remain intact. This experiment never invokes the transfer initializer, loads old replay, distills actions or installs source overlays. `initialization/` records the freshly generated actor/critics, empty replay and origin configuration. Scratch/transfer origin validation forbids silently resuming either as the other.

Both actor and critics receive 121 actual physical/PWM/task-context inputs. Unlike the transferred actor, the new actor needs no preserved 105-input matrix and uses those inputs directly. Hidden gated layers use ordinary sigmoid scaling at fresh initialization. Two 128-unit hidden layers provide a practical baseline on this machine; two independent critics estimate 25 quantiles each and drop two upper quantiles per critic from the pooled target mixture. This architecture may change only in a separate controlled fresh experiment.

## Learning and curriculum

The first 3,000 transitions use seeded generic forward-biased exploration; no track-specific action is prescribed. Thereafter collection samples the learned squashed Gaussian policy, with ordinary entropy tuning initially targeting -2. There is no inherited deterministic-policy lock, narrow teacher distribution bound or verified-state actor sampling. Actor and critic learning rates start at 3e-4. Batch size is 128, with one critic update per four collected decisions and one actor/temperature update per two critic updates. Actor delay is counted across successive training calls. Discount is 0.999 and target rate 0.005. Actor updates have a broad 0.1 action-change bound, rather than the transferred branch's microscopic bound.

Randomised quarter starts provide physically consistent states through the existing simulator curriculum. They do not specify subsequent actions. The initial section budget is at most 60,000 decisions; a 75% section-completion rate across 40 attempts advances earlier. Unused budget carries into full-track training and the revised plan is saved. The main stage continues for the remaining budget, initially two million decisions total. Full-lap evaluation always starts at the real track beginning, independently of curriculum starts. Partial section finishes do not count as full training finishes or pace milestones.

After the first reliable five-finish deterministic evaluation, entropy target moves to -4 to emphasize pace while retaining stochastic exploration. The reward objective already prioritizes finish speed from the start, avoiding an incompatible replay reward change between stages. Changes to physics/action frequency or network dimensions need controlled comparisons, not conclusions drawn from the old actor's 20-tick test.

## Reward and acceptance

Ghost/teacher action and pose guidance weights are zero. Corridor-speed, unsafe-speed, air-brake and ground-brake shaping are zero. Progress gives two raw points/metre, incomplete failure claws back one point/metre, and time costs ten points/second. This retains a partial-progress learning signal for an inexperienced policy without the transferred profile's early failure penalty. Global nonterminal contact cost is -50; it is never a positive control reward or a section-specific avoidance rule.

Finish credit is 2,000 plus 15,000*exp(-0.15*max(lap_seconds-20,0)). A 22.9-second finish earns about 11,709 raw finish points versus about 9,085 at 25 seconds. Faster completion dominates the small control/contact shaping. There is no reward for matching TQC. Existing lookahead/geometry observations and physically consistent curriculum starts remain environment knowledge; they are not action imitation.

A first reliable policy can be champion even when slower than 24.263 seconds. Subsequent champions require a faster five-finish median. The live candidate remains separate and keeps learning after ordinary weaker checks; it is not automatically reset to the transferred actor or every slightly better snapshot. Screens that already fail do not consume five laps, but every finishing screen requires a separate five-lap confirmation. Rejected policies and actual evaluation transitions remain available for diagnosis and critic learning.

Full stochastic-rollout best times are marked unverified and archived separately. Verified milestone thresholds are 25, 24.263, 24, 23.5 and 23 seconds. Progress/loss trends alone do not establish an improved lap.

## Diagnostics and running

Logs include actor/critic loss, quantile/target mean and spread, TD residual, disagreement/regularizer, entropy temperature, total/per-layer gradients and parameter updates, Adam steps, replay size, phase competence, full finishes, crashes/contact progress and evaluation best/median. Failed tested-policy experience enters compatible fresh replay. The immutable original TQC source is not read for this branch's policy training.

```powershell
.\.venv\Scripts\python.exe scripts/train_with_stop_file.py --config profiles/training/summer-1-grtqc-scratch-30.json --stop-file logs/grtqc-scratch-20261002/train.stop --retry-transport
```

Bridge 0.1.32 accepts a per-tab `polybotPort` URL parameter, tags worker initialization with that port and validates it to a loopback endpoint. Defaults remain 8765. Use `https://web.polymodloader.com/?polybotPort=8766` for this simulator after enabling the new version. The original already-running tab need not be reloaded. Parallel learners require distinct game tabs and ports; output directories alone do not isolate simulation state.

Bridge 0.1.33 additionally scopes PolyTrack's native single-instance BroadcastChannel to the selected port. Without this, the second tab is blocked even with separate websocket endpoints. The native guard remains effective for two tabs using the same port. This only partitions offline client sessions; worker physics and training controls are unchanged.
