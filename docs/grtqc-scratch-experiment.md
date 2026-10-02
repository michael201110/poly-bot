# Independent Summer 1 GRTQC scratch experiment

Success requires a deterministic five-finish evaluation with median below 23.000 seconds. The long-term project target remains below 22.000 seconds; the supervisor runs repeated two-million-step budgets until the configured 22-second goal is confirmed. Initialization, completion and critic calibration are milestones, not success. No result is established yet.

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

The initial live collection exposed a configuration mistake: `early_off_track_penalty=0` replaces, rather than adds to, the ordinary -400 off-track cost. Every quarter's early off-track failures were therefore cheaper than stalls. Early and ordinary off-track failures now receive the same global -400 cost. The first learner/replay is archived before resuming its independently learned weights with empty replay; old reward labels are not reused.

A first reliable policy can be champion even when slower than 24.263 seconds. Subsequent champions require a faster five-finish median. The live candidate remains separate and keeps learning after ordinary weaker checks; it is not automatically reset to the transferred actor or every slightly better snapshot. Screens that already fail do not consume five laps, but every finishing screen requires a separate five-lap confirmation. Rejected policies and actual evaluation transitions remain available for diagnosis and critic learning.

Full stochastic-rollout best times are marked unverified and archived separately. Verified milestone thresholds are 25, 24.263, 24, 23.5 and 23 seconds. Progress/loss trends alone do not establish an improved lap.

## Diagnostics and running

Logs include actor/critic loss, quantile/target mean and spread, TD residual, disagreement/regularizer, entropy temperature, total/per-layer gradients and parameter updates, Adam steps, replay size, phase competence, full finishes, crashes/contact progress and evaluation best/median. Failed tested-policy experience enters compatible fresh replay. The immutable original TQC source is not read for this branch's policy training.

```powershell
.\.venv\Scripts\python.exe scripts/train_with_stop_file.py --config profiles/training/summer-1-grtqc-scratch-30.json --stop-file logs/grtqc-scratch-20261002/train.stop --retry-transport --continue-until-target
```

Bridge 0.1.33 accepts a per-tab `polybotPort` URL parameter, tags worker initialization with that port and validates it to a loopback endpoint. Defaults remain 8765. Use `https://web.polymodloader.com/?polybotPort=8766` for this simulator after enabling the new version. Parallel learners require distinct game tabs and ports; output directories alone do not isolate simulation state.

Bridge 0.1.33 additionally scopes PolyTrack's native single-instance BroadcastChannel to the selected port. Without this, the second tab is blocked even with separate websocket endpoints. The native guard remains effective for two tabs using the same port. This only partitions offline client sessions; worker physics and training controls are unchanged.

## Run log — 2026-10-02

The initial independent policy reached 35,349 decisions across 137 short attempts (mostly off-track or stalled) without a full-track finish; its best sampled section progress was 62.1%. It had 8,086 critic updates and an unlocked actor. At this point, reward-term logs showed early off-track cost was zero even though stalled attempts cost -400. The complete checkpoint and its 35,845-transition replay are preserved at `models/experiments/grtqc-scratch-20261002/archive/early-off-track-zero-step-35349/latest`.

Training resumed from that scratch actor with newly initialized critic/replay and the corrected equal -400 off-track/stall costs. It uses the same single simulator tab and verified Summer 1 reference run selected in the game. At step 46,046 it had 10,932 new replay transitions, 10,011 critic updates, an unlocked actor, and no full-track finish. This is early learning evidence only, not a faster lap. Live output is `logs/grtqc-scratch-20261002/training-resume-uniform-offtrack-v3.stdout.jsonl` and detailed episode telemetry remains in `logs/summer-1-grtqc-*.jsonl`.

Deterministic one-lap screens remained unfinished through step 180,163. Stochastic attempts reached at most 44.2% of the route; none finished, and 197 of 235 attempts ended in `airborne_roll_failure`. The active profile gave only 0.1 seconds before this terminal, shorter than one 30-tick observation interval (~0.5 seconds), so one tilted sample could end an attempt before another action. That roll timeout is now 1.0 seconds, with a -400 terminal cost matching other failures. The old 148,639-transition replay and optimizer state are preserved in an archived checkpoint because replay rewards predate this correction. Resume the saved scratch actor and critics with fresh replay so corrected failure returns are consistent, then continue two-million-step budgets until the five-finish 22.000-second target or user stop.

After the reward correction, 14 of 51 attempts had roll failures (27%), compared with 197/235 (84%) before it; stalls and off-track exits remain common. The one-lap screen at step 190,803 failed at 15.97% progress, and the screen at 201,171 reached 23.50%; neither finished. Training continues from the same scratch actor and critics with new reward-consistent replay. The lower roll-failure share is evidence the termination change is helping failure diversity, not evidence of pace or route completion yet.

A paired deterministic frame-skip check on the saved 231,632-step policy found no gain from finer control: five episodes at skip 30 and five at skip 20 all went off track near 23.4% progress, with 0% finishes in both. Skip 20 reduced measured barrier-contact steps from 60 to 15, but median route progress was 23.41% versus 23.50%. The learner therefore resumed at the established skip 30 with its compatible replay; do not attribute the contact-count difference to improved completion or pace.

After returning to training at skip 30, deterministic one-lap screens reached 39.91% at step 241,808 and 43.55% at 251,904, then regressed to 23.41% at 262,064. All three had zero finishes; each unfinished candidate was rejected for champion promotion while training continued, preserving the checkpoints. This shows route-progress exploration beyond the prior point, but also high policy variance and no lap-time result yet.

The five-paired-evaluation traces showed repeated contact at progress 0.234 in every run (60 contact steps total at skip 30). Since the old global contact cost of -50 was small at reward scale 0.01 and did not prevent this barrier-bashing policy, it is raised to -200 for all contacts, without progress-specific behavior. The 282,369-step actor/critic snapshot and replay are archived; continuation uses fresh replay so its contact returns match the new cost.

A same-seed five-run check compared the step-251,904 partial-progress checkpoint with the current step-312,538 actor under identical current settings. Both had 0% finishes; the earlier checkpoint reached 43.55% progress in all five runs before stalling, while the current actor stalled at 16.64% in all five. The regressed step-312,538 actor/replay is preserved at `archive/barrier-cost-regressed-step-312538/latest`; training resumes from the better partial-progress checkpoint with fresh replay and the global contact cost.

The resumed step-251,904 actor's first screen under the corrected reward, at global step 262,688, reached 27.23% with zero barrier-contact steps and zero finishes. In its first 25 training attempts there was one airborne-roll failure; off-track and stall exits still dominate. Continue monitoring before treating this as a stable improvement.

Subsequent deterministic screens reached 42.21% at step 272,896 (stalled) and 41.38% at 283,280 (off track), each with zero barrier-contact steps and no finish. The policy now repeatedly traverses farther without recorded barrier contact, but is not yet completing this section reliably.

A paired five-run comparison then tested step 272,896, step 293,424, and latest step 300,213 under identical current settings. The step-293,424 actor was clearly regressed (19.01%, crash and 5 contact steps); step 272,896 reached 42.21% with no contacts but stalled in every run. Latest reached 43.74% with five contact steps per run and stalled every time. Continue from latest with its compatible replay and the -200 global contact cost rather than rewind to a lower-progress candidate; no policy has completed a lap yet.

At the following 10k-step check, deterministic progress fell to 20.50% with an airborne-roll crash, then recovered to 43.74% with a stall and one contact step; the next screen fell to 30.59% off track with three contact steps. The actor's proposed action drift remained 0.19-0.23 while the per-update action bound clipped it at 0.10. The 342,806-step policy/replay is archived at `archive/actor-clipped-step-342806/latest`; reduce only the actor learning rate from 3e-4 to 1e-4 and retain the compatible replay and critic learning rate to test whether smaller policy updates stabilize progress.
