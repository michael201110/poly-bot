# Summer 1 TQC-to-PPO transfer diagnosis

The directly distilled PPO student does **not** reproduce the frozen TQC
teacher's actions accurately enough to stay on its line. This is visible before
the two cars separate. The simulator is also highly sensitive to even much
smaller persistent action changes, so ordinary behavioral cloning error is a
poor initialization for this particular 24.263-second path. The earlier PPO
probe was stopped at its safe boundary at 130,143 steps. A later guarded
continuation evaluated a slower 24.716-second candidate at 57,344 steps, then
rejected five candidates at 61,440, 65,536, 69,632, 73,728, and 77,824 steps
(0/5 finishes each). The 24.263-second champion was restored after every
regression. A stop request was honored after the final evaluated checkpoint;
that run is no longer active, and the champion was not replaced. A separate
stagewise residual PPO run is now active from the same teacher initialization.
At 90,112 steps its latest candidate evaluates at 24.332s with 5/5 finishes;
the isolated 24.263s champion remains intact and no sub-22s lap is confirmed.

## Reproduce

From the repository root, with the game worker available for live mode:

```powershell
.\.venv\Scripts\python.exe -m polybot.training.compare_teacher_student --live --experiments
```

The command reads the immutable teacher checkpoint, the directly distilled PPO
checkpoint, and the 20 successful teacher laps in the dataset. It writes a
machine-readable report, same-observation arrays, and policy-decision JSONL
trajectories under `runs/teacher-student/transfer-diagnosis/`. `--live` drives
both policies from the same seed, and `--experiments` additionally replays the
teacher's exact actions and tests signed perturbations. Without `--live`, it
only runs the offline comparison. It never updates either model.

## Checkpoints and conditions

| Item | Frozen teacher | Direct PPO student |
| --- | --- | --- |
| Path | `models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion` | `models/experiments/ppo-wallspin-standard-20260930/summer-1/ppo/teacher-student/pretrained` |
| `policy.zip` SHA-256 | `FFBEA4CA57116CD2586C17CCEC4FC761600E0D1EE5C6D94E31B98122220DAECE` | `8C0CE7D9D875803BFD30D8030C4A635D488B81D381718337090A7E828A14D0F2` |
| Algorithm | TQC | PPO, 0 on-policy training steps |
| Observation | `polybot.observation.v2`, 105 float32 values, 12 lookahead points | Same |
| Action | `continuous-pwm-v2`, signed steering and longitudinal | Same |
| Track | Summer 1, `current` | Same |
| Frame skip | 30 physics ticks per decision | Same |

The teacher metadata lists seven policy overlays: two air-brake windows and
five small steering/drive biases. The student metadata lists the two air-brake
windows; its actor was trained against labels with the other overlays baked in.
The teacher dataset's SHA-256 is
`CDE4DD766FE9FFEF62D0231C6AF1D6FC01EC0BD4C9B7AAD3E540EBDC439FC12D`.
The teacher checkpoint tree hash matches the dataset's teacher ID. The teacher
and student use different reward profiles, but rewards are not consulted during
deterministic inference. The strict barrier-contact threshold in the student's
saved metadata was separately checked with a permissive threshold and did not
explain its failure at 26% progress.

## Same-observation actor error

All 16,180 observations come from 20 successful teacher laps. The exact same
array is fed to the teacher and student; neither drives during this test.

| Axis | MAE | RMSE | p50 | p90 | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Steering | 0.1092 | 0.1556 | 0.0719 | 0.2502 | 0.3384 | 0.4923 | 0.8517 |
| Longitudinal | 0.0684 | 0.1241 | 0.0273 | 0.1950 | 0.3178 | 0.5105 | 0.6724 |

The first 5% of track has steering MAE 0.221, more than twice the overall
average. High-speed states have steering MAE 0.0953 and longitudinal MAE
0.0759. The sampled air-brake region has steering MAE 0.1032. Teacher labels
recomputed from the raw TQC actor and bakeable overlays agree with dataset
targets to within 0.0000023. This rules out training against raw, pre-overlay
actions as the main cause. PPO deterministic predictions are repeatable and
use the clipped Gaussian mean; there is no sampled evaluation action. Both
algorithms use the same `ContinuousActionAdapter` and deterministic pulse-density
controller. There is no legacy discrete PPO action bucket in this path.

## Live same-seed comparison

On seed `20260929`, the teacher finished in **24.263s** in 809 decisions. The
student stopped at **25.998% progress** after 8.13s and 271 decisions with an
airborne-roll failure. On the first decision, teacher steering was 0.79643
and PPO steering was 0.89451, a difference of 0.09808 before any trajectory
separation. Across the first 30 ticks, these generated 23 and 26 right-steer
ticks respectively. Their longitudinal actions differed by only 0.00334.

| First decision with difference | Decision index |
| --- | ---: |
| Action >0.001, >0.005, >0.01, >0.05 | 0 |
| Action >0.1 | 1 |
| Position >0.01m, >0.05m, >0.1m | 12, 17, 20 |
| Position >0.5m, >1m | 47, 75 |
| Heading >0.1°, >0.5°, >1°, >5° | 3, 16, 24, 105 |
| Speed >0.01m/s | 3 |
| Wheel contacts | 126 |

At decision 47 the cars were 0.51m apart, which is the first concrete
half-metre racing-line departure. At decision 75 they were 1.01m apart. The
student's failure at decision 270 occurred with a 27.97m position gap and a
51.86m/s speed deficit relative to the teacher. These thresholds compare the
same decision indices; the JSONL trajectories retain simulator tick and elapsed
time for each sample.

## Replay and sensitivity

Replaying the teacher's recorded high-level actions and air-brake base actions
from the same seed reproduced its **24.263s finish**. No action, position,
heading, speed, or wheel-contact difference crossed the smallest detector
threshold. Thus the live action path is sufficiently deterministic for this
diagnosis; a simulator/replay mismatch is not needed to explain the failure.

Persistent one-axis offsets were applied to a live teacher without changing its
checkpoint. Steering `+0.0001` failed at 56.2% progress; `-0.0001` finished
in 26.901s. Steering `+0.0005` failed at 19.3%; `-0.0005` failed at 53.8%.
Both `+0.0001` and `-0.0001` longitudinal offsets failed before 24%.
The complete signed `0.0001`, `0.0005`, `0.001`, `0.005`, and `0.01` results
for each axis are in `report.json`. These are single-seed sensitivity tests,
not finish-rate estimates. Some larger offsets happened to finish, so the
outcome is nonmonotonic. Even so, the PPO's first steering error of 0.098 is
hundreds of times larger than offsets that can derail this trajectory.

## Root cause and next experiment

The primary classification is **G: behavioral cloning error too large** and
**H: extreme closed-loop sensitivity**. Observation schema, action sign and
range, frame skip, adapter, and evaluation determinism match in code and in
the same-seed trace. The observed action error precedes the physical
divergence. Exact teacher-action replay succeeds. This does not support more
ordinary supervised epochs or repeated DAgger rounds as the next default move.

There is a secondary, late-track air-brake semantic difference to resolve
before claiming full transfer parity: TQC applies overlapping air-brake
overlays in sequence and captures its touchdown base action at each layer;
the PPO wrapper keeps the actor's unbraked action as the touchdown base.
This window starts near 69% progress, after the student's current failure,
so it cannot explain the 26% crash. Keep the teacher checkpoint frozen while
testing a shared low-level controller on the same high-level action trace.

The simulator protocol resets to a seeded race start or curriculum start; it
does not expose arbitrary saved-state restoration. Therefore a literal
one-step counterfactual from a teacher mid-lap state was not possible. The
same-seed replay and signed perturbation experiments provide the available
causal evidence. The reference ghost identifier is not serialized in either
checkpoint; both live laps used the same connected Summer 1 game session.

After identifying the late-track air-brake mismatch, the PPO wrapper was
corrected to resume the preceding overlay action on touchdown. A batch-only
TQC air-brake condition was also corrected; scalar live teacher inference is
unchanged. The supervised pretraining CLI now passes its advertised learning
rate to the optimizer. One isolated re-distillation with the advertised
`1e-4` rate and 100 epochs was saved at
`models/experiments/ppo-transfer-parity-20260930/summer-1/ppo/teacher-student/pretrained`
(`policy.zip` SHA-256
`B1B899F031619D97DEF872C1B1B7B176314BD053F88C181CC4102C8A07041257`).
It performed worse: steering MAE 0.1759, longitudinal MAE 0.0899, and 0/5
live finishes with median progress 16.45%. The teacher still finished 5/5 at
24.263s. This checkpoint was **not** promoted and no further training was
started. The control fix is too late in the track to resolve the student's
early crash, and a lower supervised rate did not improve actor fidelity in
the 100-epoch budget.

The next training strategy should start only after the low-level air-brake
parity test, then target much tighter first-section imitation and evaluate the
student on repeated identical starts before any PPO fine-tuning. If sub-0.001
actor differences still derail the line, an actor with teacher-assisted
closed-loop correction or a less brittle trajectory is needed. The 22-second
goal remains unverified; no PPO finish was produced by this direct student.

## Continued goal work (30 September 2026)

The 100-epoch supervised fit was still improving at its final epoch. Extending
the original `3e-4` fit to 400 epochs reduced same-observation errors to
0.0369 steering MAE and 0.0168 longitudinal MAE (held-out validation 0.0384
and 0.0188). It still went off track 5/5 times at about 24.9% progress. Its
paired trace shows the PPO steering right while TQC requests left as early as
18.7% progress on the PPO's own observations; near the failure PPO saturates
at +1 while teacher actions switch rapidly. This is consistent with compounding
state-distribution error at a sharp transition rather than a simple global
action bias.

Three supervised DAgger attempts started from this stronger clone:

| Candidate | Median progress | Result |
| --- | ---: | --- |
| Deterministic round 1 | 52.9% | Passed the first difficult transition; all five evaluations crashed at the new point. |
| Deterministic round 2 | 53.9% | Best checkpoint; all five evaluations crashed during an airborne 360 at the next transition. |
| Deterministic round 3 | 15.9% | Regressed; kept out of champion slot. |
| Stochastic round 1 (`std=0.02`) | 53.3% | Added trajectory variety but did not beat the deterministic round-two checkpoint. |

DAgger error on the states specifically visited near 25% fell below 0.004 in
round one. The remaining airborne 360 at 53.9% still shows a large online
teacher/student action gap despite low held-out DAgger error, so a low offline
mean error alone is not a reliable success gate.

A short anchored PPO run was then launched from the deterministic round-two
student in the isolated registry
`models/experiments/ppo-transfer-rl-20260930`. It uses a 0.1 KL anchor, learning
rate `1e-5`, and rollback to the saved best evaluation when a candidate
regresses. At 12,288 steps it produced a clean 25.419s lap, 5/5 finishes. A
later candidate regressed to 29.5% progress and was rolled back. At 27,648
steps it improved to a clean 25.358s lap, again 5/5; this remains the best
observed PPO result. Blocks have continued automatically from the champion,
with no confirmed sub-22 lap yet. This registry is isolated from the normal
track PPO champion until a candidate is faster and reliable.

The run has since completed another 10,240-step block at 63,488 total steps;
the champion remains 25.358s and finished 5/5. At the next intermediate
evaluation, 68,608 steps, the candidate reached only 23.5% progress and was
rolled back. Training continued from the saved champion. The 22-second target
is still unmet.

At 78,848 steps another candidate reached 27.8% progress and was restored to
the 25.358s checkpoint. A subsequent candidate completed all five eval laps
but was slower at 27.637s, so it was not promoted. At 99,328 steps a candidate
reached 64.5% progress without finishing. By 104,448 steps the runner was
again on the saved 25.358s, 5/5 champion and continued training. No sub-22s lap
has been confirmed.

At 109,568 steps PPO improved to 25.310s with a 5/5 finish evaluation. At
114,688 steps it improved again to **25.086s, 5/5 finishes**, now the best
verified PPO lap. Evaluations at 119,808 and 124,928 steps regressed to 23.4%
and 20.0% progress respectively and were rolled back. The run continues from
the 25.086s champion; the sub-22s target remains unmet.

The next PPO block ended at 155,648 steps without a faster lap. One intervening
five-episode candidate completed at 26.148s and was not promoted. The reliable
25.086s champion remains the resume point while training continues.

At 176,128 steps the run still had a 100% finish rate but no lap faster than
25.086s. The simulator worker remained active, so training was left running.

Evaluations from 114,688 through 217,088 steps confirm a plateau rather than a
reliable pace gain: the 25.086s lap at 114,688 remains the champion; three later
candidates finished at slower 26.414s, 25.785s, and 26.534s. The other seven
reached only 18.8–57.0% median progress and were rolled back. The isolated
champion remains intact, and PPO fine-tuning continues.

At 222,208 steps PPO established a new 24.971s champion, confirmed across five
finishes with no off-track episodes. This is 0.708s behind the frozen TQC
baseline and 2.971s above the 22s target. The 227,328-step candidate regressed
to 27.209s and was rolled back; subsequent fine-tuning continues from 24.971s.
The next two evaluations, at 232,448 and 237,568 steps, also failed to finish
(median progress 58.8% and 52.6%) and were rolled back to that champion.

An offline same-observation comparison was then run between the frozen TQC
teacher and the 24.971s PPO champion on 16,180 successful teacher states. Both
checkpoints use the same 105-feature observation schema, continuous action
schema, and frame skip 30. PPO deterministic prediction exactly matched its
clipped policy mean; recalculating the overlay-inclusive teacher labels matched
the saved targets within 2.3e-6. The student's errors against those targets
were 0.0246 steering MAE (0.0685 p95) and 0.0219 longitudinal MAE (0.0738 p95).
In the 20-30% track interval around the early jump, steering MAE was 0.0195-
0.0252 and p95 was 0.0492-0.0767. These results rule out a deterministic
predict-path error and a teacher-label mismatch in this dataset; they show a
remaining closed-loop action gap but do not alone prove its trajectory effect.
Recent PPO training episodes often ended with `airborne_roll_failure`, so the
early-flight recovery states merit targeted follow-up.

The comparison used teacher SHA-256
`FFBEA4CA57116CD2586C17CCEC4FC761600E0D1EE5C6D94E31B98122220DAECE`, PPO
champion SHA-256 `7A3C8CB59646276A783F57D9861DFA0A091CF20769970801F43073D1515EEB6B`,
and teacher dataset SHA-256
`CDE4DD766FE9FFEF62D0231C6AF1D6FC01EC0BD4C9B7AAD3E540EBDC439FC12D`.

Training has since reached 278,528 steps without beating 24.971s. The
268,288-step candidate finished at 25.673s and was not promoted; candidates at
273,408 and 278,528 steps finished 0/5 (median progress 23.4% and 57.0%), and
were rolled back. The first of those had off-track failures in every evaluation
episode. The run remains live from the isolated 24.971s champion.

A stochastic recovery-DAgger round from the 24.971s champion collected 2,493
states across 10 episodes. Eight episodes encountered crash, off-track, or stall
outcomes, mostly between 18.6% and 52.8% progress. The resulting distilled
candidate achieved only 0/5 finishes and 23.6% median progress, so it was not
used as the PPO seed. The pipeline retained the independently validated
24.971s baseline and began a new PPO run in
`models/experiments/ppo-transfer-rl-recovery-20261003`.

That first recovery retry revealed two control-flow/data issues: the reliable
baseline gate skipped requested later DAgger rounds, and stochastic PPO actions
were not seeded per collection round, making three rounds identical. Commits
`8beabd9` and `733f840` fix these behaviors. In the corrected three-round run,
the datasets varied (4,725, 5,630, and 4,831 samples), but all distilled
candidates still failed the 5-episode completion gate and were discarded. PPO
fine-tuning restarted from the validated 24.971s baseline; at 234,496 steps its
first two candidates both finished 0/5 at 23.2% and 22.8% median progress and
were rolled back. The new run remains active in
`models/experiments/ppo-transfer-rl-recovery-20261003-r4`.

The r4 PPO run reached 285,696 steps without beating 24.971s. Its 244,736-step
candidate completed at 25.233s, a later full candidate at 260,096 steps took
32.589s, and the other recent evaluations failed around 54-59% progress. To test
the logged air-roll failures, the 24.971s champion was evaluated on matched
seeds with its existing progress-window air-brake overlays and with an added
full-airborne brake. The baseline reconfirmed 5/5 finishes at 24.971s; the
full-airborne guard produced 0/5 finishes, 53.6% median progress, and a 100%
crash rate. That guard was rejected. The r4 run was cleanly stopped at a saved
boundary with the 24.971s champion preserved; the next PPO search uses a
moderately larger update size with rollback still enabled.

## Direct TQC-to-PPO transfer diagnosis (30 September 2026)

The transfer investigation supersedes the planned PPO update-size trial: no more
DAgger or PPO fine-tuning should run until the mismatch is understood. A
10-episode stochastic DAgger round was already underway when this instruction
arrived. It completed and was preserved (7,372 samples); the follow-on PPO job
was stopped at its next saved checkpoint. Its `latest` checkpoint has no
evaluation result, while the isolated 24.971s champion is unchanged. The stop
path exposed a bug: target confirmation constructed an `EvaluationResult` from
that incomplete metadata and raised `TypeError`. `_evaluation_confirms_target`
now treats missing core metrics as a non-confirmation, with regression tests.

The exact frozen TQC teacher is
`models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion`, policy SHA-256
`FFBEA4CA57116CD2586C17CCEC4FC761600E0D1EE5C6D94E31B98122220DAECE`. The
directly distilled PPO seed is
`models/experiments/ppo-transfer-fidelity-20260930/summer-1/ppo/teacher-student/pretrained`,
policy SHA-256 `22B8500CA5C1588183EFDC2F3B6AD672F2863F6BA023E4F859B2ADD4B2942A2E`.
Both use the 105-value `polybot.observation.v2` observation, the
`continuous-pwm-v2` action schema, Summer 1, and frame skip 30. The saved
teacher dataset hash is
`CDE4DD766FE9FFEF62D0231C6AF1D6FC01EC0BD4C9B7AAD3E540EBDC439FC12D`.

A fresh matched-seed live comparison reconfirmed the teacher at 24.263s and
finished, while the direct PPO seed went off track at 24.9% progress after
14.07s. In the decision-aligned traces, PPO's action first differed by more
than 0.001 at decision 0 and by more than 0.05 at decision 3. Heading differed
by over 0.1 degrees at decision 8, position by over 1 cm at decision 14, and
wheel contact at decision 126. Speed differed on the first transition. Thus
meaningful control error appears before the observed route separation; it is
not solely a case where two identical action streams somehow produce different
simulator outcomes.

On the same 16,180 successful teacher observations, the direct PPO seed has
steering MAE 0.0369 (p95 0.1192) and longitudinal MAE 0.0168 (p95 0.0662).
In the first 5% of the track, steering MAE is 0.0745 and p95 is 0.2212. Around
the early jump at 20-30%, steering MAE is 0.0204-0.0466 and p95 is
0.0581-0.0985. The high-speed-state steering MAE is 0.0328. This is a
materially imperfect clone, especially in steering, rather than near-exact
imitation. A separate PPO recovery collection also shows its largest teacher
action errors immediately before its early off-track/crash states.

The checks rule out several suspected transfer defects for these checkpoints:

- The calculated overlay-inclusive teacher action matches saved teacher labels
  to 2.3e-6 maximum error. The teacher's non-air-brake progress overlays are
  represented in those labels; the PPO keeps only the tick-level air-brake
  wrapper because the remaining action overlays are baked into the actor.
- Deterministic PPO prediction is repeatable and equals the clipped policy
  mean exactly. There is no stochastic evaluation, PWM bucketing, or extra
  action noise in the PPO prediction path.
- The teacher and student schemas and frame skips match. Both use the same
  environment and continuous action adapter; a regression test confirms that
  equal actions produce equal low-level tick sequences for both backends.
- Replaying the teacher's recorded actions from the same seed reproduced its
  complete 24.263s run with no measured position, heading, contact, speed, or
  action divergence. That run was deterministic under this seed.

There is also direct evidence of closed-loop sensitivity: in the earlier
controlled TQC experiment, persistent signed perturbations as small as
0.0001 in one action channel changed whether the fixed-seed run finished or
failed. This was a per-decision perturbation experiment, not a one-step
counterfactual, so it demonstrates accumulated sensitivity rather than proving
that a single tiny error causes a crash. The global-air-brake experiment also
showed that imposing full brake in every airborne state makes performance
worse, so no such controller change is indicated.

Root-cause classification: **G (behavioral-cloning error is still too large)
and H (closed-loop driving is sensitive to accumulated action error)** are
supported. A, B, C, D, E, F, and I have no supporting evidence in these
comparisons; the matching schemas, label audit, deterministic output test,
shared adapter test, and exact teacher replay specifically reduce those
suspicions. The diagnosis does not identify a proven observation/action-path
implementation defect, so the PPO seed was not re-distilled and training was
not restarted. Future work should measure supervised holdout error by route
section and high-speed steering before spending more simulator time; improved
clone fidelity is the next hypothesis to test, not a guaranteed fix.

The final artifacts are in the ignored run directory
`runs/teacher-student/transfer-diagnosis-final` (report and teacher/student
JSONL trajectories). The corrected DAgger round is in
`runs/teacher-student/rl-lr3e5-kl01/round-001.npz` and its candidate model is
preserved under `models/experiments/ppo-transfer-dagger-20260930`. Neither
changes the saved TQC teacher nor the validated 24.971s PPO champion.

## Fidelity refinement and PPO restart (30 September 2026)

Because the original offline fit was still improving at epoch 400, actor-only
regression was continued from that seed for 1,000 epochs at learning rate
5e-5. On the exact same held-out teacher laps, combined MSE fell from
0.00396 to 0.000923; steering MAE fell from 0.0369 to 0.0175 and longitudinal
MAE from 0.0168 to 0.0085. The critic was left unchanged. The refined model is
isolated at
`models/experiments/ppo-transfer-bc-refine-20260930/summer-1/ppo/teacher-student/pretrained`.

Better teacher-state imitation alone did not improve driving: its independent
5-episode validation crashed 5/5 at 55.2% median progress. A matched-seed trace
showed action error over 0.05 by decision 3, then position error over 1 cm by
decision 20 and an airborne-roll failure at 55.1%. The refinement was therefore
not promoted as a reliable policy. This directly verifies the diagnosis that
the failure is dominated by compounding state-distribution error, rather than
only teacher-state regression loss.

A deterministic recovery-DAgger pass from that refined model collected the
teacher's labels over the 3 seconds before the repeatable 55.1% crash. The
resulting student completed **5/5** independent validation laps at median
25.025s (best 25.025s), with no crashes or off-track events. It is now the
reliable, isolated starting point for continuous PPO search; the original
24.971s PPO champion and 24.263s TQC teacher remain unchanged. The sub-22s
success criterion has not yet been met.

The lower-rate PPO search was then restarted from the faster existing
24.971s 5/5 PPO checkpoint, using learning rate 1e-5, target KL 0.003, anchor
KL 0.1, and zero lap-slowdown tolerance. At 329,728 total timesteps it
produced a **24.838s PPO champion, 5/5 finishes**, 0.133s faster than the
previous best. The policy SHA-256 is
`68782998A1EE6A99034535105D5973F983ACF4706E0121F765DDB9A0C9D927ED`. The
checkpoint is preserved at
`models/experiments/ppo-transfer-rl-20260930/summer-1/ppo/verified-champions/ppo-24.838`
and committed on `main`.

Subsequent fine-tune evaluations at 334,848, 339,968, and 345,088 steps were
0/5 at 57.2%, 5/5 at 26.566s, and 0/5 at 24.2% progress. Strict rollback
restored the 24.838s champion each time. Fine-tuning then produced a **24.675s
champion, 5/5 finishes** at 360,448 total timesteps, improving the best by
0.163s. The evaluation reported no crashes, off-track laps, or stalls. This
checkpoint is preserved at
`models/experiments/ppo-transfer-rl-20260930/summer-1/ppo/verified-champions/ppo-24.675`,
with policy SHA-256
`CEF598297A97ABCC471C9F06156FE597F88776186E2D777B76B875752FBD8AE8`.
The confirmed sub-22s target is still unmet.

## PPO continuation checkpoint validation

Repeated early crashes near 23% progress were traced to an unverified continuation
checkpoint, not the saved 24.675s champion or a new evaluation seed. A run stopped
between evaluations had written partially updated weights to `latest` while
retaining the previous champion's 24.675s 5/5 evaluation in its metadata. The
next run saw that stale evaluation tied with the champion and resumed the newer,
untested weights. Directly evaluating the saved champion on the newer run's exact
five seeds still produced 5/5 finishes at 24.675s.

The runner now omits evaluation metadata from `latest` whenever its policy has
advanced past the last evaluation. The continuation gate then recognizes it as
unverified and selects the evaluated champion. A regression test covers that
metadata rule. The unverified local `latest` was marked as unevaluated, and PPO
fine-tuning was restarted from the verified champion with learning rate 3e-6,
target KL 0.001, and champion-anchor KL 0.25. That update failed three
consecutive evaluations at roughly 23%, 53%, and 57% progress, so the regression
guard stopped it and restored the champion. The exact saved champion was also
re-evaluated on the newer run's seed and passed 5/5, confirming the loss belongs
to updated policies rather than that seed. A second anchoring defect was found:
the fine-tuning jobs reused a 2,048-step PPO snapshot with no successful
evaluation as their KL reference, even after the 24.675s champion had improved.
Anchor directories are now versioned by their starting policy hash, so each new
continuation is anchored to the exact validated champion it resumes.

A controlled one-update PPO probe from 24.675s (learning rate 3e-7, target KL
0.0001, and the matching champion anchor) changed mean actions on 16,180
successful teacher states by steering MAE 0.000369 (p95 0.000863) and
longitudinal MAE 0.000376 (p95 0.000662). On the same five fixed seeds, the
champion remained 5/5 at 24.675s while the updated policy failed 0/5 at median
23.6% progress. Weight interpolation confirmed a very narrow safe region: a
1% and larger blends failed 0/5; a 0.1% blend completed 5/5 but slowed to
26.027s.

The PPO champion's learned action standard deviation is about 0.15. Stochastic
training episodes under that noise frequently crashed or left the track. A
single on-policy update with a fixed 0.05 action standard deviation instead
completed 5/5 at 25.859s, still slower than the champion but stable enough to
continue searching. The trainer now exposes a fixed PPO action-standard-
deviation setting. Its first full-run check found a mismatch: the actor used
0.05 noise while its anchor retained 0.15, driving measured teacher KL to 5.8.
The backend now applies the same fixed standard deviation to the in-memory
anchor, keeping the mean-action anchor aligned; a regression test covers both
policies. The next sustained run uses 0.05 noise, the 3e-7 learning rate, target
KL 0.0001, and the 24.675s hash-matched anchor. The sub-22s criterion remains
open.

## Frozen-transfer audit follow-up (30 September 2026)

The active PPO continuation was stopped cleanly at 371,912 steps after its
first five-seed candidate evaluation at 365,568 steps finished 0/5 (median
progress 23.0%). The runner restored the saved 24.675s, 5/5 champion before
stopping. No teacher, dataset, or champion files were changed. Further PPO
training is paused until the transfer audit is understood.

I reran the offline same-observation comparison against the exact frozen
teacher and saved reports under the ignored
`runs/teacher-student/transfer-diagnosis-current-*` folders. The teacher policy
hash is still
`FFBEA4CA57116CD2586C17CCEC4FC761600E0D1EE5C6D94E31B98122220DAECE`; its
metadata specifies Summer 1/current, observation schema v2 with 105 float32
features, 12 lookahead samples, frame skip 30, continuous action schema v2,
and seven overlays. The teacher dataset hash is unchanged. Its action labels
recomputed from the raw actor plus the bakeable overlays still agree within
2.3e-6.

The directly distilled, zero-PPO-step model remains a poor copy on the 16,180
successful teacher observations: steering MAE 0.1092 (p95 0.3384) and
longitudinal MAE 0.0684 (p95 0.3178). The live paired run already recorded for
this exact checkpoint has the teacher finishing at 24.263s while the student
fails at 26.0% progress. Its steering differs by 0.098 on the very first
decision, before the cars move apart. This is a measured behavioral-cloning
fidelity failure, not just an inference from lap outcomes.

The current fine-tuned 24.675s PPO champion is closer but still not identical
to TQC on those teacher states: steering MAE 0.0286 (p95 0.0755) and
longitudinal MAE 0.0178 (p95 0.0541). Its `predict()` output is deterministic
(repeat difference 0), equals the clipped Gaussian mean, and uses the same
105-value observation schema, 30-tick decision interval, continuous action
adapter, and saved air-brake overlays. The TQC and PPO backends return the same
tick-control sequence for identical continuous actions in the adapter
regression test. No distinct PPO action bucketing, rounding, exploration noise,
frame-skip mismatch, or observation-normalization layer was found.

The archived same-seed direct-student trace and teacher-action replay add the
closed-loop evidence: the teacher's exact action sequence reproduces its
24.263s lap, with no difference crossing the smallest measured position,
heading, speed, contact, or action threshold. In contrast, persistent
teacher-steering offsets as small as 0.0001 caused failures on some signs and
seeds, though outcomes were nonmonotonic and this is only a single-seed
sensitivity experiment. The student's initial 0.098 steering error is vastly
larger. Taken together, this supports **G (cloning error too large)** and **H
(extreme closed-loop sensitivity)**; the evidence does not support a hidden
observation/action conversion defect as the primary cause.

The current live game worker was not listening after the safe stop, so I did
not claim a new paired lap for the 24.675s champion. The 24.263s teacher result
and direct-student trace above are the existing matched-seed live comparison
for the same frozen teacher hash. No implementation fix was justified by this
follow-up, and no training or re-distillation was started. Next work should
first produce a much closer initial actor match and gate it on repeated
same-start closed-loop tests; ordinary PPO updates at the current safe-region
scale remain too disruptive to justify running blindly. The under-22s goal is
still open.

## Exact actor graft and live parity result (30 September 2026)

The audit exposed two transfer gaps that the earlier comparison did not
isolate. First, the teacher's TQC actor is two 128-unit ReLU layers, while the
PPO architecture label saved for earlier students resolved to two 256-unit
Tanh layers. Second, TQC carries a learned speed-bias schedule as well as its
overlay stack. The previous PPO path neither copied the actor weights nor
applied the complete saved schedule. A new `tqc_compatible` PPO policy uses a
squashed Gaussian with the same 128x128 ReLU actor, and the transfer command
can now copy TQC actor parameters exactly while keeping PPO's trainable critic
and action distribution.

The resulting zero-update graft, with the teacher's speed schedule and overlays
applied, matches all 16,180 successful teacher observations to maximum action
error 2.265e-6 (steering MAE 3.35e-7; longitudinal MAE 1.96e-7). The first live
paired test still differed at one air-brake window boundary. Its cause was a
precision mismatch: the PPO wrapper promoted float32 progress to Python
float64 before computing the overlay fade. At a boundary, that produced a tiny
positive fade where TQC's float32 calculation produced zero. Since the wrapper
then changed the whole frame-skip block to per-tick braking, this tiny numeric
difference caused a real control change. The wrapper now keeps progress in
float32, matching TQC.

After that fix, the same-seed live test completed 809 decisions on both
policies. TQC and grafted PPO each finished in 24.263s at progress 0.999935;
the traces had no action, position, heading, wheel-contact, speed, or steering
divergence at any recorded threshold. This verifies that the earlier student
failures came from a large actor/transfer mismatch, not inherent inability of
PPO to reproduce this teacher. It does not yet meet the sub-22s criterion: the
validated PPO is currently a faithful 24.263s initialization, and fine-tuning
should proceed only from this parity checkpoint with evaluation safeguards.

## First on-policy probe (30 September 2026)

The exact graft passed five fresh live validation episodes at 24.263s (5/5,
zero crash/off-track/stall events), was registered as the isolated PPO
champion, and was promoted over the slower global PPO champion. An initial
guarded fine-tuning run used learning rate 3e-6, target KL 0.0003, anchor KL
0.1, and action standard deviation 0.05. Its first deterministic evaluation
slowed to 24.715s (5/5); the next two candidates finished 0/5, with median
progress 56.4% and 23.4%. Each regression was restored to the verified
24.263s champion. The run was stopped cleanly at 20,480 steps, before further
updates could accumulate.

The first rollout logs showed actor approximate KL below 0.0004, while critic
explained variance remained near zero. A 10,240-step value-only warmup then
kept actor parameters byte-for-byte unchanged and again evaluated 5/5 at
24.263s, but final critic explained variance was still only about 0.012. Two
further PPO candidates using learning rate 3e-7, target KL 0.0001, anchor KL
0.25, and action standard deviation 0.02 both evaluated 0/5, at median
progress 14.3% and 30.4%. Both were rolled back. This shows the short critic
warmup did not resolve the policy update instability; deterministic transfer
and base champion remain intact, but no sub-22s PPO lap has been achieved.

The runner now saves a rejected PPO candidate and its failed evaluation under
`checkpoints/step-<n>-rejected` before restoring the champion. The next probe
can compare that exact failed actor against TQC on the same teacher states and
trace the live divergence, instead of losing the candidate during rollback.

The preserved 15,360-step candidate was replayed live against TQC on the same
seed. It failed at 14.2% progress, while TQC finished in 24.263s. Their
executed tick schedules first differed at decision 8 even though the continuous
steering actions differed by only 0.000045 and the reported per-block steering
duty was identical. `ContinuousPwmControls` carries fractional pulse error
between blocks, so that small action perturbation moved two steering pulses
within the 30-tick block. The resulting tiny state offset reached 0.0001m and
0.005 degrees by decision 13; on those neighboring states, both neural actors
changed steering by about 0.182. This identifies stateful PWM phase plus the
teacher actor's local sensitivity as the closed-loop amplification path.

Resetting PWM phase at every block was tested and rejected: it made the frozen
TQC policy itself fail at 23.0% progress, so that would alter the teacher's
control semantics too much. The follow-up implementation instead adds a
TQC-residual PPO architecture: it freezes the transferred ReLU trunk and mean
head, and learns a zero-initialized linear correction on top. A fresh graft
passed five live evaluations at 24.263s with identical action and telemetry
traces, so the initialization is stable.

Residual fine-tuning exposed the same amplification at a much smaller scale.
With a 0.1 action-correction limit, the 5,120-step candidate's offline action
MAE was only 0.000137 steering and 0.000095 longitudinal. On a live identical
start, the first PWM tick schedule difference occurred at decision 3: teacher
and student requested steering 0.933984 and 0.933989, with the same average
duty, but two pulses landed on different ticks. Their positions differed by
only 0.000006m then. At decision 13, nearby observations made the teacher's
steering responses differ by about 0.182; the student rolled over at 53.9%
progress while TQC finished in 24.263s. Same-observation teacher/PPO action
differences at those early states remained below 0.001. This confirms the
dominant failure is extreme closed-loop sensitivity to pulse-phase divergence,
not a large policy-output or observation-schema mismatch.

Bounded PPO probes at learning rates 3e-7 and 3e-6 were both rolled back. The
3e-7 candidates repeatedly failed near 54% progress; the 3e-6 candidate failed
near 23.5%. Air-brake use fell to 6-10% of airborne time on the former,
compared with 45.3% for TQC. The residual bound is retained as a configurable
safety limit, but it has not yet made on-policy optimization useful. The
validated PPO and global champion remain at 24.263s; no sub-22s PPO lap has
been confirmed. Future progress needs a rollout strategy that avoids letting
microscopic PWM phase changes destabilize the teacher's high-sensitivity
steering states, while still allowing larger corrections in recoverable
states.

The next bounded search tried smaller exploration noise and longer PPO
rollouts. Action standard deviation 0.0001 caused a non-finite policy-loss
metric and aborted before saving; 0.005 remained finite but its candidate
failed at 53.4% progress (20.0% airborne braking). A fresh exact graft using
4,096-step rollouts, batch size 256, and three epochs passed the same 5/5
24.263s initialization gate. After 20,480 training steps, its rejected
candidates finished 0/5 at 58.9%, 55.5%, and 23.5% progress, with airborne
braking between 0.8% and 9.7%. Longer rollouts and lower exploration alone do
not resolve the failure. Continue from the validated checkpoint only; the
next experiment should provide teacher-labeled recovery data or constrain
updates to states where a deviation can be recovered, then confirm a complete
PPO lap before increasing the search range.

## DAgger transfer follow-up (30 September 2026)

The bounded recovery DAgger run exposed two implementation defects in the
supervised initialization path. The optimizer included the residual action head
in the forward pass but did not update it; and its loss compared the raw,
pre-squash Gaussian mean with teacher actions even though live `predict()` uses
the squashed deterministic action. It also trained against overlay-adjusted
targets while overlays are applied again after the policy at inference. The
training loss now uses the policy's deterministic post-squash action and the
teacher's raw action labels, and includes the residual head among trainable
parameters.

The progress window for residual corrections was also only held in runtime
state. Saving/reloading a student silently dropped it. The window is now part
of PPO policy kwargs and is restored when a DAgger student is loaded. Tests
cover optimizer updates to the residual head, squashed-action/overlay label
semantics, and persistence of the progress window.

These fixes do not establish that DAgger can produce a viable policy. In the
corrected v6 run, every one of eight recovery episodes in rounds 2–4 left the
track at 56.7% progress. The round-3 student then had median validation
progress 23.5%; round 4 recovered to 52.9%, still with zero finishes. In the
55–60% recovery window, the student’s mean longitudinal action error against
the teacher was 0.193 (maximum 0.402), while steering error remained about
0.0045. This sharp error increase aligns with the failure region and suggests
that the current DAgger dataset and loss do not yet teach the necessary
closed-loop recovery. The zero-update graft remains the only validated PPO
policy; the TQC teacher remains 24.263s. No sub-22s PPO lap has been confirmed.

## Residual actor freeze after checkpoint load (30 September 2026)

A late-track PPO probe gated the residual to progress 60–100%, but its first
deterministic evaluation still failed at 56.6%. The run log showed 30,596
trainable actor parameters at startup; the verified graft metadata lists only
258 trainable actor parameters. The discrepancy identifies the cause: SB3
checkpoints preserve tensor values and optimizer state, but not PyTorch
`requires_grad` flags. The exact TQC trunk and mean head had been frozen when
grafted, then silently became trainable when the PPO checkpoint was loaded.
Consequently PPO changed the base actor before the gated residual could act.

The PPO backend now re-freezes the base actor whenever it loads a
`tqc_residual` checkpoint and again when configuring resumed training. The
residual and value network remain trainable, while the TQC feature extractor,
policy trunk and mean head stay fixed. A save/load regression test checks these
parameter flags. The late-track probe was stopped after restoring the verified
24.263s checkpoint; its interrupted candidate is not being reused. The next
on-policy run must start from that exact checkpoint under the corrected load
path and prove the actor count remains 258 before evaluating its pace.

The corrected gated probe reported exactly 258 trainable actor parameters.
Three 5-episode evaluations at steps 8,192, 16,384 and 24,576 completed 5/5
but measured 24.276s, 24.316s and 24.385s; each was rejected and the 24.263s
champion restored. A second probe raised the learning rate from 3e-7 to 1e-5
and target KL from 1e-4 to 1e-3, with anchor coefficient 0.1. Its candidates
were 24.340s, 24.331s and 24.385s, also 5/5 and slower, so they were rolled
back. These runs confirm stable early-track imitation but do not show a pace
gain. Because 5,000-step evaluation spacing yielded checks only every 8,192
PPO decisions with a 4,096-step rollout, the next run increases the evaluation
spacing to 20,000 decisions and tolerates up to 0.5s slowdown for continued
training. Promotion remains strict: only an actually faster champion replaces
the 24.263s seed, and a loss of more than 0.5s or completion stability still
restores it.

## Transfer isolation recheck (30 September 2026)

The current frozen teacher and saved teacher dataset were rechecked against
three PPO artifacts using the same 16,180 teacher observations. The teacher
`policy.zip` SHA-256 is
`FFBEA4CA57116CD2586C17CCEC4FC761600E0D1EE5C6D94E31B98122220DAECE`; its
metadata reports 24.263s and a 100% finish rate. Recomputed high-level teacher
targets differ from the saved post-overlay labels by at most `2.27e-6`, so the
dataset labels are correct.

The ordinary PPO student at
`models/experiments/ppo-transfer-fidelity-20260930/summer-1/ppo/teacher-student/pretrained`
has SHA-256
`22B8500CA5C1588183EFDC2F3B6AD672F2863F6BA023E4F859B2ADD4B2942A2E`. Its
same-observation action error is smaller than the older wall-spin checkpoint,
but still measurable on most decisions: steering MAE/RMSE are `0.0369`/`0.0538`
(p99 `0.1851`), and longitudinal MAE/RMSE are `0.0168`/`0.0327` (p99
`0.1428`). Behavior cloning reduces, but does not eliminate, the control error.
The later refined checkpoint reduces those to steering MAE `0.0175` (p99
`0.0844`) and longitudinal MAE `0.0085` (p99 `0.0590`), yet its same-seed
live run still ends in an airborne-roll failure at 55.1% progress after 14.82s.

The actor-copy control at
`models/experiments/ppo-tqc-actor-graft-20260930/summer-1/ppo/teacher-student/pretrained`
has SHA-256
`A4C48A885A054836801AAA9F9BE2583FAFDC404F644E4D9D2F98427E8CB42C0F`. Its
same-observation action MAE is below `3.4e-7` steering and `2.0e-7`
longitudinal. In the saved same-seed live traces it finishes in 24.263s just
like TQC: all 809 observation vectors and tick-control sequences match exactly,
and the maximum position difference is zero. This is direct evidence against an
observation-pipeline, frame-skip, overlay, air-brake, PWM, or simulator-replay
mismatch in the shared path.

For the ordinary PPO student, action differences appear first: the first
steering difference above `0.01` is decision 1, and the combined action
difference exceeds `0.05` by decision 3. Heading differs by `0.1°` at decision
8; position differs by 1cm at decision 14. The same-seed episode fails
off-track at 24.9% progress after 14.07s. The refined student's first action
differences above `0.005` also occur by decision 3, before its position differs
by 1cm at decision 20. These
results answer the ordering question: the cloned actors already choose
different controls before the car paths visibly separate.

The ordinary PPO artifact uses SB3's unsquashed Gaussian policy, whose public
`predict()` clips deterministic means to `[-1, 1]`; the TQC-compatible PPO
control uses the same tanh-squashed deterministic mean as TQC. The ordinary
student is deterministic and repeatable at inference. On the saved teacher
states, its raw Gaussian mode differs from the clipped public action by as much
as `0.2217`, while public `predict()` matches the correctly transformed mode
exactly. This clipping is expected behavior, not an inference bug; the action
error table and live comparison use the public clipped prediction. It does not
explain the whole failure: even a `0.0001` fixed TQC action offset can
derail a live lap, while exact replay of the unmodified TQC actions reproduces
the finish. In a separate early residual probe, a `5e-6` requested steering
difference moved two PWM pulses to different ticks; the state was only
`6e-6m` apart then, before a later state-dependent steering response amplified
the trajectory difference.

The updated classification is **G: behavioral cloning error too large** and
**H: extreme closed-loop sensitivity**, with an additional policy-output
transform difference for unsquashed standard PPO. The shared simulator control
path itself is verified by the exact actor-copy control and exact action replay.
Do not resume ordinary BC/DAgger or unconstrained PPO updates on the failed
student. If training resumes toward the sub-22s goal, start from the validated
TQC-compatible actor-copy checkpoint; use bounded updates with deterministic
teacher comparisons and reject a candidate at the first repeatable pulse-phase
or trajectory regression. The current 24.263s actor-copy PPO has zero on-policy
updates and is not a sub-22s result.

## Current-session confirmation (30 September 2026)

The read-only comparison was repeated against the frozen teacher and the
ordinary directly distilled student. The teacher SHA-256 remains
`FFBEA4CA57116CD2586C17CCEC4FC761600E0D1EE5C6D94E31B98122220DAECE`; the
student is
`models/experiments/ppo-wallspin-standard-20260930/summer-1/ppo/teacher-student/pretrained`,
SHA-256 `8C0CE7D9D875803BFD30D8030C4A635D488B81D381718337090A7E828A14D0F2`.
On seed `20260929`, TQC again finished in 24.263s. The ordinary student failed
at 25.998% progress after 8.13s from an airborne-roll failure. Its first action
already differed by more than 0.05, before measurable vehicle separation;
heading differed by 0.1 degrees at decision 3 and position by 1cm at decision
12.

The exact TQC-compatible actor-copy PPO is a useful control: its SHA-256 is
`A4C48A885A054836801AAA9F9BE2583FAFDC404F644E4D9D2F98427E8CB42C0F`. In the
same-seed live comparison, both it and TQC finished in 24.263s in 809 decisions.
No action, position, heading, speed, or wheel-contact difference crossed the
smallest detector threshold, and replaying the teacher's saved actions again
reproduced the same finish. Its same-observation action MAE was below
`3.4e-7` steering and `2.0e-7` longitudinal.

The TQC-compatible candidate after the three rejected update blocks had
same-observation MAE `0.00036` steering and `0.00028` longitudinal (maximum
error `0.0024`). Despite those small errors, its deterministic evaluations
finished 0/5 times, with median progress `23.5%`, `56.7%`, and `23.5%`. The
single-seed teacher perturbation test independently showed failures from fixed
offsets as small as `0.0001`. These are not estimates of general finish
probability, but together they demonstrate that the line is acutely sensitive
to small actor changes.

The report's deterministic-output check was corrected in this recheck. It now
compares public `predict()` with the distribution mode after the policy's
actual transform: tanh/unscale for squashed policies and action-bound clipping
for ordinary Gaussian policies. For both tested PPO architectures the maximum
difference is zero. The previous report field had compared an unsquashed
policy's clipped output against its *unclipped* mode and mislabeled that as a
clipped-mean discrepancy. A regression test now covers both transforms.

## Stagewise residual continuation (30 September 2026)

The residual progress gate originally masked its learned mean correction but
still sampled Gaussian exploration noise before the gate. Even an inactive
standard deviation of `1e-4` could change PWM pulse phase and cause early
airborne failures; reducing it to near zero made PPO's squashed-Gaussian
likelihood numerically unusable. The policy now uses the exact deterministic
mean for rollout actions outside the residual window, while retaining a finite
likelihood for the PPO update. A regression test checks exact inactive actions,
active stochastic actions, and a finite on-policy PPO update.

The exact TQC actor graft with a 75%-progress gate passed five live laps at
24.263s. After guarded residual updates its best evaluated candidate reached
24.300s, still slower than the teacher. Widening the gate to 60% passed an
independent 5/5 live check at 24.362s. On-policy evaluation at 81,920 steps
slowed to 24.435s and was rolled back; later 5/5 candidates evaluated at
24.318s at 86,016 steps and 24.332s at 90,112 steps. The run continues from
the isolated best PPO checkpoint. Rollout episodes now finish consistently,
but no candidate has yet beaten the 24.263s baseline or reached the 22-second
target.

## Full-track impact-cost PPO (30 September 2026)

The stagewise run stopped at 118,784 steps with the exact graft still the best
policy (24.263s). Its 60%-progress gate could not learn from the early contact,
and its actor residual remained narrowly bounded. A new isolated PPO experiment
starts from the frozen TQC actor graft with a full-track progress window and a
larger but bounded residual. The objective remains lap time and completion; no
section has prescribed actions.

The `Summer 1 - 20s pace impact` profile adds a small global cost for collision
impulses above threshold, excluding landing transitions. Such an impact is now
nonterminal so the policy can learn from what follows and complete the lap. The
fresh actor graft passed 5/5 live validation at 24.263s before PPO updates. The
first five blocks used action standard deviation 0.02. Every candidate either
failed the five-lap reliability evaluation or ran slower, so guarded rollback
restored the graft each time; the best remains 24.263s at 49,152 timesteps. This
exploration scale was too disruptive for the sensitive route. The next blocks
used 0.01 action standard deviation and a 0.05 residual limit across the full
track. The next two deterministic candidates still failed 0/5 live laps and
were rolled back. Failures clustered around 23% progress at the first jump and
chicane. A 25%-gate trial let rollouts through the first chicane, but many then
failed around 53–57% progress. Its first candidate finished 5/5 at 24.773s; the
next two failed 0/5, so all were rejected. The next curriculum trial keeps the
grafted behavior through that second failure cluster (gate starts at 58%) and
lets PPO learn over the finish section. These progress gates prescribe no
control action or line; earlier sections can be opened once the later section
shows reliable improvement. At the 58% gate, four candidates finished 5/5 with
median lap times of 24.347s, 24.405s, 24.375s, and 24.321s. Each was rolled
back because it remained slower than 24.263s. PPO's approximate KL stayed near
`6.7e-6`, well below the `0.001` target, so the next trial raises the learning
rate and update epochs while retaining the same gate and rollback protection.
With the higher rate, four more 5/5 candidates evaluated at 24.394s, 24.362s,
24.364s, and 24.442s. The larger update rate increased KL but did not
produce a faster policy. The next trial widens the bounded residual to 0.1 so
the actor can make a larger state-dependent correction in the late section.
Two candidates with the 0.1 residual limit still finished 5/5 at 24.373s and
24.341s, slower than the graft. Their changes were small, so the next trial
raises action standard deviation from 0.005 to 0.01 within the same 58%-to-finish
window while keeping rollback strict. Four 5/5 candidates then evaluated at
24.373s, 24.341s, 24.390s, and 24.330s, still slower than baseline; mean late
action drift remained about `2.4e-4`. The next trial raises the learning rate to
`1e-4`, the KL target to `0.01`, and PPO epochs to five to let on-policy updates
move farther while preserving the same gate and rollback. The first four
five-lap candidates were 24.370s, 24.446s, 24.370s, and 24.369s, all slower than
the graft and rolled back. The next trial tests standard more exploratory PPO
settings (learning rate `3e-4`, action standard deviation `0.02`, KL target
`0.02`) with the same late-section gate and exact rollback. Three candidates
then finished 5/5 at 24.365s, 24.393s, and 24.363s, followed by 24.383s; each
was rolled back. A full-track low-noise trial produced 28.523s and 29.818s
5/5 candidates followed by a 0/5 candidate; all were rolled back. Strict
zero-tolerance rollback also restarted PPO from the same exact actor each block,
so small on-policy changes could not accumulate. The next trial returns to the
reliable 58% gate and permits candidates within 0.25s of the graft to continue
learning. The exact TQC graft stays the champion and rollback point; larger
regressions still restore it.

The first accumulation blocks then evaluated at 24.338s, 24.394s, and
24.349s, each 5/5 finishes. They remain slower than the graft, and their
episode summaries reported zero `barrier_contact` reward. The existing -50
impact cost was weak relative to the roughly 5,000-point finish reward, and it
did not appear in those logged rollouts. The profile now uses a global -1,000
cost for detected non-landing impacts, without ending an episode or choosing
any control action. Training episode records also report raw collision-impulse
peak/count, split into landing and non-landing peaks. A fresh run from the
exact graft will show whether the simulator sends impact signals for the
visually observed barrier contacts before PPO is judged on whether it learns to
avoid them.
