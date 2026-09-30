# Summer 1 TQC-to-PPO transfer diagnosis

The directly distilled PPO student does **not** reproduce the frozen TQC
teacher's actions accurately enough to stay on its line. This is visible before
the two cars separate. The simulator is also highly sensitive to even much
smaller persistent action changes, so ordinary behavioral cloning error is a
poor initialization for this particular 24.263-second path. No additional
DAgger or PPO training was run during this diagnosis.

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
restored the 24.838s champion each time. Fine-tuning remains active from the
champion; the confirmed sub-22s target is still unmet.
