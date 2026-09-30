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
