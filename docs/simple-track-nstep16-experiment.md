# Simple Track: 16-step continuation

Prepared on 2026-10-11 after the 8-step scratch GRTQC learner plateaued.

- Protected champion: 34.348 seconds, five deterministic finishes.
- Best observed training lap: 34.217 seconds, not verified for promotion.
- Most recent deterministic evaluation: 34.673 seconds, five finishes at step 725,262.
- Resume source: Latest at step 729,315.
- Change: `training_config.grtqc.n_step_return` from 8 to 16 in local Latest metadata.
- Full original Latest snapshot, including policy and replay, preserved under
  `models/simple-track/grtqc/archive/nstep8-before16-step-729315-20261011T193918Z/latest`.

Rewards, actor/critic learning rates, frame skip 30, replay contents, champion,
evaluation gates and the one-million-step session budget are unchanged.
The existing resume path converts the replay sampling horizon while preserving
raw transitions and temporarily closes the actor gate for critic readiness.
The serialized policy still records the previous horizon until continuation loads
the updated configuration and saves another checkpoint.

Continue Latest in the GUI. Compare verified evaluations after approximately
30,000 additional decisions (around global step 759,315); training laps alone
do not establish an improvement. No 16-step result is established yet.

## Contact regression and actor update cap (2026-10-14)

The protected Simple Track Champion remains 33.345 s, verified over five clean laps. After the disk recovery, Latest at step 3,275,472 produced a clean five-lap median of 33.768 s and was rejected; the actor had again saturated the cumulative reference-drift ceiling of 0.05. A preceding clean check at step 3,270,960 was 33.483 s, also slower than Champion. No new champion was promoted.

To reduce how quickly the actor consumes its allowed cumulative drift between 2,000-step checks, the active contact-aware profile changes only `actor_step_action_limit` from 0.0025 to 0.00125. Actor LR (5e-6), cumulative drift limit (0.05), 16-step return, rewards, replay, critic settings, evaluation gate, and 50,000-step checkpoint cadence are unchanged. The update is intended to reduce regression/contact risk; it is an experiment, not an established pace improvement.

Training resumed from Latest at step 3,275,472 on the NVIDIA T500, retaining the 250,000-transition replay and using a fresh 1,000,000-decision session budget. Initial telemetry shows the actor accepted a 0.00125-scale step after resuming from the saturated boundary. Judge this change only from later deterministic five-lap evaluations, barrier telemetry, and the protected Champion gate. Preserve Champion unless a candidate is faster and passes its verification gate.
