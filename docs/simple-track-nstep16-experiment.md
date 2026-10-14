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

The continued run (`logs/simple-track/grtqc/20261014T102624Z.jsonl`) has been screened at steps 3,276,608, 3,278,864, 3,281,120, 3,283,376, 3,285,632 and 3,287,888. The first five deterministic five-lap checks were respectively 33.620, 33.583, 33.557, 33.564 and 33.564 seconds, with zero barrier-contact steps each. All remain slower than the 33.345-second protected Champion and were rejected for promotion. The next five-lap check was 33.877 seconds with 10 total contact steps; it was also rejected. This shows the smaller per-update cap has not yet produced a verified pace gain and has not eliminated contact regressions.

The active training replay index is `models/simple-track/grtqc/visual_replays/5810cd679aa14ab78bbd0bf6fe6e7b06/index.json`; captures include both completed and interrupted segments with observations and HUD telemetry. Inspection of a completed training replay confirms the raw policy can emit exactly -1 and +1. A normalized request around 0.97 is already near saturation: with 30-tick pulse-density control it averages about 29 active ticks per 30. Do not force such near-limit commands to +/-1 globally without evidence from a controlled comparison, as this changes steering and signed throttle/brake equally and can increase contact risk.

Further five-lap checks completed at step 3,294,608 (33.384 s, zero contacts) and 3,296,864 (33.628 s, zero contacts). Both were rejected against the protected 33.345-second Champion. The weak-actor counter is 3/5, and Latest continues training under the profile's existing recovery gate. Neither screen is a pace breakthrough; retain the Champion.

At step 3,299,120, the five-lap evaluation was 33.814 s with one barrier-contact lap on each of the five attempts. The contact regression gate rejected this candidate and restored the protected Champion actor; it preserved Latest's critics and 250,000-transition replay, then entered the configured 1,000-update critic cooldown. Training remained active from the recovered actor at step 3,299,689. This confirms the rollback is working, but also that the 0.00125 per-update cap alone does not prevent the actor from learning a contact-causing policy.

The active training replay bundle had 31 episode files at this check, including 21 completed runs and 10 short interrupted evaluation segments, with observations recorded. The larger Simple Track visual-replay library remains available for swarm selection.

A second contact rollback occurred at step 3,303,584: the five-lap median was 33.801 s with one contact on each lap, and the gate again restored Champion while retaining critics/replay. After its critic cooldown, the recovered actor reached a clean five-lap 33.570 s at step 3,308,064; it remained slower than Champion and was rejected without rollback. The following screen at step 3,310,320 was 33.600 s with zero contacts and awaits its five-lap result. Training remains active. The current run's visual replay bundle contains 44 episodes (31 completed, 13 interrupted), in addition to the existing larger swarm library.
