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

After the second contact rollback, a later five-lap check at step 3,312,576 improved to 33.468 s with zero contacts. It remains 0.123 s slower than the protected 33.345-second Champion, so it was rejected. This is the best clean verified screen in the post-recovery segment so far, but not a promotion. Training continued with the actor drift at roughly 0.024; the active replay bundle had 48 episodes (34 finished, 14 interrupted).

The next checks after recovery were 33.722 s at step 3,314,832 and 33.475 s at step 3,317,088, both five-lap verified with zero barrier contacts and both slower than Champion. The latter is the closest post-recovery result so far but remains 0.130 s behind; both were rejected and training continued. At this point the active replay bundle contains 53 episodes (37 completed, 16 interrupted), with observations and HUD controls/reward telemetry retained.

At step 3,319,344 the candidate evaluated at 33.890 s with one barrier-contact step in each of five laps. All five contacts occurred at normalized progress 0.993742 (maximum impulse 95.21), immediately before the finish; the gate rejected it and restored the Champion actor again. Earlier contact-causing candidates also hit near the end (0.9359 and 0.9923 at step 3,287,888; 0.9937 at step 3,299,120; 0.9874 at step 3,303,584). This repeatable late-lap location points to a specific end-of-track control weakness in candidate actors, while the protected Champion remains clean. Training continues after recovery; no track-specific action script was introduced.

After the step-3,319,344 finish-area contact rollback, the recovered actor produced clean five-lap results of 33.637 s at step 3,323,824 and 33.575 s at step 3,326,064. Both had zero contacts and were rejected against Champion; training continued with the recovery safeguards intact. The active run remains in progress at step 3,326,565, and its replay bundle has 63 episode files (45 finished, 18 interrupted).

The following clean five-lap checks were 33.722 s at step 3,328,320 and 33.654 s at step 3,330,576, both with zero barrier contacts. Each was slower than the Champion and rejected; Latest continued training. At this check the session had advanced beyond step 3,330,845 and the active replay bundle contained 69 episodes (49 finished, 20 interrupted).

The next clean five-lap check at step 3,332,832 improved to 33.448 s with zero contacts, but remained 0.103 s slower than Champion and was rejected. This is the closest candidate since the actor recovery sequence; training continued. The active replay bundle had 72 episodes (51 finished, 21 interrupted).

Two further deterministic five-lap checks completed at steps 3,335,088 (33.582 s) and 3,337,344 (33.680 s), both with zero barrier contacts. Both were rejected against the protected Champion. The weak-actor counter is 2/5 and training continues. The current run replay bundle contains 79 episodes (56 finished, 23 interrupted).

At step 3,339,600 the actor passed a clean five-lap check at 33.551 s with zero barrier contacts. It remained slower than the protected Champion and was rejected. Training continued; the active replay bundle contained 84 episodes (59 finished, 25 interrupted).

At step 3,341,856, the five-lap evaluation was 33.641 s with zero barrier contacts. It was rejected against the protected Champion; the weak-actor counter is 1/5 and Latest continues training. The active training run's replay bundle currently has 84 episodes (59 finished, 25 interrupted); the broader Simple Track replay library already contains thousands of finished runs for later swarm selection.

At step 3,344,112 another five-lap evaluation completed at 33.643 s with zero contacts. It was rejected against Champion and advanced the weak-actor counter to 2/5; training continues. The active replay bundle now contains 87 episodes (61 finished, 26 interrupted).

At step 3,346,368 the actor achieved a clean five-lap median of 33.399 s with zero contacts. It is 0.054 s slower than Champion, so it was not promoted, but the learner classified it as a clean local pace improvement and reset the weak-actor counter. Training continued. The active replay bundle contains 90 episodes (63 finished, 27 interrupted).

At step 3,348,624, the latest deterministic five-lap screen was 33.580 s with zero barrier-contact steps. It is slower than the 33.345-second protected Champion, so it was rejected; the Champion remains unchanged. Training continues in the same session. By step 3,350,867, its replay bundle contained 96 episodes (67 finished, 29 interrupted).

I audited the newest completed captured run (`episode-000096`, 33.640 s, 1,122 decisions) after the near-maximum control concern. Raw policy actions were at least 0.96 magnitude on 33.3% of steering decisions and 43.0% of longitudinal decisions; exactly ±1 occurred on 2.2% and 2.5%, respectively. HUD fields distinguish model output/requested duty from the per-decision applied fraction. For example, the model requested 0.99995 throttle, while that 30-tick window displayed applied 0.96667 (29 pulses). The pulse-density controller carries its fractional phase across decisions, so one window's 29/30 average is not evidence of a persistent 3.3% power loss. Do not snap near-limit requests to ±1 globally: it affects steering and both pedals, and no controlled pace/contact comparison supports it. The complete captured action stream remains saved with observations for later replay/swarm use.

The next five-lap check completed at step 3,350,880 in 33.487 s with zero barrier-contact steps. It remains slower than the 33.345-second Champion and was not promoted, but the actor evaluation marked it as a local pace improvement. Training resumed without rollback; by 3,351,515 it was progressing at about 25.1 decisions/s. The active replay bundle still contains the captured run and evaluation data for later swarm use.
