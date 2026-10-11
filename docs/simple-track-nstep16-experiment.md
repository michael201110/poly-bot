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
