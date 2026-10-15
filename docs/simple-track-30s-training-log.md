# Simple Track 30-second training log

## 2026-10-14 continuation and disk recovery

The protected Simple Track GRTQC Champion remains **33.324 s**, verified in
five clean laps. The older training log records a best training lap of
33.105 s; this has not replaced the Champion because it was not a faster
verified five-lap result. The user also observed a PolyTrack personal best of
33.106 s. A saved visual training replay at step 2,616,954 has a 33.105 s
finish and final simulation tick 33,105. The one-millisecond difference is
consistent with a one-tick timer offset. The bridge reports elapsed time as
`session.tick * 0.001`; its native recorder assigns controls to the previous
state frame. This is the likely source, but the original PolyTrack PB replay
has not been matched byte-for-byte to that training capture, so no timing
correction has been applied.

On 2026-10-14, training resumed from the last complete `latest` checkpoint
with the existing full-track, one-million-step
`simple-track-grtqc-contact-aware-30.json` profile, CUDA/T500, port 8765, and
visual replay capture. Startup replay retention reduced numbered checkpoint
buffers from 35 to five. The first resumed episode finished in 33.595 s with
a late barrier contact. A subsequent deterministic five-lap check at step
3,596,896 finished 5/5 in 33.674 s with zero contacts and was correctly
rejected against Champion. Saving replay state then failed with
`[Errno 28] No space left on device`; the incomplete step checkpoint was not
used for recovery. The last complete `latest` was at step 3,596,640.

To keep training and preserve the swarm archive, completed visual replay runs
are being moved losslessly to `D:\PolyBot-Visual-Replays\simple-track\grtqc`.
Junctions at their original model paths keep the replay GUI's paths stable.
The active replay run remains on C:. The first completed 5.13 GB run has been
moved and linked; archival of the other completed runs is continuing in the
background. At the latest check, C: had about 12.1 GB free. This is a
removable-volume fallback, so those archived replays require D: to be
connected when loaded.

Training restarted from `latest` at step 3,596,640 with the same one-million
step budget. It reached a clean one-lap screen of 33.635 s; the corresponding
five-lap evaluation was also 33.635 s and was rejected against the 33.324 s
Champion. At step 3,600,272, another clean five-lap evaluation improved to
33.601 s with zero barrier contacts. This is a 0.034 s improvement over the
previous clean evaluation, but still slower than Champion, so it was rejected
for promotion. At step 3,602,528, a further clean five-lap evaluation improved
to 33.495 s with zero contacts, 0.106 s faster than the preceding clean
candidate. It remains slower than Champion and was not promoted. Training then
continued; at the latest check it was active on port 8765 at step 3,602,652,
reporting 10.3 steps/s and retaining its best training lap of 33.105 s. No
new Champion has been promoted.

The new visual replay run `3df0a4c5db0c4f5ca9e2d91062360b75` remains a normal
directory on C: and contains 12 captured episodes so far. The latest completed
training episode finished in 33.797 s; training captures are not champion
evaluations. A deterministic five-lap check finished in 33.693 s with zero
contacts, slower than the 33.495 s best clean candidate, and was rejected. A
later clean five-lap check improved to 33.623 s with zero contacts. At step
3,609,296, another clean five-lap check improved to 33.582 s with zero
contacts. Both remain slower than the 33.495 s best clean candidate, so no
Champion changed. At step 3,611,536, a five-lap evaluation regressed to
33.736 s; at step 3,613,792 it recovered to 33.509 s (5/5 clean, zero
contacts), just 0.014 s slower than the best candidate. Neither replaced
Champion. At step 3,616,048, a clean five-lap evaluation was 33.602 s, slower
than the best candidate, and was rejected. At step 3,618,304, another clean
five-lap evaluation improved to 33.592 s with zero contacts, but it remains
slower than the 33.495 s candidate. `latest` now records this evaluation. The
replay archive is complete: 91 of the 93 existing replay directories are on D:
with junctions at their original paths; the current and previous runs remain
local on C:. At the latest check, training was connected on port 8765 at step
3,618,769, reporting 11.7 steps/s. The current replay run contains 28
episodes. C: had about 30.8 GB free and D: had about 144.3 GB free.

## Racing-line update behavior

The trainer records a candidate line from evaluation telemetry, but promotes
that line only with an eligible verified policy candidate. For scratch GRTQC,
the candidate must pass the full five-lap finish gate, be faster than the
current Champion, and avoid a regression in barrier-contact rate. When a line
change is eligible, the trainer runs another five-lap evaluation using the
candidate line before saving it. Thus the line used for guidance can remain
unchanged while the actor explores; the actor's actions can still take a
different path. At step 3,613,403 the actor proposed action drift of `0.0233`,
but executed drift was `0.001245`, close to its `0.00125` per-update limit;
cumulative reference drift was `0.0361` of the `0.05` limit. The live process
loaded its configuration at startup and continues with those original limits.
For the next continuation, the profile on disk now allows a `0.0025`
per-update action change and `0.075` cumulative reference drift. Actor learning
rate remains `5e-6`; five-lap evaluation and protected Champion gates are
unchanged. This profile edit does not alter the active process and takes effect
when a later process loads the profile.

## 2026-10-14 continued training checkpoint

The existing training process (PID 10892) remained connected on port 8765 and
continued from the 2026-10-14 `latest` checkpoint. At training step 3,629,503,
a captured training episode finished in 33.396 s. The subsequent deterministic
five-lap evaluation at step 3,629,504 completed 5/5 laps in 33.324 s with zero
barrier-contact steps. This matches the protected 33.324 s Champion but does
not beat it, so the Champion remains unchanged. The `latest` checkpoint was
saved at 17:35:59 UTC with that evaluation. The active visual replay index had
42 episodes at the check; replay recording continued.

The active process still has the actor limits loaded at startup (0.00125 per
update and 0.05 cumulative reference drift). The profile on disk now specifies
0.0025 and 0.075, respectively, but those values are not hot-reloaded into the
current process. The checkpoint metadata still carries the historical
33.105 s best training lap; it is not a new verified evaluation. No training
code, reward settings, champion files, or replay-retention settings were
changed for this checkpoint update.


## 2026-10-14 apply the actor exploration limits

Replay analysis of the live training run found three consecutive 33.324 s
training episodes whose saved actions, observations, positions, and wheel
telemetry were byte-for-byte identical. The latest actor diagnostic showed
proposed action drift of 0.02265 but accepted drift of 0.001248, effectively
at the active 0.00125 per-update limit. This supports the earlier observation
that the actor's line changes were being strongly constrained; it does not by
itself establish the cause of the pace plateau.

After a fresh `latest` checkpoint was saved at training step 3,631,760
(33.577 s, five of five clean, zero contact steps), training was relaunched
from that checkpoint with the already committed contact-aware profile. The
165,018,603-byte replay buffer was restored at its full 250,000-transition
capacity (`fresh_replay=false`); the protected Champion and evaluation gates
were left untouched. The resumed process acquired port 8765 and created a new
visual replay run `3b613adbff564f53ac9d95e8ab6bd9b1`.

The first resumed actor diagnostics confirm that the profile's larger limit is
active: for a proposed action drift of 0.02247, accepted drift is now 0.002495,
versus 0.001248 before continuation. Cumulative reference drift restarted at
0.002495 from the restored checkpoint. At the latest check, training had
advanced to step 3,632,881 at about 35.2 steps/s. This is an implementation
check of the new actor update bound, not yet evidence of a faster evaluated
lap. The previous 33.324 s Champion remains protected.


The first deterministic five-lap evaluation after applying the larger actor
update limit finished 5/5 with zero barrier contacts at 33.545 s (step
3,632,896; saved at 17:41:34 UTC). This is 0.032 s faster than the resumed
checkpoint's 33.577 s evaluation, but 0.221 s slower than the protected
33.324 s Champion. It is an early clean improvement over the resume point,
not a Champion promotion. The saved checkpoint records the active limits as
0.0025 per update and 0.075 cumulative reference drift. At step 3,633,204,
proposed actor drift was 0.02284 and accepted drift 0.002496, showing the
larger per-update bound is active and still binding. Training and new visual
replay capture remain active on port 8765.


The resumed actor then approached the 0.075 cumulative reference-drift cap:
at step 3,635,143 it proposed 0.02270 action drift but accepted only
0.0000648, with cumulative drift 0.0749969 of 0.075. To let the actor continue
exploring alternate actions and lines, the contact-aware profile now raises
only the cumulative allowance to 0.5. The per-update limit stays at 0.0025,
and five-lap clean-evaluation and protected-Champion promotion gates remain
unchanged. Training will resume from the saved 33.545 s clean checkpoint, which
includes the existing full replay buffer; the intervening visual replay files
remain saved in the prior run's replay folder.


With cumulative actor drift widened to 0.5, the first clean five-lap evaluation
was 33.663 s at step 3,634,032. A later clean five-lap evaluation improved to
33.616 s at step 3,636,288, a 0.047 s gain; both remained slower than the
protected 33.324 s Champion. The latter was saved as `latest` at 17:49:06 UTC.
At step 3,636,700, actor proposed drift was 0.02586 and executed drift
0.002496, so the per-update cap remains the active constraint. Cumulative
drift was 0.01486 after the evaluation, below the 0.5 allowance. The new replay
run `7fd2814c78e049589f4d65f1debe9818` had five indexed entries, including
the training episodes and evaluation boundary. Training remained connected on
port 8765; no Champion promotion occurred.


At step 3,638,750, the actor proposed 0.03965 action drift while the 0.0025
per-update cap admitted 0.002494; the latest five-lap evaluation at step
3,638,528 was clean at 33.628 s, slower than the preceding 33.616 s result.
Since the per-update cap is still strongly binding, the profile now doubles
that cap to 0.005 for the next continuation. The cumulative allowance remains
0.5. This is a bounded search increase; clean five-lap evaluation, contact
screening, and Champion rollback/preservation gates are unchanged. The next
continuation will resume from the saved 33.628 s checkpoint and its replay
buffer; the active process is not changed by this profile edit alone.


The profile change was activated from the clean latest checkpoint at step
3,638,528 (33.628 s, 5/5, zero barrier contacts); resume restored all 250,000
replay transitions. The new trainer reconnected on port 8765 and started visual
replay run `b65f8d3033ef43d9a20fb1ebeeedf6d3`. At step 3,639,653, proposed
actor drift was 0.03162 and executed drift 0.004994, confirming the 0.005
per-update cap is active. The first deterministic five-lap evaluation under
this cap finished 5/5 with zero contacts at 33.609 s (step 3,639,664), 0.019 s
faster than its resume checkpoint but 0.064 s slower than the previous
33.545 s clean candidate and 0.285 s slower than Champion. No promotion was
made; replay recording and training remain active.


The wider per-update search produced a 33.599 s five-lap evaluation at step
3,646,432, but with 10 barrier-contact steps (two per lap) at progress ratios
0.99162 and 0.99233 and a maximum impulse of 449.2. The contact gate rejected
the candidate and restored the actor from protected Champion; critic replay
was preserved and a 1,000-update critic cooldown began. The following two
training episodes were each 33.324 s, consistent with the restored Champion
policy rather than a faster candidate. At step 3,649,657 the actor remained
locked with 136 recovery updates remaining. The current visual replay run
contains the driving episodes and evaluation boundary. Training remains live;
no Champion files were overwritten.


After the contact rollback and critic recovery, the actor unlocked from the
protected Champion and resumed updates. Its clean five-lap evaluation at step
3,655,392 improved from 33.630 s to 33.401 s (0.229 s faster), with five of
five finishes and zero barrier contacts. This is now the best clean candidate
in the current continuation, 0.077 s slower than the protected 33.324 s
Champion, so it was saved as `latest` and the Champion was not replaced. The
weak-evaluation counter reset from two; the active replay run had 23 indexed
episodes at the latest check. Training remains active on port 8765.


A second contact regression occurred during the 0.005 per-update search. At
step 3,664,384 the five-lap candidate finished in 33.691 s but had 10
barrier-contact steps, repeated at progress ratios 0.93210 and 0.93273, with
maximum impulse 567.5. The candidate was rejected, Champion's actor was
restored, and a second 1,000-update critic recovery began with replay
preserved. This contact location differs from the earlier rollback's
0.99162-0.99233 progress region. The best clean candidate remains 33.401 s;
Champion remains 33.324 s. Training and visual replay capture continue.


After the second contact rollback, actor recovery completed and training
returned to the protected Champion. The first clean five-lap candidate after
that recovery was 33.736 s at step 3,668,880. Continued actor updates then
improved the clean candidate to 33.615 s at step 3,671,136 (5/5 finishes,
zero contacts), a 0.121 s gain. This is still 0.291 s slower than Champion
and did not replace it. `latest` saved the evaluation at 18:29:02 UTC. At step
3,672,333 the actor was unlocked with 51 Adam steps, accepted action drift
0.004998 under the 0.005 cap, and cumulative drift 0.0943 of 0.5. The active
visual replay run had 42 indexed entries and continued recording.

## 2026-10-14 live pace and replay checkpoint

At 19:34 local, the trainer remained active as PID 17144 on WebSocket port
8765. Its latest saved `latest` checkpoint is step 3,673,392 with a clean
five-lap evaluation of 33.525 s (5/5 finishes, zero barrier contacts). This
improves 0.090 s over the preceding 33.615 s evaluation, but remains 0.201 s
slower than the protected Champion at 33.324 s (step 3,371,072); no promotion
was made. The checkpoint metadata still records the historical best training
lap of 33.105 s, which is not a verified evaluation. Training advanced beyond
the checkpoint: a subsequent episode finished in 33.517 s at step 3,674,510,
and the active process continued afterward.

Visual replay capture is active in
`models/simple-track/grtqc/visual_replays/b65f8d3033ef43d9a20fb1ebeeedf6d3`.
Its index contains 47 episodes through episode ID 48; the newest entry is a
partial/interrupted capture at step 3,675,648 while training itself remains
live. C: had 30.1 GB free and D: had 144.3 GB free, so no space cleanup was
needed at this checkpoint.

## 2026-10-14 recovered 33.105 s action trace

A search of the D: replay archive located the exact historical 33.105 s training
lap: run `14d4064ea20449b8928e5bd65d27239c`, episode 000234, at training step
2,616,954. Its NPZ contains 1,104 decision-level observations (121 features)
and two-component actions, in addition to the lower-rate visual telemetry.
This confirms the fast training lap has a usable recorded action trace. The
active GRTQC learner does not automatically train from visual replay NPZ files;
it uses its separate 250,000-transition RL replay buffer. The archived lap is
therefore available for a future offline imitation experiment but has not been
fed into the running learner.

Comparing that trace with the current run's episode 000050 (33.510 s, step
3,677,885) shows a 0.405 s total gap. At ten-percent progress splits, the
current lap loses about 0.12 s in the 40-50% section and 0.105 s in the final
10%, with smaller losses across several other sections. The recorded
normalized mean longitudinal action is 0.674 on the fast trace versus 0.643 on
the current trace; median local speed is 62.24 versus 60.75 m/s. These are
trace comparisons, not proof that any single control difference caused the
time gap. The demonstration supplies a concrete, data-derived reference for
future pace work without encoding a track-specific action script.

The live actor screen at step 3,675,648 was 33.553 s (5/5, zero contacts),
slower than the 33.525 s latest checkpoint, so it was rejected. Subsequent
training episodes finished in 33.587 s at step 3,676,768 and 33.510 s at step
3,677,885. At that step, the actor was unlocked, the critic recovery cooldown
was zero, and the RL replay buffer remained full. Champion remains 33.324 s.
The active visual replay index continued recording through episode ID 50.
C: had about 30.1 GB free and D: about 144.3 GB; no disk cleanup was needed.

## 2026-10-14 fast-checkpoint policy comparison

The saved actor checkpoint at step 2,616,006 was loaded on CPU and asked for
deterministic actions on all 1,104 observations from the 33.105 s replay. Its
mean absolute action error against the recorded controls was 0.0009 steering
and 0.0011 longitudinal, with per-axis correlation 1.0. The recorded lap is
therefore closely reproduced by that checkpoint's actor on its recorded
states; the pace is not explained only by a lucky random action sequence.

On the same observations, Champion's deterministic actor differs by 0.0017
steering and 0.0098 longitudinal MAE. The current `latest` actor differs by
0.0145 and 0.1311, respectively; its mean longitudinal output is 0.543 on
these states versus 0.674 for the fast trace. This supports actor drift away
from the faster trace as a reason current weights do not repeat it. These are
open-loop same-state action comparisons, not a live lap test, so step 2,616,006
must pass a deterministic five-lap simulator evaluation before promotion.
Champion remains protected at 33.324 s.

Meanwhile, the active run's five-lap clean evaluation improved from 33.502 s
at step 3,677,888 to 33.360 s at step 3,680,128 (5/5 finishes, zero contacts).
This is 0.036 s slower than Champion. Training and replay recording remained
active; the current replay index reached 54 entries through episode ID 55.
The fast historical checkpoint is preserved at
`models/simple-track/grtqc/checkpoints/step-2616006/`; it was not loaded into
the live trainer because that process currently owns the simulator bridge on
port 8765. A separate verified evaluation remains necessary before changing
which actor is trained or promoted.

## 2026-10-14 separate evaluation of the 33.105 s actor

To verify the historical fast lap without interrupting the live trainer, a
separate copy of checkpoint `step-2616006` was evaluated on WebSocket port
8766 while the trainer retained port 8765. The candidate completed all five
deterministic laps cleanly, with zero barrier contacts, but its best and median
were both 33.755 s. That is 0.431 s slower than the protected 33.324 s
Champion, so the checkpoint was not promoted. The five evaluation replays are
preserved under
`models/fast-checkpoint-evaluation-20261014/simple-track/grtqc/visual_replays/`.

At 19:55 local, the active trainer was still PID 17144 on port 8765. Its latest
logged screen was at step 3,695,840: one clean finish at 33.617 s, with zero
barrier contacts. The latest training episode finished in 33.663 s at step
3,695,830. The 33.105 s replay remains a genuine recorded training lap, but
its actor did not reproduce that time in a controlled five-lap run. Together
with the close open-loop action match, this points to state-distribution
sensitivity: small divergences from the demonstrated trajectory can change
the closed-loop lap outcome. It does not establish which section or dynamics
caused the divergence. The active visual replay index had 56 entries through
episode ID 57; 96 replay directories were present overall. C: had 29.8 GB free
and D: 144.3 GB. Training and replay capture remain active; Champion is
unchanged.

At 20:00 local the run produced a new single-episode actor screening result of
33.484 s at step 3,698,096 (finished cleanly, no barrier contacts). This is
0.133 s faster than the previous 33.617 s latest five-lap screen and 0.160 s
faster than Champion, but the one-episode screen is not yet a five-lap
verified promotion result. The process remained live on port 8765; episode 53
had finished at 33.615 s. The active replay index contained 74 entries, with
new recordings still being written. Champion remains 33.324 s pending the next
full deterministic evaluation.

## 2026-10-14 actor exploration check

At 19:59 local, training remained live as PID 17144 on port 8765 and had
advanced to step 3,698,085 (episode 53 in progress). The latest finished
training lap was 33.783 s; the latest five-lap screen remained 33.617 s, clean
5/5 with zero barrier contacts. The current replay index had 76 entries,
including finished training attempts and an interrupted short entry as the
next evaluation began. Replay capture is still active. C: had 29.8 GB free and
D: 144.3 GB.

The live profile keeps exploration low (`exploration_std=0.0025`) and delays
actor updates (`actor_update_interval=16`). Each actor step is also limited to
0.005 action drift, with a 0.5 cumulative reference-drift cap. The recent
progress log showed 63 actor Adam updates and entropy coefficient about
0.00167. These values give a plausible explanation for the actor's reluctance
to explore a substantially different line, but do not prove that exploration
is the only cause of the plateau. The saved 33.105 s replay is still not a
verified reproducible policy lap. I left the loaded profile and live process
unchanged; these settings can only affect a future trainer launch, and changing
them during this run would require a restart. Champion remains 33.324 s.

The saved `latest` metadata was subsequently written at step 3,698,096 and
records a five-episode clean evaluation at 33.484 s (5/5 finishes, zero
barrier contacts). This confirms the pace beyond the preliminary one-lap
screen, although it remains 0.160 s slower than the protected Champion. At
20:01 local, PID 17144 was still training on port 8765 and had advanced to
step 3,698,501, episode 54. The active replay index still held 74 entries
while that next episode was being recorded; C: had 29.8 GB free and D: had
144.3 GB. No checkpoint was promoted.

## 2026-10-14 continued training after 33.484 evaluation

The learner continued from step 3,698,096 without changing the protected
Champion. At 20:03 local, PID 17144 still owned WebSocket port 8765 and had
reached step 3,700,339 (episode 55). Episode 54 finished at 33.602 s and
episode 55 at 33.640 s. The latest single-episode screen was 33.692 s, clean
with no barrier contact; it is slower than the saved 33.484 s five-lap
checkpoint, so that weaker actor remains only a training proposal. The replay
index reached 77 entries, including the finished episode 54 and newer capture
entries; visual replay recording continued. The latest protected Champion
remains 33.324 s. Disk headroom remained C: 29.8 GB and D: 144.3 GB.

The GRTQC runner's current scratch policy saves rejected pace proposals as
checkpoints and leaves the candidate training, while Champion promotion
requires a better reliable evaluation. It recovers the actor after five weak
evaluations (this profile's `recovery_weak_evaluations` setting), so a single
33.692 s screening result does not trigger a rollback or risk Champion.

The full five-lap evaluation for that proposal completed at step 3,700,352:
33.692 s median and best, 5/5 finishes, zero barrier contacts. It was rejected
as slower than Champion and preserved at
`models/simple-track/grtqc/checkpoints/step-3700352-rejected/`. The preceding
33.484 s five-lap candidate is also preserved at
`models/simple-track/grtqc/checkpoints/step-3698096-rejected/`; it remains the
best recent candidate, although still 0.160 s behind Champion. The weak-actor
counter is 1 of 5. At 20:05 local, PID 17144 remained live and had resumed
training through step 3,700,741. Replay capture had 77 indexed entries, with
the next training episode in progress; C: remained at 29.8 GB free and D: at
144.3 GB. No cleanup or trainer restart was needed.

## 2026-10-14 episode trace comparison

I compared the archived 33.105 s lap (run
`14d4064ea20449b8928e5bd65d27239c`, episode 000234) with current training
episode 000081 (33.508 s, step 3,702,599). Both contain per-decision HUD
telemetry and 1,100+ observations. Interpolating elapsed time at each 10% of
track progress gives a 0.403 s total gap: the new lap was 0.105 s slower from
40-50%, 0.098 s slower from 50-60%, and 0.179 s slower over the final 10%; it
was slightly quicker in a few earlier sections. Median speed in those three
larger-loss sections was 2.81, 3.34, and 4.83 m/s lower, respectively. The
recorded lateral offsets also differed by section, but this comparison cannot
establish that line choice alone caused the pace loss. It identifies the
middle and final sections as useful places to examine in future replay/actor
comparisons, without encoding a fixed racing line or track-specific actions.

At 20:06 local, PID 17144 remained live on port 8765 at step 3,702,599. The
33.692 s five-lap proposal at step 3,700,352 remains the latest rejected
candidate and the weak-evaluation counter is 1/5; the faster 33.484 s rejected
candidate at step 3,698,096 remains preserved. The replay index had 80 entries
and episode 000081 was saved as a finished 33.508 s run. C: had 29.7 GB free,
D: 144.3 GB. Champion remains 33.324 s.

## 2026-10-14 pace returned close to Champion

At step 3,702,608 the actor screening lap was 33.354 s and the subsequent
five-lap evaluation matched it: 5/5 finishes, zero barrier contacts. The
candidate was rejected by the Champion gate because 33.354 s is 0.030 s
slower than the protected 33.324 s Champion. Its checkpoint and evaluation
are preserved at `models/simple-track/grtqc/checkpoints/step-3702608-rejected/`.
The previous weaker-evaluation count was cleared after this clean pace gain.

The next training lap, episode 58 at step 3,703,724, finished in 33.454 s and
was saved as visual replay `episode-000083.npz`. Relative to the earlier
33.508 s episode 57, this gained 0.054 s, mainly in the 50-60% section
(0.083 s) and final 10% (0.040 s). Relative to the historical 33.105 s lap,
the new lap still loses most time in 40-50% (+0.090 s), 70-80% (+0.059 s),
and the final 10% (+0.139 s). The trace shows continuing pace improvement in
some of the previously slow sections, but those comparisons are observational
and do not isolate a control change as the cause.

At 20:10 local, PID 17144 was still training on port 8765 and had reached step
3,704,709. The replay index had 81 entries, and visual replay recording
continued. C: had 29.7 GB free and D: 144.3 GB. The latest verified actor is
0.030 s off Champion; no promotion or rollback occurred.

## 2026-10-14 barrier-contact recovery

At step 3,704,848, the next actor screening lap was 33.672 s with one barrier
contact step. The full deterministic evaluation repeated that contact on all
five laps (5/5 finishes, 5 total contact steps), so the contact-regression
gate rejected the proposal and restored the protected Champion actor. The
rejected actor is preserved at
`models/simple-track/grtqc/checkpoints/step-3704848-rejected/`. Recovery kept
the critic replay buffer and began the configured 1,000-update critic cooldown;
the trainer itself stayed live on port 8765.

Training resumed immediately from the restored actor. Its first post-recovery
training episode (episode 60, step 3,705,959) finished in 33.324 s with a clean
finish, and the visual replay is saved as `episode-000086.npz`. This is a
single training lap, not a new five-lap verification. At 20:13 local, the
trainer had advanced to step 3,706,443 and the replay index contained 84
entries. C: had 29.7 GB free and D: had 144.3 GB. Champion remains 33.324 s.

The next post-recovery actor screen at step 3,709,296 matched Champion at
33.324 s with zero contacts. Its full five-lap evaluation also matched
33.324 s (5/5 finishes, zero contacts); it was retained as a rejected
checkpoint because it did not strictly beat Champion. After further actor
updates, the next candidate screened at 33.482 s and passed another full
5/5 clean evaluation at step 3,711,552. This candidate is preserved at
`models/simple-track/grtqc/checkpoints/step-3711552-rejected/`, 0.158 s slower
than Champion; it raised the weak-evaluation count to 2/5. Training continued
past step 3,712,161, and the replay index reached 91 entries. C: remained at
29.7 GB free and D: at 144.3 GB. The recovery restored clean Champion pace,
but the post-recovery actor has not yet found a faster verified lap.

After more updates, the actor's next screen was 33.957 s at step 3,713,792.
Its five-lap check confirmed 33.957 s (5/5 finishes, zero barrier contacts),
and the proposal was saved at
`models/simple-track/grtqc/checkpoints/step-3713792-rejected/`. This is
0.633 s slower than Champion and raised the weak-evaluation counter to 3/5.
The preceding post-recovery training lap (episode 67) was 33.662 s. At 20:24
local, PID 17144 remained live on port 8765 and had resumed training at step
3,713,941. The active visual replay index contained 93 entries, with the
33.662 s episode saved as `episode-000095.npz`. C: had 29.6 GB free and D:
144.3 GB. The actor recovery gate remains armed for two further weak full
evaluations; Champion is still protected at 33.324 s.

The next actor screen at step 3,716,048 was 33.785 s, and the five-lap check
confirmed 33.785 s (5/5 finishes, zero barrier contacts). It improved 0.172 s
over the preceding 33.957 s candidate and reset the weak-evaluation counter,
but remains 0.461 s slower than Champion. The checkpoint is preserved at
`models/simple-track/grtqc/checkpoints/step-3716048-rejected/`. At 20:28 local,
PID 17144 was still live on port 8765 and continued training beyond step
3,716,213. The active visual replay index held 96 entries. C: had 29.6 GB
free and D: 144.3 GB. Champion was not changed.

The following actor screen, at step 3,718,304, was 33.709 s. Its five-lap
evaluation confirmed 33.709 s (5/5 finishes, zero barrier contacts). This
improves 0.076 s over the previous 33.785 s result and is preserved at
`models/simple-track/grtqc/checkpoints/step-3718304-rejected/`; it remains
0.385 s slower than Champion. The clean pace improvement reset the weak-actor
counter. Episode 71 immediately before the evaluation finished in 33.521 s.
At 20:31 local, PID 17144 was live again on port 8765 at step 3,718,357. The
replay index held 99 entries, C: had 29.6 GB free, and D: had 144.3 GB.

The next evaluation at step 3,720,560 was 33.588 s and verified identically
over five clean finishes (zero barrier contacts). This improved 0.121 s over
the previous 33.709 s candidate; it remains 0.264 s slower than the protected
33.324 s Champion. The checkpoint is saved at
`models/simple-track/grtqc/checkpoints/step-3720560-rejected/`, and the clean
pace gain reset the weak-actor counter. Episode 73 before evaluation finished
in 33.700 s. At 20:35 local, training was still active as PID 17144 on port
8765 at step 3,721,013. The replay index had 102 entries. C: had 29.4 GB free
and D: had 144.3 GB; Champion remains unchanged.

The next proposal at step 3,725,104 screened at 33.797 s with a barrier
contact. Its full evaluation repeated one contact on each of five laps (5/5
finishes, five total contact steps); every contact occurred at 35.76% track
progress, with peak impulse 126.58. The contact-regression gate rejected it
and restored the Champion actor, preserving the critic replay and starting the
configured 1,000-update cooldown. The rejected checkpoint is preserved at
`models/simple-track/grtqc/checkpoints/step-3725104-rejected/`. At 20:42 local,
PID 17144 resumed training on port 8765 and had advanced to step 3,725,957.
The replay index held 107 entries. C: had 29.3 GB free and D: 144.3 GB.

The next actor candidate screened at 33.631 s at step 3,722,816 and passed a
five-lap clean evaluation (5/5 finishes, zero barrier contacts). It was
preserved at `models/simple-track/grtqc/checkpoints/step-3722816-rejected/`;
it is 0.043 s slower than the preceding 33.588 s result and 0.307 s slower
than Champion, raising the weak-evaluation counter to 1/5. The next training
lap, episode 76 at step 3,723,950, was 34.006 s and saved to the visual replay
index. At 20:39 local, PID 17144 remained live on port 8765 at step 3,724,501,
with 106 indexed replay entries. C: had 29.3 GB free and D: 144.3 GB.

## 2026-10-14 20:47 local â€” provenance of the 33.105 s training best

The resumed run's first progress record already reported
`best_training_lap_s=33.105` at step 3,638,529. Its startup record shows it
resumed `models/simple-track/grtqc/latest` from step 3,638,528, so that number
was inherited from an earlier run; this log contains no episode event at
33.105 s. The active visual replay index has 112 entries and no replay with
that lap time. The number therefore does not identify a currently loadable,
deterministically verified policy. Champion remains 33.324 s from a clean
five-lap evaluation.

After the step-3,725,104 contact rollback, episodes 78, 79, and 80 each
finished in 33.324 s, and episode 81 finished in 33.270 s. The actor screen at
step 3,729,552 was 33.657 s with no barrier contact; a full five-lap
verification for this candidate had not yet appeared in the log at this
snapshot. No candidate has replaced Champion. At 20:47 local, training PID
17144 remained live and owned the WebSocket listener on port 8765. The replay
index held 112 entries. C: had 29.3 GB free and D: had 144.3 GB.

### Verification follow-up, 20:48 local

The five-lap evaluation for step 3,729,552 completed at 33.657 s (5/5
finishes, zero barrier contacts). The candidate was rejected as slower than
Champion and preserved at
`models/simple-track/grtqc/checkpoints/step-3729552-rejected/`. This confirms
that the stored 33.105 s training best is not reproduced by this candidate.
Episode 82 finished at 33.612 s and is included in the visual replay index,
which now has 113 entries. PID 17144 remains live on port 8765.

### Continued training, 20:51 local

The next candidate, screened at step 3,731,808, completed five clean laps at
33.660 s and was rejected as 0.336 s slower than Champion. Its checkpoint is
preserved at `models/simple-track/grtqc/checkpoints/step-3731808-rejected/`.
Training continued without restarting: at 20:51, PID 17144 was on port 8765
at step 3,732,641, episode 84 (73.0% progress). The visual replay index held
115 entries, and C: had 29.3 GB free.

### Clean pace gain, 20:54 local

The next actor evaluation at step 3,734,064 verified 33.499 s over five laps,
all finished with zero barrier contacts. This is 0.161 s faster than the
33.660 s candidate and 0.175 s slower than Champion; it was kept as a pace
proposal rather than promoted over Champion. The runner recorded
`actor_pace_improved`, clearing the weaker-candidate streak. Training resumed
on the same PID and port; at 20:54 it had reached step 3,734,405, episode 86
(26.3% progress). The visual replay index had 118 entries and C: had 29.2 GB
free.


### Contact rollback and replay telemetry, 20:58-20:59 local

The next candidate at step 3,736,320 screened at 33.969 s with four contact
steps, then completed five laps at the same median with 20 contact steps
(four per lap). All occurred between 63.21% and 63.64% of the route; peak
impulse was 1,088.83. The candidate was rejected and preserved at
`models/simple-track/grtqc/checkpoints/step-3736320-rejected/`. The recovery
restored Champion, kept critic replay, and started the 1,000-update critic
cooldown. Training remained live on PID 17144.

Visual replay comparison around this section shows the candidate did try a
different path: the normalized lateral-offset observation averaged +0.495 in
three post-recovery Champion laps and about -0.10 in the two preceding
candidate laps. Candidate steering reached at least 0.98 absolute magnitude
on 26-31% of the sampled decisions there, with mean throttle 0.966-0.974.
This is evidence of a line shift associated with the contacts, not proof that
the line shift alone caused them. At 20:59:53 local, the same trainer had
resumed at step 3,739,181 in episode 90 (55.2% progress); port 8765 was owned
by PID 17144. The visual replay index had 123 entries and C: had 29.2 GB free.

### First candidate after contact recovery, 21:03 local

Episodes 88-90 after recovery reproduced Champion's 33.324 s training pace.
The first post-cooldown five-lap candidate at step 3,740,784 was 33.744 s,
with 5/5 finishes and zero barrier contacts. It was preserved at
`models/simple-track/grtqc/checkpoints/step-3740784-rejected/` and counted as
the first weaker evaluation since recovery. At 21:03, PID 17144 still owned
port 8765 and had reached step 3,741,633, episode 92 (74.4% progress). The
visual replay index had 126 entries and C: had 29.2 GB free.

### Second clean candidate after contact recovery, 21:06 local

The next five-lap evaluation at step 3,743,040 completed in 33.785 s, 5/5
finishes with zero barrier contacts. It was rejected as 0.461 s slower than
Champion and counted as the second weaker evaluation since recovery. The
candidate is preserved at
`models/simple-track/grtqc/checkpoints/step-3743040-rejected/`. Training
continued on PID 17144; at 21:06 it had reached step 3,743,365 in episode 94
(24.7% progress), with the simulator listener reacquired on port 8765. The
visual replay index had 129 entries and C: had 29.2 GB free.

### Second contact recovery cycle, 21:11 local

The next candidate at step 3,745,312 screened at 33.838 s with one contact at
99.23% progress. Its five-lap evaluation confirmed 5/5 finishes but one
contact per lap (five total), all at the same progress, with peak impulse
243.02. The checkpoint was rejected and kept at
`models/simple-track/grtqc/checkpoints/step-3745312-rejected/`. Contact
regression recovery again restored Champion, preserved critic replay, and
started a 1,000-update cooldown. At 21:11, PID 17144 still owned port 8765 and
had advanced to step 3,747,677, episode 98 (6.9% progress). The visual replay
index held 134 entries and C: had 29.1 GB free.

### Post-recovery Champion match, 21:15 local

After the second contact recovery, episodes 97 and 98 returned to 33.324 s,
and episode 99 completed in 33.191 s. The deterministic five-lap check at
step 3,749,776 then matched Champion at 33.324 s (5/5 finishes, zero
contacts). It was not promoted because it did not beat Champion; this is the
first weaker check since the recovery. At 21:15, PID 17144 had resumed on port
8765 at step 3,750,277, episode 100 (41.4% progress). The visual replay index
held 137 entries and C: had 29.1 GB free.

### Next clean but slower actor check, 21:18 local

At step 3,752,032 the five-lap candidate evaluated at 33.724 s (5/5 finishes,
zero barrier contacts). The clean anchor refreshed, but the candidate remained
slower than the 33.324 s Champion and was rejected; this is the second weaker
evaluation since the latest recovery (2/5 toward automatic actor recovery).
Training continued on PID 17144 at step 3,752,349, episode 102 (24.6%
progress), with port 8765 live. The visual replay index had 140 entries and
C: had 29.1 GB free.

### Third clean slower check since recovery, 21:21 local

The five-lap actor evaluation at step 3,754,288 completed at 33.747 s with
5/5 finishes and zero contacts. It was rejected as slower than Champion, and
the weaker-check count reached 3/5. The trainer remains active on PID 17144;
at 21:21 it had resumed collection on port 8765 at step 3,754,321, episode
104. The visual replay index held 143 entries and C: had 29.1 GB free.

### Clean pace gain, 21:25 local

The five-lap check at step 3,756,560 completed at 33.589 s (5/5 finishes,
zero contacts). This improved 0.158 s over the previous 33.747 s candidate
and reset the weaker-check streak, although it remains 0.265 s slower than
Champion. The clean actor checkpoint was retained at
`models/simple-track/grtqc/checkpoints/step-3756560-rejected/`. Episode 105
immediately before screening had contacted at 63.08-63.33% and finished in
34.098 s; the deterministic five-lap candidate remained contact-free. At
21:25, PID 17144 had resumed on port 8765 at step 3,756,869, episode 106
(23.3% progress). The visual replay index had 146 entries and C: had 29.0 GB
free.

### Repeated finish-approach contact rollback, 21:28 local

The actor screen at step 3,758,832 was 33.751 s with one contact at 98.95% of
the route. Five-lap verification repeated that same contact once per lap
(5/5 finishes, five contact steps; peak impulse 77.19). The candidate was
preserved at `models/simple-track/grtqc/checkpoints/step-3758832-rejected/`;
the contact gate restored Champion and preserved critic replay with another
1,000-update cooldown. At 21:28, PID 17144 had resumed on port 8765 at step
3,758,945, episode 108. The visual replay index held 149 entries and C: had
29.0 GB free.

### Later clean but slower candidates, 21:33-21:38 local

The five-lap candidate at step 3,763,296 evaluated at 33.664 s (5/5 finishes,
zero contacts); it was rejected because it did not beat the 33.324 s Champion.
The next candidate at step 3,765,552 evaluated at 33.863 s, also 5/5 clean,
and was retained at `models/simple-track/grtqc/checkpoints/step-3765552-rejected/`.
Neither candidate replaced Champion. By 21:38, PID 17144 was still training
episode 115 at step 3,766,965 (21.2% progress). The active replay index had
158 entries, and C: had 29.0 GB free (D: 144.3 GB).

### Clarification: historical 33.105 s lap provenance

The 33.105 s lap is a genuine saved training replay, not a fabricated or
timer-only statistic: it is run `14d4064ea20449b8928e5bd65d27239c`, episode
000234 at step 2,616,954. The associated step-2,616,006 actor closely matches
the recorded actions when evaluated against the lap's saved observations.
However, a separate deterministic five-lap rollout of that checkpoint
finished at 33.755 s (5/5 clean), slower than the protected 33.324 s Champion.
This shows the lap is not reproducible as a closed-loop policy result from
that checkpoint under the controlled evaluation; a close action match on the
recorded state sequence does not guarantee the car will remain on that same
state sequence when driven again. The active run resumed from a later
`latest` checkpoint at step 3,638,528, whose inherited `best_training_lap_s`
field still says 33.105 s. The one-tick timer offset accounts for only about
0.001 s, so it cannot explain the larger rollout difference. An earlier
status reply incorrectly said no replay was tied to the historical lap; the
archived replay and separate checkpoint evaluation above are the correct
evidence.

### Cleaner pace gain, 21:41 local

The candidate at step 3,767,808 completed five deterministic laps at 33.450 s,
with 5/5 finishes and zero barrier contacts. This is 0.413 s faster than the
prior 33.863 s candidate, but 0.126 s slower than Champion, so it was rejected
and preserved at `models/simple-track/grtqc/checkpoints/step-3767808-rejected/`.
The preceding free-driving episodes finished at 33.774 s with a finish-line
contact and 33.853 s with a contact around 63.4% of the route; those contacts
did not recur in the deterministic candidate check. Training stayed active on
PID 17144, port 8765; at 21:42 it was at step 3,768,652 in episode 116 (73.4%
progress). The active visual replay index held 160 entries. C: had 28.9 GB free
and D: had 144.3 GB.

### Repeated mid-route contact rollback, 21:45 local

The next actor screen at step 3,770,080 was 34.057 s with three contacts. Its
deterministic five-lap evaluation confirmed 5/5 finishes but 15 contact steps:
the same three contacts repeated on every lap around 62.96%-63.14% progress.
Peak collision impulse was 1,427.94. The contact gate rejected the candidate
and automatically restored Champion, preserving critic replay. At 21:45,
training remained live on PID 17144 with port 8765 owned by the trainer; it had
resumed at step 3,771,219 in episode 119 after recovery, with 716 critic
cooldown updates remaining. The active visual replay index held 164 entries.
C: had 28.9 GB free and D: had 144.3 GB.

### First post-recovery clean screen, 21:49 local

The actor's first deterministic five-lap check after the 3,770,080 contact
rollback matched Champion at 33.324 s (5/5 finishes, zero contacts). It was
not promoted because it did not beat Champion. The checkpoint is preserved at
`models/simple-track/grtqc/checkpoints/step-3774544-rejected/`. At 21:50,
training was active on PID 17144 and port 8765 at step 3,775,477 in episode
122 (82.0% progress). The active visual replay index held 168 entries. C: had
28.8 GB free and D: had 144.3 GB.

### Clean slower check after recovery, 21:52 local

The next deterministic five-lap evaluation, at step 3,776,800, was 33.676 s
(5/5 finishes, zero contacts). It was rejected as slower than the 33.324 s
Champion; the saved `latest` metadata now records this evaluation. The actor
recovery cooldown had completed, and at 21:53 training remained live on
PID 17144, port 8765, at step 3,777,077 in episode 124 (20.0% progress). The
active visual replay index held 171 entries. C: had 28.8 GB free and D: had
144.3 GB.

### Repeat contact at 63.5%, 21:56 local

The actor screen at step 3,779,056 was 33.990 s with one contact. Five-lap
verification confirmed 5/5 finishes but one barrier contact at 63.52% on every
lap (five total; peak impulse 188.52). The candidate was rejected and preserved
at `models/simple-track/grtqc/checkpoints/step-3779056-rejected/`; automatic
recovery restored Champion and preserved critic replay. At 21:56, PID 17144
still owned port 8765 and had resumed collection at step 3,779,153 in episode
126 (3.8% progress), with 976 critic cooldown updates remaining. The active
replay index held 174 entries. C: had 28.8 GB free and D: had 144.3 GB.

### Clean but slower post-recovery candidate, 22:01 local

The next five-lap candidate at step 3,783,520 evaluated at 33.762 s, with
5/5 finishes and zero contacts. It was rejected as slower than the 33.324 s
Champion; no rollback was needed. The preceding free-driving laps were 33.324,
33.192, and 33.762 s. At 22:01, training remained active on PID 17144 and
port 8765 at step 3,783,683 in episode 130 (9.7% progress), with the critic
recovery cooldown complete. The active visual replay index held 179 entries.
C: had 28.7 GB free and D: had 144.3 GB.

### Finish-line contact rollback, 22:04 local

The actor screen at step 3,785,776 was 33.787 s with one contact at 99.30%.
Five-lap verification confirmed 5/5 finishes but repeated that finish-approach
contact on every lap (five contacts; peak impulse 1,062.84). The candidate was
rejected and saved at `models/simple-track/grtqc/checkpoints/step-3785776-rejected/`;
automatic recovery restored Champion and preserved critic replay. At 22:05,
PID 17144 still owned port 8765 and had resumed at step 3,787,041 in episode
133 (8.9% progress), with 684 critic cooldown updates remaining. The active
visual replay index held 183 entries. C: had 28.7 GB free and D: had 144.3 GB.

### Repeated contact at 93.4%, 22:09 local

The actor screen at step 3,790,240 was 33.853 s with one contact at 93.40%.
The deterministic five-lap check confirmed 5/5 finishes but repeated that
contact once on every lap (five contacts; peak impulse 297.71). The candidate
was rejected at `models/simple-track/grtqc/checkpoints/step-3790240-rejected/`;
automatic recovery restored Champion and retained critic replay. At 22:10,
PID 17144 was still live and collecting episode 137 at step 3,791,541 (11.1%
progress), with 675 critic cooldown updates remaining. The visual replay index
held 188 entries. C: had 28.7 GB free and D: had 144.3 GB.

### Post-recovery clean Champion match, 22:13 local

The five-lap candidate at step 3,794,720 matched the protected Champion at
33.324 s (5/5 finishes, zero contacts). It was rejected because it did not
improve Champion; the checkpoint is preserved at
`models/simple-track/grtqc/checkpoints/step-3794720-rejected/`. At 22:14,
training remained active on PID 17144 and port 8765, at step 3,794,773 in
episode 140 (1.2% progress); the recovery cooldown had completed. The active
visual replay index held 192 entries. C: had 28.7 GB free and D: had 144.3 GB.

### Clean but slower candidate, 22:17 local

The deterministic five-lap check at step 3,796,976 was 33.667 s (5/5
finishes, zero contacts), so it was rejected as slower than the 33.324 s
Champion. No recovery was needed. At 22:18, training was live on PID 17144 and
port 8765 at step 3,797,973 in episode 142 (88.4% progress), with cooldown
complete. The active visual replay index held 195 entries. C: had 28.6 GB free
and D: had 144.3 GB.

### Another clean but slower candidate, 22:20 local

The deterministic five-lap check at step 3,799,232 evaluated at 33.721 s
(5/5 finishes, zero contacts). It was rejected as slower than Champion and
preserved at `models/simple-track/grtqc/checkpoints/step-3799232-rejected/`.
At 22:21, PID 17144 still owned port 8765 and had resumed training at step
3,800,100 in episode 144 (76.0% progress). The active visual replay index held
198 entries. C: had 28.6 GB free and D: had 144.3 GB.

### Repeated finish-approach contact, 22:24 local

The actor screen at step 3,801,488 was 33.808 s with three finish-approach
contact steps. Five-lap verification repeated those contacts on every lap
(15 total, at 99.02%-99.37% progress; peak impulse 283.91). The candidate was
rejected and preserved at `models/simple-track/grtqc/checkpoints/step-3801488-rejected/`;
automatic recovery restored Champion while retaining critic replay. At 22:24,
PID 17144 remained active at step 3,801,521 in episode 146 (0.5% progress),
with 992 critic cooldown updates remaining and port 8765 owned by the trainer.
The active visual replay index held 201 entries. C: had 28.6 GB free and D:
had 144.3 GB.

### Post-recovery clean Champion match, 22:29 local

At step 3,805,952 the next deterministic five-lap check matched Champion at
33.324 s (5/5 finishes, zero contacts), so it was not promoted. The checkpoint
was preserved at `models/simple-track/grtqc/checkpoints/step-3805952-rejected/`.
At 22:29, PID 17144 still owned port 8765 and had resumed at step 3,806,181 in
episode 150 (15.1% progress), with critic cooldown complete. The active visual
replay index held 206 entries. C: had 28.6 GB free and D: had 144.3 GB.

### Clean slower candidate after recovery, 22:32 local

The five-lap candidate at step 3,808,208 evaluated at 33.593 s (5/5 finishes,
zero contacts). It was rejected as slower than the 33.324 s Champion and saved
at `models/simple-track/grtqc/checkpoints/step-3808208-rejected/`. At 22:33,
training was active on PID 17144 and port 8765 at step 3,809,189 in episode 152
(87.0% progress), with recovery cooldown complete. The active visual replay
index held 209 entries. C: had 28.5 GB free and D: had 144.3 GB.

### Clean pace gain, 22:36 local

The five-lap candidate at step 3,810,464 evaluated at 33.504 s (5/5 finishes,
zero contacts). This improved by 0.089 s over the prior 33.593 s candidate,
but remained 0.180 s slower than Champion and was not promoted. It was saved
at `models/simple-track/grtqc/checkpoints/step-3810464-rejected/`. At 22:36,
PID 17144 still owned port 8765 and was training at step 3,810,804 in episode
154 (26.0% progress). The active replay index held 212 entries. C: had 28.5 GB
free and D: had 144.3 GB.

### Clean slower candidate, 22:39 local

The five-lap evaluation at step 3,812,720 was 33.751 s (5/5 finishes, zero
contacts). It was rejected as slower than both the 33.504 s preceding candidate
and 33.324 s Champion; it was preserved at
`models/simple-track/grtqc/checkpoints/step-3812720-rejected/`. At 22:40,
training continued on PID 17144 and port 8765 at step 3,813,749 in episode 156
(90.9% progress), with cooldown complete. The active visual replay index held
215 entries. C: had 28.5 GB free and D: had 144.3 GB.

### Clean slower candidate, 22:42 local

The five-lap check at step 3,814,960 evaluated at 33.756 s (5/5 finishes,
zero contacts), slower than Champion, so it was rejected and preserved at
`models/simple-track/grtqc/checkpoints/step-3814960-rejected/`. At 22:43,
training remained active on PID 17144 and port 8765 at step 3,816,053 in
episode 158 (96.9% progress), with cooldown complete. The active replay index
held 218 entries. C: had 28.4 GB free and D: had 144.3 GB.

### Two repeated finish-approach contacts, 22:46 local

The actor screen at step 3,817,200 was 33.720 s with two contact steps at
98.38%-98.52% of the route. Five-lap verification confirmed 5/5 finishes but
repeated both contacts every lap (10 total; peak impulse 397.53). The candidate
was rejected and preserved at
`models/simple-track/grtqc/checkpoints/step-3817200-rejected/`; automatic
recovery restored Champion while keeping critic replay. At 22:46, PID 17144
still owned port 8765 and was collecting at step 3,817,505 in episode 160
(23.5% progress), with 924 critic cooldown updates remaining. The active replay
index held 221 entries. C: had 28.4 GB free and D: had 144.3 GB.

### Clean but slower post-recovery check, 22:51 local

The five-lap candidate at step 3,821,648 evaluated at 33.705 s (5/5 finishes,
zero contacts) and was rejected as slower than the 33.324 s Champion. It was
preserved at `models/simple-track/grtqc/checkpoints/step-3821648-rejected/`.
At 22:51, the trainer remained live on PID 17144 and port 8765 at step
3,822,761 in episode 164 (98.8% progress), with recovery cooldown complete.
The active visual replay index held 225 entries. C: had 28.4 GB free and D:
had 144.3 GB.

### Clean but slower candidate, 22:54 local

The five-lap evaluation at step 3,823,904 was 33.697 s (5/5 finishes, zero
contacts). It was rejected as slower than the 33.324 s Champion and saved at
`models/simple-track/grtqc/checkpoints/step-3823904-rejected/`. At 22:54,
training remained active on PID 17144 and port 8765 at step 3,823,944 in
episode 166 (0.7% progress), with recovery cooldown complete. The active replay
index held 228 entries. C: had 28.3 GB free and D: had 144.3 GB.

### Finish-approach contact rollback, 22:58 local

The actor screen at step 3,826,160 was 33.633 s with a contact at 98.95% of
the route. Five-lap verification repeated the contact once on each lap (five
contacts; peak impulse 54.61). The candidate was rejected and saved at
`models/simple-track/grtqc/checkpoints/step-3826160-rejected/`; automatic
recovery restored Champion while preserving critic replay. At 22:58, PID 17144
still owned port 8765 and had resumed at step 3,827,257 in episode 168 (98.4%
progress), with 726 critic cooldown updates remaining. The active replay index
held 231 entries. C: had 28.3 GB free and D: had 144.3 GB.

### Repeated finish-line contact, 23:02 local

The actor screen at step 3,830,624 was 33.653 s with three contacts in the
last 0.6% of the track. Five-lap verification confirmed 5/5 finishes but
repeated all three contacts on each lap (15 total; peak impulse 193.50). The
candidate was rejected at
`models/simple-track/grtqc/checkpoints/step-3830624-rejected/`; recovery
restored Champion while preserving critic replay. At 23:03, PID 17144 still
owned port 8765 and was collecting episode 173 at step 3,831,912 (10.7%
progress), with 679 critic cooldown updates remaining. The visual replay index
held 237 entries. C: had 28.3 GB free and D: had 144.3 GB.

### Post-recovery Champion match, 23:07 local

The five-lap candidate at step 3,835,088 matched Champion at 33.324 s (5/5
finishes, zero contacts). It was not promoted because it did not improve the
protected Champion, and was saved at
`models/simple-track/grtqc/checkpoints/step-3835088-rejected/`. At 23:08,
training remained live on PID 17144 and port 8765 at step 3,836,213 in episode
176 (99.9% progress), with cooldown complete. The active replay index held 241
entries. C: had 28.2 GB free and D: had 144.3 GB.

### Slower full-track evaluation, 23:11 local

At step 3,837,344, the actor screen and five-lap deterministic verification
both measured 33.716 s (5/5 finishes). The evaluation reported zero barrier
contact steps, but its max barrier impulse was 49.14. The candidate was
rejected as slower than the protected 33.324 s Champion. Training resumed on
PID 17144; at 23:11:50 it had reached step 3,838,376 in episode 178. The visual
replay index held 243 episodes. C: had 28.2 GB free and D: had 144.3 GB.

### Repeated finish-line contact, 23:15 local

The actor screen at step 3,839,600 was 33.726 s. Five-lap verification
finished 5/5 at the same median pace, but each lap contacted the barrier at
98.95% progress (five contacts total; peak impulse 81.80). The candidate was
rejected against the 33.324 s Champion. At 23:15:38, PID 17144 had resumed
training at step 3,841,137. The visual replay index held 247 episodes. C: had
28.2 GB free and D: had 144.3 GB.

### Clean but slower verification, 23:19 local

At step 3,844,048, the deterministic five-lap evaluation completed 5/5 at
33.760 s with zero barrier contacts. It was rejected against Champion at
33.324 s. At 23:19:37, PID 17144 was training at step 3,844,457. The visual
replay index held 251 episodes. C: had 28.2 GB free and D: had 144.3 GB.

### Actor action-limit diagnosis and Champion re-match, 23:28 local

At step 3,845,763, learner diagnostics showed proposed action drift 0.01514
but executed drift 0.00500, exactly at the active per-update limit; cumulative
reference drift was 0.09744 against its 0.5 cap. This indicates the per-update
limit, rather than the cumulative reference cap, is clipping actor changes.
The next continuation profile now raises only that per-update limit from
0.005 to 0.01. Actor learning rate (5e-6), cumulative drift cap, critics,
replay, five-lap evaluation, and Champion rollback gates remain unchanged.

At step 3,850,752 the five-lap candidate matched Champion at 33.324 s (5/5,
zero barrier contacts) and was rejected because it was not faster. The latest
checkpoint was saved at this step at 22:27:50 UTC. At 22:28:37 UTC PID 17144
had resumed at step 3,851,881 in episode 190. The visual replay index held 259
episodes. C: had 28.1 GB free and D: had 144.3 GB. The active process still
has the old 0.005 action limit until it is continued from latest with the
updated profile.

### GUI continuation with increased actor movement, 23:32 local

After saving the 3,850,752-step checkpoint, the prior CLI process was stopped
and the GUI continued Latest with the selected actor settings. The GUI log
confirms it restored Latest's policy, critics, replay, rewards, and curriculum,
using actor LR 5e-6, per-update action limit 0.01, and cumulative drift limit
0.5. It started GRTQC on the NVIDIA T500 with a 1,000,000-step budget and full
track phase. At 23:32:19, the run had reached step 3,853,314 at about 25.4 TPS;
a policy check was in progress. The visual replay index held 259 episodes. C:
had 28.1 GB free and D: had 144.3 GB.

### Early evaluations after actor-cap increase, 23:37 local

The GUI-run policy check at step 3,854,144 completed five clean finishes at
33.429 s and was rejected against Champion. The next actor candidate at step
3,856,400 evaluated at 33.656 s (5/5, zero barrier contacts) and was also
rejected. At step 3,856,392, diagnostics showed proposed action drift
0.03352 versus executed 0.00999 at the new 0.01 cap; cumulative drift was
0.3071 of 0.5. At 23:37:31, GUI PID 19940 was still training on port 8765 at
step 3,856,589. The visual replay index held 262 episodes. C: had 28.0 GB
free and D: had 144.3 GB.

### Faster candidate with repeated barrier contact, 23:40 local

At step 3,858,656 the actor screen was 33.375 s. Five-lap verification
finished 5/5 at that pace, but contacted the barrier at 98.52% on every lap
(five contacts; peak impulse 153.97), so the candidate was rejected and
automatic actor recovery restored the verified policy with a 1,000-update
critic cooldown. At 22:40:56 UTC, GUI PID 19940 was still training at step
3,860,167 on port 8765. Replay index: 262 episodes. C: had 28.0 GB free; D:
had 144.3 GB. Critics and replay were preserved through recovery.

### Post-recovery checks and guarded cap adjustment, 00:01 local, 15 Oct

After the 23:40 contact rollback, a clean five-lap check at step 3,863,120
matched Champion at 33.324 s; later clean checks were 33.823 s (step 3,865,360)
and 33.556 s (step 3,867,600). The candidate at step 3,869,840 was 33.674 s
but contacted the barrier at 93.21% on all five laps (peak impulse 579.81),
triggering another 1,000-update critic cooldown. After recovery, step 3,874,288
again matched Champion at 33.324 s and step 3,876,544 was clean at 33.556 s.
At step 3,878,784, the candidate was 33.481 s but repeated two late-track
contacts (99.23% and 99.44%) on every lap (10 contact steps; peak impulse
67.27), so it was rejected and the Champion actor restored.

The repeated contacts after changing the per-update action cap to 0.01 justify
backing it down to 0.0075 for the next continuation. Actor LR (5e-6), cumulative
drift limit (0.5), n-step return (16), critics, replay, five-lap verification,
and Champion rollback gates remain unchanged. At 23:01:36 UTC (00:01:36 local),
GUI PID 19940 was still training at step 3,879,119. Its active visual replay
archive `models/simple-track/grtqc/visual_replays/d1ff01715b3646d6b11b97baa7a19d9c/`
held 31 episodes; prior archives remain intact. C: had 27.8 GB free and D: had
144.3 GB. The GUI clean-stop/resume will apply the adjusted cap from Latest.

## 2026-10-15 00:06 local — resumed with moderated actor updates

The historical 33.105 s lap is a genuine episode-234 replay (step 2,616,954),
but not a reproducible result from its nearby checkpoint: the saved actor's
five-lap deterministic rollout completed cleanly at 33.755 s on all five laps.
Its near-exact open-loop action match only shows that it copied the actions on
the replay's saved states. In closed loop, tiny state deviations can move it off
that trajectory; the evidence does not isolate one specific cause. The current
latest actor is also farther from those replay actions. The protected Champion
remains 33.324 s.

The GUI had been cleanly stopped at step 3,881,843. I loaded
`profiles/training/simple-track-grtqc-contact-aware-30.json` and verified the
Learning settings page showed actor step action limit 0.0075, actor LR 5e-6,
and cumulative reference drift limit 0.5. I resumed **Latest**, not Champion,
with the GUI's “Continue latest with selected actor settings” action. The new
run log `logs/simple-track/grtqc/20261014T230607Z.jsonl` records resume source
`models/simple-track/grtqc/latest`, replay preserved (`fresh_replay=false`),
CUDA/NVIDIA T500, and a 1,000,000-step full-track session. Its first progress
record is step 3,881,844; visual replay capture remains enabled by the WebSocket
training setup for later swarm use. The GUI reports Active · Training and the
simulator bridge is listening on port 8765. Champion and all existing replay
archives were left intact.

The first post-resume episode finished at step 3,882,964 in 33.603 s (full-track
finish). Its visual replay index was created and had two entries by 00:06:48
local, confirming capture has resumed. This is an ordinary training episode,
not a five-lap candidate verification; Champion remains unchanged.

### First resumed deterministic screen

At step 3,882,979, the first post-resume five-lap deterministic screen measured
33.725 s median and best, with all five finishes and zero barrier-contact steps.
It was rejected as slower than Champion. The protected 33.324 s actor was not
overwritten. Training continued past step 3,883,297, the T500 remained connected
on port 8765, and this run's WebSocket visual replay index continues to hold the
new episode recordings for future swarm playback.

### Contact-guard recovery after resumed exploration

After the first 33.725 s clean screen, ordinary training laps were 33.739 s
and 33.492 s. The next deterministic five-lap evaluation at step 3,885,235 had
a 33.733 s median, 5/5 finishes, but five barrier-contact steps (maximum
impulse 70.90). The contact gate rejected the candidate and triggered automatic
actor recovery; this is not a clean pace improvement. Training was observed
continuing at step 3,885,456 after recovery began. The visual replay archive for
this resumed run had five entries at 00:09:37 local, preserving the episodes
and evaluation traces for later swarm inspection. Port 8765 remained active,
and C: had 26 GB free (D: 144.3 GB).

### Post-recovery pace check, 23:15 UTC

After the 1,000-update critic recovery completed, the actor unlocked and resumed
full-track training. Free-driving episodes at steps 3,888,522, 3,888,969, and
3,890,824 were 33.492 s, 33.536 s, and 33.745 s. The deterministic five-lap
screen at step 3,889,699 was 33.759 s (5/5 finishes, zero barrier contacts),
so it remained slower than Champion and was not promoted. Training continued
through step 3,890,959 with actor unlocked. The resumed replay archive had 10
entries at 00:13:05 local. Port 8765 remained bound to GUI process 19940; disk
headroom was C: 25.9 GB and D: 144.3 GB. The 33.324 s Champion remains intact.

### Second contact regression under the moderated cap

At step 3,891,955, the resumed actor's deterministic screen was 33.864 s
(5/5 finishes) with five barrier-contact steps, so the gate rejected it and
started another automatic recovery. At step 3,892,492 it was still advancing
with 866 critic-recovery updates remaining and actor updates locked. This
confirms that the 0.0075 cap has not yet produced a better reproducible actor.
Historical evidence also cautions against treating a smaller cap as a complete
fix: the earlier 0.005-cap search had a clean 33.609 s screen at step 3,639,664,
but later regressed to 10 repeated contact steps at step 3,664,384. No cap
change was made during this live run; its recovery and Champion guard remain in
control. Replay capture is active, port 8765 is listening, and C: still has
25.9 GB free.

### Contact locations and recovery baseline, 23:20 UTC

The next actor screen at step 3,896,403 measured 33.711 s (5/5 finishes) but
had 15 barrier-contact steps and a maximum impulse of 111.08. Contacts repeated
at progress 98.9489%, 99.2327%, and 99.3742% on each lap. The guard rejected
that actor and entered recovery. During the ensuing actor lock, episodes 12 and
13 completed cleanly at 33.324 s and 33.338 s, matching Champion pace; at step
3,897,168, recovery had 809 critic updates remaining and actor updates were
still locked. This separates the stable baseline behavior from the late-track
contacting proposal. The current run's replay index contains the corresponding
training and evaluation traces for swarm/replay analysis. Training stayed live
on port 8765; C: had 25.9 GB free and D: 144.3 GB.

### Clean evaluation after the second recovery, 23:24 UTC

After the contact-triggered recovery at step 3,896,403, the actor unlocked and
continued training. Its next deterministic five-lap screen, at step 3,900,851,
finished 5/5 at 33.324 s median and best with zero barrier contacts. The clean
anchor was refreshed, but the candidate was not promoted because it only
matched the protected 33.324 s Champion. Training continued to step 3,902,160
with actor updates unlocked and port 8765 still active. The resumed visual
replay archive had 24 entries at 00:24:33 local. C: had 25.8 GB free and D:
144.3 GB. The 30-second target remains unverified.

### Repeated late-track contact after clean Champion match

The next actor proposal, screened at step 3,903,091, completed all five laps at
33.747 s median but contacted the barrier once on every lap (five contact steps,
maximum impulse 67.11), each at 98.6637% progress. The contact gate rejected it
and automatically started another recovery. At step 3,904,444, the run was
still advancing with 662 recovery updates remaining and the actor locked. This
matches the recurring near-finish contact pattern seen in earlier proposals;
the candidate was not promoted. The replay archive had 26 entries at 00:27:05
local. Port 8765 remained live; C: had 25.8 GB free and D: 144.3 GB.

### Clean but slower post-recovery candidate, 23:29 UTC

After the next recovery cycle, the deterministic screen at step 3,907,587 was
33.774 s median (5/5 finishes, zero contacts). The screen was clean but slower
than Champion, so the candidate was rejected and Champion remains protected.
Free-driving episode 23 had finished in 33.366 s immediately before it. The
visual replay index had grown to 30 entries, including the recent training and
evaluation records. Port 8765 remained bound to the active GUI trainer; C: had
25.7 GB free and D: 144.3 GB.

### Latest clean pace screen, 23:32 UTC

The five-lap screen at step 3,909,843 was 33.749 s median and best, with 5/5
finishes and zero barrier contacts. It was rejected as slower than Champion;
free-driving episode 25 immediately before it was 33.679 s. The learner
continued to step 3,910,504 in episode 26 with actor updates unlocked. The
visual replay archive had 33 entries at 00:31:03 local. No champion or training
configuration change was made. Port 8765 stayed active; C: had 25.7 GB free and
D: 144.3 GB.

### Actor update cap telemetry, 23:33 UTC

I inspected the last 67 actor-update progress records at or after step 3,910,000.
Mean proposed action drift was 0.00898 while mean executed drift was 0.00748;
maximum executed drift was 0.00750, exactly the active per-update cap. Maximum
cumulative reference drift in this sample was 0.1594 against the 0.5 allowance.
This is direct evidence that the local per-update cap is constraining actor
movement while the cumulative budget is not. It is a plausible contributor to
slow line adaptation, not proof that relaxing it will improve lap time: the
higher 0.01 setting previously coincided with contact regressions. I left the
live run and its safeguards unchanged. Latest deterministic pace remains
33.749 s clean, with Champion protected at 33.324 s.

### Actor rollback after a faster but contacting screen, 23:35 UTC

At step 3,912,083, the five-lap screen improved to 33.658 s median (5/5
finishes), but each lap contacted the barrier at 98.5208% progress (five contact
steps, peak impulse 272.39). The result was still slower than Champion and was
rejected by the contact gate; automatic actor recovery began immediately. The
trainer continued through step 3,912,692 with 848 critic-recovery updates left
and actor updates locked. This reinforces that the late-track contact is tied
to the more exploratory actor and that a faster one-off candidate is not yet a
safe verified pace improvement. Replay capture remains on; the archive held 35
entries at the previous index refresh. Port 8765 remained active and C: had
25.7 GB free.

### Repeatable clean Champion match after contact recovery, 23:38 UTC

After recovery from the step-3,912,083 contact regression, the next
five-lap screen at step 3,916,547 matched Champion at 33.324 s (5/5 finishes,
zero barrier contacts). It was retained as a clean anchor, not promoted because
it tied the Champion. The preceding free-driving lap was slower at 33.908 s,
showing why deterministic screening remains necessary. Training continued in
episode 33 through step 3,918,124 with actor updates unlocked. The replay index
had 39 entries at 00:37:07 local. Port 8765 remained active; C: had 25.6 GB free
and D: 144.3 GB. The 30-second target remains open.

### Clean candidate pace movement, 23:41 UTC

The next deterministic screen at step 3,918,803 improved to 33.639 s median
(5/5 finishes, zero barrier contacts), 0.110 s faster than the previous clean
screen but still 0.315 s slower than the protected 33.324 s Champion. It was not
promoted. Training remained active through step 3,920,596 in episode 35 with
the actor unlocked. Replay capture advanced to 43 indexed entries at 00:41:41
local. Port 8765 remained active; C: had 25.6 GB free and D: 144.3 GB.

### Further clean pace gain, 23:43 UTC

The deterministic screen at step 3,921,059 improved to 33.502 s median and best,
with 5/5 finishes and zero barrier contacts. This is 0.137 s faster than the
prior 33.639 s clean screen and 0.178 s slower than Champion, so it was not
promoted. Free-driving episode 35 immediately before it was slower at 33.847 s.
The run continued through step 3,922,608 in episode 37 with actor updates
unlocked. Replay capture reached 46 entries at 00:44:13 local. Port 8765 stayed
active; C: had 25.6 GB free and D: 144.3 GB.

### Follow-up clean screen, 23:46 UTC

The deterministic screen at step 3,923,315 measured 33.526 s median and best,
with 5/5 finishes and zero barrier contacts. It was 0.024 s slower than the
preceding 33.502 s result and 0.202 s slower than Champion; no promotion was
made. Free-driving episode 37 immediately before the screen was 33.850 s. The
trainer continued through step 3,924,595 in episode 39 with actor updates
unlocked. The replay index reached 49 entries at 00:46:46 local. Port 8765
remained active; C: had 25.6 GB free and D: 144.3 GB.

### Closest clean screen in this continuation, 23:48 UTC

The deterministic five-lap screen at step 3,925,571 improved to 33.401 s median
and best, with 5/5 finishes and zero contacts. It is now 0.077 s slower than
the protected 33.324 s Champion and is the closest clean candidate in this
continuation so far; it was not promoted. Free-driving episode 39 immediately
before the screen was 33.550 s. Training advanced through step 3,926,996 in
episode 41 with actor updates unlocked. Replay capture reached 51 entries at
00:49:25 local. Port 8765 remained active; C: had 25.5 GB free and D: 144.3 GB.

### Close clean follow-up screen, 23:51 UTC

At step 3,927,827, the deterministic five-lap screen was 33.454 s median and
best (5/5 finishes, zero contacts). It was 0.053 s slower than this
continuation's 33.401 s best and 0.130 s slower than Champion. The candidate was
not promoted. Free-driving episodes 41 and 42 were 33.864 s and 33.804 s. The
learner continued to step 3,929,024 in episode 43 with actor updates unlocked;
replay capture reached 55 entries at 00:51:52 local. Port 8765 remained active;
C: had 25.5 GB free and D: 144.3 GB.

### Contact regression after close clean screens, 23:53 UTC

At step 3,930,083, the next five-lap screen regressed to 33.894 s median
(5/5 finishes) and had ten contact steps: two repeated contact locations on
each lap at 92.8367% and 99.3035% progress. Peak impulse was 959.93. The
candidate was rejected and automatic actor recovery began. Training was still
advancing through step 3,931,396 with 672 critic-recovery updates remaining and
the actor locked. This candidate did not beat Champion and illustrates why the
close 33.401 s clean screen remains only a candidate until the clean pace is
reproduced. Replay capture reached 57 entries at 00:52:24 local. Port 8765
remained active; C: had 25.5 GB free and D: 144.3 GB.

### Post-recovery clean screen, 23:57 UTC

After the step-3,930,083 contact rollback and critic recovery, the next
five-lap screen at step 3,934,563 was 33.567 s median (5/5 finishes, zero
contacts). It remained slower than the 33.401 s best candidate from this
continuation and the 33.324 s Champion, so no promotion occurred. Training
continued through step 3,936,576 in episode 49 with actor updates unlocked and
another policy check pending. Replay capture reached 62 entries at 00:57:59
local. Port 8765 remained active; C: had 25.4 GB free and D: 144.3 GB.

### Near-finish contact rollback and clean locked lap, 23:59 UTC

At step 3,936,803, the next deterministic screen was 33.583 s median (5/5
finishes) but contacted the barrier once on every lap at 98.7351% progress
(five contact steps, peak impulse 124.54). The contact gate rejected the actor
and began recovery. During the actor lock, episode 51 completed cleanly at
33.324 s, matching Champion. At step 3,939,082 recovery still had 431 critic
updates remaining and training continued. The replay archive had 65 entries at
01:00:59 local. Port 8765 remained active; C: had 25.3 GB free and D: 144.3 GB.

### Clean Champion match after the next recovery, 00:03 UTC

After the rollback at step 3,936,803, episode 52 finished at 33.324 s while the
actor was still locked. Recovery then completed; episode 53 ran at 33.365 s,
and the next deterministic five-lap screen at step 3,941,251 matched Champion
at 33.324 s (5/5 finishes, zero contacts). It was not promoted because it tied
the protected Champion. Training continued through step 3,941,928 in episode 54
with actor updates unlocked. The replay archive had 68 entries at 01:01:57
local. Port 8765 remained active; C: had 25.3 GB free and D: 144.3 GB.

### Latest clean screen after matched Champion, 00:05 UTC

The next deterministic screen at step 3,943,507 measured 33.690 s median and
best, with 5/5 finishes and zero contacts. It was not faster than the
continuation's 33.401 s best or the 33.324 s Champion. Training continued
through step 3,944,336 in episode 56 with actor updates unlocked. Replay capture
reached 71 entries at 01:06:22 local. Port 8765 remained active; C: had 25.3 GB
free and D: 144.3 GB.

### Clean but slower follow-up, 00:08 UTC

The five-lap screen at step 3,945,763 was 33.700 s median and best, with 5/5
finishes and zero contacts. It did not improve on the 33.401 s best candidate or
the 33.324 s Champion. Training continued through step 3,946,708 in episode 58
with actor updates unlocked. The visual replay archive had 74 entries at
01:07:04 local. Port 8765 remained active; C: had 25.3 GB free and D: 144.3 GB.

### Latest clean regression, 00:10 UTC

At step 3,948,019, the deterministic screen was 33.782 s median and best, with
5/5 finishes and zero contacts. It was slower than the continuation's 33.401 s
best and the protected 33.324 s Champion, so it was not promoted. Training
continued in episode 60 through step 3,949,120 with actor updates unlocked. The
visual replay archive had 76 entries at 01:09:37 local. Port 8765 remained
active; C: had 25.2 GB free and D: 144.3 GB.

### Contact-triggered rollback, 00:17 UTC

The next screened actor at step 3,952,531 completed its deterministic
five-lap evaluation at 33.754 s median/best, but hit a barrier on every lap
(five total contacts, one per lap). All five contacts occurred at the same
63.455% track progress, with maximum impulse 590.68, so the candidate was
rejected and the actor was restored from Champion. The trainer applied its
1,000-update critic cooldown; it kept the replay and critics. The next recorded
training lap finished cleanly at 33.324 s. This is evidence of repeatable
candidate instability at one section, rather than a pace improvement hidden by
the 33.105 s historical training-best field. That historical lap's nearby
checkpoint already failed deterministic reproduction at 33.755 s (see above).

Training remained active on port 8765 after rollback. Episode 65 was underway
at step 3,953,740 in the latest observed log; the active replay index had 90
entries. C: had about 25.2 GB free and D: about 144.3 GB. The protected
Champion remains 33.324 s.

### Rejected line-change trace and continued training, 00:28 UTC

I compared the rejected step-3,952,531 actor and protected Champion on the
saved observations from episode 000087 around the repeated 63.455% contact
section. The rejected actor's predicted actions closely matched the recorded
training controls (whole-lap MAE 0.0057 steering and 0.0053 longitudinal).
At the nearest contact-zone sample, its steering was -0.346 versus Champion's
-0.071, while both predicted longitudinal actions were about 0.99. Across the
20 samples within one percentage point of the section, the rejected actor
steered 0.147 more negative on average. This shows a material alternate line
choice in the saved training states, but does not prove that line is viable
in closed loop: its deterministic five-lap rollout contacted the barrier at
that exact progress on all five laps. The comparison is open-loop on saved
states and cannot identify the cause of the later trajectory divergence.

Post-recovery five-lap screens were 33.324 s at step 3,956,995 (clean tie
with Champion), 33.621 s at step 3,959,251 (clean), and 33.657 s at step
3,961,507 (clean). The latter two candidates were rejected as slower; Champion
remains 33.324 s. Training was active at step 3,963,763, with visual replay
capture at 94 indexed episodes through episode 000102. Port 8765 was live;
C: had 25.1 GB free and D: had 144.3 GB.

### Best post-recovery candidate remains unpromoted, 00:36 UTC

Training continued from the recovered actor. The clean five-lap screen at
step 3,963,763 measured 33.749 s and was rejected. The next deterministic
screen at step 3,966,035 improved to 33.392 s median/best with 5/5 finishes
and zero barrier contacts. It is 0.009 s faster than the continuation's prior
33.401 s best candidate, but remains 0.068 s slower than Champion, so it was
not promoted. Its policy checkpoint is preserved at
models/simple-track/grtqc/checkpoints/step-3966035-rejected/.

The following clean screen at step 3,968,307 was 33.679 s and was rejected.
At 01:36 local the trainer remained active through step 3,970,560, and the
visual replay archive contained 103 entries through episode 000111. C: had
25.0 GB free and D: had 144.3 GB. Champion remains 33.324 s; the 30 s goal
is still unmet.

### Post-recovery pace gain followed by near-finish rollback, 00:46 UTC

After the step-3,970,563 contact rollback and critic cooldown, the next
clean five-lap actor screen at step 3,977,283 improved to 33.618 s. It was
still slower than the 33.324 s Champion, so Champion remained protected.
The following candidate at step 3,979,555 measured 33.695 s and contacted
the barrier once on each of its five deterministic evaluation laps at
99.233% progress (peak impulse 1,050.63). This is a second failure location,
separate from the earlier 63.4-63.7% corner. The contact gate restored the
Champion actor again. Training continued through step 3,980,196, and the
visual replay index had 114 entries through episode 000122. Port 8765 remained
live; the protected Champion is unchanged at 33.324 s.

### Near-finish action trace after contact regression, 00:49 UTC

I compared the rejected step-3,979,555 actor and Champion on the saved
observations from episode 000121 near the finish. At the closest saved sample
(99.304% progress, just after the evaluation's 99.233% contact point), the
candidate predicted steering -0.356 and longitudinal action 0.726; Champion
predicted -0.431 and 0.641. Over 15 saved samples from 98.2-99.9% progress,
candidate action MAE against recorded training controls was 0.003 steering
and 0.0005 longitudinal. This is an open-loop comparison on a clean training
replay, not the exact collision state, so it suggests but does not prove that
the candidate's final approach was more throttle-heavy and less corrective.
The deterministic five-lap evaluation still contacted at 99.233% on every lap
and the actor was restored from Champion.

The recovery cooldown completed; the actor was unlocked again at step
3,984,002 with cumulative action drift 0.2247 of the 0.5 limit. Training was
active and visual replay capture had 119 entries through episode 000127. C:
had 24.8 GB free and D: 144.3 GB. Champion remains 33.324 s.

### Repeated 63% rollback and post-recovery checks, 01:02 UTC

The candidate at step 3,988,531 repeated the mid-track contact regression: its
five-lap evaluation was 33.830 s with ten barrier contacts, two on each lap at
63.268% and 63.644% progress (peak impulse 904.78). The runner restored the
Champion actor and applied the critic cooldown. After recovery, the clean
five-lap check at step 3,992,979 was 33.548 s, followed by 33.902 s at step
3,995,251; both were 5/5 finishes with zero contacts and slower than Champion.
They were rejected, and training continued. The latest observed state was
step 3,996,776 with replay capture through episode 000142 (134 indexed runs).
Port 8765 remained live. C: had 24.7 GB free and D: had 144.3 GB. Champion
remains protected at 33.324 s; the 30 s target remains unmet.

### Preparing a critic-uncertainty actor experiment, 01:14 UTC

The latest rollout at step 4,006,531 failed with 20 contacts across five
evaluation laps at the 62.961-63.206% corner; its pre-evaluation actor
diagnostics reported Q uncertainty 0.580 and critic disagreement 3.144. Other
contact regressions had lower Q uncertainty, so uncertainty awareness is only
a hypothesis, not an established fix. I created
profiles/training/simple-track-grtqc-contact-aware-30-uncertainty1.json as an
exact copy of the active profile except for actor_uncertainty_coefficient=1.0
(previously 0.0). No reward, actor/critic architecture, replay, evaluation gate,
or Champion setting is changed. The trainer was active at step 4,010,235 on
port 8765 with 150 indexed visual replays. The profile will be applied by
cleanly saving Latest and continuing from it; its effect must be judged by
deterministic five-lap screens before any pace claim.
\r\n### Repeated contact rollback and uncertainty trial decision, 01:23 UTC\r\n\r\nThe run remained active and replay capture continued. The clean five-lap check at step 4,013,267 was 33.699 s, slower than the protected 33.324 s Champion. At step 4,015,523, a five-lap check measured 34.029 s on the first screen and 34.0 s median over five laps, with two barrier contacts versus zero per Champion lap. The guard restored the Champion actor and kept the critics and replay. Pre-rollback actor Q uncertainty reached 0.748 while actor_uncertainty_coefficient remained 0.0. With the actor locked in critic recovery after rollback, I chose to apply the already prepared coefficient 1.0 profile as a one-variable continuation from Latest. This is a hypothesis to evaluate, not a pace result. The protected Champion and deterministic five-lap gate remain unchanged; visual replay capture must continue.\r\n\r\n### Latest actor-setting override gap, 01:27 UTC\r\n\r\nLoading the prepared profile showed Actor uncertainty coefficient=1.0 in Learning settings, but the existing Continue latest with selected actor settings handler only copied learning rate and action drift limits. That would silently restore coefficient 0.0 from Latest. Continue latest with selected rewards was unsuitable because it refills replay. Extended the selected actor-settings continuation to carry the uncertainty coefficient while retaining Latest policy, critics, rewards, curriculum, and replay. Its activity log now reports the selected coefficient. The champion and evaluation gates are unchanged. After restarting the GUI, load the same profile again and use Continue latest with selected actor settings; confirm the resumed run says Latest, fresh_replay=false, and actor uncertainty coefficient=1.0.\r\n\r\n### Uncertainty-aware Latest continuation started, 01:34 UTC\r\n\r\nAfter extending the GUI actor-settings continuation, I restarted the GUI, reloaded the 1M-step uncertainty1 profile, and chose Continue latest with selected actor settings. The new run log is logs/simple-track/grtqc/20261015T013341Z.jsonl. Its started event confirms resume_source=Latest, seed/starting step 4,021,156, fresh_replay=false, CUDA, and NVIDIA T500. At step 4,022,282, the progress event confirms actor_uncertainty_coefficient=1.0 and a nonzero uncertainty penalty (0.347); rollback count is 0. The new visual-replay index already contains 2 captures. Champion remains protected at 33.324 s, 5/5 clean. This is early configuration evidence only; pace impact awaits deterministic evaluation.\r\n
### Uncertainty-aware continuation check, 02:37 UTC

The Latest continuation is still active on CUDA. After the first candidate's
33.745 s one-lap screening result, the deterministic evaluation/diagnostic
cycle rejected that candidate, kept Champion protected, and refreshed a clean
anchor. Training resumed: the live log reached step 4,023,625 in episode 3,
with actor uncertainty coefficient 1.0, actor unlocked, cumulative reference
action drift 0.153, and zero rollbacks. The visual replay index contains full
captures for episodes 1 and 3 (about 9.4 MB each); episode 2 has a short 84 KB
capture. This confirms replay capture continues. The Champion remains 33.324 s
(5/5 clean, zero contacts). No pace improvement has been verified yet; wait for
the next deterministic evaluation before tuning further.

### Contact rollback during uncertainty-aware continuation, 02:39 UTC

The second candidate screening reached 33.973 s with two barrier contacts. Its
full five-episode follow-up also had median 33.973 s but accumulated ten
barrier-contact steps, so the regression guard rejected the actor, emitted
`contact_regression_rollback`, and began actor recovery. At step 4,025,191 the
same run was still advancing; the actor was locked, 840 critic recovery updates
remained, actor uncertainty coefficient was 1.0, and the live visual replay
index held five captures. The 33.324 s clean Champion remains protected. This
trial has not improved pace and has increased contact risk; keep training
through the configured recovery and evaluate the restored anchor before
changing actor constraints again.

### Clean tie after contact recovery, 02:43 UTC

After the contact-triggered actor rollback and 840 critic-recovery updates, the
restored policy's next deterministic evaluation at step 4,028,996 completed
5/5 laps at a 33.324 s median with zero barrier contacts. It tied the protected
Champion and was therefore not promoted. The trainer returned to normal
collection by step 4,029,329, episode 8, with the actor unlocked and the
uncertainty coefficient still 1.0. The current run's visual replay index has
10 captures, including the evaluation replay. This confirms recovery restored
the clean baseline but the uncertainty trial has produced no faster champion.
Continue collecting and retain the existing Champion gate.

### Section-level line exploration diagnosis, 02:49 UTC

The deterministic five-lap check at step 4,031,252 was 33.614 s with zero
contacts, slower than the 33.324 s Champion. The adjacent saved training replay
from the same policy step ran 33.698 s. Comparing that replay with the Champion
replay at step 4,027,881, it covered about 4.7 m more path; normalized-distance
splits place about 0.196 s of its 0.374 s deficit in the 10%-20% section. In
that section its mean absolute steering was 0.569 versus 0.709 in the Champion
replay. This is one pair of runs and is diagnostic evidence, not proof that
steering magnitude caused the loss. A later five-lap check at step 4,033,524
was 33.999 s with 15 total contact steps; the guard rejected it, rolled back
the actor, and began another critic recovery. Training remained live at step
4,035,277, episode 13, with 562 recovery updates remaining.

To test whether the actor needs more generic line exploration, prepared
`profiles/training/simple-track-grtqc-contact-aware-30-uncertainty1-explore1e-2.json`.
It changes only unlocked-actor `exploration_std` from 0.0025 to 0.01; the
critic-collection noise, uncertainty term, LR, action-drift guards, reward,
replay, and five-lap/contact gates remain as before. This profile has not yet
been applied. If applied, continue from Latest with the selected exploration
setting and require five-lap clean checks before promotion.

### Wider actor-exploration trial started, 02:54 UTC

After the 33.999 s / 15-contact regression had triggered rollback, I used the
GUI's clean stop at step 4,037,413; Latest saved and retained the 250,000-item
replay. Loaded the committed one-setting profile and chose Continue latest
with selected exploration. The new run is
`logs/simple-track/grtqc/20261015T015214Z.jsonl`; its startup record confirms
resume from Latest at step 4,037,413, `fresh_replay=false`, CUDA/T500. The GUI
activity log confirms selected actor exploration 0.01 (four times the prior
0.0025); all other settings remain as in the uncertainty-aware profile.

The first five-lap evaluation at step 4,038,533 tied the protected Champion at
33.324 s, with 5/5 finishes and zero barrier contacts. It was rejected as a
tie, not promoted, and its evaluation replay was saved. Training returned to
collection at step 4,038,602 in episode 2 with replay retained and actor
unlocked. This initial check verifies the restored baseline under the new
setting; it does not yet show a pace gain. Await later policy checks to assess
whether wider exploration produces a faster line without contacts.

### Wider-exploration candidate checks, 03:00 UTC

The first actor-policy check after the clean tie at step 4,038,533 completed five
laps at 33.786 s with zero contacts; it was rejected as slower than Champion.
After further learning under exploration_std=0.01, the next five-lap check at
step 4,043,061 was 34.023 s with 20 total barrier-contact steps. The gate
rejected it, restored the verified actor, and started critic recovery. At step
4,043,970 the same run remained live in episode 6 with 773 recovery updates
remaining. The new run's visual replay index continued to grow (8 captures at
the last index check); no Champion or reward change occurred. This trial has
not improved pace and its wider exploration raised contact risk. Let critic
recovery finish and retain the clean Champion gate before judging further
samples.

### Post-recovery check in wider-exploration run, 03:04 UTC

After critic recovery from the 34.023 s / 20-contact candidate, the next screen
at step 4,047,541 and its full five-lap confirmation were 33.798 s, 5/5 clean,
zero contacts. The candidate was rejected as slower than the 33.324 s Champion.
The trainer resumed at step 4,048,314 in episode 10 with the actor unlocked;
visual replay capture remains enabled. Under exploration_std=0.01, observed
candidate outcomes so far are a 33.786 s clean check, a 34.023 s check with 20
contacts and rollback, then a 33.798 s clean post-recovery check. There is no
verified pace gain; continue the guarded trial while retaining Champion.

### Reduce exploration after repeated contact regressions, 03:06 UTC

The wider-exploration run's next candidate at step 4,049,797 evaluated at
33.771 s with five total barrier-contact steps across five laps. The guard
rejected it and began critic recovery; the actor remains protected. Across
exploration_std=0.01 the results have been 33.786 s clean, 34.023 s with 20
contact steps, 33.798 s clean after recovery, and 33.771 s with five contacts.
None beats the 33.324 s Champion, while two candidates caused contact rollback.
Prepared `profiles/training/simple-track-grtqc-contact-aware-30-uncertainty1-explore5e-3.json`,
changing only exploration_std from 0.01 to 0.005 (still twice the old 0.0025)
to reduce contact risk while preserving some additional line exploration. Next
continue from Latest with selected exploration, retaining replay, critics,
rewards, curriculum, uncertainty objective and all evaluation/rollback gates.

### Locate the 33.105 s lap and resume moderate exploration, 02:13 UTC

Correction to the earlier replay search: the 33.105 s lap is saved. It is
`episode-000234.npz` in visual-replay run
`models/simple-track/grtqc/visual_replays/14d4064ea20449b8928e5bd65d27239c/`.
The index records a finished 1,104-decision episode from training steps
2,616,954â€“2,618,058, with observations. Its 33,105 simulator ticks match the
reported 33.106 s PolyTrack personal best to a one-tick (1 ms) display offset.

This was a training rollout, not the deterministic Champion score. At step
2,618,070, immediately after the replay, the trainer completed a deterministic
five-lap evaluation at 33.635 s with 5/5 finishes and zero barrier contacts.
That candidate was rejected against the then-protected 33.324 s Champion. The
33.105 one-off therefore was real and recorded, but its policy did not reproduce
that pace under the promotion evaluation. The 33.105 value remains visible as
the historical `best_training_lap_s` statistic; it does not mean Latest or
Champion can currently repeat the lap. The latest screen before this trial was
33.762 s at step 4,054,261, also clean but slower than Champion.

Resumed Latest from step 4,054,261 using the prepared
`simple-track-grtqc-contact-aware-30-uncertainty1-explore5e-3.json` setting.
Run `logs/simple-track/grtqc/20261015T021312Z.jsonl` confirms CUDA on the NVIDIA
T500, `fresh_replay=false`, the existing 250,000-transition replay, and a
1,000,000-decision full-track budget. The GUI's WebSocket listener is active on
port 8765; automatic visual replay capture remains enabled. The initial progress
record was at step 4,054,262 and the newest observed progress was step 4,054,842
at about 28.9 decisions/s. No post-resume five-lap result is available yet.


### First screen after resuming, 02:14 UTC

The resumed run reached step 4,055,397 and produced a one-lap screen of
33.548 s with zero contact steps. The log emitted `evaluation_screen`, so the
configured deterministic five-lap confirmation is now pending; this single lap
is not a promotion result. The GUI remains active on port 8765 and the same run
continues with its replay capture enabled.


The confirmation completed at 02:15 UTC: 33.548 s median, 5/5 finishes, and
zero barrier contacts. It was rejected because it remains 0.224 s slower than
the 33.324 s Champion, while the clean pace-improvement tracker reset against
the preceding 33.762 s Latest reference. The same run continued collecting
training data after the screen (step 4,056,522); its visual replay bundle was
created and continued receiving captures. No Champion change.


### Second moderate-exploration evaluation, 02:18 UTC

At step 4,057,669 the next deterministic five-lap check was 33.544 s, 5/5
finishes, with zero barrier contacts. It was rejected: this is 0.220 s slower
than Champion and 0.004 s slower than the preceding 33.548 s screen, so it did
not reset the clean pace reference. Training continued with replay capture on;
by 02:19 UTC it had reached step 4,059,142. The active WebSocket listener stayed
on port 8765. No Champion update.


### Contact regression and actor recovery, 02:21 UTC

The third candidate screen at step 4,059,941 confirmed at 34.094 s median with
5/5 finishes but 10 barrier-contact steps across the five laps. The contact
regression gate rejected it, restored the protected actor, and emitted
`actor_recovery` while retaining training and replay state. The run remained
active and continued collection (step 4,063,710 by 02:23 UTC); this was an actor
rollback, not a stopped session. Retain the 33.324 s Champion; no promotion.


### Post-recovery tie with Champion, 02:24 UTC

The first candidate after actor recovery screened at step 4,064,405 and
confirmed at 33.324 s, 5/5 finishes, zero contacts. This ties Champion exactly,
so the deterministic gate rejected promotion; no verified pace improvement was
made. The evaluation replay was recorded and training resumed, reaching step
4,066,198 by 02:25 UTC with the same 250,000-transition replay. The GUI and
WebSocket listener remained active. Champion stays protected at 33.324 s.


### Clean slower check after recovery, 02:27 UTC

The next five-lap evaluation at step 4,066,661 was 33.781 s, 5/5 finished,
with zero contact steps. It was rejected as slower than both Champion and the
post-recovery clean tie; the weak-actor guard recorded the decline. Training
continued beyond step 4,067,882 with automatic replay capture. This sample does
not show pace improvement; Champion remains unchanged.


### Repeated contact rollback in 0.005 exploration trial, 02:30 UTC

The next candidate screened at step 4,068,917 and confirmed at 33.744 s, 5/5
finishes, with five barrier-contact steps total. The contact gate rejected it,
restored the Champion actor, and started another critic recovery while retaining
training and replay. Training continued past step 4,070,642. The 0.005
exploration setting has not yielded a verified pace gain and has now produced
multiple contact rollbacks (including the 34.094 s screen at step 4,059,941).
Keep the deterministic five-lap and contact gates; no Champion change.


### Further clean decline after recovery, 02:33 UTC

The evaluation at step 4,073,365 returned 33.873 s across five clean laps
(5/5 finishes, zero contacts). It was rejected and classified as a weaker actor
than the current clean reference. Since the second contact rollback at step
4,068,917, the recovered actor has produced one 33.324 s Champion tie followed
by 33.781 s, 33.744 s with five contact steps (rollback), and 33.873 s. The
0.005 exploration trial has not improved the protected pace and continues to
show actor instability. Training and replay capture remained active beyond step
4,075,178; no Champion change.


### Contact rollback and actor-objective check, 02:36 UTC

At step 4,075,621, the five-lap evaluation was 33.833 s with 10 total contact
steps (two on the initial screen, repeated across the five-lap confirmation).
The contact guard rejected the actor and started the configured critic cooldown;
training and replay collection continued. During this run's 722 unlocked-actor
progress samples, mean `actor_q_mean` was -50.03 and mean
`actor_uncertainty_penalty` was 0.689 (coefficient 1.0). Mean critic disagreement
was 5.55 with a long tail to 72.91. These raw metrics do not indicate that the
uncertainty term numerically dominates the Q scale, but they do not establish a
causal relationship with pace. I left the uncertainty coefficient and reward
unchanged. The training bridge remained active and the replay bundle retained
the rejected candidate and evaluation.


### Prepare a lower uncertainty penalty test, 02:40 UTC

The 0.005 exploration run continued after the clean 33.324 s tie at step
4,080,085; Latest has since advanced to step 4,081,442. Its previous screens
included repeated actor regressions and two contact-triggered rollbacks, while
the one-off 33.105 s rollout still has not been reproduced by a deterministic
actor. The saved diagnostics show mean actor uncertainty penalty 0.689 against
mean actor Q -50.03 (raw scale only; not causal evidence). To test whether the
uncertainty-aware actor objective is limiting adoption of faster, less familiar
actions, prepared
`profiles/training/simple-track-grtqc-contact-aware-30-uncertainty5e-1-explore5e-3.json`.
It changes only `actor_uncertainty_coefficient`, from 1.0 to 0.5; exploration
0.005, actor LR 5e-6, update/drift guards, rewards, curriculum, critic setup,
and five-lap/contact rollback gates are unchanged. The active run still uses
coefficient 1.0 until it is cleanly continued with selected actor settings.

The planned comparison resumes Latest and preserves its critics, 250,000-item
replay, reward state, Champion, and five-lap verification. It will be judged on
successive deterministic five-lap pace checks and contacts, not one-off training
laps. No change to Champion or reward shaping is intended by this experiment.


## 2026-10-15: lower actor uncertainty weight

The 33.105 s training lap is a valid recorded rollout, but it was not a repeatable verified result: its separate five-lap deterministic evaluation of the nearby saved actor checkpoint was 33.755 s. It therefore did not replace the protected 33.324 s 5/5 Champion. Recent scratch runs repeatedly proposed slower or contact-heavy actors and rolled the actor back while keeping critics and replay, limiting stable policy changes.

Resumed Latest at decision 4,088,364 through **Continue latest with selected actor settings** using `simple-track-grtqc-contact-aware-30-uncertainty5e-1-explore5e-3.json`. Changed only `actor_uncertainty_coefficient` from 1.0 to 0.5; actor learning rate remains 5e-6, per-update action limit 0.0075, cumulative reference drift limit 0.5, exploration std 0.005, and the pace-polish rewards and gates remain selected. The run log confirms `resume_source=models\simple-track\grtqc\latest`, `fresh_replay=false`, replay size 250,000, CUDA on the NVIDIA T500, and coefficient 0.5. New run: `20261015T025008Z`; first resumed progress record is step 4,088,365. Replay recording remains enabled. Champion was not changed.

This is a controlled experiment, not evidence of a pace gain yet. Compare the next deterministic five-lap evaluations and barrier contacts against 33.324 s before promoting anything.


### Live result for the reduced uncertainty trial

The five-lap checks at steps 4,089,484 and 4,091,756 both finished 5/5 with zero barrier-contact steps, off-track events, and crashes. Their medians were 33.549 s and 33.698 s, respectively, so both were rejected against the protected 33.324 s Champion. Each weaker-actor check left Champion intact while training continued. These results do not support the hypothesis that reducing the uncertainty penalty alone unlocks a faster repeatable line; the 33.105 s rollout remains a stochastic one-off.

At step 4,092,326 the 1,000,000-decision Latest continuation remained active on CUDA, still using actor uncertainty coefficient 0.5 and a 250,000-item replay buffer. Its new visual-replay bundle `34c873e331904d93810c93ee8efe367a` contains five recorded episodes. Port 8765 remained listening with an established simulator connection. C: had about 23.6 GiB free, so no cleanup was needed. The trial continues; no pace improvement is claimed yet.


The next evaluation at step 4,094,012 also finished 5/5 at 34.082 s, with 15 total barrier-contact steps (three per lap), zero off-track terminations, and zero crashes; it was slower than Champion and rejected. At step 4,094,673 the same run was still active with the 0.5 uncertainty coefficient, full replay buffer, and port 8765 connected. Its visual-replay bundle had grown to seven episodes. C: still had about 23.6 GiB free. The reduced-uncertainty trial has not produced a pace improvement; training continues and the 33.324 s Champion remains protected.


The 34.082 s check at step 4,094,012 triggered `contact_regression_rollback`: three barrier-contact steps per lap versus zero for Champion. The runner restored the Champion actor and preserved the critic and replay, then started its configured 1,000-update critic recovery cooldown. At step 4,097,117 the run remained active at 14.4 decisions/s with 224 recovery updates remaining and `actor_unlocked=0`; actor updates were temporarily gated while critic recovery continued. This explains the current pause in actor policy improvement after the contact-heavy candidate. No training process was stopped.


After critic recovery completed, the evaluation at step 4,098,476 finished 5/5 at 33.584 s with zero barrier contacts, off-track events, or crashes. It was rejected as slower than the 33.324 s Champion. Actor updates were unlocked again, and by step 4,098,821 the same training run had resumed; its visual-replay bundle contained 12 episodes. The uncertainty 0.5 trial still has no verified pace improvement.


The five-lap evaluation at step 4,100,732 measured 33.701 s, with 5/5 finishes and zero barrier contacts, off-track events, or crashes. It was rejected against the 33.324 s Champion. At step 4,102,293, training remained active at 14.4 decisions/s with actor updates unlocked and replay capacity still 250,000; the active visual-replay bundle contained 16 episodes. This is the fifth completed five-lap result in the 0.5 uncertainty trial; none has improved on Champion, so the trial remains inconclusive and continues without changing the protected checkpoint.


The next candidate at step 4,102,988 finished its five-lap check at 33.874 s with 10 total barrier-contact steps (two per lap), zero off-track events, and zero crashes. It was rejected, and the contact gate restored Champion's actor while preserving critics and replay. At step 4,103,468 training was active in critic recovery with 881 of 1,000 cooldown updates remaining and actor updates temporarily locked. Port 8765 stayed connected, the replay bundle had 17 episodes, and C: had about 23.5 GiB free.


### Historical timing correction and live recovery state

Correction: the nearby saved actor associated with the recorded 33.105 s episode was separately tested at step 3,695,830 on port 8766 and completed five clean deterministic laps at 33.755 s. The 33.635 s five-lap result recorded earlier in this document belongs to a different continuation and was mistakenly attributed to the 33.105 s episode in the uncertainty-trial note. The 33.105 s replay remains a genuine stochastic training episode, not a verified actor pace.

At 03:10:05 UTC, the active run `20261015T025008Z` had reached step 4,106,957 (episode 17), with the original 1,000-update critic cooldown nearly complete (8 updates remaining; actor temporarily locked). The replay buffer remained full, coefficient 0.5 remained active, and port 8765 was listening with a simulator connection. C: had about 23.5 GiB free. Training was left running.


### Stage an intermediate actor update cap

The evaluation at step 4,107,436 finished 5/5 at 33.846 s with zero barrier contacts, off-track events, or crashes. It was rejected as slower than Champion. At step 4,108,827, the existing run was active with actor updates unlocked, coefficient 0.5, and full replay. Its latest replay bundle held 23 episodes.

Actor diagnostics at step 4,107,434 showed proposed action drift 0.0368 but executed drift 0.00748 under the 0.0075 per-update limit, so the cap is constraining actor movement. A prior 0.01 cap produced a 33.375 s candidate but repeated barrier contact on every verification lap and still did not beat the 33.324 s Champion. I staged `simple-track-grtqc-contact-aware-30-uncertainty5e-1-cap8p75e-3-explore5e-3.json` as a measured intermediate: it changes only the per-update action cap from 0.0075 to 0.00875, keeping actor LR, cumulative drift cap, rewards, replay, and all evaluation/rollback gates unchanged. It has not been applied to the live run; training remains active.


### Candidate pace gain after recovery

The five-lap check at step 4,109,692 measured 33.753 s, 5/5 finishes with zero barrier contacts, off-track events, or crashes. It remained slower than the 33.324 s Champion and was not promoted, but it improved the active scratch candidate from 33.846 s; the trainer logged `actor_pace_improved`, reset its weak-evaluation count, and kept the candidate training. At step 4,111,839 the actor was unlocked, replay capacity remained 250,000, and the active replay bundle had 26 episodes. This modest gain is reason to continue the present coefficient 0.5 / action cap 0.0075 run through further checks before using the staged 0.00875 cap.


The next candidate at step 4,111,964 screened at 34.070 s and completed five laps at the same median, 5/5 finishes but 15 total barrier-contact steps (three per lap). There were zero off-track events or crashes. It was rejected and triggered another contact rollback; the Champion actor was restored, critics and replay retained, and a fresh 1,000-update critic cooldown began. At step 4,113,889, the run remained active with 519 recovery updates left and actor updates locked. The active replay bundle contained 29 episodes. The staged 0.00875 profile remains unapplied pending further evidence.


The first post-recovery five-lap check at step 4,116,428 tied the protected Champion at 33.324 s (5/5 finishes, zero barrier contacts, off-track events, or crashes). It was not promoted because the gate requires a faster lap, but this shows the actor returned to anchor pace after the prior rollback. At step 4,118,671, training continued with actor updates unlocked, full replay, and coefficient 0.5; the active replay bundle contained 35 episodes. Port 8765 remained connected. The 0.00875 next-continuation profile is still staged, not applied.


The candidate at step 4,118,684 completed five-lap confirmation at 33.777 s, 5/5 clean with zero barrier contacts, off-track events, or crashes. It was rejected as slower than Champion; no pace gain was claimed. Training resumed and remained active at step 4,120,297, with actor unlocked and the 250,000-item replay full. The current visual-replay bundle had 36 episodes. The prepared 0.00875 cap remains staged and unapplied.


The five-lap check at step 4,120,940 completed at 33.728 s, 5/5 clean with zero barrier contacts, off-track events, or crashes. The trainer recorded a pace improvement over the previous clean candidate (33.777 s) but did not promote it because Champion remains 33.324 s. At step 4,121,701, actor updates were unlocked, replay remained full, and the latest visual-replay bundle contained 38 episodes. The current profile continues unchanged; the intermediate cap remains staged.


The candidate at step 4,123,228 confirmed at 33.943 s over five laps (5/5 finishes, 20 total barrier-contact steps, zero off-track events or crashes). The contact gate rejected it and restored Champion's actor while preserving critic and replay, starting another 1,000-update recovery. At step 4,123,597, training remained active with 908 cooldown updates left and actor updates locked. The current visual replay bundle contained 41 episodes; the staged 0.00875 cap is unchanged and unapplied.


The screen at step 4,127,708 was confirmed by five deterministic laps at 33.324 s (5/5, zero barrier contacts, off-track events, or crashes). It tied, but did not beat, Champion and was not promoted. At step 4,128,917, the active learner resumed with actor unlocked and full replay; its visual-replay bundle had 47 episodes. The current 0.0075 actor cap remains in use; the staged 0.00875 profile has not been loaded into the running trainer.


The five-lap confirmation at step 4,129,964 measured 33.744 s (5/5 clean, zero barrier contacts, off-track events, or crashes), slower than Champion and therefore rejected. At step 4,131,501, the run was still active with actor unlocked, replay full, and the 0.5 uncertainty setting. The active replay bundle contained 50 episodes. The preceding screen's proposed action drift was below the 0.0075 cap, so there is no evidence that raising the cap would affect every actor update; the staged 0.00875 profile remains untested and unapplied.


The five-lap result at step 4,132,220 was 33.781 s (5/5 finishes, five barrier-contact steps, zero off-track events or crashes). The candidate was rejected and triggered Champion-actor rollback plus a 1,000-update critic recovery. Its diagnostic showed proposed action drift 0.0322 versus 0.00749 executed at the current 0.0075 limit; despite that clipping, the candidate still contacted barriers. At step 4,133,649, the live run continued with 643 cooldown updates remaining, actor locked, full replay, and 53 recorded replay episodes. This weakens the case for raising the action cap: the staged 0.00875 profile remains unapplied pending safer evidence.


The five-lap evaluation at step 4,136,684 completed 5/5 at 33.699 s with zero barrier contacts, off-track events, or crashes. It was rejected as slower than Champion. The preceding exploratory training episode (step 4,136,681) did hit a barrier late at 98.7% progress; the evaluation replay did not, so the training contact is not evidence that the deterministic policy made the same contact. The temporary absence of an established 8765 connection was an episode-reset gap: the listener remained alive, the connection returned, and there was no disconnect error. At step 4,138,064 training had resumed with actor unlocked and replay full; the active visual-replay bundle contained 58 episodes. C: had about 23.1 GiB free.


The five-lap evaluation at step 4,138,940 measured 33.808 s, 5/5 finishes with zero barrier contacts, off-track events, or crashes. It was rejected as slower than Champion. By step 4,140,737, training had resumed with the actor unlocked and replay full; the active visual-replay bundle held 61 episodes. The 0.00875 cap remains staged and unapplied.


The candidate at step 4,141,196 confirmed at 33.986 s over five laps (5/5 finishes, 25 barrier-contact steps, zero off-track events or crashes). It was rejected and triggered actor rollback with another 1,000-update critic cooldown. The current 0.0075 cap clipped proposed drift 0.0324 to 0.00750, yet the candidate remained contact-heavy. Therefore the previously staged 0.00875 cap is withdrawn from the GUI profile list; increasing the cap is not justified by this evidence. The live trainer remains on its original settings, running at step 4,142,052 with 787 cooldown updates remaining and full replay; training was not interrupted.


### Resume verification and clean actor screens, 03:56 UTC

The GUI process was still active after the previous status note; its Stop cleanly control remained enabled. The training log showed that the run had continued through step 4,145,651 and then screened the actor at step 4,145,660. Five-lap evaluation measured 33.669 s (5/5 finishes, zero barrier contacts), slower than the 33.324 s Champion, so the candidate was rejected and training continued. A later screen at step 4,147,916 measured 33.816 s (5/5, zero barrier contacts) and was also rejected. Episode 53's exploratory lap was 33.784 s.

The earlier 33.105 s remains a one-off exploratory episode, not the pace of a verified saved actor. The actor associated with that historical run had previously measured 33.755 s over five clean deterministic laps. These new screens likewise show a gap between exploratory laps and the policy that can be repeatedly evaluated; no Champion promotion was made.

At the time of this note, the same uncertainty-0.5 continuation was active at step 4,147,978, with replay capacity 250,000, actor learning rate 5e-6, per-update action limit 0.0075, cumulative drift limit 0.5, and n-step return 16. Visual replay recording remained enabled; the current bundle 34c873e331904d93810c93ee8efe367a contained 71 episodes. During inspection, selecting the ordinary game tab (default port 8765) restored a live simulator connection after Edge had been focused on the separate replay tab (port 8766); the run advanced without being restarted. Champion remains 33.324 s, and the 30 s goal is not yet reached.


### Post-rollback recovery and clean pace checks, 04:06 UTC

The five-lap confirmation at step 4,150,172 measured 34.002 s (5/5 finishes, ten total barrier-contact steps, zero off-track events or crashes). The contact gate restored Champion's actor and started another 1,000-update critic recovery while preserving critics and replay. The cooldown fell from 660 to 168 updates and then completed; the actor unlocked with cumulative reference drift 0.229 and executed update drift at the 0.0075 cap.

The first post-recovery check at step 4,154,652 tied Champion at 33.324 s, 5/5 with zero contact steps, off-track events or crashes. It was not promoted because the gate requires a faster verified result. The next check at step 4,156,908 measured 33.585 s (5/5 clean) and was also rejected as slower. This remains a plateau around the current Champion rather than a verified improvement toward 30 s.

At step 4,157,268 the learner was continuing on the same full replay with actor updates unlocked. The visual-replay bundle remained active and had 82 episodes. No actor settings, reward settings, evaluation gates or protected Champion files were changed.


### Continued clean ties and contact rollbacks, 04:25 UTC

The current continuation kept running and saved visual replays. After the prior note, five-lap checks were: 34.035 s at step 4,159,180 with 10 contact steps (rollback); 33.684 s at step 4,163,660 with zero contacts (rejected as slower); 33.804 s at step 4,165,916 with five contacts (rollback); 33.324 s at step 4,170,380 with zero contacts (clean tie, not promoted); 33.764 s at step 4,172,636 with zero contacts (rejected); and 33.849 s at step 4,174,892 with 20 contact steps (rollback). Each evaluation finished 5/5; Champion stayed 33.324 s.

The run is still active at step 4,177,146, episode 80, about 15.4 decisions/s. Following the last contact rollback, actor updates are locked during critic recovery with 437 updates remaining; critics and the 250,000-entry replay are retained. The active visual-replay bundle contains 105 episodes. No policy, reward or evaluation settings were changed. This evidence continues to show no verified pace improvement toward 30 s.


The next post-recovery candidate, screened at step 4,179,356 and five-lap confirmed at 4,179,356, measured 33.692 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It was rejected as slower than Champion. At step 4,181,101 the same run remained active with actor updates unlocked and full replay; its visual-replay bundle held 110 episodes. The protected 33.324 s Champion and current actor settings remain unchanged.


### Continued evaluations after step 4,179,356, 04:51 UTC

The checks since the prior note were: 33.744 s at step 4,181,612 (5/5 clean, rejected); 34.080 s at step 4,183,884 (5/5, 15 barrier-contact steps, rollback); 33.685 s at step 4,188,348 (5/5 clean, rejected); 33.691 s at step 4,190,604 (5/5 clean, rejected); 33.789 s at step 4,192,876 (5/5 clean, rejected); 33.963 s at step 4,195,164 (5/5, 20 contact steps, rollback); and 33.324 s at step 4,199,612 (5/5 clean tie, not promoted). No candidate beat the protected 33.324 s Champion.

At 04:51 UTC the same full-track learner remained active at step 4,201,213 with actor updates unlocked, replay size 250,000, and no recovery cooldown. Its visual replay bundle held 135 episodes. C: had 22.4 GiB free and D: had 144.3 GiB free. Training continued without changing actor settings, rewards, critics, evaluation gates, or Champion.


### Evaluations after step 4,199,612, 05:15 UTC

The subsequent five-lap checks were: 33.807 s at step 4,201,868 (five contact steps, rollback); 33.324 s at step 4,206,332 (clean tie, not promoted); 33.547 s at step 4,208,588 (clean, rejected as slower); 34.118 s at step 4,210,860 (ten contact steps, rollback); 33.762 s at step 4,215,324 (clean, rejected); 33.801 s at step 4,217,580 (ten contact steps, rollback); 33.590 s at step 4,222,060 (clean, rejected); and 33.786 s at step 4,224,316 (clean, rejected). All completed 5/5. The best clean candidate in this interval was 33.547 s; Champion remains 33.324 s.

At 05:15 UTC the run was active at step 4,224,531, episode 122, with actor updates unlocked and the 250,000-entry replay retained. The visual-replay bundle contained 164 episodes. C: had 22.2 GiB free and D: had 144.3 GiB free. The full-track profile, rewards, critic/replay state, Champion, and evaluation gates were unchanged.


### Clean candidate then renewed contact regressions, 05:31 UTC

The policy checks after step 4,224,316 were: 33.601 s at step 4,226,588 (5/5 clean; the trainer marked actor pace improved but it remained slower than Champion); 34.032 s at step 4,228,844 (5/5, 20 contact steps, rollback); 33.616 s at step 4,233,308 (5/5 clean, rejected); 33.821 s at step 4,235,564 (5/5 clean, rejected); and 34.017 s at step 4,237,852 (5/5, 20 contact steps, rollback). Champion remains 33.324 s. The 33.601 s result is the best clean candidate in this interval but is not a promotion.

At 05:31 UTC training remained active at step 4,239,209, episode 135. The actor was locked for critic recovery with 661 updates remaining; replay size stayed 250,000 and the visual-replay bundle had 182 episodes. C: had 22.2 GiB free and D: had 144.3 GiB free. No training settings, critics, replay, evaluation gates, or Champion files were changed.

### Contact rollback after the 33.105 s rollout, 05:38 UTC

At step 4,244,556 the candidate completed five deterministic laps with a 33.869 s median and 5/5 finishes, but it made two barrier contacts on each lap (10 contact steps total, concentrated near 99% progress). The contact gate rejected it and restored the protected Champion actor (33.324 s); critics and replay were preserved and a 1,000-update critic recovery began. A preceding single deterministic screen tied the Champion at 33.324 s, which was not sufficient to promote. This does not reproduce the earlier 33.105 s exploratory training lap; the nearby saved actor had already measured 33.755 s in its own five-lap check.

At 05:38 UTC the same training process was active at step 4,246,949, episode 142, about 15.7 decisions/s. Actor updates remained locked with 402 critic recovery updates left; replay stayed full at 250,000 transitions. The active visual replay bundle `models/simple-track/grtqc/visual_replays/34c873e331904d93810c93ee8efe367a` contained 190 episodes. Champion and training settings were unchanged.

### Clean evaluation after critic recovery, 05:44 UTC

The candidate screened at step 4,251,276 completed five deterministic laps at 33.839 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It was rejected because it remained slower than the protected 33.324 s Champion; no actor promotion occurred. This is a clean result after recovery, but it does not establish further pace gain.

At 05:44 UTC the learner was active at step 4,251,461, episode 146, about 15.6 decisions/s, with actor updates unlocked and no remaining critic cooldown. The 250,000-transition replay remained full, and visual replay bundle `models/simple-track/grtqc/visual_replays/34c873e331904d93810c93ee8efe367a` contained 197 episodes. Champion and training settings remain unchanged.

### Contact-heavy actor rollback, 05:47 UTC

The candidate screened at step 4,253,548 completed five laps at a 34.115 s median (5/5 finishes, 35 barrier-contact steps total, seven per lap). It was rejected and the Champion actor restored; critic weights and replay were preserved, with another 1,000-update critic recovery started. The protected Champion remains 33.324 s.

At 05:47 UTC the live run had resumed at step 4,253,614, episode 148, about 15.6 decisions/s. The actor was locked with 984 critic recovery updates remaining; replay remained full at 250,000 transitions. The visual replay bundle `models/simple-track/grtqc/visual_replays/34c873e331904d93810c93ee8efe367a` contained 200 episodes. No training settings or Champion files were changed.

### Contact rollback after the clean pace check, 05:51 UTC

The candidate screened at step 4,257,996 completed five laps at 33.772 s (5/5 finishes, 15 barrier-contact steps total, three per lap). Despite the faster lap time relative to the preceding 33.839 s clean check, the contact gate rejected it and restored the protected 33.324 s Champion actor. Critics and replay were retained; another 1,000-update critic recovery began.

At 05:51 UTC the trainer was active at step 4,258,641, episode 152, about 15.7 decisions/s, with 839 recovery updates left and the actor locked. Replay capacity remained full at 250,000 transitions. The current visual replay index last updated at 05:49 UTC and listed 205 episodes. C: had about 22 GiB free and D: about 144.3 GiB. No training settings or Champion files were changed.

### Faster contact-affected candidate rejected, 05:55 UTC

At step 4,262,460 the candidate completed five deterministic laps at 33.716 s (5/5 finishes) but had five barrier-contact steps total, one per lap. The contact regression gate rejected it despite the faster time relative to recent clean candidates, restored the protected 33.324 s Champion actor, preserved critics and replay, and started another 1,000-update critic recovery. This confirms the actor is finding some pace at the cost of repeatable barrier contact; it is not a safe verified improvement.

At 05:55 UTC training was active at step 4,263,393, episode 156, about 15.8 decisions/s, with actor updates locked and 767 critic recovery updates remaining. The replay buffer remained full at 250,000 transitions, and the active visual replay bundle contained 210 episodes. No actor settings or Champion files were changed.

### Clean scratch-anchor refresh, 05:59 UTC

The evaluation at step 4,266,924 completed five deterministic laps at 33.747 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It refreshed the scratch clean anchor but was rejected against the 33.324 s protected Champion. No contact rollback was triggered; the actor remained eligible for training updates.

At 05:59 UTC the live run had resumed at step 4,268,141, episode 161, about 15.9 decisions/s, with actor updates unlocked and no recovery cooldown. Replay remained full at 250,000 transitions. The visual replay bundle contained 215 episodes. No training settings or Champion files were changed.

### Small clean scratch pace improvement, 06:02 UTC

The evaluation at step 4,269,180 completed five deterministic laps at 33.684 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). This improved on the previous clean scratch candidate at 33.747 s, so the trainer marked actor pace improved and refreshed the clean anchor. It remains 0.360 s slower than the protected 33.324 s Champion and was not promoted.

At 06:02 UTC training had resumed at step 4,270,569, episode 163, about 15.9 decisions/s, with actor updates unlocked and no recovery cooldown. Replay capacity remained full at 250,000 transitions; the active visual replay bundle contained 219 episodes. No training settings or Champion files were changed.

### Contact regression on latest actor screen, 06:08 UTC

The candidate screened at step 4,273,724 completed five laps at a 33.935 s median (5/5 finishes, 20 barrier-contact steps total, four per lap). It was rejected by the contact gate; the protected Champion actor was restored and the critics and replay were retained for another 1,000-update recovery.

At 06:08 UTC the training process remained active at step 4,274,921, episode 167, about 15.8 decisions/s. Actor updates were locked with 701 critic recovery updates remaining; replay capacity was still full at 250,000. The visual replay bundle contained 225 episodes. No training settings or Champion files were changed.

### Clean scratch pace improves to 33.571 s, 06:11 UTC

The evaluation at step 4,278,172 completed five deterministic laps at 33.571 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). This improved the active scratch pace after the 33.684 s clean candidate and refreshed the clean anchor. It remains 0.247 s slower than the protected 33.324 s Champion, so it was not promoted.

At 06:11 UTC the run was active at step 4,279,385, episode 171, about 15.8 decisions/s; actor updates were unlocked and no recovery cooldown remained. The replay buffer was full at 250,000 transitions, and the active visual replay bundle contained 230 episodes. No training settings or Champion files were changed.

### One-contact-per-lap candidate rollback, 06:14 UTC

The candidate screened at step 4,280,428 completed five deterministic laps at 33.900 s (5/5 finishes, five barrier-contact steps total, one per lap). The contact gate rejected it and restored the Champion actor, preserving critics and replay while starting a 1,000-update critic recovery.

At 06:14 UTC the live run continued at step 4,282,413, episode 173, about 15.8 decisions/s, with actor updates locked and 504 recovery updates remaining. Replay was full at 250,000 transitions, and the visual replay bundle contained 233 episodes. C: had about 21.7 GiB free and D: about 144.3 GiB. No training settings or Champion files were changed.

### Post-recovery candidate remains slower, 06:17 UTC

The evaluation at step 4,284,908 completed five deterministic laps at 33.773 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It was rejected as slower than the 33.571 s clean scratch candidate; it did not trigger a contact rollback or promote over the 33.324 s protected Champion.

At 06:17 UTC training had resumed at step 4,284,933, episode 176, about 15.8 decisions/s, with actor updates unlocked and no critic cooldown. The 250,000-transition replay remained full, and the visual replay bundle contained 237 episodes. No settings or Champion files were changed.

### Repeated one-contact-per-lap regression, 06:20 UTC

The candidate evaluated at step 4,287,164 completed five deterministic laps at 33.771 s (5/5 finishes, five barrier-contact steps total, one per lap). The contact regression gate rejected it, restored the 33.324 s Champion actor, and began a 1,000-update critic recovery while preserving critics and replay. The result remains far from the 30 s objective and does not improve on the clean 33.571 s scratch candidate.

At 06:20 UTC the run remained active in recovery, with the actor locked. The full 250,000-transition replay and visual replay recording were retained; the visual bundle contained 240 episodes. No training settings or Champion files were changed.

### Clean candidate after one-contact rollback, 06:24 UTC

The evaluation at step 4,291,628 completed five deterministic laps at 33.694 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). The clean anchor was refreshed, but this candidate was rejected as slower than the best clean scratch pace of 33.571 s and remained slower than the 33.324 s protected Champion. No contact rollback occurred.

At 06:24 UTC the same learner was active at step 4,292,354, episode 182, about 15.9 decisions/s, with actor updates unlocked and no recovery cooldown. The replay buffer remained full at 250,000 transitions and the visual replay bundle contained 245 episodes. Training settings and Champion files were unchanged.

### Contact rollback after the latest evaluation, 06:27 UTC

The candidate at step 4,293,884 completed five deterministic laps at 33.864 s (5/5 finishes, five barrier-contact steps total, one per lap). The contact gate rejected it, restored the protected Champion actor, and began another 1,000-update critic recovery while preserving critics and replay.

At 06:27 UTC the run was active at step 4,294,617, episode 184, about 15.8 decisions/s, with the actor locked and 817 recovery updates remaining. Replay capacity stayed full at 250,000 transitions, and the visual replay bundle contained 248 episodes. Champion and training settings were unchanged.

### Recurring one-contact rollback, 06:31 UTC

The candidate at step 4,298,348 completed five laps at 33.697 s (5/5 finishes, five barrier-contact steps total, one per lap). The contact gate rejected it and restored the protected Champion actor while retaining critics and replay, starting another 1,000-update critic recovery.

At 06:31 UTC training remained active at step 4,299,753, episode 189, about 15.9 decisions/s, with 649 recovery updates remaining and actor updates locked. Replay remained full at 250,000 transitions; the visual replay bundle contained 253 episodes. No training settings or Champion files were changed.

### Marginal clean pace improvement, 06:35 UTC

The evaluation at step 4,302,828 completed five deterministic laps at 33.567 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). The trainer refreshed the clean anchor. This is 0.004 s faster than the prior best clean scratch result (33.571 s), but remains 0.243 s slower than the protected 33.324 s Champion and was not promoted.

At 06:35 UTC the learner was active at step 4,304,978, episode 193, about 16.0 decisions/s, with actor updates unlocked and no critic cooldown. Replay remained full at 250,000 transitions; the visual replay bundle contained 259 episodes. Settings and Champion files were unchanged.

### New clean scratch pace: 33.520 s, 06:38 UTC

The evaluation at step 4,305,084 completed five deterministic laps at 33.520 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It improved the previous best clean scratch result of 33.567 s by 0.047 s, refreshed the clean anchor, and was marked `actor_pace_improved`. It remains 0.196 s slower than the protected 33.324 s Champion and was not promoted.

At 06:38 UTC training remained active at step 4,306,093, episode 194, about 15.9 decisions/s, with actor updates unlocked and no critic cooldown. Replay remained full at 250,000 transitions; the visual replay bundle contained 261 episodes. Training settings and Champion files were unchanged.

### Follow-up clean candidate slower, 06:40 UTC

The evaluation at step 4,307,340 completed five deterministic laps at 33.718 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It refreshed the clean anchor but was rejected as slower than the 33.520 s scratch candidate and remained slower than the protected 33.324 s Champion. No contact rollback occurred.

At 06:40 UTC training had resumed at step 4,308,097, episode 196, about 15.9 decisions/s, with actor updates unlocked and no recovery cooldown. Replay remained full at 250,000 transitions; the visual replay bundle contained 264 episodes. Settings and Champion files were unchanged.

### Scratch pace rebounds to 33.662 s, 06:43 UTC

The evaluation at step 4,309,596 completed five deterministic laps at 33.662 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It improved on the immediately preceding 33.718 s candidate, refreshed the clean anchor, and was marked `actor_pace_improved`. It did not beat the current best clean scratch result of 33.520 s and remains slower than the protected 33.324 s Champion.

At 06:43 UTC training was active at step 4,310,513, episode 198, about 15.9 decisions/s, with actor updates unlocked and no cooldown. Replay remained full at 250,000 transitions; the visual replay bundle contained 267 episodes. No training settings or Champion files were changed.

### Clean scratch gain to 33.480 s, 06:49 UTC

The evaluation at step 4,314,092 completed five deterministic laps at 33.480 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). This improves the clean scratch pace from 33.520 s by 0.040 s, but remains 0.156 s slower than the protected 33.324 s Champion, so it was not promoted. The 33.105 s metadata field remains a best training lap and was not reproduced by this five-lap evaluation.

At 06:49 UTC training was active at step 4,314,943, episode 202, about 15.8 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 890 files (5.65 GiB); C: had 21.3 GiB free. No training settings or Champion files were changed.

### Contact regression and critic recovery, 06:52 UTC

The evaluation at step 4,316,364 completed five deterministic laps at 33.955 s (5/5 finishes, 10 barrier-contact steps total, zero off-track events or crashes). The contact gate rejected the candidate and locked actor updates for critic recovery; the protected 33.324 s Champion remains unchanged.

At 06:52 UTC training had resumed at step 4,317,405, episode 204, about 15.7 decisions/s, with 740 critic recovery updates remaining. Replay remained full at 250,000 transitions; the visual replay bundle contained 892 files. C: had 21.3 GiB free. No training settings or Champion files were changed.

### Clean post-recovery evaluation, 06:55 UTC

After critic recovery from the contact regression, the evaluation at step 4,320,828 completed five deterministic laps at 33.754 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). This is slower than the best clean scratch result of 33.480 s and the protected 33.324 s Champion; neither checkpoint was replaced. Actor updates were unlocked again after recovery.

At 06:55 UTC training was active at step 4,320,856, episode 208, about 15.8 decisions/s, with no critic recovery remaining. Replay remained full at 250,000 transitions. The visual replay bundle contained 897 files (5.70 GiB); C: had 21.3 GiB free. Training settings and Champion files were unchanged.

### Repeated one-contact-per-lap rollback, 06:58 UTC

The evaluation at step 4,323,084 completed five deterministic laps at 33.786 s (5/5 finishes, five barrier-contact steps total, zero off-track events or crashes). The contact gate rejected this candidate and began another critic recovery; it did not replace the clean anchor or protected 33.324 s Champion.

At 06:58 UTC training was active at step 4,323,322, episode 210, about 15.7 decisions/s, with actor updates locked and 941 critic recovery updates remaining. Replay remained full at 250,000 transitions. The visual replay bundle contained 900 files (5.72 GiB); C: had 21.2 GiB free. Training settings and Champion files were unchanged.

### Clean evaluation after contact recovery, 07:02 UTC

The evaluation at step 4,327,532 completed five deterministic laps at 33.637 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It improved on the immediately preceding 33.786 s contact-affected candidate, but did not beat the clean scratch best of 33.480 s or the protected 33.324 s Champion.

At 07:02 UTC training was active at step 4,327,678, episode 214, about 15.8 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 905 files (5.75 GiB); C: had 21.2 GiB free. No training settings or Champion files were changed.

### Clean but slower evaluation, 07:05 UTC

The evaluation at step 4,329,788 completed five deterministic laps at 33.891 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It remained clean but was slower than the clean scratch best of 33.480 s and the protected 33.324 s Champion; neither was replaced. No contact recovery was triggered.

At 07:05 UTC training was active at step 4,330,149, episode 216, about 15.8 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 908 files (5.77 GiB); C: had 21.2 GiB free. Training settings and Champion files were unchanged.

### Contact rollback after another slower candidate, 07:08 UTC

The evaluation at step 4,332,060 completed five deterministic laps at 33.987 s (5/5 finishes, 10 barrier-contact steps total, zero off-track events or crashes). The contact gate rejected it and began critic recovery; the protected 33.324 s Champion and the 33.480 s clean scratch best remain unchanged.

At 07:08 UTC training was active at step 4,332,564, episode 218, about 15.7 decisions/s, with actor updates locked and 875 critic recovery updates remaining. Replay remained full at 250,000 transitions. The visual replay bundle contained 911 files (5.79 GiB); C: had 21.1 GiB free. Training settings and Champion files were unchanged.

### Clean scratch candidate ties Champion, 07:12 UTC

The evaluation at step 4,336,556 completed five deterministic laps at 33.324 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). This matches the protected Champion's verified 33.324 s, but does not beat it, so the Champion slot remains untouched. The latest checkpoint records the same five-lap clean evaluation.

At 07:12 UTC training was active at step 4,337,581, episode 222, about 15.8 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 916 files (5.82 GiB); C: had 21.1 GiB free. No training settings or Champion files were changed.

### Clean follow-up after Champion tie, 07:15 UTC

The evaluation at step 4,338,812 completed five deterministic laps at 33.651 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It did not repeat the 33.324 s Champion-tie evaluation and remains slower than both the protected Champion and the 33.480 s clean scratch best. The Champion was not changed.

At 07:15 UTC training was active at step 4,339,053, episode 224, about 15.8 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 919 files (5.84 GiB); C: had 21.1 GiB free. Training settings and Champion files were unchanged.

### Small clean pace rebound, 07:18 UTC

The evaluation at step 4,341,068 completed five deterministic laps at 33.567 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It improved on the previous 33.651 s result by 0.084 s, but remains 0.087 s slower than the best clean scratch result (33.480 s) and 0.243 s slower than the protected 33.324 s Champion. No checkpoint was promoted.

At 07:18 UTC training was active at step 4,341,525, episode 226, about 15.7 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 922 files (5.86 GiB); C: had 21.0 GiB free. Training settings and Champion files were unchanged.

### Clean but marginally slower follow-up, 07:20 UTC

The evaluation at step 4,343,324 completed five deterministic laps at 33.666 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It is 0.099 s slower than the preceding 33.567 s evaluation and remains slower than the 33.480 s clean scratch best and protected 33.324 s Champion. The Champion was not changed.

At 07:21 UTC training was active at step 4,343,469, episode 228, about 15.7 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 925 files (5.88 GiB); C: had 21.0 GiB free. Training settings and Champion files were unchanged.

### Clean plateau around 33.7 s, 07:26 UTC

The evaluation at step 4,347,836 completed five deterministic laps at 33.697 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). This is 0.008 s faster than the preceding 33.705 s evaluation, but remains 0.217 s slower than the best clean scratch result (33.480 s) and 0.373 s slower than the protected 33.324 s Champion. No promotion occurred.

At 07:26 UTC training was active at step 4,347,876, episode 232, about 15.6 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 931 files (5.91 GiB); C: had 20.9 GiB free. Training settings and Champion files were unchanged.

### Contact rollback at 33.930 s, 07:29 UTC

The evaluation at step 4,350,124 completed five deterministic laps at 33.930 s (5/5 finishes, 15 barrier-contact steps total, zero off-track events or crashes). The contact gate rejected this actor and restored the protected reference actor while retaining critics and replay.

At 07:30 UTC training was active at step 4,350,941, episode 234, about 15.6 decisions/s, with actor updates locked and 796 critic recovery updates remaining. Replay remained full at 250,000 transitions. The visual replay bundle contained 935 files (5.94 GiB); C: had 20.9 GiB free. Training settings and Champion files were unchanged.

### Faster but contact-affected evaluation, 07:33 UTC

The evaluation at step 4,354,572 completed five deterministic laps at 33.606 s (5/5 finishes, five barrier-contact steps total, zero off-track events or crashes). It was faster than the immediately prior 33.930 s result but still slower than the clean scratch best (33.480 s) and protected 33.324 s Champion. The contact gate rejected it and began critic recovery.

At 07:33 UTC training was active at step 4,354,913, episode 238, about 15.7 decisions/s, with actor updates locked and 915 critic recovery updates remaining. Replay remained full at 250,000 transitions. The visual replay bundle contained 939 files (5.96 GiB); C: had 20.8 GiB free. Training settings and Champion files were unchanged.

### Clean pace after contact recovery, 07:37 UTC

The evaluation at step 4,359,052 completed five deterministic laps at 33.543 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It recovered clean pace after the 33.606 s contact-affected candidate but remains 0.063 s slower than the best clean scratch result (33.480 s) and 0.219 s slower than the protected 33.324 s Champion. No promotion occurred.

At 07:37 UTC training was active at step 4,359,321, episode 242, about 15.7 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 944 files (6.00 GiB); C: had 20.8 GiB free. Training settings and Champion files were unchanged.

### Contact rollback at 33.847 s, 07:40 UTC

The evaluation at step 4,361,308 completed five deterministic laps at 33.847 s (5/5 finishes, five barrier-contact steps total, zero off-track events or crashes). The contact gate rejected it and initiated critic recovery; the protected Champion remains unchanged.

At 07:40 UTC training was active at step 4,361,732, episode 244, about 15.7 decisions/s, with actor updates locked and 895 critic recovery updates remaining. Replay remained full at 250,000 transitions. The visual replay bundle contained 947 files (6.02 GiB); C: had 20.8 GiB free. Training settings and Champion files were unchanged.

### Second clean Champion tie, 07:43 UTC

The evaluation at step 4,365,756 completed five deterministic laps at 33.324 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). This exactly ties the protected Champion, but does not beat it; the Champion slot was left unchanged.

At 07:44 UTC training was active at step 4,366,721, episode 248, about 15.8 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 952 files (6.05 GiB); C: had 20.7 GiB free. Training settings and Champion files were unchanged.

### Contact rollback after clean Champion tie, 07:46 UTC

The evaluation at step 4,368,012 completed five deterministic laps at 33.802 s (5/5 finishes, 10 barrier-contact steps total, zero off-track events or crashes). It did not repeat the prior clean 33.324 s Champion tie. The contact gate rejected it and started critic recovery; the Champion remains unchanged.

At 07:47 UTC training was active at step 4,368,921, episode 250, about 15.7 decisions/s, with actor updates locked and 773 critic recovery updates remaining. Replay remained full at 250,000 transitions. The visual replay bundle contained 955 files (6.07 GiB); C: had 20.7 GiB free. Training settings and Champion files were unchanged.

### Clean evaluation after recovery, 07:50 UTC

The evaluation at step 4,372,460 completed five deterministic laps at 33.600 s (5/5 finishes, zero barrier-contact steps, zero off-track events or crashes). It is clean but remains slower than the 33.480 s best scratch result and the protected 33.324 s Champion. The Champion was unchanged.

At 07:50 UTC training was active at step 4,372,673, episode 254, about 15.8 decisions/s, with actor updates unlocked and no critic recovery. Replay remained full at 250,000 transitions. The visual replay bundle contained 960 files (6.10 GiB); C: had 20.6 GiB free. Training settings and Champion files were unchanged.


### Historical 33.105 s lap versus the current policy, 07:55 UTC

The current continuation inherited `best_training_lap_s = 33.105` from its checkpoint; the first progress record for this run already contained that value. The current deterministic five-lap evaluation at step 4,374,716 was 33.555 s (5/5 finishes, zero contacts, zero crashes, zero off-track events), so this policy has not repeated the historical lap. The evaluation was rejected against the protected 33.324 s Champion. GRTQC training rollouts add correlated exploration noise, while evaluation calls the policy deterministically; a fast exploratory episode is therefore not evidence that the deterministic policy can reproduce that pace. No Champion files or training settings were changed.

At 07:55 UTC the GUI trainer was still alive, the bridge was listening on port 8765, and training had resumed at step 4,376,721, episode 257, about 15.8 decisions/s. Replay capacity remained full at 250,000 transitions. Visual replay capture contained 5,698 files totaling 36.71 GiB, including the evaluated run; C: had 20.07 GiB free. The latest hour added about 0.44 GiB, so no space cleanup was needed and training remained active.


### Clean but slower evaluation; actor update cap observed, 07:57 UTC

The next five-lap evaluation completed at step 4,376,988 with a 33.752 s median (5/5 finishes, zero barrier-contact steps, zero crashes, zero off-track events). Its evaluation replay was saved. The candidate was rejected against the protected 33.324 s Champion, and training resumed without recovery at step 4,378,345, episode 259, about 15.7 decisions/s. The retained historical best training lap remains 33.105 s; it has not been reproduced by deterministic evaluation.

At step 4,378,345 the actor proposed action drift of 0.02182 while the applied drift was 0.00749, showing that the configured per-update action-drift cap is binding. This is direct evidence that actor changes are being clipped, though it does not prove that increasing the cap would improve lap time. No live setting change or restart was made. Replay capture remained active: 5,702 files totaling 36.74 GiB, with 20.04 GiB free on C:. Training stayed active on port 8765.


### Successive clean candidate remained slower, 08:00 UTC

At step 4,379,244 the five-lap evaluation completed at 33.796 s (5/5 finishes, zero barrier contacts, zero crashes, zero off-track events). The evaluation replay was saved. This candidate was rejected against the protected 33.324 s Champion; no critic recovery was needed. Training resumed at step 4,379,969, episode 260, about 15.7 decisions/s. It remains roughly 3.8 s above the 30 s target, and the inherited 33.105 s exploratory training lap remains unverified by deterministic evaluation.

The current per-update action-drift limit is being reached in actor diagnostics, but the historical 0.01-cap experiment produced repeated barrier contact during verification. I left the running actor/configuration unchanged because the prior faster candidate was unsafe and there is no verified live-tuning path. The five-lap run replay was captured for later swarm use. Training and port 8765 remain active; C: had 20.02 GiB free. No cleanup was necessary.


### Continued actor-update saturation, 08:00 UTC

The live continuation advanced to step 4,380,893, episode 261, at about 15.7 decisions/s. Its latest progress record shows proposed actor action drift 0.03965 versus executed drift 0.00749, confirming that the active per-update cap is substantially clipping actor changes. The last deterministic evaluation remains 33.796 s; the protected Champion remains 33.324 s. This diagnostic supports the hypothesis that line adaptation is conservative, but does not prove that a higher cap would yield a faster clean policy. The earlier 0.01-cap trial had repeated contact, so I made no live setting changes and did not interrupt training.

Replay capture remains active (5,705 files, 36.76 GiB); C: had 20.01 GiB free. The GUI trainer PID 21944 and websocket bridge on port 8765 were live at the check.


### Contact regression rolled back, 08:03 UTC

The five-lap evaluation at step 4,381,500 reported 33.995 s with 5/5 finishes. Although the initial screen showed three contact steps, the completed evaluation replay and aggregate recorded 15 barrier-contact steps across the five laps (zero crashes and off-track events). The candidate was rejected, the actor was rolled back, and critic recovery started. At the latest progress check the trainer had resumed at step 4,382,444, episode 262, about 15.7 decisions/s, with 765 critic recovery updates remaining. The evaluation replay was saved.

This contact regression occurred with the existing conservative update cap, so relaxing it now has no evidential basis and could increase contact risk. I left settings and protected Champion unchanged. C: had 19.99 GiB free, and the trainer/bridge remained active.


### Clean post-recovery evaluation and exploratory 33.313 s lap, 08:07 UTC

After contact rollback and the 1,000-update critic cooldown, the actor unlocked. Recorded training episodes at steps 4,383,722 and 4,384,846 included a 33.713 s lap and a 33.313 s lap respectively; both replays are saved in the active visual-replay bundle. The 33.313 s result is a single training rollout, not a verified deterministic result. The following clean five-lap actor evaluation at step 4,385,964 was 33.521 s (5/5 finishes, zero barrier contacts, zero crashes/off-track events). It was rejected because it remains slower than the protected 33.324 s Champion; no promotion occurred.

Training resumed at step 4,387,064, episode 266, about 15.7 decisions/s, with actor updates unlocked and no recovery cooldown. Replay capture remains active. The exploratory lap shows the run can occasionally reach Champion pace, but repeatability is still unproven and the 30 s target remains unmet. C: had 19.95 GiB free. No training settings or protected checkpoints were changed.


### Second post-recovery contact regression, 08:10 UTC

The actor evaluation at step 4,388,220 completed five laps at 33.912 s (5/5 finishes, 15 barrier-contact steps total, zero crashes and off-track events). It was rejected and `contact_regression_rollback` restored the actor; a new critic recovery cooldown started. At step 4,389,885, episode 269, training was still active at about 15.7 decisions/s with actor updates locked and 584 recovery updates remaining. The five-lap evaluation replay was saved. This is the second consecutive contact-heavy actor candidate after a recovery cycle, reinforcing that the current actor updates are not yet yielding repeatable clean pace; the Champion remains protected at 33.324 s.

The current actor step cap remains 0.0075, cumulative reference drift limit 0.5, actor LR 5e-6, uncertainty coefficient 0.5, n-step return 16, and replay capacity 250,000. I did not change these during the active cooldown. Visual replay capture remains enabled; C: had 19.93 GiB free. Training and port 8765 remained live.


### Contact locations shift with actor candidates, 08:12 UTC

Reviewing the two latest rejected five-lap evaluations shows the contact regressions occurred in different parts of the track, rather than at one fixed finish-line location. At step 4,381,500, the 15 contact steps repeated at progress ratios 0.6308, 0.6314, and 0.6333 on each lap, with maximum impulse 1,216.9. At step 4,388,220, contacts repeated at 0.9359, 0.9866, and 0.9966, with maximum impulse 219.7. Both candidates were rolled back by the existing guard. This points to actor-dependent line regressions across different corners, not a single deterministic finish-line fault; no scripted line or action was added.

The 1,000-update recovery after the latter rollback completed. At 08:11:54 UTC the trainer was active at step 4,392,667, episode 271, about 15.8 decisions/s, actor unlocked, and no recovery updates remaining. Replay capture continues and the evaluation replays are saved. C: had 19.87 GiB free. Champion remains protected at 33.324 s; target 30 s is unmet.


### Clean actor evaluation after second rollback, 08:14 UTC

After the second 1,000-update critic recovery, the five-lap evaluation at step 4,392,668 completed at 33.811 s (5/5 finishes, zero barrier contacts, crashes, or off-track events). It was rejected as slower than the protected 33.324 s Champion. Its deterministic replay was saved. The active run resumed at step 4,394,901, episode 273, about 15.7 decisions/s, with actor updates unlocked and no recovery cooldown. Replay capture remains active; the archive contained 5,720 files totaling 36.86 GiB and C: had 19.86 GiB free. No settings or Champion files were changed.


### Incremental clean pace improvement, 08:16 UTC

The five-lap deterministic evaluation at step 4,394,924 completed in 33.719 s (5/5 finishes, zero barrier contacts, crashes, or off-track events). This is 0.092 s faster than the prior clean candidate at 33.811 s and refreshed the clean scratch anchor, but remains 0.395 s slower than the protected Champion at 33.324 s; no Champion promotion occurred. The evaluation replay was saved. Before this screen, cumulative actor reference-action drift was 0.237 against the configured 0.5 limit; the action change was clean on this evaluation.

At 08:16:53 UTC training had resumed at step 4,395,457, episode 274, about 15.7 decisions/s, actor unlocked and no recovery cooldown. Replay capture remains active. C: had 19.85 GiB free. Settings and Champion files remain unchanged.


### Clean screen remained just behind anchor, 08:19 UTC

The deterministic five-lap evaluation at step 4,397,196 completed at 33.737 s with 5/5 finishes and zero barrier contacts, crashes, or off-track events. It was 0.018 s slower than the 33.719 s clean anchor, so it was rejected as weaker; no contact rollback occurred. Its evaluation replay was saved. The preceding exploratory training episode recorded three barrier contacts clustered around 63.08?63.27% progress, while the deterministic actor screen remained clean.

Training resumed at step 4,397,373, episode 276, about 15.6 decisions/s, actor unlocked with no recovery cooldown. Visual replay capture remains active (5,725 files, 36.89 GiB); C: had 19.82 GiB free. No training settings or Champion files were changed.


### Repeated 63% corner contact regression, 08:22 UTC

The five-lap evaluation at step 4,399,484 completed at 34.200 s (5/5 finishes, 15 barrier-contact steps, zero crashes/off-track events). Contacts repeated at progress ratios 0.6266, 0.6278, and 0.6314 on every lap; maximum barrier impulse was 2,017.6. This repeats the earlier 63.1% corner contact cluster seen at step 4,381,500, after a clean intervening actor evaluation. The contact gate rejected the candidate and rolled back the actor; the evaluation replay was saved.

At step 4,400,281, episode 278, the trainer remained active at about 15.6 decisions/s, actor locked with 801 critic recovery updates remaining. The clean Champion and clean anchor protections remain active; no hyperparameters were changed. C: had 19.80 GiB free.


### Faster candidate rolled back for finish-sector contacts, 08:26 UTC

The deterministic five-lap evaluation at step 4,403,948 completed in 33.659 s (5/5 finishes, 10 barrier-contact steps, zero crashes/off-track events). This is 0.060 s faster than the 33.719 s clean anchor but remains 0.335 s slower than Champion and made two contacts on every lap near progress 0.9909 and 0.9923; maximum impulse was 183.7. The contact gate rejected the candidate and restored the actor. The five-lap replay was saved.

At 08:26:20 UTC training was active at step 4,404,273, episode 282, about 15.7 decisions/s, with actor locked and 919 critic recovery updates remaining. The deterministic contact cluster differs from the earlier 63% corner regression, confirming that recent faster actor variants are unsafe in multiple sectors. I did not alter the active profile or bypass the rollback. C: had 19.76 GiB free.


### Clean pace improvement after finish-sector rollback, 08:30 UTC

Following recovery from the 33.659 s contact rollback, the five-lap evaluation at step 4,408,396 completed in 33.683 s (5/5 finishes, zero barrier contacts, crashes, or off-track events). It improved the clean anchor by 0.036 s from 33.719 s, but remains 0.359 s slower than the protected 33.324 s Champion. The evaluation replay event was recorded; the training visual-replay archive remains active. No Champion promotion occurred.

At 08:30:13 UTC training was active at step 4,408,777, episode 286, about 15.7 decisions/s, actor unlocked and no recovery cooldown. C: had 19.71 GiB free. No settings or protected checkpoint files were changed.


### Finish-sector contact regression after clean 33.683 s anchor, 08:33 UTC

The five-lap evaluation at step 4,410,652 completed at 33.772 s (5/5 finishes, five barrier-contact steps, zero crashes/off-track events). Each lap contacted the barrier near progress 0.9895; maximum impulse was 116.6. It was slower than the 33.683 s clean anchor and triggered candidate rejection plus actor rollback. The evaluation replay was saved.

At 08:33:11 UTC the trainer continued at step 4,411,157, episode 288, about 15.7 decisions/s, actor locked with 874 critic recovery updates remaining. The current 33.324 s Champion and clean anchor are preserved. C: had 19.69 GiB free. No settings were changed.


### Rollback restored clean Champion pace, 08:37 UTC

After recovery from the 33.772 s finish-contact regression, the five-lap evaluation at step 4,415,116 completed in exactly 33.324 s (5/5 finishes, zero barrier contacts, crashes, or off-track events), tying the protected Champion. The clean anchor was refreshed; no promotion occurred because the result did not beat Champion. The evaluation replay event was recorded.

At 08:37:12 UTC the run resumed at step 4,415,933, episode 292, about 15.7 decisions/s, actor unlocked and no recovery cooldown. The visual replay archive remains enabled. C: had 19.64 GiB free. Training settings and Champion files were unchanged.


### Clean but slower actor candidate after Champion tie, 08:39 UTC

The next deterministic five-lap evaluation at step 4,417,372 completed in 33.717 s (5/5 finishes, zero barrier contacts, crashes, or off-track events). It was rejected as weaker than the protected 33.324 s Champion; no contact rollback was triggered. The evaluation replay event was recorded.

At 08:39:52 UTC training resumed at step 4,417,705, episode 294, about 15.7 decisions/s, actor unlocked with no recovery cooldown. Replay capture remains active. C: had 19.61 GiB free. No settings or Champion files were changed.


### Repeated near-finish contact rollback, 08:43 UTC

The five-lap evaluation at step 4,419,628 completed in 33.786 s (5/5 finishes, five barrier-contact steps, zero crashes/off-track events). Every lap had one contact at progress 0.98878; maximum impulse was 119.6. This candidate was slower than the clean 33.324 s Champion tie and triggered contact rollback. The evaluation replay was saved.

At 08:43:04 UTC training remained active at step 4,420,449, episode 296, about 15.7 decisions/s, actor locked with 795 critic recovery updates remaining. The Champion was unchanged. C: had 19.59 GiB free. I did not change the active configuration or stop the run.


### Small clean pace gain after rollback, 08:46 UTC

The deterministic five-lap evaluation at step 4,424,108 completed in 33.671 s (5/5 finishes, zero barrier contacts, crashes, or off-track events). This improves the preceding clean candidate at 33.717 s by 0.046 s and refreshed the clean scratch anchor, but remains 0.347 s slower than the protected 33.324 s Champion. No Champion promotion occurred. The evaluation replay event was recorded.

At 08:46:55 UTC the run had resumed at step 4,424,837, episode 300, about 15.7 decisions/s, actor unlocked and no recovery cooldown. Replay capture remains active. C: had 19.55 GiB free. Training settings and Champion files were unchanged.
