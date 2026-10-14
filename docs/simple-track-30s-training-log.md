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

## 2026-10-14 20:47 local — provenance of the 33.105 s training best

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
