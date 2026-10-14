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
