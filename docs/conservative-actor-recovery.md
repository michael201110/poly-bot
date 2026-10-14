# Recovering a regressed GRTQC actor

Use **Recover champion actor** in Overview or Run setup after stopping training.
This resumes Latest, restores the verified champion actor, and preserves Latest's
critics, replay buffer, reward settings, return horizon, and total training step.
The selected session budget is retained. The champion slot remains protected by
the existing deterministic five-lap promotion gate.

Recovery now limits actor learning rate to 0.000005, delays actor updates to at
least every eight critic updates, caps individual action drift at 0.005 and
cumulative reference action drift at 0.05. Actor evaluation occurs at least every 2,000
decisions, with recovery after at most two consecutive weaker checks. Existing
stricter limits are retained. The existing critic recovery cooldown still applies.

These settings constrain further regressions; they do not guarantee faster laps.
Ordinary Continue actions still restore the checkpoint's saved settings.
The recovery action now samples actor updates from states on the champion's
verified lap. Critic updates still sample the full replay buffer. This targets
the part of the track where pace must improve without prescribing actions or a
scripted line. It is an experiment; only five-lap evaluations can establish a
gain. The actor reference is captured before updates on every recovery resume.

On 2026-10-12, Simple Track's protected champion was 33.591 seconds over five
completed deterministic laps. Latest was stopped safely at step 1,544,539 before
recovery, with its policy and metadata archived locally. The active run had actor
LR 0.0001, update interval 2, no cumulative reference drift limit, and five weak
checks before recovery. Its critics and 250,000-transition replay were retained.

## First recovery outcome and next run

The conservative run started at step 1,544,539, with 250,000 replay transitions.
Its first recovery settings were actor LR 0.00002, update interval four, step
action cap 0.01, and cumulative drift cap 0.1.
Its first two five-lap evaluations, at steps 1,555,179 and 1,557,675, were
34.823 seconds. The second weak result restored the champion actor automatically.
Training then stopped safely at step 1,564,516 for the verified-state actor
sampling experiment. The champion remained 33.591 seconds, five clean laps.

The recorded training run `5e77f00d8849497b9ed21143405c67e0` contains 632
replays: 534 finishes, 42 failures, and 56 interrupted attempts. Its fastest
training lap is 33.494 seconds at step 1,056,265. The first conservative run
`25d4b9fbaaec478b8769eaff72077ce9` also has its own replay directory.
Both are under `models/simple-track/grtqc/visual_replays/` with `index.json`
and full `.npz` telemetry for the GUI swarm selector. These generated replay
files stay in the local model workspace because they are large; this document
records their stable run IDs on GitHub.

Resume from Latest with the GUI's **Recover champion actor** action and a
2,321,682-decision budget. This preserves the original three-million-decision
session total begun at step 886,198. Compare the next deterministic evaluations
with the 33.591-second champion before claiming improvement.

The verified-state sampling run began at step 1,564,516. Its first two
deterministic five-lap evaluations, at steps 1,575,508 and 1,577,700, both
finished in 35.190 seconds. Each evaluation recorded 55 barrier-contact steps
across five laps, compared with five for the 33.591-second champion. The actor
was restored automatically after the second weaker result; training then stopped
safely at step 1,577,700. The experiment did not improve pace and visibly
increased barrier contact. Its saved replay run is
`5c4ede5e367a4cca85714dd0313a8c85` under `visual_replays/`.

The next recovery reduces actor learning rate and action drift again. A
finishing candidate with more than two extra barrier-contact steps per lap now
immediately restores the verified actor, even if it is faster. Promotion of a
faster candidate also requires at most two
extra barrier-contact steps per lap relative to the champion. These are
track-independent evaluation gates; they do not prescribe controls or a route.
The next budget is 2,308,498 decisions, retaining the original session total.

## Resume exploration correction

The contact-guard recovery at step 1,577,700 still repeated the first three
training attempts from the previous recovery exactly: a 33.960-second roll
failure, then 36.014- and 35.851-second finishes. The fourth attempt was also
slow. The GUI stopped safely at step 1,583,345 to correct this repeated
exploration. GRTQC resume now seeds its stochastic learner with the configured
seed plus the saved training step, as PPO already did. Evaluation remains
deterministic with its fixed evaluation seeds; replay, actor/critic weights,
and champion gates are unchanged. This changes the next training attempts but
does not itself establish a faster policy.

The next GUI continuation uses **Recover champion actor** and a 2,302,853-
decision budget. The newest saved replay run remains local under
`models/simple-track/grtqc/visual_replays/` and can be selected for swarms.

Reseeding was active at step 1,583,345 (`resume_rng_seed` was recorded in the
new run log), yet the first attempts still took the same 33.960, 36.014 and
35.851 seconds. The run stopped safely at step 1,589,550. Reseeding alone did
not fix the repeated barrier hits; the earlier inference was too strong.

The deeper issue is rollout collection in scratch pace mode. It still called
SAC's stochastic action sampler, ignoring the configured zero exploration
standard deviation. On representative saved champion states, repeated
stochastic predictions had steering standard deviation up to about 0.18 and
longitudinal-control standard deviation up to about 0.23. This is large enough
to make live training look much more aggressive than deterministic champion
evaluation. The next recovery experiment uses deterministic actor output plus
small, correlated action noise: 0.005 while collecting for a locked critic and
0.02 after actor unlock, with 0.95 correlation. New scratch models and pace
models without an explicit positive exploration standard deviation retain their
previous stochastic behavior. Critic targets and actor gradients remain
stochastic; only environment collection changes. GRTQC resume also reseeds
its custom exploration generator. The next budget is 2,296,648 decisions.

The fastest saved training lap (`episode-000165.npz` in run
`5e77f00d8849497b9ed21143405c67e0`) took 33.494 seconds over about
2,009 metres of credited route progress. A 30.000-second lap would require
about 11.6% more average progress per second on the same route. The recorded
lap spent 5.01 seconds in its first 10% of progress, then roughly 2.94–3.36
seconds per 10% section. This is one training lap, not a verified split or
proof that any single section accounts for the pace gap.

## Champion line repair and promotion gate

The smooth-rollout recovery stopped at step 1,600,791 after a one-lap
deterministic screen failed at 90.2% progress. Its first three training laps
were 35.215, 35.786, and 35.215 seconds; the repeated 35.215 was the actual
champion actor's deterministic behavior with the line stored in champion
metadata. A direct five-lap evaluation of that checkpoint and stored line
finished 5/5 at 35.215 seconds with 50 barrier-contact steps, contradicting
the stored 33.591-second, five-contact verification.

The checkpoint policy itself was sound. Its line before the step-1,064,134
promotion was preserved in `checkpoints/step-1056198/metadata.json`. An
isolated copy of the champion checkpoint using that earlier 34.158-second
reference line again finished 5/5 at 33.591 seconds with five contact steps.
The bad champion metadata was backed up under
`archive/before-champion-line-repair-20261012T164200Z/`; champion metadata
now uses the five-lap reverified earlier line. Actor weights, critic weights,
replay, and recorded evaluation remain unchanged.

The bug was in line promotion: candidates with any barrier contact could
replace the racing line without first re-evaluating the actor on that new
line. This was possible because the earlier saved-line check required zero
contacts, while the champion promotion rule allowed a small number. Every
five-lap candidate that could beat the champion now receives a deterministic
five-lap check with the proposed line. The new line is kept only if it improves the result without
excess additional contact; otherwise a faster actor can still be promoted
with the existing, actually verified line. The 30-second target remains unmet.

The repaired continuation resumed from step 1,600,791 with 2,285,407 decisions
remaining in the original session budget. Its first three training laps exactly
matched the restored 33.591-second champion, with one contact step each. At
step 1,610,903, the candidate passed a one-lap screen at 33.418 seconds and
then a deterministic five-lap evaluation at **33.418 seconds, 5/5 finishes,
zero barrier-contact steps**. The proposed new line failed its own five-lap
check at 35.950 seconds with 30 contact steps. The new gate discarded that
line and promoted only the faster actor with the still-verified existing line.
The protected champion is now 33.418 seconds. Training continues toward 30.

The next two deterministic five-lap candidates, at steps 1,613,159 and
1,615,415, measured 33.775 and 33.753 seconds respectively, with zero barrier
contacts. Both were rejected as slower than the protected champion. After two
weak evaluations the actor was restored from champion again; critics and the
250,000-transition replay were retained. Training is continuing from step
1,615,415 under the remaining original budget. The associated new visual
replay session is `models/simple-track/grtqc/visual_replays/36ce837fd80a494cb157a6fe0634f3b2/`;
its index currently contains eight episodes for replay and swarm selection.

The first wider-search continuation started from step 1,623,680 with the
250,000-transition replay intact. The actor reached the 0.10 reference-drift
limit. Its first deterministic five-lap check was 33.802 seconds; after another
search interval it improved to 33.627 seconds, still slower than the champion.
Both checks had zero contact steps, and the second weak check restored the
protected actor. A second bounded-search experiment now raises the cumulative
reference-drift limit to 0.20 while retaining the 0.005 per-update action cap,
8-update actor interval, five-lap gate, and barrier-contact promotion rule.
Its purpose is to let the actor test a meaningfully wider policy region; any
faster verified champion remains protected by the evaluation and rollback
gates.

The 0.20-search continuation resumed at step 1,637,992 with 2,248,206
decisions remaining and the full replay buffer retained. At step 1,642,817 the
actor had unlocked after critic recovery and reached 0.114 reference drift.
Its first four live laps were 33.418, 33.418, 33.823, and 33.393 seconds, all
without barrier contacts. These are training laps, not a verified improvement;
the deterministic champion check is still pending. Its visual replay session
is `models/simple-track/grtqc/visual_replays/24353014d37a43838643c6ad3e5f79ce/`.

The deterministic checks for that search measured 33.668 seconds at step
1,648,104 and 33.765 seconds at step 1,650,360, each 5/5 with zero barrier
contacts. Both were slower than the 33.418-second champion, so the actor was
restored again. Since the actor reached the 0.20 cumulative drift bound after
63 Adam updates, the next recovery experiment increases the cumulative limit
to 0.40; its 0.005 per-update cap, evaluation cadence, and promotion gates
stay in place.

The 0.40 experiment began at step 1,652,237 with the protected actor restored.
Its deterministic five-lap results at steps 1,662,333 and 1,664,589 were
33.764 and 33.781 seconds with zero barrier contacts. Both were rejected and
the champion actor was restored, with critic and replay retained. The actor
reached the 0.40 cumulative limit before the first check. The next recovery
widens that cumulative limit to 0.60, keeping the same 0.005 per-update action
cap and deterministic contact and pace gates.

The 0.60 search was stopped at step 1,681,373 after the candidate began
contacting the barrier repeatedly in the last 1.5% of the lap. Its five-lap
check at step 1,677,198 was 33.579 seconds with five barrier-contact steps;
it was slower than the protected 33.418-second champion and was rolled back.
Three subsequent clean laps on the restored actor were recorded before the
clean stop. To reduce how forcefully the next recovery moves away from the
champion, the next attempt caps cumulative reference drift at 0.40 and lowers
post-verification action exploration from 0.02 to 0.005. The champion,
critics, and replay buffer remain preserved.

The first attempted cap reduction was ineffective: recovery used `max(saved,
0.40)`, so a Latest checkpoint that already stored 0.60 continued to train at
0.60 even while the GUI status claimed 0.40. The run was stopped safely at step
1,690,426. Its eight recorded training laps had four finish-line barrier
contacts, including the last three consecutive laps; telemetry reached 0.60
cumulative action drift. The champion remains 33.418 seconds. Recovery now uses
the smaller of the saved limit and 0.40, so this checkpoint cannot silently
retain its old, wider limit. The 250,000-transition replay and critics are kept.

The corrected 0.40 run repeatedly produced slower deterministic checks and
occasionally reintroduced barrier contacts; its last recorded five-lap check
at step 2,021,914 was 33.657 seconds with five barrier-contact steps. The
protected 33.418-second champion was not replaced. The worker later failed to
save at step 2,030,419 because the drive ran out of space. To keep recovery
conservative, subsequent champion recovery is capped at 0.20 cumulative
reference drift; the five-lap contact and pace gates remain unchanged. Older
checkpoint replay buffers can be pruned while preserving each checkpoint's
policy and metadata. The last complete checkpoint's replay buffer is retained
for resume; the newest incomplete checkpoint is kept for diagnosis.

The 0.20 recovery resumed from Latest at step 2,021,914 with its 250,000-item
replay and critics intact. Its deterministic five-lap checks at steps 2,032,010
and 2,034,266 were 33.770 and 33.669 seconds, each with five finishes and zero
barrier-contact steps. Both were slower than the protected 33.418-second
champion; the second check restored the champion actor while retaining critics
and replay. The run is continuing under the 0.20 cumulative drift cap.

The previous session's 345 visual replays (steps 1,700,506–2,030,426, with
recorded observations) remain available. The current continuation has also
written six indexed replay episodes with observations, actions, transforms,
and HUD frames; the indexed payloads load and validate. Full finished laps in
this new session include 33.592, 33.757, 33.730, 33.746, and 33.418 seconds.
These are training replays, not deterministic champion verifications.

The next five-lap candidate check at step 2,044,378 measured 33.712 seconds
with five finishes and zero barrier-contact steps. It remains slower than the
champion and is the first weak check in the current recovery cycle; the
two-check rollback gate is still active. The live continuation has since
recorded 13 indexed replay episodes.

The second five-lap check after that recovery, at step 2,046,634, measured
33.943 seconds with zero barrier-contact steps. It was the second weak check,
so the actor was restored from the protected champion; critics and replay
remain in place. The active replay index has since grown to 19 episodes. A new
recovery cycle is active from the restored champion; at step 2,050,923 it had
reached 0.044 cumulative reference drift, before its next candidate check.

That cycle's five-lap check at step 2,056,746 measured 33.674 seconds with
five finishes and zero barrier-contact steps. It is the first weak check in
this cycle, so the two-check rollback gate remains active. The indexed replay
session has grown to 26 episodes through step 2,056,746.

At step 2,059,002 the next five-lap check measured 33.750 seconds with five
finishes and five barrier-contact steps. The contact gate restored the
protected champion actor immediately while keeping critics and replay; a
1,000-update critic cooldown began. Training continued from Latest. The replay
index has grown to 30 episodes through step 2,060,116.

The same 0.20 recovery cycle was stopped cleanly at step 2,067,759 after the
actor again reached the 0.20 drift ceiling. The preceding five-lap evaluation
at step 2,059,002 had barrier contact on all five laps and triggered actor
rollback, so this repeated search was paused before another candidate check.
Earlier 0.10-bound checks were slower than champion but had zero contacts;
recovery is therefore capped at 0.10 for the next continuation. Latest was
saved at the stop point, retaining critics and replay, and the 33.418-second
champion remains protected.

The GUI resumed Latest at step 2,067,759 with a fresh 1,000,000-decision
budget. Champion actor recovery is now limited to 0.10 cumulative reference
drift; the run log confirms the actor was restored from Champion and the saved
critics and replay were retained. Training is active on CUDA.

The first five-lap check under the 0.10 cap, at step 2,077,871, measured
33.586 seconds with five finishes and zero barrier-contact steps. It is slower
than Champion, so the actor was kept only for the second weak-evaluation check;
the protected actor and pace gate remain intact. The active replay run
`350a625c0301432fb4bcca32af84ac57` has recorded two episodes with observations
through step 2,078,999.

The next five-lap check at step 2,080,127 measured 33.728 seconds with zero
barrier contacts. Both checks in this 0.10 cycle were slower than Champion, so
the two-weak-check gate restored Champion again while preserving critics and
replay. The 1,000-update critic cooldown has restarted. Replay run
`350a625c0301432fb4bcca32af84ac57` continues recording the active training
episodes for later swarm playback.

The saved Pace reward target was still 32 seconds despite the 30-second goal.
The GUI Target lap time was changed to 30 seconds; progress, speed, finish,
barrier, failure, and other Pace coefficients were left unchanged, and no new
reward profile was created. Latest confirms `finish_target_s: 30.0`. Continuing
Latest with selected rewards correctly refilled replay, but its first startup
did not restore a champion reference actor. That short attempt was stopped
after three clean training laps at step 2,089,369. It was restarted through
Recover champion actor, retaining the newer critics and 30-second reward data
while restoring the verified actor. The new log `20261013T180925Z.jsonl`
confirms `rollback_on_regression: true`, the 0.10 reference limit, and 512
sampled champion guard states. The one-million-decision run is active; the
champion remains 33.418 seconds.

Under the corrected 30-second finish target, the first deterministic five-lap
check at step 2,099,481 measured 33.848 seconds, with five finishes and zero
thresholded barrier-contact steps. It is the first weak check in this cycle;
the candidate remains subject to the second-check rollback gate. The
33.418-second Champion is unchanged, and training continues from Latest.

The second five-lap check at step 2,101,737 measured 33.632 seconds, again
with five finishes and zero thresholded barrier contacts. This is 0.216 seconds
faster than the first candidate check but remains slower than Champion; the
two-weak-check guard therefore restored Champion and retained critics and
replay. The following run was stopped cleanly at step 2,106,958 with Latest
saved. The verified Champion remains 33.418 seconds, 5/5.

Inspection of the run's episode telemetry found non-landing collision impulses
peaking between 73 and 152 near the finish. The selected reward threshold was
250, so those impacts were logged as collisions but did not activate the
barrier-contact or speed-loss penalties. The two-second landing grace already
filters touchdown impulses. For the next run, the saved training configuration
`profiles/training/simple-track-grtqc-contact-aware-30.json` lowers the barrier
threshold to 50 and raises the raw contact penalty from -100 to -600, while
keeping the 30-second target and other reward terms unchanged. It keeps the
same 1,000,000-step budget, frame skip 30, five-lap evaluations, and conservative
actor recovery limits.

The automatic visual replay run
`7450c46351274490804f6e6d0fc78ecc` contains six complete laps from the
candidate and restored Champion actors, plus screening and stopped partial
episodes. The cleanly stopped checkpoint remains available in Latest; Champion
was not overwritten.

The contact-aware configuration was started from Latest at step 2,106,958 on
CUDA with a fresh replay buffer, a 1,000,000-decision full-track budget, and
Champion rollback enabled. The verified Champion actor and Latest critics were
loaded, and the 512-state reference guard was captured from Champion. The first
nine training laps remained around 33.4 seconds. One lap recorded a 120.6
non-landing impulse at 99.0% progress and correctly applied the -600 contact
term. After the 1,000-update critic cooldown, the actor unlocked; its first
one-lap screen measured 33.787 seconds. Five-lap confirmation is pending, so
Champion remains unchanged. The active log is
`logs/simple-track/grtqc/20261013T182820Z.jsonl`.

The run's visual replay directory is
`models/simple-track/grtqc/visual_replays/3ee7a78fcc5b4fa4b19d102bc75ab509`.
It contains two completed attempts and one short interrupted attempt, with full
telemetry for swarm playback.

## Finish-line contact rollback correction

The five-lap check at step 2,119,294 measured 33.801 seconds and recorded one
barrier contact on each lap, all at 98.81% progress (peak impulse 56.9). This
confirmed a repeatable finish-line collision; the preceding evaluation at step
2,117,022 was 33.787 seconds with zero contacts. The slower candidate was
rejected, and after the second weak check the actor was restored from Champion
while retaining the trained critics and replay. The protected Champion remains
33.418 seconds, five laps, zero contacts.

The scratch promotion gate had allowed up to two extra contacts per lap, and
its contact rollback waited for more than three contacts per lap. That left a
repeatable one-contact-per-lap regression below the rollback threshold. The
gate now compares contact rate directly with Champion: a candidate can only
promote if its verified five-lap contact rate is no worse, and a reliable
contact regression restores the Champion actor immediately. This does not
prescribe a track-specific action or change the reward profile. Latest was
saved at step 2,119,294 with Champion's actor restored; its critics, replay,
and -600 contact penalty remain available for continued training.

The guarded restart used the **Recover Champion actor with selected rewards**
action. The first attempted Continue Latest session had rollback disabled, so it
was stopped safely after one clean 33.418-second lap at step 2,120,741. Recovery
then restarted from Latest with rollback enabled, no reward change, and replay
preserved. Its startup log `20261013T184429Z.jsonl` confirms CUDA/T500, a
1,000,000-decision budget, Champion actor restoration, 1,114 reference
observations, and 512 guard states. The first nine training laps were
33.418–33.937 seconds with no thresholded contacts.

At step 2,130,837, the first post-unlock deterministic screen measured 33.860
seconds with one contact at 99.37%. The full five-lap evaluation reproduced
that contact on all five laps. The new gate rejected the candidate and emitted
`contact_regression_rollback`, restoring the Champion actor while keeping
critics and replay. Latest remains at this recovered state while training
continues. Champion is still 33.418 seconds, 5/5, zero contacts. The active
replay session is
`models/simple-track/grtqc/visual_replays/b039f710069640bfb71aa64d2bf128d5`;
it currently contains 14 attempts, including 12 complete laps, for later swarm
use.

The next five-lap evaluation at step 2,140,917 was 33.825 seconds with zero
contacts. The candidate was rejected as slower than Champion and recorded as
the first weak pace check. At step 2,143,173, a second five-lap evaluation was
33.765 seconds, also with zero contacts. After the second weak check, the
Champion actor was restored again while the critics and replay were retained.
Training remains active from this recovered checkpoint; the Champion remains
33.418 seconds, 5/5, with zero contacts. Late contacts appeared in two
exploratory training laps between evaluations and received the -600 term, while
both deterministic evaluations stayed contact-free. No 30-second lap has been
verified yet.

## Stronger contact shaping after repeat contacts

The run later reached step 2,160,421. Episodes 27, 28, and 31 incurred the
-600 contact term, and a five-lap evaluation at step 2,155,525 showed a contact
on every lap near the finish. The contact gate rejected that actor and
restored Champion. Additional exploratory contacts after rollback show that
-600 was not discouraging the risky behavior enough. The next continuation
raises the contact penalty to -1,500, restores the Champion actor again, and
keeps Latest critics while rebuilding replay under the changed rewards. Pace
recovery allows five consecutive clean-but-slower checks before restoring the
actor; verified contact regressions still roll back immediately. The budget
remains 1,000,000 decisions, and promotion still requires the deterministic
five-lap gate. The protected Champion remains 33.418 seconds, 5/5, with zero
contacts.

Recovery was started from the GUI at 19:16 UTC on 2026-10-13. The startup
record `logs/simple-track/grtqc/20261013T191636Z.jsonl` confirms CUDA on the
NVIDIA T500, resume from Latest at step 2,160,421, a fresh replay buffer, and
champion rollback protection enabled. Latest metadata now records the -1,500
barrier-contact penalty and five weak pace evaluations. The first deterministic
five-lap evaluation at step 2,170,485 finished 5/5 at 33.883 seconds with zero
barrier contacts. It was rejected as slower than Champion and counted as weak
check 1/5; Champion remains unchanged. Two exploratory training episodes had
one and two late contacts near 98.9% progress and received the stronger contact
penalty. The second deterministic five-lap evaluation at step 2,172,741 was
33.962 seconds with zero contacts (weak check 2/5). The next three training
episodes completed at 33.857, 33.690, and 33.648 seconds without contacts.
The third deterministic five-lap evaluation at step 2,174,997 was 33.743
seconds, again 5/5 with zero contacts (weak check 3/5). The next training lap
was 33.738 seconds with zero contacts. Champion remains 33.418 seconds and
unchanged. The fourth deterministic five-lap evaluation at step 2,177,253 was
33.683 seconds, 5/5 with zero contacts (weak check 4/5). The new replay session
has nine entries so far, six complete and three short interrupted evaluation
captures.
Visual replays remain in
`models/simple-track/grtqc/visual_replays/b039f710069640bfb71aa64d2bf128d5`
for swarm playback. The new recovery session is
`models/simple-track/grtqc/visual_replays/8983e2d52ba84229bcfc52fc2b5e1a77`;
it has nine replay entries, including clean evaluation telemetry and finished
training laps.

## Tighter actor trust limits after a contact regression

The fifth five-lap check at step 2,179,509 regressed to 33.941 seconds and
recorded ten barrier-contact steps across five laps, with peak impulse 149.1.
Each lap had contacts at 93.66% and 99.09% progress. The gate rejected the
candidate and restored Champion immediately. The next training lap completed
cleanly at 33.418 seconds. Training was then stopped safely at step 2,181,866;
Latest retains the critics and replay, while Champion remains 33.418 seconds,
5/5, zero contacts. The recovery profile now lowers actor LR to 0.000002,
updates the actor every 16 critic updates, caps each action update at 0.0025,
caps reference drift at 0.05, and halves the smooth exploration noise to
0.0025. The -1,500 contact penalty, five-evaluation pace recovery window,
immediate contact rollback, 1,000,000-decision session budget, and existing
five-lap promotion gate remain in place. The completed run's visual replay
session contains 15 entries (10 finished, 5 short interrupted captures).

The tightened recovery was relaunched from the GUI at step 2,181,866 on the
NVIDIA T500. Startup record `logs/simple-track/grtqc/20261013T193818Z.jsonl`
confirms Champion actor restoration, preserved Latest critics and 55,302 replay
transitions, fresh replay disabled, and Champion rollback enabled. The first
episode is collecting with the reduced actor update limits; the profile keeps
the 1,000,000-decision full-track budget.

That run stopped cleanly at step 2,191,175 with Latest saved and 64,609 replay
transitions. Episodes 2 and 4 hit the barrier at exactly 99.23% progress, with
peak impulses 222.9 and 1,863.6. The other six finished, but the last four
slowed from 33.616 to 33.843 seconds. No evaluation occurred: the actor's
2,000-decision evaluation interval was only activated after the ordinary
10,000-decision evaluation, while scratch-origin runs did not schedule an
immediate check when the actor unlocked. Thus the actor reached its 0.05
reference-drift limit without a screening evaluation. This was a scheduling
gap, not a failed five-lap Champion gate.

The runner now schedules the first actor evaluation on unlock for scratch and
transferred GRTQC alike. If the one-lap screen records any barrier contact
while the verified Champion has zero contacts, it immediately restores the
Champion actor while retaining critics and replay. This catches the repeatable
finish contact before several thousand more actor decisions; all Champion
promotions still require the existing deterministic five-lap gate.

The action path was checked after a report that controls often read 0.97. The
policy action space and PWM controller both accept the full [-1, 1] range; no
0.97 clamp exists. On a saved 1,114-decision Champion lap, predicted steering
and longitudinal actions reached 0.9998 and 1.0000, and 31%/39% of samples
were at least 0.97 in absolute value. A 0.97 demand is therefore a real 97%
PWM duty request, not a hidden scale limit. We did not force controls to 1.0;
the policy should choose full demand only where the verified pace/contact gate
supports it.

The stopped run did not create a new visual replay index, despite automatic
WebSocket replay recording being enabled. This is being checked on the next
run; older replay sessions remain available, including the 15-entry recovery
session `8983e2d52ba84229bcfc52fc2b5e1a77`.

## Guarded restart and replay capture

After the runner correction, the GUI resumed Latest at step 2,191,175 on the
NVIDIA T500. It restored Champion's actor, kept Latest's critics and 64,610
replay transitions, and kept the one-million-decision full-track budget. The
first three training laps exactly matched Champion at 33.418 seconds with no
contacts. The fourth was 33.506 seconds and recorded a thresholded contact at
98.81% (six impulse steps, peak 80.7); the -1,500 contact penalty applied.
The next laps were 33.590 and 33.819 seconds without contacts.

The new unlock scheduling worked: the actor was held for a screen after its
critic cooldown. At step 2,195,639, the full deterministic five-lap evaluation
finished 5/5 at 33.759 seconds with zero contacts. It remains slower than the
33.418-second Champion and counted as weak pace check 1/5; no promotion
occurred. The next one-lap screen, at step 2,197,895, measured 33.603 seconds
with zero contacts. Its deterministic five-lap confirmation also measured
33.603 seconds, 5/5 finished with zero contacts (weak pace check 2/5). The
following training lap was 33.763 seconds without a contact; training continues
with actor reference drift at 0.0495 of the 0.05 limit. Champion remains
protected.

The active visual replay session is
`89c98caff4bf4e8f86fa49dd938e82e9`. Its index currently has four entries,
three completed and one interrupted. The compressed observation and HUD
payloads arrived asynchronously several minutes after the corresponding laps;
the indexed session is saved for later swarm playback.

## Contact rollback and action range check

The same live run continued past the checkpoint above. At step 2,202,407 the
actor's one-lap screen completed in 33.716 seconds but recorded a barrier
contact at 99.23% progress (peak impulse 886.9). The deterministic five-lap
confirmation also measured 33.716 seconds and recorded contacts on all five
laps. The candidate was rejected and `contact_regression_rollback` restored
the Champion actor while retaining the critics and replay. At that point the
trainer was active again at step 2,202,728. The protected Champion remained
33.418 seconds, 5/5, with zero contacts.

To quantify the reported near-maximum control values, the seven finished
episodes in replay session `89c98caff4bf4e8f86fa49dd938e82e9` were inspected
(7,857 policy decisions). Steering reached both -1 and +1; 31.2% of decisions
had absolute steering demand at least 0.97. Signed longitudinal demand reached
1.0 but only -0.910 on braking; 41.9% of decisions requested throttle at or
above 0.97. Values within 0.965–0.975 were only 1.4% positive and 2.0% negative
for steering, and 2.7% positive for longitudinal demand. The action adapter
has no 0.97 ceiling: it accepts [-1, 1] and converts continuous requests to
pulse-density controls across the 30-tick frame skip. These observations show
frequent near-maximum requests, but do not establish that replacing them with
full demand would improve a lap; the candidate's finish contact is a concrete
counterexample to accepting pace without the contact gate.

At the time of this check, the replay index contained seven finished training
laps, including 33.698- and 33.746-second laps, plus two interrupted episodes.
This session remains available for swarm playback. Training was still running;
no model or control code was changed during this diagnosis.

Subsequent guarded actor checks remained clean but did not improve on
Champion. At step 2,206,871, the five-lap result was 33.479 seconds, 5/5, with
zero contacts; at step 2,209,127, it was 33.547 seconds with the same clean
completion rate. Both candidates were rejected for Champion promotion, and
the clean learning anchor was refreshed. The 33.418-second Champion remains
protected. At step 2,209,500 the trainer was active and collecting another
episode. The replay index for this session had 14 records (ten finished,
four interrupted); the finished training runs remain available for swarm
playback.

At step 2,211,383, another deterministic five-lap candidate evaluation
finished at 33.589 seconds, 5/5 with zero contacts. It was rejected as slower
than the protected 33.418-second Champion, and the clean learning anchor was
refreshed. Training continued; at step 2,211,816 it was collecting episode
19. The visual replay session grew to 20 indexed episodes, with 14 finished
laps and six interrupted entries, preserving the additional laps for swarm
playback.

At step 2,215,895, the next five-lap evaluation was 33.699 seconds, 5/5 with
zero contacts. This was the fifth consecutive clean pace check slower than
Champion, so the configured weak-evaluation recovery restored the 33.418s
Champion actor while preserving the critics and replay. The protected
Champion metadata remains at 33.418 seconds, 5/5, zero contacts. Training
resumed from the recovered policy; at step 2,217,092 the actor was locked for
critic recovery with 701 updates remaining. The visual replay session now
contains 27 indexed records (19 finished and eight interrupted) for later
swarm use.

After critic recovery, the first five-lap evaluation at step 2,220,359 matched
Champion exactly at 33.418 seconds, 5/5 with zero contacts. Since it did not
beat Champion, it was not promoted and counted as the first weak pace check
after recovery. The protected checkpoint remains unchanged. Training
continued past step 2,221,364, and the replay index grew to 32 records (23
finished and nine interrupted), retaining the additional clean laps for
swarm playback.

The following candidate at step 2,265,287 screened at 33.964 seconds and
confirmed with ten barrier-contact steps across five laps; it was rejected
and the Champion actor was restored. After critic recovery, full clean checks
at steps 2,269,751 and 2,272,007 measured 33.662 and 33.610 seconds,
respectively (each 5/5, zero contacts). Both remain slower than Champion and
were not promoted. At step 2,273,800 the GUI trainer was still active; the
visual replay index held 96 records (69 finished, 27 interrupted).

Two subsequent candidates remained clean but slower: step 2,274,263 at
33.747 seconds and step 2,276,503 at 33.604 seconds, each 5/5 with zero
contacts. Neither replaced the Champion; training continued with the actor
unlocked. At step 2,277,373, the live process was active. Replay storage had
reached 100 indexed episodes (72 finished and 28 interrupted), retaining
over 70 completed laps for swarm playback.

Further confirmations stayed clean but did not surpass Champion. At step
2,278,759 the candidate was 33.610s (5/5, zero contacts). A later check at
2,283,239 tied Champion at 33.418s (also 5/5, zero contacts) and therefore
was not promoted. Subsequent clean checks were 33.655s at step 2,285,495,
33.704s at 2,287,751 and 33.763s at 2,290,007. The actor remains bounded by
the 0.05 action-drift guard; training was active at step 2,291,080. Replay
storage had 117 records at that point (84 finished, 33 interrupted).

The next post-recovery candidate checks were 33.562, 33.704, 33.673 and
33.634 seconds; all four completed five laps without contacts and remained
slower than the 33.418-second Champion. The fifth weak check (33.634 seconds
at step 2,229,383) restored Champion again while preserving critics and
replay. At step 2,231,984, training was active in critic recovery with 350
updates remaining. The replay index had grown to 45 records (32 finished, 13
interrupted).

The actor diagnostics also exposed an effective-update bottleneck. Before the
next evaluation was pending, several progress records (steps 2,226,647–
2,226,856) showed cumulative action drift at 0.0499999 against the 0.05 limit,
proposed action changes around 0.01, but zero executed actor update and zero
parameter update. The saturation metric remained false. This indicates the
actor could not use its proposed gradient step while the cumulative guard was
exhausted. As a single-parameter follow-up, the training profile now sets the
cumulative drift limit to 0.08; the 0.0025 per-update action limit, evaluation
cadence, and contact rollback remain unchanged. This setting is not live in
the already-running GUI process; it requires a safe stop and restart from
Latest. The champion remains protected, and no runtime model state has been
changed by this profile edit.

The subsequent five-lap check at step 2,233,863 again matched Champion at
33.418 seconds, 5/5 with zero contacts. It was not promoted because it did not
beat the protected time; the post-recovery weak-check count reset to one.
Training continued to step 2,234,452. At the last replay-index inspection,
the session contained 48 records (34 finished, 14 interrupted). The profile
edit is committed, but the active GUI run still uses its loaded 0.05 drift
limit until it is safely stopped and restarted from Latest.

The next five-lap screen/evaluation at step 2,242,855 measured 33.854 seconds
and 15 barrier-contact steps across five laps (the one-lap screen had three
contacts and a peak impulse of 513.8). The candidate was rejected and the
contact-regression rollback restored Champion. Since this occurred with the
existing 0.05 limit, it does not prove what a looser limit would do, but it
leaves no evidence to justify widening the actor's exploration now. The 0.08
follow-up was never applied to the live process and has been reverted in the
profile to 0.05. At step 2,244,532, the active run remained in
critic recovery with 581 updates left. Champion is still 33.418 seconds,
5/5, zero contacts; the replay index contains 61 records (43 finished, 18
interrupted).

At step 2,247,335, the five-lap candidate check measured 33.599 seconds,
5/5 with zero barrier-contact steps. It was clean but slower than the
protected 33.418-second Champion, so it was not promoted. The next candidate
at step 2,249,591 measured 33.712 seconds, 5/5, with two contacts on the
screen lap and ten barrier-contact steps across confirmation. The contact
gate rejected it and restored the Champion actor. Training continued in
critic recovery; at step 2,250,252 the GUI trainer was active with 835
recovery updates remaining. The visual replay index had 68 episodes (48
finished, 20 interrupted), including the recent laps.

Action-range diagnosis on the saved 33.414-second Champion replay
(episode-000064) found no action-scale cap: steering reached +1.0 and
-0.9989, and signed longitudinal demand reached +1.0. About 31% of steering
decisions had magnitude at least 0.97 and 39% of longitudinal decisions were
at least +0.97. The continuous PWM adapter converts a 0.97 duty request into
almost all active ticks in its 30-tick frame; a 1.0 request fills every tick.
This small remaining gap does not justify forcing near-maximum values to
full input, which would change the controller's learned action mapping and
could increase barrier contacts. The current pace limitation remains lap
quality and safe line execution, not inability to command full throttle or
steering. No control or reward code was changed.

The next post-recovery candidate at step 2,254,055 screened at 33.728 seconds
with one contact step, then confirmed at 33.728 seconds, 5/5 with five
barrier-contact steps. The contact gate rejected it and restored Champion
again. The live GUI trainer remained active in critic recovery immediately
after rollback. This is further evidence that the remaining pace gap is not
caused by a hard action ceiling: bounded actor changes are currently failing
the contact gate before they improve the 33.418-second protected lap.

After that rollback and critic recovery, three successive actor screens
confirmed clean full five-lap results at steps 2,258,519 (33.776s), 2,260,775
(33.514s), and 2,263,031 (33.464s), all with 5/5 finishes and zero barrier
contacts. Each was rejected as slower than Champion, while the clean anchor
was refreshed and training continued. The newest result narrows the gap to
0.046 seconds but does not replace the protected 33.418s checkpoint. At step
2,263,556 the GUI trainer was active again. The replay index held 84 records:
60 finished laps and 24 interrupted episodes, preserving the new runs for
swarm playback.

The repeated bounded actor cycles still failed to beat the 33.418-second
Champion, so the next controlled experiment increases only post-verification
actor rollout exploration from 0.0025 to 0.01. Reward settings, actor update
limits, cumulative drift guard, critic collection noise, replay, and all
evaluation and rollback gates remain unchanged. This tests whether the prior
noise was too small to sample actions near saturation; it does not force
controls to full input. The GUI's ordinary Resume latest restores saved
training settings, so a dedicated “Continue latest with selected
exploration” action was added to override only this value while preserving
Latest's critics and replay.

The old run was stopped cleanly at step 2,298,544 after saving a finished
33.5-second episode, and Latest was verified at that step. The new GUI loaded
the updated Simple Track profile and resumed Latest on CUDA (NVIDIA T500)
with `fresh_replay: false`, selected exploration 0.01, and a 1,000,000-decision
session budget. Its first deterministic check at step 2,299,680 measured
33.423 seconds, 5/5 with zero contacts. It was not promoted because it is
0.005 seconds slower than Champion; actor updates have now resumed under the
new exploration setting. This first check is a clean baseline, not evidence
that the new setting improved pace. Champion remains 33.418 seconds, 5/5,
zero contacts.

The next screen at step 2,301,936 was 33.652 seconds and recorded two contact
steps. Its deterministic five-lap confirmation was also 33.652 seconds and
recorded ten contact steps; the contact gate rejected it and restored the
verified Champion actor. Critic recovery then began with a 1,000-update
cooldown. At step 2,304,369, the run was still active with 392 cooldown
updates remaining; actor learning was locked during recovery. The current
Champion remains 33.418 seconds, 5/5 with zero contacts.

The latest captured post-change training lap (step 2,300,801; 33.708 seconds)
was also checked against the report that the policy stops at 0.97. Its saved
raw actions span -1 to +1 for steering and -0.8404 to +1 for longitudinal
demand. About 24.0% of steering actions and 44.8% of longitudinal actions
have magnitude at least 0.97; only 6.0% and 2.3%, respectively, fall in the
narrow 0.965–0.975 band. The policy does request full-scale actions, so these
data do not support a hard 0.97 action cap. The latest check is descriptive,
not a counterfactual proof that full steering or braking would improve pace.
At frame skip 30, a 0.97 demand is approximately 29 active digital ticks per
30, while 1.0 fills all 30.

The 0.01 exploration trial was stopped cleanly at step 2,308,733 after its
second contact-regression rollback. The step-2,306,400 candidate measured
33.677 seconds over five finished laps and accumulated five contact steps;
the preceding candidate at step 2,301,936 measured 33.652 seconds and
accumulated ten contact steps. Both were slower than Champion and failed its
zero-contact gate. Latest preserved the trained critics and replay. The GUI
profile's actor exploration is returned to 0.0025 for the next continuation;
the next run will resume Latest without replacing its replay. This controlled
trial does not establish that exploration alone caused the contacts, but it
does not justify keeping the higher value.

Latest was then resumed from step 2,308,733 on the NVIDIA T500 with the
profile restored to 0.0025 exploration and `fresh_replay: false`. By step
2,310,978 it had completed two full-track episodes, the critic recovery
cooldown had ended, and actor updates were executing again (measured update
norm 0.000103). The replay buffer and visual replay recording remain enabled.
The first post-restart actor evaluation completed 5/5 at step 2,310,989 in
33.620 seconds with zero contacts. It was rejected as 0.202 seconds slower
than Champion; the clean anchor was refreshed and training continued. At
step 2,311,206 the run was active again.

The next candidate at step 2,313,245 screened at 33.550 seconds with one
contact; its five-lap confirmation finished 5/5 at the same median and
recorded five contact steps. The contact gate restored Champion and retained
replay/critics. At step 2,316,530 the trainer was active in its 1,000-update
critic recovery cooldown, with 179 updates remaining.

The deterministic evaluation replay from that five-lap check includes a
33.418-second lap. Its stored actions range from -0.9989 to +0.9998 for
steering, and -0.8936 to +0.99998 for signed longitudinal demand. About
30.6% of steering and 38.7% of longitudinal actions have magnitude at least
0.97; only 5.4% and 2.6%, respectively, lie in 0.965–0.975. Evaluation uses
deterministic policy predictions, so this confirms the actor can request
near-full inputs without rollout noise. The data do not show that full
braking is optimal: negative longitudinal demand did not saturate, and the
five-lap candidate still failed the contact gate.

After that contact rollback and critic cooldown, the step-2,317,709 five-lap
check reproduced Champion exactly at 33.418 seconds, 5/5 with zero contacts.
It was retained as the clean reference rather than promoted; actor updates
resumed afterward. At step 2,318,438 the trainer was active on the next
full-track episode.

The following actor candidate at step 2,319,965 measured 33.642 seconds over
five finished laps and recorded five contact steps; the contact guard restored
Champion again. After cooldown, the 2,324,445 evaluation was clean (33.847
seconds, 5/5, zero contacts) but slower than Champion. The run was stopped
cleanly there with Latest saved. Repeatedly allowing up to 0.05 cumulative
action drift has produced no pace gain and multiple contact regressions, so
the next isolated experiment tightens only that limit to 0.02. Exploration
stays at 0.0025, the per-update cap at 0.0025, and reward, replay, critics,
evaluation cadence and Champion rollback gates remain unchanged. A new GUI
continuation action applies the profile's selected drift limit while keeping
Latest's policy, critics and replay. This is a bounded hypothesis, not an
expected pace improvement. The GUI can invoke the same action automatically
at startup with `--continue-latest-with-selected-actor-guard`, which is useful
for unattended recovery after a GUI restart.

The guarded continuation was started from the GUI on 2026-10-13 using the
1,000,000-decision Simple Track profile. Run log:
`logs/simple-track/grtqc/20261013T221624Z.jsonl`. The runner confirmed
`resume_source=models/simple-track/grtqc/latest`, `fresh_replay=false`, CUDA
on the NVIDIA T500, and the WebSocket bridge on port 8765. Its first finished
training episode was 33.754 seconds with zero collision impulse steps; that
episode was saved under `models/simple-track/grtqc/visual_replays/` for later
swarm use. This is training telemetry, not a deterministic five-lap promotion
result. The 33.418-second protected Champion remains unchanged.

The first deterministic five-lap evaluation of this resumed actor finished all
five laps at a 33.634-second median, but registered five barrier-contact steps
(maximum impulse 92.30). The candidate was rejected under the existing contact
gate, so Champion was not changed. Training continued with the rejected actor;
its next training lap was 33.418 seconds with zero collision impulse steps.
That episode result is not a five-lap verification. The next deterministic
check at step 2,330,045 was 33.418 seconds (5/5, zero contacts), tying Champion
and registering weak-pace check 1/5. The following check at step 2,332,301 was
33.733 seconds (5/5, zero contacts), check 2/5. The actor reached the selected
0.02 cumulative drift boundary by step 2,332,163 with 31 Adam updates; at step
2,334,534 it had made 62 updates and remained just under the boundary. The
30-second target is present in both the selected profile and Latest metadata.
At step 2,334,557 the next five-lap check was 33.703 seconds (5/5, zero
contacts), weak-pace check 3/5. The candidate remained below the protected
Champion in pace, so it was not promoted. The run then resumed automatically;
at step 2,336,813 its five-lap check was 33.697 seconds (5/5, zero contacts),
weak-pace check 4/5. The next check at step 2,339,069 measured 33.773 seconds
and recorded five contact steps, so contact regression restored the verified
Champion actor. The recovery event confirms critics and replay were preserved
and starts the configured 1,000-update critic cooldown. At the latest check,
training was active in cooldown at step 2,339,546 with 881 updates remaining
and the actor locked. Visual replays continue to be recorded. Collect data
after cooldown before changing another training variable.

The second recovery cycle repeated the pattern. After cooldown, the step
2,343,533 evaluation tied Champion at 33.418 seconds (5/5, zero contacts), but
did not promote because it was not faster. The next candidate at step 2,345,789
measured 33.842 seconds with ten contact steps and triggered another rollback
to the Champion actor, preserving critics and replay and restarting the
1,000-update cooldown. After that cooldown, the step-2,350,269 evaluation was
33.731 seconds (5/5, zero contacts), weak-pace check 1/5. At the latest check
the trainer was active at step 2,351,210, episode 24, and its current visual
replay index contained 31 episode records. Latest retains the 30-second target
and 0.02 drift guard; Champion remains 33.418 seconds. The following check at
step 2,352,525 improved the resumed actor to 33.577 seconds (5/5, zero
contacts), weak-pace check 2/5. It remains 0.159 seconds slower than Champion
and was not promoted. The actor is unlocked with 0.0048 cumulative action drift
against the 0.02 limit; training continues. The next candidate at step
2,354,781 screened at 33.594 seconds with one contact at 98.81% progress. Its
five-lap evaluation repeated the contact on all five laps; the gate restored
Champion and preserved critics/replay again. At the latest check training was
active at step 2,357,114 in the 1,000-update cooldown (417 remaining, actor
locked). The current visual replay bundle contains 39 entries.

The next post-recovery evaluation at step 2,359,261 tied Champion at 33.418
seconds (5/5, zero contacts), weak-pace check 1/5. At step 2,361,517, the
five-lap median improved to 33.612 seconds (5/5, zero contacts), still slower
than Champion and weak-pace check 2/5. Training remains active; at step
2,362,619 the actor is unlocked and again near its 0.02 drift limit. The live
visual replay bundle had 45 indexed attempts; Champion remains unchanged. The
next check at step 2,363,773 was 33.674 seconds (5/5, zero contacts), weak-pace
check 3/5. Latest saved that evaluation; by step 2,364,286 training had
resumed, the actor had made 71 Adam updates and was near the drift limit
(0.0196/0.02). The replay bundle has 48 indexed attempts. The 30-second goal
remains unmet; no candidate has passed the faster-than-Champion gate.

The next two five-lap checks under the same actor saturated the 0.02 action
drift limit: step 2,366,029 measured 33.998 seconds (5/5, zero contacts), and
step 2,368,269 measured 33.789 seconds (5/5, zero contacts). Together with
33.674 at step 2,363,773, these were weak checks 3/5, 4/5, and 5/5 after the
previous clean tie. The configured recovery gate restored Champion's actor
while preserving critics and replay. The GUI was closed cleanly at step
2,371,404 with Latest saved and its 247 MB replay intact. The 56-record visual
replay session remains available for swarms.

For the next isolated search, the profile raises only the cumulative actor
action-drift limit from 0.02 to 0.03. The per-update cap remains 0.0025,
exploration 0.0025, actor learning rate 0.000002, actor update interval 16,
1,000,000-decision budget, 30-second target, contact penalty, critics, replay,
five-lap evaluation, and Champion rollback gates remain unchanged. This gives
the policy room for a broader measured adjustment after repeated 0.02-limit
pace plateaus; no improvement is assumed. The next run resumes Latest with
`fresh_replay=false` via the GUI's selected-actor-guard continuation.

The new 0.03 run started from the GUI on 2026-10-14 at saved step 2,371,404.
Startup log `logs/simple-track/grtqc/20261013T230203Z.jsonl` confirms Latest
was restored, the existing replay was kept (`fresh_replay=false`), the full
1,000,000-decision budget was selected, and CUDA initialized on the NVIDIA
T500. At step 2,372,205 the run was actively collecting its first episode;
the prior critic cooldown had 17 updates remaining. The existing 56-record
visual replay session remains intact while the new session begins recording.

The first actor candidate under the 0.03 limit screened at step 2,374,780 at
33.790 seconds with a contact at 99.37% progress (peak impulse 1,880.61). Its
five-lap evaluation repeated that contact on all five attempts, so the existing
gate restored Champion's actor, preserved critics/replay, and started the
1,000-update cooldown. At the latest check, training was active at step
2,376,369 with 603 cooldown updates remaining and the actor locked. This trial
is already showing the larger trust region needs the unchanged contact gate.

After recovery, the clean five-lap checks at steps 2,379,260 and 2,381,516
measured 33.586 and 33.656 seconds respectively, both slower than the protected
33.418-second Champion. Neither was promoted. The second check raised the
weaker-actor count to 2/5. Clean checks reset the local action reference for the
next trust-region window; they do not replace Champion. At step 2,382,413 the
actor was unlocked and had rebuilt nearly all of its 0.03 local action-drift
allowance, but its measured parameter update norm was only 1.7e-7. Latest was
still training and the live replay bundle had 14 indexed episodes, including
the current episode. This small update near the guard is worth monitoring; the
pace target remains unmet and Champion is unchanged.

Action telemetry from the saved training laps confirms the controller does
not have a 0.97 output ceiling: deterministic policy outputs reach both
steering extremes and throttle 1.0. Depending on the episode, about 30–35% of
decisions request steering magnitude above 0.97, and about 39–43% request
longitudinal magnitude above 0.97. These are normalized demands, not a measure
of lost engine power. The adapter converts each demand into 30 digital ticks;
for example, a 0.97 magnitude averages roughly 29 pressed ticks per 30, with
the exact schedule carried across decisions. Forcing every demand to 1.0 would
change steering and throttle behavior globally, so no such remapping was made.

The next five-lap evaluation at step 2,383,756 verified 33.372 seconds, 5/5
finishes, and zero barrier contacts. This is 0.046 seconds faster than the
previous Champion, so it was promoted and saved in the protected Champion slot.
The training replay bundle has 15 indexed episodes, including the completed
33.672-second episode immediately preceding the evaluation; evaluation replay
data was also recorded. Because this trust-region cycle produced a verified
improvement, the planned increase to the 0.03 action-drift limit was not made.
Training remains active from the improved Champion with the existing contact
and five-lap gates. At step 2,384,406 the training process was responsive, the
actor was unlocked, and actor updates had resumed (update norm 8.5e-5; local
action drift 0.02496/0.03). The run's visual replay recorder remained active.

The next candidate at step 2,386,012 measured 33.867 seconds, 5/5 finishes,
and one repeated barrier-contact step per lap at 99.37% progress (peak impulse
840.28). The contact gate rejected it and restored the 33.372-second Champion
actor, preserving critic and replay state and starting the configured 1,000-
update critic cooldown. At step 2,387,101 the GUI was still active, with 728
cooldown updates remaining and the actor locked. The visual replay index had
18 records; the interrupted candidate attempt is retained there as well.

After cooldown, the five-lap evaluation at step 2,390,476 tied Champion at
33.372 seconds (5/5 finishes, zero contacts), so it was not promoted; the
weaker-actor counter restarted at 1/5 and training continued from the
candidate. At step 2,390,747 the actor was unlocked with 0.00995 action drift
within the 0.03 limit. The visual replay bundle had 23 indexed attempts,
including the new evaluation laps and an interrupted attempt. Champion remains
33.372 seconds and training continues toward the 30-second target.

At step 2,392,732 the 0.03-guard candidate measured 33.825 seconds (5/5,
zero contacts) and was weak-pace check 2/5. After another saturated actor
window, training was stopped cleanly at step 2,394,185; Latest saved there and
Champion remained 33.372 seconds, 5/5 with zero contacts. The selected profile
now raises only `actor_reference_drift_limit` from 0.03 to 0.05. The 0.0025
per-update action cap, actor/critic rates, replay, rewards, exploration,
evaluation frequency, five-lap promotion rule, and contact rollback remain
unchanged.

The GUI restarted with `--continue-latest-with-selected-actor-guard` and the
one-million-step budget. Log `20261013T232611Z.jsonl` confirms it resumed
Latest at step 2,394,185 on the NVIDIA T500 with `fresh_replay=false`. By step
2,394,222 cumulative actor drift had moved from its saved 0.03 boundary to
0.03250, with an actor update norm of 9.2e-5, confirming the selected 0.05
limit is active. No evaluation result is established yet. The visual replay
index continues recording; it contains 28 indexed attempts at this check, and
the previous 56-record session remains available.

After the contact rollback and cooldown, the next five-lap evaluation at step
2,399,785 tied Champion at 33.372 seconds (5/5 finishes, zero contacts). It was
not promoted; the weak-candidate counter restarted at 1/5. At step 2,400,216
the actor was unlocked and changing again (0.01748 local action drift, update
norm 8.5e-5 under the selected 0.05 limit). The protected Champion remains
33.372 seconds and training/replay recording continue.

The first post-resume policy check was already due at step 2,394,249, only 64
decisions after restart, and held actor updates until that episode completed.
Consequently, the step-2,395,305 candidate tested the resumed actor at only
0.03250 cumulative drift, not at the selected 0.05 limit. Its five-lap median
was 33.745 seconds with one repeated contact per lap at 99.445% progress (peak
impulse 53.16). Contact rollback restored Champion and preserved critics and
replay, then started the 1,000-update cooldown. At step 2,398,382 training was
still active with 231 cooldown updates remaining; this early check does not
establish the result of a full 0.05-drift actor search.

The next 0.05-window candidate at step 2,402,041 completed five clean laps at
33.488 seconds. This is 0.116 seconds slower than Champion, so it was rejected
and counted as weak-pace check 2/5; no contact rollback was needed. At step
2,402,308 the actor was unlocked again, with 0.00999 drift from the new clean
reference and update norm 1.17e-4. The visual replay bundle contained 10
indexed attempts and continued recording; Champion remains 33.372 seconds.

The subsequent candidate reached the full 0.05 local drift boundary and was
screened at step 2,404,297. Its five-lap median was 33.801 seconds with one
barrier-contact step on every lap at 98.949% progress (peak impulse 64.70).
The contact gate restored the protected 33.372-second Champion actor while
preserving critic and replay state and started the 1,000-update cooldown. At
step 2,406,422 training remained active with 469 cooldown updates remaining;
the failed candidate was not promoted.

After that rollback cooldown, the step-2,408,761 five-lap check was clean at
33.539 seconds (5/5, zero contacts), so it remained below the 33.372-second
Champion and counted as weak check 1/5. At step 2,409,348 the actor was
unlocked and updating from the clean reference (0.02214 action drift, update
norm 1.73e-4). The replay bundle had 18 indexed attempts and remained active.

The next five-lap check at step 2,411,017 completed at 33.703 seconds with
5/5 finishes and zero contacts. This was weak-pace check 2/5 against the
33.372-second Champion. At step 2,411,054 the actor was unlocked and had
resumed updates from its clean reference (0.00250 local action drift, update
norm 1.04e-4). The replay bundle had 21 indexed attempts and continued
recording.

The next check at step 2,413,273 completed 5/5 clean laps at 33.696 seconds,
another slower candidate (weak-pace check 3/5). At step 2,414,186 the actor
remained unlocked and was updating (0.03475 local action drift; update norm
1.41e-4). The replay index contained 24 attempts and recording continued;
Champion remains protected at 33.372 seconds.

At step 2,415,529 the next 0.05-window candidate regressed to 33.828 seconds
and contacted a barrier on all five laps at 98.735% progress (peak impulse
128.04). Contact rollback restored Champion and began the 1,000-update
cooldown, retaining critics and replay. The following one-lap actor screen at
step 2,419,993 tied Champion at 33.372 seconds with no contact; its ordinary
five-lap confirmation also tied at 33.372 seconds (5/5 finishes, zero
contacts). The tie was not promoted. Training resumed from the saved Latest
checkpoint at step 2,419,993; replay capture remains enabled, and session
`1b81b6f8f0ad455795d83d8e23ade0aa` contains the recent training attempts.

The later 0.05-window screen at step 2,422,249 was slower at 33.7 seconds and
contacted the barrier on all five laps at 98.878% progress. The contact gate
restored Champion and began another 1,000-update critic cooldown. After that
rollback, I stopped cleanly at step 2,426,006, changed only the profile drift
limit to 0.04, and resumed the same Latest checkpoint with its 250,000-item
replay, critics, and one-million-decision session budget (`fresh_replay=false`,
CUDA on the NVIDIA T500). The first 0.04-window five-lap evaluation at step
2,429,398 was 33.454 seconds, with the same contact on every lap at 99.233%
progress (peak impulse 1,074.05). It was rejected and Champion was restored;
the 1,000-update cooldown began again. At step 2,431,140 training remained
active with 565 cooldown updates left. The new replay session
`14d4064ea20449b8928e5bd65d27239c` contains its training and evaluation
attempts. Neither candidate improved the protected 33.372-second Champion.

The first progress telemetry after lowering the limit still displayed the old
0.05 cumulative-drift value. Inspecting the saved Latest policy after resume
showed the actual reference drift was approximately 2.8e-6 with an empty actor
optimizer state, and the new actor subsequently reached its 0.04 trust limit.
The stale metric was not evidence that actor updates had been frozen.

## Requested versus applied controls

Inspection of the saved replay `episode-000031.npz` resolves the apparent
0.97 action ceiling. The recorded actor outputs are the requested continuous
controls; the replay HUD also stores the controller demand and the per-window
applied tick fraction. For example, a steering request of -0.981 and a
longitudinal request of +0.99998 were applied as -0.9667 and +0.9667 in that
30-tick frame-skip window: the displayed applied fraction is quantized to
29/30 or 30/30 binary ticks. The deterministic PWM scheduler carries its
fractional remainder into the next window, so this one-window rounding is not
a persistent reduction in input. Across the complete 1,116-decision replay,
mean absolute requested/applied duty was 0.73793/0.73735 for steering and
0.74650/0.74401 for signed longitudinal control. The saved controls therefore
show no 3% action clamp; the average per-lap discrepancy is below 0.25
percentage points. No action remapping was applied, since forcing every
near-bound steering value to exactly 1 could increase overshoot without
evidence of a pace gain.

After cooldown, the step-2,433,862 candidate completed a clean five-lap
confirmation at 33.372 seconds (5/5, zero contacts). It tied rather than beat
Champion, so it was not promoted; the clean reference was refreshed. At step
2,433,983 the actor was unlocked and moving within the new reference window
(0.00499 cumulative action drift of the configured 0.04). Replay capture for
the restarted run remained active with 10 indexed attempts.

The following five-lap check at step 2,436,118 was clean at 33.694 seconds,
0.322 seconds slower than Champion. It was rejected as weak pace, and the clean
reference was refreshed. At step 2,436,744 the actor remained unlocked and was
updating (0.02499 cumulative action drift); the replay session contained 13
indexed attempts and recording continued.

The next clean five-lap evaluation at step 2,438,390 was 33.844 seconds
(5/5, zero contacts), another weaker candidate that was not promoted. At step
2,438,839 the actor remained unlocked at 0.01720 cumulative drift under the
0.04 limit. The visual replay session contained 16 attempts and continued
recording.

At step 2,440,646 the next candidate evaluated at 33.877 seconds (5/5) but
repeated a barrier contact on all five laps at 99.233% progress (peak impulse
1,870.02). The contact gate rejected it, restored Champion, and started the
1,000-update cooldown with critics and replay preserved. At step 2,441,599 the
actor was locked with 762 updates remaining; the new replay session had 19
indexed attempts.

After recovery, the next clean five-lap check at step 2,445,126 was 33.499
seconds, 0.127 seconds slower than Champion, and was not promoted. At step
2,445,907 the actor was unlocked and updating (0.02816 cumulative drift).
Replay capture remained active with 24 indexed attempts.

## Action-range concern and current guarded continuation

The report that controls often show `0.97` does not indicate an actuator cap.
Earlier saved training and deterministic evaluation replays contain steering
and longitudinal requests at or very near both action limits. At frame skip
30, the controller realizes fractional requests with digital PWM ticks, so a
single window may show 29/30 (`0.9667`) even when the requested duty is 0.97;
the fractional scheduler carries the remainder forward. No snap-to-1 action
mapping was added: maximum steering or throttle is not always the fastest
choice, and full steering can increase contact risk.

The current continuation remained live on the T500. At step 2,449,638 its
deterministic five-lap evaluation was clean (5/5, 33.823 seconds, zero barrier
contacts), but weaker than the protected 33.372-second Champion, so it was
rejected without promotion. This is the third weaker actor evaluation toward
the five-result rollback threshold; the candidate was allowed to continue.
At step 2,450,359 the actor was unlocked and updating again (update norm
0.000154, cumulative reference drift 0.02683); the replay archive had 30
indexed attempts. The 0.04 drift guard can temporarily hold actor changes at
its limit while an evaluation is pending, then actor updates resume after the
evaluation refreshes the clean reference. Training remains active, with
Champion and replay data preserved. The 30-second target remains unmet.

The next five-lap actor evaluation at step 2,451,894 measured 33.759 seconds
and one barrier contact per lap, versus the clean 33.372-second Champion. The
contact gate restored Champion and began critic recovery; at step 2,452,999,
724 recovery updates remained and actor learning was locked. The visual replay
archive had grown to 33 indexed attempts. This contact regression is further
evidence against forcing near-maximum controls to exactly 1.0; the policy can
already request full scale, and the failed candidate needs better line quality.

After the contact rollback, critic recovery completed by step 2,456,361 and
actor updates resumed (update norm 0.000095). The next deterministic check at
step 2,456,374 was clean across five laps (33.658 seconds, zero contacts), but
slower than Champion; it was rejected without rollback. At step 2,456,515 the
actor remained unlocked and updating (cumulative reference drift 0.0050), and
replay capture continued. Champion remains 33.372 seconds, 5/5 and contact
free; the 30-second target is still unmet.

After the clean but slower 33.658-second evaluation, the next five-lap
candidate at step 2,458,630 measured 33.861 seconds and repeated two barrier
contacts near the finish (99.16% and 99.23% progress) on every lap, for ten
contact steps total. The contact gate restored Champion and restarted critic
recovery; at step 2,458,715, 979 recovery updates remained and actor updates were
locked. The visual replay archive contained 41 attempts. Champion's five-lap
metadata remains 33.372 seconds with zero contacts; the 30-second goal remains
open.

The next actor evaluation at step 2,463,110 completed five clean laps with a
33.597-second median and zero barrier contacts. It remains 0.225 seconds slower
than Champion, so it was not promoted; the clean reference was refreshed. At
step 2,463,239 actor updates had resumed (update norm 0.000227, cumulative
action drift 0.00490), with training still active. The visual replay archive
contained 46 attempts. This is a clean pace screen, not a verified Champion
improvement; the target remains 30 seconds.

After the clean 33.597-second evaluation, the next five-lap check at step
2,465,366 was 33.627 seconds and recorded one barrier contact on every lap at
98.88% progress (peak impulse 153.45). Although closer in pace than the last
contacting candidate, it was slower than Champion and failed the zero-contact
gate; Champion was restored. At step 2,465,691 critic recovery had 919 updates
remaining and actor changes were locked. The visual replay archive contained
49 attempts. The repeating late-lap contact is now the main observed failure
mode; no steering/throttle remapping or reward change was made while the run's
critic and replay remain trained on the current reward definition.

The actor resumed after recovery but its next five-lap evaluation at step
2,469,814 was slower (33.683 seconds) and again contacted the barrier once on
every lap, this time at 98.81% progress (peak impulse 69.73). The contact gate
restored Champion; at step 2,471,669 the actor was locked with 537 critic
recovery updates remaining. The visual replay archive had 56 attempts. This
repeats the same late-lap failure after a fresh actor trial, so no candidate was
promoted and no reward or action remapping was made.

The following post-recovery evaluation at step 2,474,294 was clean (5/5,
33.734 seconds, zero contacts), but 0.362 seconds slower than Champion; it was
rejected without rollback and the clean reference was refreshed. At step
2,474,479 the actor was again updating (norm 0.000093, cumulative drift
0.00750), and the GUI trainer remained active. The replay archive contained 59
attempts. Champion remains 33.372 seconds; the target remains 30 seconds.

At step 2,476,550 the next five-lap check was clean (33.774 seconds, zero
contacts), but slower than both Champion and the prior clean check; it was
rejected without rollback and refreshed the clean reference. The actor resumed
updates by step 2,476,640 (update norm 0.000138, cumulative drift 0.00248),
with the trainer active and 62 replay attempts archived. The protected
33.372-second Champion remains unchanged.

At step 2,478,806 the five-lap candidate screened faster at 33.504 seconds but
had one barrier contact per lap at 99.02% progress (peak impulse 178.86). The
contact gate rejected it and restored Champion despite the 0.132-second pace
gain; critic recovery restarted, with 977 updates remaining at step 2,478,899.
The replay archive had 65 attempts. This is a concrete speed/contact tradeoff
at the finish: the faster candidate is not clean enough for promotion.

The post-recovery five-lap check at step 2,483,270 tied Champion at 33.372
seconds with zero contacts. It was not promoted because it did not beat the
protected model; the clean anchor was refreshed and training continued. At
step 2,483,271 the trainer remained active, and the replay archive contained
70 attempts. The target remains 30 seconds.

At step 2,485,526 the five-lap candidate was clean at 33.848 seconds but slower
than Champion and its prior clean tie; it was rejected without rollback and the
reference was refreshed. Training remained active with actor learning unlocked
at step 2,485,527. The replay archive contained 73 attempts; Champion is still
33.372 seconds, 5/5 and contact free.

The next five-lap check at step 2,487,782 was clean at 33.790 seconds but
slower than Champion and the previous clean results. It was rejected without
rollback and the reference was refreshed. Actor updates resumed immediately
afterward (update norm 0.000138, drift 0.00250 by step 2,487,795). The replay
archive contained 76 attempts; no model was promoted.

At step 2,490,038 the deterministic five-lap check was clean at 33.601 seconds
with zero contacts. It improved on the immediately preceding 33.790-second
check but remained 0.229 seconds slower than Champion, so it was not promoted;
the clean reference was refreshed. The trainer continued after evaluation,
and the replay archive contained 79 attempts.

The five-lap evaluation at step 2,492,294 was clean at 33.610 seconds but
slower than Champion. It was the fifth weaker actor check in the recovery
window, so the verified actor was restored; the event confirms critics and
replay were preserved. At step 2,494,907, 347 critic recovery updates remained
and actor learning was locked. The visual replay archive contained 84 attempts.
Champion remains 33.372 seconds, five laps, zero contacts.

After the weak-check recovery, actor updates resumed. The next five-lap check at
step 2,496,774 was clean (33.716 seconds, zero contacts) but slower than
Champion, so it was rejected and the clean reference refreshed. At step
2,498,071 training remained active with actor learning unlocked; the replay
archive contained 88 attempts. Champion remains verified at 33.372 seconds.

At step 2,499,030 another deterministic five-lap check finished clean at
33.901 seconds with zero barrier contacts. It was slower than Champion and was
rejected; the clean reference was refreshed. At step 2,500,405 the trainer was
still active with actor learning unlocked. The visual replay archive contained
91 attempts. The protected five-lap Champion remains 33.372 seconds.

At step 2,501,286 the candidate passed five clean laps at 33.691 seconds and
zero contacts, but remained slower than the protected 33.372-second Champion.
It was rejected and did not replace the Champion. The latest trainer progress
at step 2,503,532 showed actor training unlocked but zero executed update norm
and cumulative reference drift near 0.04; the run remains active. This merits
continued monitoring of the trust-boundary updates. The replay archive had 96
attempts.

The next five-lap candidate at step 2,503,542 was 33.695 seconds with one
barrier contact on every lap at 98.95% progress (peak impulse 73.79). The
contact gate restored Champion; at step 2,505,575 the trainer was active in
critic recovery with 492 updates remaining, and the replay archive contained
98 attempts. The repeated finish-area contact persists despite sub-0.04 actor
updates; no candidate was promoted.

At step 2,508,006 the next five-lap candidate measured 33.602 seconds but
recorded two finish-area barrier contacts on every lap (99.23% and 99.30%
progress; peak impulse 341.54). The contact gate restored Champion. At step
2,505,575, 492 critic recovery updates remained; training and replay capture
continued with the archive at 102 attempts. This candidate did not improve the
33.372-second clean Champion.

The post-recovery check at step 2,512,486 tied Champion at 33.372 seconds
(5/5 finishes, zero contacts). It did not replace the protected model; the clean
reference was refreshed and the trainer continued. At step 2,514,734 the actor
was unlocked, with its current window at the 0.04 cumulative drift boundary
and zero executed actor-update norm. The replay archive contained 108 attempts.

The next five-lap check at step 2,514,742 was clean at 33.757 seconds, slower
than Champion, and was rejected without rollback. After the clean reference was
refreshed, actor updates resumed; at step 2,514,887 the update norm was
0.000087 with cumulative drift 0.00492. Training remained active, with 108
visual replay attempts preserved.

The five-lap candidate at step 2,516,998 measured 33.773 seconds with one
barrier contact on every lap at 99.02% progress (peak impulse 135.73). The
contact gate restored Champion. At step 2,518,387 critic recovery had 653
updates remaining and the actor was locked; the replay archive contained 112
attempts. Champion remains 33.372 seconds, 5/5 and zero-contact verified.

After the finish-area rollback, the next five-lap evaluation at step 2,521,478
was clean at 33.847 seconds. It was slower than Champion and was rejected; the
clean anchor refreshed and actor updates resumed (norm 0.000107, drift 0.00984
at step 2,521,745). The active replay archive contained 116 attempts.

The next five-lap evaluation at step 2,523,734 measured 33.622 seconds and
repeated one barrier contact on every lap at 98.95% progress (peak impulse
134.96). The gate restored Champion. At step 2,524,751 critic recovery had 746
updates remaining; actor changes were locked and the replay archive contained
120 attempts. No candidate has improved the verified 33.372-second pace.

The next five-lap evaluation at step 2,528,198 was clean at 33.752 seconds,
but slower than Champion, so it was rejected without rollback. Actor updates
resumed after the clean reference refresh (norm 0.000098, drift 0.00747 at step
2,528,395). Training remained active; the replay archive contained 124
attempts.

The candidate at step 2,530,454 passed five deterministic laps cleanly at
33.487 seconds. It improved on the previous 33.752-second check, but remains
0.115 seconds slower than the protected 33.372-second Champion, so it was not
promoted. Actor updates resumed; at step 2,531,439 the trust-window drift was
0.03987 with update norm 0.000095. The replay archive contained 128 attempts.

The next deterministic five-lap check at step 2,532,710 was clean at
33.537 seconds, but 0.165 seconds slower than Champion; it was rejected and
the clean anchor refreshed. At step 2,534,035 training was still active. The
replay archive contained 131 attempts; no Champion replacement occurred.

At step 2,534,966 the candidate completed five clean laps at 33.453 seconds.
This is the closest clean result in the current actor search, but remains 0.081
seconds slower than the protected 33.372-second Champion and was not promoted.
The clean reference refreshed and training continued. The replay archive held
135 attempts.

At step 2,537,238 the candidate was 33.591 seconds but made three barrier
contacts on every lap near the finish (99.16%, 99.23%, and 99.37% progress;
peak impulse 150.34). The gate restored Champion. At step 2,539,811 critic
recovery had 357 updates remaining and actor learning was locked. The replay
archive contained 137 attempts; Champion remains 33.372 seconds.

After the contact rollback, the five-lap evaluation at step 2,541,702 tied the
protected Champion at 33.372 seconds with zero contacts. It was not promoted;
the clean reference refreshed and training resumed. At step 2,542,819 the
actor remained unlocked, and the replay archive contained 141 attempts.

The next five-lap check at step 2,543,958 completed cleanly at 33.789 seconds,
slower than Champion, and was rejected without rollback. The clean reference
refreshed; actor updates resumed (update norm 0.000112, drift 0.01984 at step
2,544,467). Replay capture reached 143 attempts.

At step 2,546,214 the five-lap candidate was 33.785 seconds and recorded one
barrier contact on each lap at 98.95% progress (peak impulse 87.96). The contact
gate restored Champion; at step 2,546,885 the trainer remained active in critic
recovery with 833 updates remaining. The visual replay archive contained 146
attempts.

At step 2,550,694 the next five-lap evaluation tied Champion at 33.372 seconds
with zero contacts. It was not promoted; the protected Champion remains intact.
At step 2,552,379 training remained active, but the actor had reached its 0.04
reference-drift boundary with zero executed update norm. The replay archive
contained 152 attempts.

The candidate at step 2,552,950 evaluated at 33.683 seconds but contacted the
barrier once on each lap at 99.445% progress (peak impulse 106.42). The contact
gate restored Champion. At step 2,554,745 critic recovery had 552 updates
remaining, with the trainer still active and the actor locked. The replay
archive contained 155 attempts.

Following the contact rollback, the five-lap evaluation at step 2,557,446 tied
the protected 33.372-second Champion with zero contacts. It was not promoted;
training continued from the clean reference. At step 2,558,319 the actor was
unlocked and updating (norm 0.000112, drift 0.03482). The replay archive
contained 159 attempts.

The candidate at step 2,559,702 evaluated at 33.877 seconds and made one barrier contact on every lap (five total). The contact gate restored Champion. After critic recovery, the next check at step 2,564,182 tied Champion at 33.372 seconds, clean across all five laps; it was correctly not promoted. The subsequent clean checks at steps 2,566,438 and 2,568,694 were 33.642 and 33.700 seconds and were rejected as slower.

At step 2,570,966 the candidate measured 33.721 seconds and accumulated ten contact steps across five laps. The contact gate again restored Champion. The next full evaluation at step 2,575,430 tied Champion at 33.372 seconds with zero contact steps and was rejected without replacing it. A clean 33.420-second training episode followed at step 2,575,430; it is a training lap, not a verified promotion. Training remained active near step 2,576,000 with the actor unlocked. The visual replay archive held 180 indexed episodes at the latest check. No change to the protected Champion or active training configuration was made.

After the clean tie at step 2,575,430, the next deterministic evaluation at step 2,577,686 was 33.763 seconds with zero contacts. It was rejected as slower than Champion, and actor updates resumed. A subsequent ordinary training episode at step 2,579,937 finished in 33.758 seconds but hit the barrier twice at 99.16% and 99.23% progress (peak impulse 2,152.65); this is not a verified candidate result. Replay capture indexed 186 episodes, including the latest interrupted/active episode. At the latest check the trainer remained active and the protected 33.372-second Champion was unchanged.

Two further evaluations were rejected after the previous update. At step 2,579,942 the candidate was clean at 33.850 seconds, slower than the protected Champion. At step 2,582,198 it measured 33.815 seconds but made one contact in each of the five laps; the contact gate restored Champion. At step 2,584,001 the trainer was still active in critic recovery with 550 of 1,000 cooldown updates remaining and the actor locked. The replay archive contained 190 indexed episodes; the protected 33.372-second Champion remains unchanged.

The next candidate evaluation at step 2,588,918 was 33.862 seconds and recorded exactly one barrier contact on each of five laps, all at 98.8777% progress (peak impulse 135.79). The contact gate rejected the actor and restored Champion again. At step 2,591,139 critic recovery had 445 updates remaining and actor updates were locked. Replay capture held 198 indexed episodes. The fixed contact location near the finish is recurring; no track-specific action or controller workaround was introduced.

After the step 2,588,918 rollback, three clean checks at steps 2,593,398, 2,595,654, and 2,597,910 measured 33.635, 33.776, and 33.599 seconds respectively; all were slower than Champion and rejected. At step 2,600,166 the candidate measured 33.746 seconds and again contacted once per lap at exactly 98.8777% progress (five contacts, peak impulse 135.15). The contact gate rolled it back. At step 2,600,923 critic recovery had 811 updates remaining and the actor was locked. The replay archive contained 210 indexed episodes. Champion remains 33.372 seconds and untouched.

After the clean 33.372-second tie at step 2,604,630, the next candidate at step 2,606,886 measured 33.587 seconds and made three contacts on each lap at 99.162%, 99.233%, and 99.374% progress (15 total, peak impulse 94.42). This repeated the same three finish-area contact points from earlier actor regressions. The gate restored Champion; at step 2,608,423 critic recovery had 616 updates remaining with actor updates locked. Replay capture contained 219 indexed episodes.

After the clean tie at step 2,611,350, the next candidate at step 2,613,606 was 33.660 seconds and contacted the barrier once on every lap at 99.3742% progress (five total, peak impulse 766.81). It was rejected and rolled back. At step 2,615,232 the trainer remained active in critic recovery with 594 updates remaining and the actor locked. Replay capture contained 227 indexed episodes; Champion remains 33.372 seconds.

The first raw training lap faster than 33.2 seconds was 33.105 seconds at step 2,618,058. It is not a verified candidate result. The following deterministic checks at steps 2,618,070 and 2,620,326 measured 33.635 and 33.601 seconds cleanly, both slower than Champion. At step 2,622,582 the candidate measured 33.656 seconds and contacted once on each lap at 98.8064% progress (five total, peak impulse 81.48); the gate rolled it back. At step 2,624,875 recovery had 427 updates remaining and the actor was locked. Replay capture held 238 indexed episodes. The protected Champion remains 33.372 seconds.

Following the 33.656-second contact rollback, the next clean five-lap checks at steps 2,627,046 and 2,633,782 were 33.687 and 33.559 seconds. The candidate at step 2,629,302 measured 33.899 seconds with one contact each lap at 99.3035% progress (five total, peak impulse 176.87) and was rolled back. Subsequent clean checks at steps 2,636,038 and 2,638,294 measured 33.646 and 33.511 seconds. The 33.511-second candidate is the closest result in this search but remains 0.139 seconds slower than the protected 33.372-second Champion, so it was rejected. Training resumed with actor updates unlocked; replay capture indexed 257 episodes.

## Higher actor step size trial

The verified Champion remains 33.372 seconds. The best recent clean actor candidate was 33.511 seconds at step 2,638,294, while subsequent candidates mostly ranged from 33.6 to 33.9 seconds and several repeatedly regressed into finish-area contacts. The active checkpoint config used a 2e-6 actor learning rate, actor updates every 16 critic updates, a 0.0025 per-update action-drift cap, a 0.04 reference-drift cap, and actor checks every 2,000 environment steps. To test whether the very small actor learning rate is limiting pace search, the next continuation changes only actor learning rate to 5e-6. The previous 5e-6 run also used looser update and reference-drift caps, so it did not isolate learning rate. This continuation retains the current action and reference-drift limits, critic cooldown, contact rollback, deterministic five-lap evaluation, replay buffer, critics, and Champion gate. The current run must stop cleanly and resume from its latest saved state before the new rate takes effect.

### Applying the selected actor rate through the GUI

The first GUI continuation after setting the profile to 5e-6 did not apply that rate: “Continue latest with selected actor guard” restored the checkpoint's 2e-6 rate and only overrode the selected drift caps. Its two clean five-lap checks measured 33.743s at step 2,653,990 and 33.667s at step 2,655,126; neither beat the 33.372s Champion. The run stopped safely at step 2,657,535. The GUI option now preserves the selected actor learning rate and both action drift limits while continuing to load Latest's policy, critics, replay, rewards, and curriculum. Resume from that saved Latest state with the 5e-6 profile and verify the selected rate in the GUI activity log before treating the new trial as active.

## Restart at step 2,657,535

On 2026-10-14 the GUI loaded `profiles/training/simple-track-grtqc-contact-aware-30.json` and resumed Latest at step 2,657,535 with the selected 5e-6 actor learning rate, 0.0025 per-update action limit, and 0.04 cumulative reference-drift cap. The GUI activity log confirmed it retained Latest's policy, critics, replay, rewards, and curriculum; the run uses the one-million-step full-track budget, frame skip 30, n-step 16, CUDA on the NVIDIA T500, and five-lap evaluation gate. Champion remains 33.372 seconds and was not modified. The first live episode reached 74.9% of the track at step 2,658,385; this is startup evidence only, not a pace result. Automatic WebSocket replay recording remains enabled. Confirm the episode in the replay index after it finishes.

### Near-maximum controls

The report that actions often read 0.97 is consistent with a genuine near-maximum request, not an adapter ceiling. At frame skip 30, the pulse-density controller translates 0.97 to approximately 29 commanded ticks per 30; exactly 1.0 commands all 30. It accepts the full `[-1, 1]` range, including both signs. The existing deterministic replay audit found both steering extrema and +1 longitudinal output, so full output is available to this actor. Steering at an extreme is not automatically faster or safer, and the existing five-lap clean promotion gate remains necessary. No snapping, clipping, or reward change has been applied based on the display value alone. Evaluate the resumed actor on saved action traces and the next clean lap/evaluation before changing action semantics; retain the current controller, critics, replay, and protected Champion during that comparison.

The first saved replay in the resumed session is
`models/simple-track/grtqc/visual_replays/db3c90bf2c5848a0a14fa2a54741da5a/episode-000001.npz`
(step 2,657,535; 1,118 decisions). It records a clean 33.529-second finish with
no barrier contacts, which is 0.157 seconds slower than Champion; it is a
training episode and not a five-lap promotion evaluation. The recorded policy
requests reached +1.0 longitudinal and +1.0 steering. In this lap, 42.7% of
longitudinal requests were at least 0.97 in magnitude, 27.5% of steering
requests were at least 0.97, and exact +1 occurred for 3.5% of longitudinal
and 0.4% of steering samples. Thus the near-maximum display corresponds to
actual near-maximum demands and the policy sometimes selects the endpoint. This
single lap does not show that changing its requests to full demand would improve
pace; do not apply a blanket endpoint snap from these counts.

### First resumed evaluations

The deterministic five-lap screen at step 2,658,655 was clean at 33.775 seconds
and was rejected as slower than Champion (weak pace check 1/5). The next screen at
step 2,660,911 measured 33.740 seconds but recorded ten barrier-contact steps
across its five laps. The existing contact gate rejected the actor and restored
the Champion actor; it kept the critics and replay. Later recorded training
laps were 33.372 seconds, matching Champion after that rollback. No candidate
has promoted and Champion remains 33.372 seconds. The active replay session
`db3c90bf2c5848a0a14fa2a54741da5a` held eight entries at step 2,664,960,
including finished runs and interrupted entries; capture continues during
training.

After rollback, a recorded training episode finished at step 2,665,362 in
33.339 seconds with no contacts, 0.033 seconds faster than Champion. It is one
exploratory training lap, not a deterministic five-lap verification, so it does
not qualify for promotion. Its replay was saved as `episode-000009.npz`; the
active session had ten indexed records (nine finished and one interrupted) at
this checkpoint. Training remained active.

The next deterministic five-lap check at step 2,665,375 was clean at 33.735
seconds. It was rejected as slower than Champion and counted as weak pace check
2/5; no model promotion occurred. Training continued with Champion protected.
The replay session reached 13 entries (nine finished, including evaluation and
training laps, plus in-progress/interrupted records) at step 2,667,632.
