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
