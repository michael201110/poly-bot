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

The fastest saved training lap (`episode-000165.npz` in run
`5e77f00d8849497b9ed21143405c67e0`) took 33.494 seconds over about
2,009 metres of credited route progress. A 30.000-second lap would require
about 11.6% more average progress per second on the same route. The recorded
lap spent 5.01 seconds in its first 10% of progress, then roughly 2.94–3.36
seconds per 10% section. This is one training lap, not a verified split or
proof that any single section accounts for the pace gap.
