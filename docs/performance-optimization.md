# Performance and reliability pass — 17 October 2026

## Scope and baseline

Baseline: `7f815a85ef2cfd027db2b3fa02e9b851e8807d76`, clean `main`.
Measurements use Windows 11 build 26300, eight logical Intel CPUs
(Family 6 Model 140), NVIDIA T500 4 GiB, driver 573.91, Python 3.14.7,
NumPy 2.5.3, PyTorch 2.14.0+cu126 and one Torch CPU thread.
The GPU also serves desktop applications; it is not a dedicated benchmark GPU.

Live training remains stopped. All learning measurements use new synthetic
learners and temporary files. Existing policies are only loaded for inference.
No model, profile, protocol version, bridge release, reward equation, learning
hyperparameter, evaluation gate, control cadence or PWM implementation changes.

### Architecture inspected

| Area | Existing design and audit finding |
| --- | --- |
| GRTQC/TQC/PPO | SB3 backends; gated actor/quantile critics; separate actor and critic inputs; n-step numeric replay; readiness, reference drift and rollback protection. Repeated diagnostic readbacks and deterministic actor forwards dominate avoidable work. |
| Environment/control | Synchronous validated protocol v2 transitions, normalized observations, reward components, stateful PWM over 30 ticks. Lookahead normalization dispatches many scalar NumPy operations. |
| Transport/bridge | WebSocket request/reply, JSON, reconnect/watchdog, versioned game injections. Local transport overhead is measurable; no evidence supports changing the wire format or removing validation. |
| Storage/models/tracks | Versioned metadata, cached reference lines, compressed numeric replay, track workspaces, staged champion promotion and atomic replay replacement. Compression is expensive but protects scarce disk space. |
| Training/evaluation | Runner owns replay/model mutations; deterministic evaluation and verified champion gates. Logging, visual replay capture and resume paths retain their existing semantics. |
| Replay/PML | Python selection/chunking and native renderer cars, wheel telemetry, fixed best-run camera, leader audio and interpolation. GPU scene rendering is outside a Node logic benchmark. |
| GUI | PySide6 worker boundaries, model/replay inventories and JSONL display. Standalone log viewer drains a whole backlog synchronously and updates widgets per event. |
| CLI/profiles/tools/tests | Existing mock benchmarks, doctor, renderer harnesses, reward/training profiles and historical compatibility tests. Several assertions and mocks lag current defaults/APIs. |

Relevant starting documentation includes the training, GRTQC experiment/scratch
and causal audit records, protocol, GUI, replay storage and swarm performance
guides. Supported TQC/PPO and historical bridge releases remain supported.

Baseline Python validation had **438 passes and 9 failures**; Ruff reported
**26 errors**. Diff whitespace and PML manifest validation passed. The CUDA
doctor completed 32 steps, 24 updates and a save/load smoke check.

## Method

`tools/benchmark_performance.py` measures observation/reward/JSON processing,
mock steps, actor inference, replay insertion/sampling, learner updates,
temporary checkpoint/replay I/O, an actual loopback WebSocket control cycle and
offscreen Qt log consumption. It reports median, p95, p99, sustained calls/s
and process CPU time; cProfile summaries identify cumulative costs. CUDA work
is synchronized at timing boundaries. Optional Torch profiling counts scalar
readback operators; tracemalloc and CUDA allocation snapshots cover memory.

The production-shaped learner uses the existing Simple Track contact-aware
profile: 121 observations, batch 128, capacity 250,000, 16-step returns, compact
networks, two critics and 25 quantiles. All replay slots contain bounded numeric
synthetic transitions, with episode boundaries and 1,111 reference states.
No historical replay is padded into a different schema.

For the decisive learner comparison, the tool extracts the original `train`
method from the baseline revision and interleaves baseline/updated calls,
alternating their order under identical RNG state. Each mode has three warmup
and 60 measured calls. Parameters must match bitwise after each mode. Actor
rows include the critic update in the same call. The capped row deliberately
sets a binding **test-only** 1e-7 drift limit to exercise all existing bisection
iterations; this does not change a saved profile.

Core timings use 300 calls and GUI timings ten full 5,000-event drains, with
three warmup drains. Both GUI paths consume every event. JSONL still retains
all diagnostics; only intermediate visual progress labels are coalesced.
These are synthetic CPU/device measurements, not game FPS, lap times or proof
of improved learning. Shared GPU scheduling can produce large latency outliers.

## Implemented changes

1. **GRTQC:** calculate each existing gradient norm once; transfer per-layer
   norms and scalar diagnostics to Python in batches. Keep the same reductions,
   optimizer sequence, loss histories and readiness transition. Reuse identical
   deterministic actor forwards during drift bisection and final diagnostics.
   All 12 bisection iterations, reference caps and recovery checks remain.
2. **Observations:** normalize and clip the lookahead array once. Division stays
   float64 until the existing final float32 cast; ordering and values match
   the original scalar implementation bitwise.
3. **GUI log viewer:** extract a bounded JSONL reader, process at most 256 events
   or 1 MiB per refresh, append summaries together and display the last progress
   value in the batch. A single-shot Qt timer yields between catch-up batches.
   Preserve partial UTF-8/JSON lines for retry, handle truncation and stop timers
   on close. One oversized complete event may exceed the byte budget to ensure
   forward progress. Display remains capped at 400 text blocks.
4. **Small correctness repairs:** optional single-replay age metadata falls back
   from `training_step_start` to `training_step` to zero; Scratch completion's
   GUI label now matches the reward values already loaded. Bind the watchdog's
   attempt-specific status callback explicitly. No reward values are changed.
5. **Regression infrastructure:** frozen original numerical paths, recorded
   telemetry fixture, CPU/CUDA differential tests, read-only policy/PWM
   fingerprints, current 500-car/wheel/audio renderer tests and repeatable
   benchmark commands. Update stale fixtures and resolve existing lint errors.

## Learner results

Milliseconds per full `train(1)` call. Throughput is the reciprocal of median
latency, an approximate compute ceiling rather than live simulator SPS.

| Device/work | Median before → after | p95 before → after | p99 before → after | Approx. calls/s before → after |
| --- | ---: | ---: | ---: | ---: |
| CUDA critic | 17.050 → 14.155 | 21.609 → 20.024 | 22.500 → 22.807 | 58.7 → 70.6 |
| CUDA actor due | 33.974 → 29.059 | 39.502 → 33.775 | 45.525 → 52.833 | 29.4 → 34.4 |
| CUDA binding actor cap | 81.254 → 64.843 | 91.146 → 76.941 | 93.476 → 84.164 | 12.3 → 15.4 |
| CPU critic | 15.819 → 15.611 | 16.561 → 16.358 | 16.882 → 17.016 | 63.2 → 64.1 |
| CPU actor due | 28.566 → 25.466 | 29.641 → 26.766 | 32.073 → 28.933 | 35.0 → 39.3 |
| CPU binding actor cap | 66.654 → 51.820 | 69.902 → 53.728 | 72.765 → 54.414 | 15.0 → 19.3 |

CUDA median reductions are **17.0%, 14.5%, 20.2%** respectively. CPU critic
change is small; actor medians improve by **10.9% and 22.3%**. CUDA p99 does
not consistently improve; there is no claim that desktop GPU jitter is fixed.

A separate paired Torch profiler call counted critic `aten::item` and
`aten::_local_scalar_dense` operators falling from **100 to 48** each.
`aten::_to_copy` increased from 7 to 9: batching reduces synchronous scalar
delivery, not every tensor copy. Required finite checks and training
diagnostics remain. Profiles still show forward/backward linear algebra,
quantile Huber loss, optimizer work and reference drift searches.

Process CPU time in the independent CUDA run was approximately 89% of one core
for critic calls, 92% for actor-due calls and 85% for capped calls. This is not
whole-machine CPU utilization. During a later diagnostic window, 379 `nvidia-smi`
snapshots at 250 ms intervals showed shared GPU utilization median **28%**, p95
**34%**, maximum **35%**, with median GPU memory use **1,264 MiB**. That window
included synthetic learner work, desktop apps and part of validation, so it
cannot attribute utilization to PolyBot alone. Small kernels and CPU/readback
dependencies remain a better explanation than assuming a GPU compute ceiling.

## Environment, transport and GUI

| Work | Median before → after (ms) | p95 before → after (ms) | p99 before → after (ms) |
| --- | ---: | ---: | ---: |
| Observation vector | 0.2310 → 0.0881 | 0.3655 → 0.1336 | 0.5578 → 0.2144 |
| Full mock environment step | 0.8544 → 0.6749 | 1.5882 → 1.1599 | 2.1680 → 1.6246 |
| Protocol parse | 0.0711 → 0.0709 | 0.0983 → 0.1249 | 0.1737 → 0.2492 |
| Loopback telemetry → policy → PWM → step reply | 2.5398 → 2.4055 | 3.5754 → 3.4241 | 3.9294 → 3.8269 |
| GUI complete 5,000-event drain | 203.046 → 109.052 | 227.987 → 117.698 | 236.337 → 119.344 |

Observation median falls **61.9%**, mock step median **21.0%**. Sustained mock
steps increase **984 → 1,286/s** in this workload. The WebSocket test uses real
local sockets and a mock simulator thread, includes validation/reward/control,
and has no real game or network delay. Its roughly 5% gain is modest.

The GUI's individual UI-blocking refresh slices change from **200.56 to
5.29 ms median**, **224.40 to 6.61 ms p95**, and **238.32 to 9.31 ms maximum**.
The total drain is also faster, so responsiveness is not obtained by dropping
events. Normal polling remains one second; only backlog catch-up yields more
frequently. Main training data collection is not throttled.

Unchanged paths remain cheap: JSON decode ~0.018 ms, encode ~0.020 ms;
reward calculation **0.0543 → 0.0540 ms** in a separate 200-call comparison.
Python interpolation of a 1,200-sample replay stays **0.0888 → 0.0891 ms**.
There is no claimed optimization of these unchanged implementations.

## Replay, storage and memory

Replay already uses preallocated numeric arrays and batched sampling; no new
format, migration, reduced capacity or asynchronous mutation is introduced.
CPU sample median is **0.252 → 0.236 ms**; insertion is **0.0057 ms** in a
separate 200-call measurement. These small unchanged-path variations are not
attributed to a replay optimization. Independent CUDA sample timings were
0.591 → 0.679 ms, consistent with transfer/scheduling overhead remaining.

Temporary full-capacity replay I/O retains gzip level 3, flush/fsync and atomic
replacement. Measured baseline → later run: save **8.159 → 6.606 s**, load
**1.397 → 1.292 s**, exactly **226,137,624 bytes** in both runs. Policy save
**17.65 → 15.19 ms**, load **36.65 → 27.29 ms**. Storage code is unchanged;
cache, scheduling and compression variation prevent claiming these as gains.
Replay compression is still the largest occasional blocking cost.

CUDA peak tensor allocation is **30,349,312 bytes before and after**. A short
steady-state 200-update check ends at **21,100,544 allocated CUDA bytes** in
both paths. Traced Python retention after those updates is **44,029 bytes
baseline / 40,223 updated**, with peaks **267,589 / 266,465 bytes**. These tiny
differences are not a meaningful memory reduction. Readiness histories are
bounded at 100 entries. Optional RSS collection is unavailable here because
psutil is not installed; no dependency was added for it.

A longer separate **2,000-update** check on each package ends at exactly
**21,105,664 CUDA bytes** in both. Traced Python retention at 1,000 → 2,000
updates is **69,995 → 76,561 bytes** baseline and **67,737 → 75,198 bytes**
updated, peaks **300,195 / 301,898 bytes** respectively. The histories remain
100 entries. This shows no large tensor retention in this workload, while the
small Python growth and limited duration do not establish absence of all leaks.

The 500-step mock environment retains **5,138,415 → 5,138,351 Python bytes**,
peak **5,149,323 → 5,150,603**. Its command log intentionally keeps commands
for mock tests; this is not evidence of a live transport leak. Tracemalloc
excludes pre-existing NumPy replay allocations and most native/driver memory.
No multi-hour memory or training-stability claim is made.

## PML and swarm

No published game injection or renderer source changes. The benchmark now
resolves the current manifest (0.1.44), uses correct complete payload byte
counts, wheel telemetry, cloneable native styles and moving playback frames.
It reports median/p95/p99, CPU time and heap for 50/100/250/500 cars.

Current 500-car JS logic: **1.319 ms median, 2.105 ms p95, 2.966 ms p99**,
~34.24 MiB additional Node heap for this fixture. 50/100/250-car medians are
0.150/0.309/0.703 ms. Native cars/materials/audio/Three.js/GPU drawing are mocked,
so these numbers cannot be converted into PolyTrack FPS.

An explicit wheel-field implementation tested outside the worktree achieved
1.197/1.829/2.214 ms for 500 cars, with essentially the same heap (~34.22 MiB).
It was **not shipped**: ~0.12 ms median saving does not justify duplicating
wheel indexing or a new bridge release without live FPS evidence. The prior
live investigation in [swarm-performance.md](swarm-performance.md) identifies
native scene draw calls/triangles and T500 pressure as the larger constraint;
that historical evidence was not remeasured during this task.

## Correctness and validation

Numerical tests compare against frozen baseline source, not the updated
function itself. CPU **and CUDA** tests cover frozen critics, actor updates,
binding caps, controller adapters and warmup unlocking. Policy tensors,
actor/critic/entropy optimizer state, diagnostics, readiness histories and
NumPy/Torch RNG states match **bitwise**. Production-shaped paired runs also
require exact parameter equality. Actor-forward count assertions protect the
removed duplicate work without brittle CI timing limits.

Observation tests compare uint32 bit patterns for random/clipped/signed-zero
inputs with 0/1/12/64 lookahead points and 16 actual saved-lap telemetry frames.
The fixture records its source replay hash. Four existing policies load under
both baseline and updated packages and produce identical SHA256 fingerprints
for raw actor output, transformed actions and all sequential 30-tick PWM controls:

| Existing artifact | Recorded input count / width |
| --- | --- |
| Simple Track GRTQC champion | 1,120 / 121 |
| Summer 1 GRTQC champion | 1,120 / 121 |
| Summer 1 TQC `best.zip` reference | 896 / 105 |
| Summer 1 legacy PPO champion | 896 / 105 |

The GRTQC checks use the same recorded Simple Track input sequence; Summer 1
therefore verifies tensor/control equivalence, not track-specific lap fidelity.
TQC/PPO use the existing Summer 1 demo. No dimensions are padded. SHA256 checks
also confirm **34 protected policy/metadata/reference files unchanged**.
No simulator was started and no new five-lap verification or champion pace is
claimed. The original initialization/reference and evaluation guards still run
through their existing tests.

The nine original Python failures were stale contact-guard fixtures, rollback
mocks missing a new keyword, an outdated latest-bridge expectation, optional
replay age metadata, outdated reward preset expectations, collapsed GUI section
assumptions, a fixture overriding the color default, and a missing loaded-ghost
option expectation. Corrections retain the production guards and defaults.
Historical Node harnesses also lacked cloneable car styles, retained disposed
silent cars after leader-audio replacement, indexed HUD content before resize
controls existed and expected the superseded idle message. Their assertions now
exercise live cars and actual content, including leader camera/fade/disposal.
The HUD timing test now sends distinct frames while visible instead of timing
the identical-frame fast return.

Validation commands and final results:

- `python -m pytest -o addopts='' -q`: full Python suite, including current
  renderer integration and numerical CPU/CUDA tests: **468 pass**.
- `python -m ruff check src tests tools`: pass.
- `git diff --check`: pass.
- `python tools/validate_pml_mod.py`: pass for both versioned manifests;
  actual 0.6.2/0.6.3 game bundles unavailable, so bundle hashes/live patching
  were not checked.
- `polybot-doctor --smoke grtqc`: pass on CUDA, 32 steps / 24 updates,
  replay size 32 and resumed model loaded.
- `python tools/benchmark_v2.py --steps 128 --device cpu --threads 1
  --algorithms grtqc tqc ppo`: all nine algorithm/architecture combinations pass.
- `polybot-smoke --episodes 1`: mock protocol/environment episode finishes.
- `node --test tests/*.mjs`: **21 pass**, covering historical HUD/replay,
  current wheel telemetry/audio/camera/500 cars and cached reference geometry.
- `python tools/verify_policy_equivalence.py ...`: all four policies' baseline
  and updated hashes match; protected-file SHA256 audit passes.

## Reproduce and extend

Run from the repository root with the project's virtual environment. Keep JSON
reports and profiler outputs outside the repo, and avoid other compute jobs.

```powershell
python tools/benchmark_performance.py --sections core gui --count 300 --output "$env:TEMP\polybot-core.json"
python tools/benchmark_performance.py --sections learner --device cuda --threads 1 --updates 60 --compare-revision 7f815a8 --output "$env:TEMP\polybot-paired.json"
python tools/benchmark_performance.py --sections learner --device cpu --threads 1 --updates 60 --compare-revision 7f815a8
python tools/benchmark_performance.py --sections learner --device cpu --updates 10 --count 100 --io
python tools/benchmark_performance.py --sections learner --device cuda --updates 5 --compare-revision 7f815a8 --profile-operators --memory-updates 100
python tools/benchmark_performance.py --sections learner --device cuda --updates 3 --count 10 --memory-updates 1000
python tools/benchmark_performance.py --sections transport --device cpu --count 300
python tools/benchmark_performance.py --sections transport --device cpu --count 300 --compression none
node --expose-gc tools/benchmark_replay_swarm.mjs
python tools/verify_policy_equivalence.py --algorithm tqc --checkpoint models/summer-1/tqc/best.zip --observations profiles/demos/summer-1-26.894s.npz
```

For a whole-package baseline, extract **only `src/`** at `7f815a8` into a
temporary directory, run the same benchmark with that directory in `PYTHONPATH`,
then unset it. Keep the harness, dependencies, device/thread settings and input
artifacts identical. The paired learner mode already does this comparison for
the changed `train` method. Saved reports include revision, learner path/hash
and machine information. No benchmark changes production profiles.

## Remaining costs and rejected approaches

- Quantile loss/backpropagation and reference action-cap bisection remain
  substantial. Increasing batch size, changing update ratios, reducing
  diagnostics/evaluation or loosening caps would alter the experiment.
- CPU replay sampling is already a small part of an update. Pinned staging,
  persistent GPU replay and asynchronous copies add lifetime/synchronization
  complexity without evidence of a worthwhile gain here.
- Disabling WebSocket compression measured **2.490 ms median / 4.941 ms p99**
  for the control cycle, versus **2.405 / 3.827 ms** with deflate. Rejected;
  transport defaults and protocol v2 stay intact.
- AMP, fused optimizers and compile were not enabled: no validated gain that
  justifies a numerically different default. Existing small inference paths
  stay intact. Independent CUDA inference runs varied 1.438 → 1.842 ms while
  CPU varied 0.391 → 0.368 ms; no inference-kernel speedup is claimed.
- Replay compression remains synchronous and atomic. Serializing mutable model
  or replay state in a background thread without a consistent snapshot risks
  resume integrity and extra RAM; no such concurrency was introduced.
- Native swarm rendering still needs a separate live browser/GPU profile before
  changing materials, fidelity or draw batching. Existing native behavior and
  bridge versions are preserved.
- The bounded log-reader extraction is cohesive; broader module splitting or
  deleting legacy support would add review churn without measured benefit.

The engineering gains reduce avoidable work while retaining the same numerical
updates and controls. They do not establish that the agent learns faster laps.
