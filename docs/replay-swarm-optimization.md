# Replay storage and native swarm optimization — 17 October 2026

Follow-up to the [learner/GUI performance pass](performance-optimization.md).
Starting revision: `5115240`, clean `main`. No learner, reward, control, physics,
evaluation or champion-selection changes. Live training remains stopped.

## Workload and method

Windows 11, NVIDIA T500 4 GiB, eight logical Intel CPUs, Python 3.14.7,
NumPy 2.5.3 and PyTorch 2.14.0+cu126. Timings are local measurements, not CI
thresholds. Browser tests use Edge, PolyTrack 0.6.3, installed PML bridge
0.1.46 (with a final 0.1.47 compatibility check), native graphics, hidden HUD and the same Simple Track scene.
PolyShade is temporarily disabled to isolate native rendering; its settings
are restored after measurement. No other tabs are closed.

Storage tests read the existing Simple Track champion's replay without
modifying it. Independent copied buffers have capacities 10,000, 50,000 and
250,000; corresponding raw pickle sizes are 9.88, 49.40 and 247.00 MB. They
retain all fields and use contiguous prefixes of the numeric arrays. Each
benchmark fixture sets its own capacity, position 0 and full flag true; the
original replay is untouched. Sampling and training are not involved.
Each codec has three timed writes/reads; writes include finalization/fsync.
SHA256 of decoded pickle bytes must match for every codec. A separate
production comparison invokes the actual old/new storage APIs through SB3's
file interface, including atomic replacement and unpickling. Memory peaks
use tracemalloc separately from timed runs.

Native benchmarks duplicate a complete saved failed attempt from run
`06338533209f43fca66793220b522049`, episode `episode-000010`: **all 85 recorded
transform/wheel samples**, plus all **167 leader HUD/control frames**. IDs and
age colors differ; data are neither resampled nor truncated. The usual swarm
path sends full HUD data for its fixed camera run. This short complete attempt
fits the existing 64 MiB payload gate at 500 cars; a longer recording exceeded
that gate and was rejected, not downsampled.

Both browser paths use precisely the same payload, opacity 1 and playback
speed 0.1. The installed worker/integration hooks are identical to 0.1.44;
only the renderer function is switched between baseline and optimized source
for controlled profiling. Temporary diagnostic instrumentation reads the
existing 60-frame history and adds median/p95/p99 to status, with no per-frame
work added. Three rolling windows are sampled after warmup. Larger cases warm
for 12 seconds. Playback must advance between samples; frozen/crashed renderer
results are rejected. The browser was reopened during an interruption, so
the final baseline and optimized measurements are repeated within one session.

## Replay storage

### Findings and decision

The dominant cost was gzip compression, not pickle or array layout. Arrays
are already contiguous; materializing a new NPZ/custom array schema would add
compatibility and n-step-state reconstruction work. A raw full-buffer pickle
takes approximately 131 ms in the codec sweep, compared with several seconds
of gzip compression. Keep the entire SB3 payload and replace its compression
container instead.

The sweep compared gzip levels 1/3/6, Zstandard levels 1/3/6 and level 3 with
two workers. At 250,000 recorded transitions:

| Codec | Save median (ms) | Decode median (ms) | Compressed MB |
| --- | ---: | ---: | ---: |
| gzip 1 | 3,884 | 972 | 180.99 |
| gzip 3 | 5,424 | 1,070 | 165.07 |
| gzip 6 | 7,286 | 1,065 | 161.82 |
| Zstandard 1 | 2,305 | 672 | 169.73 |
| Zstandard 3 | 1,769 | 555 | 59.28 |
| Zstandard 6 | 3,653 | 409 | 56.40 |
| Zstandard 3, two workers | 953 | 469 | 63.37 |

Decode here means reconstructing bytes; production load below also unpickles.
An independent high-entropy synthetic sweep avoids depending only on repeated
recorded telemetry: at 250,000 entries, gzip 3 versus two-worker Zstandard 3
saves in **7,264 → 2,259 ms**, with **226.13 → 225.92 MB** files. Random data
does not yield the large recorded-data compression ratio, but remains faster.
These independent sweeps experienced normal desktop scheduling variation.

### Implemented path

- Standard Zstandard frames, level 3, two libzstd workers and frame checksums.
- Stream pickle through 1 MiB memoryview slices. No full replay snapshot or
  large input copy; SB3 still serializes its existing object graph.
- Compression workers operate while the synchronous save owns replay state.
  The API waits for all jobs, closes the frame, flushes/fsyncs, then replaces
  the destination atomically. No Python background-save race is introduced.
- Keep gzip/plain readers and filename conventions. Read the complete new
  frame trailer so corrupt/truncated checksums cannot hide behind pickle STOP.
- Use native Python 3.14 support or its maintained backport for older supported
  Python versions. See [storage compatibility](replay-storage.md).

Actual production API measurements (three repetitions; decimal MB):

| Capacity | Save median before → after (ms) | Load median before → after (ms) | Size before → after (MB) |
| --- | ---: | ---: | ---: |
| 10,000 | 227.14 → 43.58 | 46.13 → 24.81 | 6.60 → 1.90 |
| 50,000 | 982.58 → 181.61 | 231.46 → 92.45 | 33.02 → 13.04 |
| 250,000 | 5,128.71 → 807.28 | 1,198.58 → 395.97 | 165.07 → 63.37 |

Full-buffer save p95 is **5,142.78 → 817.66 ms**; load p95 is
**1,326.32 → 407.89 ms**. This is approximately **6.35× faster saving**, **3.03×
faster loading**, and **61.6% less disk space** for this recorded buffer.

Traced Python full-buffer save peak falls **198,855,750 → 11,525,254 bytes**;
load peak **381,323,719 → 253,181,191 bytes**. The load figure includes newly
allocated replay arrays. Tracemalloc excludes libzstd worker memory and other
native allocations, so these are not total process RAM claims. Before chunking,
the new compressor's save peak was 62,195,422 bytes: chunking removes large
compressed-output temporaries without changing the resulting file size.

### Alternatives investigated

Lower gzip levels save too little time and increase disk use. Zstandard 6 is
smaller but substantially slower. Two compression workers add bounded native
job memory and a small ratio cost versus single-worker level 3, while reducing
recorded-buffer latency. No dependency on LZ4/Blosc/native array extensions is
needed. The old pickle schema already stores contiguous arrays efficiently;
separate array files/out-of-band formats would require a new atomic multi-file
container and migration for a comparatively small serialization saving.

Unbounded asynchronous snapshots were rejected: serializing mutable replay
after returning from save could mix transitions; copying the entire buffer
first would increase RAM pressure. Streaming libzstd jobs overlap safely
without changing replay ownership or checkpoint semantics.

## Native swarm rendering

### Actual bottleneck

The old batcher could not find Three.js `InstancedMesh`: the minified 0.6.3
namespace exports it as `ZLX`, and `isInstancedMesh` is an instance property,
not a prototype property. Live status showed **zero instanced batches**. It
also excluded material arrays, including the wheels and other common parts.
Repeated native draw submission dominated; per-car interpolation was smaller.

### Changes in bridge 0.1.46

- Resolve the actual native instancing constructor. Instance matching opaque
  single-material and material-array geometry while retaining the full native
  material description, shadow flags, layer mask and render order.
- Preserve each wheel's native morph weights in a persistent per-instance
  morph texture. A live 0.1.45 candidate exposed this missing requirement and
  halted; its timings are discarded. 0.1.46 fixes it and adds a regression.
- Cache batch membership and roots when loading. Each car hierarchy updates
  once, and matrices are copied into persistent Float32 instance buffers.
  No per-frame Matrix4/Set construction or scene-search traversal remains in
  the batch updater. Native `setMorphAt` also reuses its texture buffer after
  initialization; its actual implementation was inspected in the browser.
- Keep body paint and brake lights on native paths, including per-run colors.
  Skid marks, wheel effects, native car updates, fixed camera and leader audio
  remain active. Instancing does not replace their state/effect updates.
- Individual fades and transparency restore original native mesh draws and
  hide the corresponding instances. This retains per-car opacity and native
  blend ordering. Returning to opacity 1 restores instancing.
- Compact visible instances into the existing buffer/count. Do not draw ended
  ghosts. Keep conservative aggregate culling: disabling off-screen effects
  or shadow casters based on camera distance could change replay appearance.

### Results

Final live results, same complete recording and leader HUD on both paths:

| Cars | Draw calls before → after | Frame time before → after (ms) | Approx. FPS before → after | Frame p95 before → after (ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 58 → 49 | 16.67 → 16.67 | 60 → 60 | 16.9 → 16.9 |
| 10 | 265 → 121 | 16.67 → 16.67 | 60 → 60 | 16.9 → 16.9 |
| 50 | 1,185 → 441 | 21.79 → 16.67 | 45.9 → 60 | 25.0 → 16.9 |
| 100 | 2,335 → 841 | 41.11 → 16.67 | 24.3 → 60 | 46.6 → 16.9 |
| 250 | 5,785 → 2,041 | 99.96 → 25.19 | 10.0 → 39.7 | 133.9 → 28.3 |
| 500 | 11,535 → 4,041 | 193.50 → 48.16 | 5.2 → 20.8 | 204.6 → 55.6 |

Frame time is the median of three 60-frame rolling means; p95 is the median
of those windows' p95 values. FPS is derived from the rolling mean, not an
uncapped hardware throughput claim. Windows overlap, especially at low FPS.
The 500-car frame median itself changes **192.8 → 47.0 ms**, and p99
**210.4 → 62.3 ms**. The 100-car p99 changes **49.3 → 17.0 ms**.

Rendered triangles match exactly across both paths at every size: 86,774;
186,692; 630,772; 1,185,872; 2,851,172; and **5,626,672** respectively,
including measured render passes. No geometry/shadow reduction explains the
gain. There are three instanced batches at sizes 10–500, covering six native
parts per car; the one-car case batches its four matching wheels.

At 500 cars, renderer CPU time changes **183.78 → 37.15 ms**, while replay
update CPU changes **7.19 → 8.70 ms**. At 100 cars, those figures are
**37.30 → 7.30 ms** and **1.53 → 2.57 ms**. The extra hierarchy/morph upload
work is measurable; reduced draw submission more than compensates. Native
car/effect updates have not been skipped to obtain these results.

These are deliberately dense duplicate trajectories, not a promise of the
same FPS on every track, mixed swarm, viewport, driver or PolyShade preset.
Bodies, brake lights and native effects still submit per-car work, and the
500-car result remains substantially below 60 FPS. Transparent/fading cars
retain native draws, so their speedup can be smaller. Aggregate culling is
conservative; this pass does not discard distant geometry or effect history.

### Final installed 0.1.47 recheck

After the compatibility change, all six sizes were measured again, with the
same complete payload, opacity, playback speed and native graphics. Draw calls
and triangle totals exactly match the optimized table. Rolling mean frame
medians are **16.665, 16.665, 16.667, 16.890, 31.143 and 52.463 ms** for
1/10/50/100/250/500 cars (about 60/60/60/59/32/19 FPS). These final measurements
are slower at the two largest sizes than the earlier 0.1.46 pass; neither
run is presented as a universal or noise-free hardware speed claim. The
500-car batch update itself is 8.71 ms versus 8.70 ms in the earlier pass;
renderer CPU time varies from 37.15 to 41.28 ms. Exact draw counts and fidelity
checks remain unchanged.

Machine-readable summaries of both optimized passes, the baseline and the
production storage comparison are saved in
[benchmark results](benchmarks/replay-swarm-20261017.json).

### PolyShade / WebGPU compatibility

Restoring the original Golden Hour Capture preset exposed a WebGPU morph
uniform error in 0.1.46. Release **0.1.47** initializes a persistent ordinary
morph uniform array as well as the native per-instance morph texture. It
batches morph meshes only while all visible opaque members have identical
weights; otherwise their original native meshes render. Uniform values update
in place. This conservative fallback preserves appearance on renderers that
consume shared uniforms instead of the instance texture.

The installed 0.1.47 was checked after a clean reload with PolyShade enabled,
its WebGPU canvas active, and 100 different saved attempts. Playback advanced,
100 ghosts and 400 skid trails remained present, and no new shader errors
were logged. Regression tests cover divergent morph weights, native fallback,
return to instancing and persistent uniform-array identity. The main table
above measures 0.1.46 with PolyShade disabled; it is not a measurement of the
Golden Hour Capture preset.

### Fidelity and isolation

Renderer differential tests compare baseline/updated car state objects exactly
through interpolation, seek, fades and opacity changes. They check matrix
buffer reuse, morph values, fixed camera, audio ownership and disposal. The
500-car test exercises actual batch construction through the minified export
contract. Source tests require the new **simulation worker and integration
template to be byte-identical to 0.1.44**. Native replay constructors still
receive no physics solver, and no simulation step/reset/training operations
are added. Existing guards clear visual playback when live model driving starts.

All source transform times, wheel arrays and interpolation equations are
unchanged. GPU instancing changes draw submission, not the stored replay,
physical car state or training observations. Matrix uploads use the same
native world transforms; per-instance morph weights retain the native values.
No samples, effects, shadows, colors or geometry detail are removed for speed.

Camera-dependent object culling and eliminating native car updates entirely
were rejected because they would require replacing native wheel/skid/light/
audio behavior. Transparent instancing was rejected because it can change
blending order. Painting all bodies through a custom instance-color shader
would require separating native material groups and dynamic brake-light
behavior; existing native body draws remain the safer residual cost.

## Validation and reproduction

The original full suite passed **468 tests**. Final validation passes **477
Python tests** (90.55 seconds), **23 JavaScript tests**, Ruff and diff checks.
PML manifest validation passes; raw game-bundle source anchors were not checked
by that command. All 34 protected policy/metadata/TQC artifact hashes match the
starting values, and the learner/environment/control/model/profile paths have
no changes. The final CUDA GRTQC doctor completes 32 mock steps / 24 updates,
saves and reloads a 32-transition replay successfully. All nine CPU mock smoke
combinations (GRTQC/TQC/PPO, tiny/compact/standard, 128 steps) complete; the
existing Node mock swarm benchmark also completes. These are disposable mock
checks, not live training or evidence of faster learned lap times. Added storage tests cover exact
decoded bytes, array shapes/dtypes/NaN/signed-zero values, legacy containers,
bounded zero-copy writes, injected serialization/fsync/replace failures,
temporary-file cleanup and corrupt/truncated checksums. Real SB3 save/load
interfaces are exercised. Python 3.13 with `backports.zstd 1.8.0` decodes Python
3.14 frames exactly and vice versa; the backport is intentionally not installed
into Python 3.14. Protected artifact hashes remain unchanged.

```powershell
python tools/benchmark_replay_storage.py --source models/simple-track/grtqc/champion/replay.pkl --repeats 3 --output "$env:TEMP\codecs.json"
python tools/benchmark_replay_storage.py --sizes 10000 50000 250000 --repeats 3 --output "$env:TEMP\entropy.json"
python tools/benchmark_replay_storage.py --source models/simple-track/grtqc/champion/replay.pkl --production-comparison --repeats 3 --output "$env:TEMP\storage-api.json"
python tools/benchmark_native_swarm.py --run models/simple-track/grtqc/visual_replays/06338533209f43fca66793220b522049 --episode episode-000010 --sizes 1 10 50 100 250 500 --settle-seconds 12 --output "$env:TEMP\native.json"
node --expose-gc tools/benchmark_replay_swarm.mjs
python -m pytest -o addopts='' -q
node --test tests/*.mjs
python -m ruff check src tests tools
git diff --check
python tools/validate_pml_mod.py
polybot-doctor --smoke grtqc
python tools/benchmark_v2.py --steps 128 --device cpu --threads 1 --algorithms grtqc tqc ppo
```

Install the mod before using the native benchmark and start the matching track
with a player car. No learner is started. The tool preserves the bridge's sample
and byte limits, verifies playback advances, and clears the swarm on exit.
For baseline browser comparisons, load the existing 0.1.44 renderer into the
same installed bridge and restore the installed 0.1.47 afterwards. Both worker/template files
are identical. Keep source replay, graphics, viewport, warmup and payload
identical; read rolling distributions only after enough new frames. Reports
and browser profiles belong outside tracked source. Older 0.6.2 remains on
its existing bridge release.
