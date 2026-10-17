# Worker swarm rendering (in progress)

PolyBot 0.1.48 optionally submits packed instances to PolyShade 0.5.0 API v1.
PolyShade renders these instances in its existing WebGPU world scene, with the
same depth, lighting, shadow and post processing pipeline. Replay interpolation
runs in a dedicated worker. The native fixed best-run car supplies camera, audio
and HUD; its body is detached from the scene. No physics interface is provided to
the worker, and simulation integration and learner files are unchanged.

## Visual fidelity corrections

The first shared renderer incorrectly substituted Phong materials for native
Standard/Physical vehicle materials. It also omitted flat shading. The packed
geometry descriptor now retains the material family, flat shading, roughness,
metalness, environment intensity, clearcoat, specular and emissive properties.
Stock geometry and suspension morphs remain complete. Mirrored wheels bake
reflection and reverse winding once to avoid negative instance determinants.

Worker output is staged until the replay update boundary. Cars, chase camera,
HUD and timer use the same completed replay timestamp, while the authoritative
playback clock continues at normal speed. Seek epochs reject stale output.
Three transferred output buffers bound the queue; every pending skid update is
applied before recycling, including when more than one output arrives between
rendered frames. No samples are discarded or resampled.

Native skid strips use the same tyre-edge projection, threshold, cooldown and
999-slot ring per wheel. One shared draw retains populated strips. Smoke for
nonleader cars and standalone effect equivalence still require completion;
this prototype must not be described as fully fidelity-verified yet.

## Compatibility

| Environment | Path |
| --- | --- |
| PolyShade 0.5.0 with API v1 ready | Worker interpolation + shared WebGPU instances |
| PolyShade absent | Worker OffscreenCanvas WebGL2, pending full visual validation |
| Unsupported worker, legacy wheel data or renderer failure | Existing native renderer |

The standalone path is implemented, but its visual equivalence and complete
benchmark matrix are pending. SharedArrayBuffer is unavailable on the current
nonisolated PML origin. Bounded transfers are used instead. A packed Float64 main
copy allows native recovery after worker failure; the worker owns a second copy.

## Validation so far

- Python: 477 passed in 236.85 s after material/presentation fixes.
- PolyShade: 81 Node tests passed, including material-family and skid-ring checks.
- PolyBot: 32 Node tests passed, including client staging, seek/failure recovery
  and native byte-channel colour checks.
- Manifests validate; no model, profile, training or physics changes.
- Live isolated-car comparison at replay timestamp 10 s confirms the restored
  body surface/shape. Mirrored-wheel appearance still needs further comparison.

The earlier 100-car baseline is approximately 11.9 FPS with Golden Hour / 50%
SGSR2. User feedback reports smooth native-resolution playback in the worker
prototype. This is not a controlled final benchmark. Before/after timing across
1/25/50/100/250/500 cars, effect parity, standalone rendering, memory lifecycle and
final sustained moving-car captures remain outstanding.

## Suspension and showcase follow-up

The live compiled shader revealed that Three r181 `geometryNode` assignments
were emitted after model-view projection, leaving suspension arms visually rigid
while the velocity pass already deformed them. The renderer now supplies the
four local morph deltas through `positionNode`. Live WGSL confirms the order:
raw vertex -> weighted suspension deltas -> exactly one instance matrix ->
projection. Shadows use the same node; the existing velocity pass uses the same
deltas/weights. Per-wheel weights were observed changing during live playback.

The final showcase loads 500 distinct complete short saved attempts (80..250
samples each), selected evenly across their real training steps. No recording
is truncated/resampled and the existing 64 MiB/250,000-sample limits remain.
The unfiltered 500-run selection exceeded the byte limit. Reproduce with:

```powershell
.venv\Scripts\python.exe tools/showcase_swarm.py --run models/simple-track/grtqc/visual_replays --report "$env:TEMP/polybot-showcase-500.json"
```

Current installed pins: PolyBot `0b844c5`, PolyShade `188ccc3`. Golden Hour and
native resolution were restored after temporary SGSR2 quality measurements.
The 500-car showcase is not a claim of sustained 60 FPS at that size.

A 100-car native-resolution interval measured 59.7 FPS, median 16.7 ms,
p95 16.9 ms and p99 17.3 ms. Longer measurements varied substantially (including
one-second gaps); GPU temperature reached 91 C and paging was active. SGSR2
quality measured 44.4 FPS over 15 s with median 16.7 ms / p95 33.5 ms. These are
observations, not controlled final acceptance. The full size/quality matrix and
complete nonleader smoke/standalone effect parity remain unfinished. User asked
to wrap up after the visual/suspension fixes and final 500-car showcase.

Final checks: 477 Python tests, 32 PolyBot Node tests, 82 PolyShade Node tests;
Ruff, manifest validation and diff whitespace checks pass. All 34 protected
policy/metadata/checkpoint hashes match their pre-work snapshots.
