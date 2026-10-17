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
