# Swarm renderer handoff (bridge 0.1.49)

The 0.1.48 unavailable-renderer callback used the general crash fallback:
restore nested sample arrays, create a native car for every replay, then batch
their render parts. Re-enabling PolyShade could therefore mirror hundreds of
ordinary native ghost objects while rebuilding its device. This caused the
500-car hot-toggle stall observed in the 0.5.1 performance pass.

Bridge 0.1.49 treats renderer availability changes separately from replay/worker
corruption. Disabling PolyShade retires the shared worker source and starts
the existing standalone OffscreenCanvas instanced worker. When API v1 becomes
ready again, it returns to the shared WebGPU source. Availability is probed at
most four times per second while standalone; no repeated scene traversal or
per-ghost update is added to normal frames.

The handoff transfers ownership of the **same main-thread canonical arrays**
to the replacement client. It does not expand nested samples or create the
other native ghost cars. The replacement worker gets the ordinary transferable
worker copy. The old worker is terminated, its canvas/source detached, and
its emergency restore closure is not retained by the new client.

The replay owner keeps its clock, pause/play state, fixed camera leader,
controls, opacity and end behavior. Geometry/material extraction and standalone
occluder setup remain initialization work. The interpolation kernel, recorded
samples, physics worker, training, rewards and checkpoint logic are unchanged.
An actual worker failure still restores the existing native fallback. If the
alternate renderer cannot initialize (for example, no OffscreenCanvas), the
canonical samples are restored before that fallback.

## Compatibility

| PolyBot bridge | PolyShade | Switching behavior |
|---|---|---|
| 0.1.49 | 0.5.2+ / API v1 | Instanced handoff both directions; retired GPU device explicitly destroyed |
| 0.1.49 | 0.5.0–0.5.1 / API v1 | Instanced handoff; old PolyShade device retirement behavior remains |
| 0.1.49 | Absent | Existing standalone worker, no PolyShade dependency |
| 0.1.48 | 0.5.2 | GPU retirement fix only; old bridge still reconstructs native ghost cars |

Install both updates for the complete fix. Neither API v1 nor simulator protocol
v2 changes. Historical bridge releases are preserved.

## Verification

Node tests cover canonical array identity, no nested-array restoration during
handoff, both worker modes, unchanged clock/leader, one native camera car and
the unsupported-renderer crash fallback. GPU lifecycle and live switching
results are documented with the coordinated PolyShade 0.5.2 release.

On 18 October 2026, all **35 Node renderer tests** and **477 Python tests**
passed. Three actual 500-car off/on cycles in Edge completed with one native
car allocation total. All 500 paused positions matched exactly at 3.000 s
across shared/standalone/shared, with unchanged clock, pause state and leader.
No browser errors were recorded. The 34 protected checkpoint hashes were
unchanged. The coordinated PolyShade GPU lifecycle suite also passed.
