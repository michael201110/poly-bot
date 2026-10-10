# Live swarm performance — 2026-10-10

Measured in the user's existing Edge / PolyTrack 0.6.3 session on the NVIDIA
T500, using bridge 0.1.40 and the saved 100-run telemetry collection. No
training was started. HUD was hidden. Playback speed was 0.1 to allow sampling
the same early track segment. Render target was approximately 1234 × 947.

## Findings

The dominant limit is native scene rendering and GPU memory pressure.
Replay interpolation and applying recorded state are a much smaller cost.

With PolyShade disabled, after applying the CPU changes below:

| Playback | Draw calls/frame | Triangles/frame | Replay update CPU |
| --- | ---: | ---: | ---: |
| 1 car | 89–90 | 56,218–56,220 | 0.19–0.24 ms |
| 100 cars, after all resources rendered | 2,366 | 1,155,316 | about 2.8–2.9 ms |

The single-car sample averaged approximately 18–19 ms per frame (52–56 FPS).
The 100-car frame times were highly unstable, with long stalls. In the last
three running samples, 471–699 ms was spent inside renderer calls, versus
481–709 ms between game frames. This includes GPU/driver blocking and should
not be interpreted as pure JavaScript computation. The 60-frame history
retains startup stalls for longer when FPS is low; it is not a stable FPS
benchmark or evidence of a measured speedup.

Windows counters for Edge's T500 3D engine measured approximately 96–100%
utilization during the 100-car run. Adapter memory was approximately 3.76 GiB
dedicated plus 2.14–2.16 GiB shared. These are whole-adapter totals, not an
isolated allocation measurement for replay cars. NVIDIA utilization readings
were inconsistent in earlier passes; Windows per-engine counters were also
collected. Disabling PolyShade did not eliminate the rendering bottleneck.

Native source inspection additionally found that each car creates a separate
2048 × 2048 paint canvas/texture, despite the swarm using the same pattern.
Bridge 0.1.40 shares repeated paint textures and batches repeated non-body car
meshes into instanced draws. Batch matrices update once per car root per frame.
Body paint remains independent for per-run colors; lights and wheel/skid
effects retain their native paths. No visual features were disabled.

## Changes applied

- Paused playback skips unchanged transforms and native car updates while
  retaining chase-camera smoothing.
- Only the fixed camera car receives the native audio manager; silent cars
  avoid creating unused audio resources.
- Audio volume setters run only when volume changes.
- Reuse the interpolation scratch object and empty collision array; cache
  the swarm duration.
- Only select the chase camera when it differs from the active camera.
  Re-selecting it unnecessarily rebuilds native shadow-camera bounds.
- Share repeated car paint textures and instance repeated non-body meshes.
  Bridge status reports instanced batch and batched-part counts.
- Bridge status exposes rolling frame/update/render metrics. Render calls
  are accumulated across passes so post-processing does not hide scene cost.

The 0.1.40 bundle and syntax checks pass. The existing Edge tab is currently
in the track ghost viewer, which does not create the player car required to
connect the local replay bridge. A separate test tab is blocked by PolyTrack's
single-instance guard, so the 0.1.40 swarm load and controlled before/after
FPS measurement remain pending a live player race. The figures above describe
the earlier unbatched build and are not a measured speedup.

Raw observations are in local `logs/swarm-profile-*.json`, with Windows
counter samples in `logs/swarm-profile-100-gpu-counters.txt`.
