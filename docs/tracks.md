# Track workspaces

PolyBot gives each registered track a stable slug. The display name can change,
but the slug—and therefore the track's data location—does not. Simulator IDs
such as `current` may be shared by multiple tracks; PolyBot's workspace and
model-compatibility checks use the slug to keep those tracks separate.

## Registry and migration

The persistent track catalogue is `config/tracks.json`. The GUI remembers the
last selected slug in `config/selected-track.json`; both files are local runtime
state and are ignored by Git. On first use, PolyBot discovers existing track
identities from model metadata, visual replay indexes, and track-named model
directories. Discovery does not move, rename, or rewrite model, replay, or log
files. Previously saved training configurations and model metadata without a
slug remain readable; their track name is converted to a stable slug when
loaded.

Manage the catalogue from a terminal:

```powershell
polybot-tracks list
polybot-tracks add "Autumn 2"
polybot-tracks rename autumn-2 "Autumn Two"
polybot-tracks remove autumn-2
```

Removing a registry entry does not delete its workspace. Register it again or
use the GUI's **Manage** dialog to add it back. Renaming a track retains its
slug and existing workspace.

The GUI has a global track selector plus **Add Track** and **Manage** actions.
The Models and Replay tabs follow the selected track. Training
configurations loaded from older files register and select their track rather
than silently using the currently selected workspace.

## Workspace layout

For the default roots, data is organized as follows:

```text
models/
  <track-slug>/
    <algorithm>/
      champion/
      latest/
      initialization/
      visual_replays/
        <run-id>/
          index.json
          episode-000001.npz
logs/
  <track-slug>/
    <algorithm>/
      <UTC timestamp>.jsonl
```

Model slots retain their existing names and contents. Visual replay indexes
include the track slug and training metadata, and each run is discoverable from
the Replay tab without searching other tracks. New training and
algorithm-sidecar logs are written under the matching track and algorithm.
Readers also continue to discover legacy flat log names such as
`logs/<track-slug>-<algorithm>-*.jsonl`.

The CLI accepts either a registered display name or slug. `--track-name`
remains as a backwards-compatible alias for training and saved-model commands:

```powershell
polybot-train --algorithm grtqc --backend websocket --track summer-1
polybot-eval --algorithm grtqc --track summer-1 --slot champion --backend websocket
```

The output and log roots remain configurable. Their per-track subdirectories
are refreshed in the GUI after editing either root.
