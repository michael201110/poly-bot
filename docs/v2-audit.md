# v2 baseline and architectural audit

Recorded before changing source on 27 September 2026, on `main` at `1b2a51b`:

- `python -m pytest`: 157 passed, 1 failed. The existing failure was `test_playback_rejects_wrong_track_or_action_schema`, caused by a TQC observation-schema mismatch between saved metadata and CLI validation.
- `python -m ruff check .`: passed.
- `python tools/validate_pml_mod.py`: manifests passed; raw PolyTrack bundles were not supplied, so their pinned hashes and code anchors were not rechecked in this baseline.

The old source was organized around a large environment and flat protocol/mock/transport modules, with training split among a manager, trainer, algorithm switch, model registry, and several PPO/TQC custom subclasses. The old GUI and CLI repeated algorithm-specific parameters. `TrainingManager` applied the complete `timesteps` budget to each sequential curriculum phase. Training episodes could promote `best.zip` without a frozen full-track evaluation. TQC combined replay warmup with a forward guard, prior, safe actor snapshots, critic probing, rehearsal, and collapse recovery. The environment masked ghost observation features only for TQC and branched on action mode. V1 model migration, distillation, and bridge code remained. The PML mod manifest targets PolyTrack 0.6.2 and 0.6.3 with mod 0.1.29; its protocol is digital and independent of learning algorithm.

The v2 refactor replaces those paths with action adapters, shared observations, named reward components, typed algorithm backends, a global-budget curriculum plan, deterministic evaluation and champion selection, v2-only metadata, and CLI/GUI configuration parity. The wire protocol and PML source remain unchanged.

The user explicitly authorized removal of all old model artifacts in both `models/` and `old-models-pending-delete/`. Automatic approval review rejected the narrow recursive deletion command for those repository directories. The directories were left untouched; `.gitignore` prevents them from entering the v2 branch. No Git history was rewritten.
