# Replay buffer storage

Off-policy TQC and GRTQC replay buffers are saved as checksummed Zstandard
streams at level 3, using two compression workers. The filename remains
`replay.pkl`; the unchanged SB3 pickle payload preserves array dtype, shape,
values, buffer position, timeouts and n-step state. Readers identify the
container by magic bytes and still load historical gzip and plain pickle
files without migration. Existing model/replay files are not rewritten.

Python 3.14 uses `compression.zstd`; Python 3.11–3.13 installs
`backports.zstd` through the `train` extra. Reinstall with
`python -m pip install -e ".[train]"` after updating an older environment.
Older PolyBot revisions cannot read newly written Zstandard replay containers;
use the updated loader when continuing those checkpoints. Model metadata and
the decoded replay schema are unchanged.

Serialization streams through 1 MiB memoryview slices, avoiding whole-array
compression output temporaries. Compression jobs may overlap inside libzstd,
but saving remains synchronous: the replay cannot change while it is saved,
and all jobs finish before flush/fsync and atomic replacement. Failed saves
remove only their owned temporary file and retain the previous replay.
The loader consumes the Zstandard trailer to verify its checksum even when
pickle stops before the end of the compressed frame.

See [replay and native swarm measurements](replay-swarm-optimization.md) for
codec comparisons, memory figures and compatibility tests.

Training startup retains the five newest replay buffers in numbered
`checkpoints/step-N` directories. After each periodic checkpoint, the same
retention rule runs again. Cleanup removes only `replay.pkl` from older
numbered checkpoints. It leaves policies, metadata, Champion, Latest, rejected
checkpoints, and visual replay captures intact. This retention bounds the
space used by historical resumable buffers while preserving the model
checkpoints for evaluation or inference.
