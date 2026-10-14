# Replay buffer storage

Off-policy TQC and GRTQC replay buffers are saved as gzip streams using
compression level 3. Checkpoints written by older versions as plain pickle
files remain loadable; new saves replace the replay atomically so an
interrupted write does not leave a partial resumable buffer.

Training startup retains the five newest replay buffers in numbered
`checkpoints/step-N` directories. After each periodic checkpoint, the same
retention rule runs again. Cleanup removes only `replay.pkl` from older
numbered checkpoints. It leaves policies, metadata, Champion, Latest, rejected
checkpoints, and visual replay captures intact. This retention bounds the
space used by historical resumable buffers while preserving the model
checkpoints for evaluation or inference.
