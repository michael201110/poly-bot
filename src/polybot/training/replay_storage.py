"""Atomic, backward-compatible compressed replay-buffer storage."""

from __future__ import annotations

import gzip
import io
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_GZIP_MAGIC = b"\x1f\x8b"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_COMPRESSION_LEVEL = 3
_CHECKPOINT_NAME = re.compile(r"^step-(\d+)$")
DEFAULT_CHECKPOINT_REPLAY_RETENTION = 5
_STREAM_CHUNK_BYTES = 1024 * 1024


def _zstd():
    # Python 3.14 provides the same API as its maintained backport. Import only
    # for replay I/O so inference-only installations need no codec dependency.
    try:
        from compression import zstd
    except ImportError:
        try:
            from backports import zstd
        except ImportError as exc:
            raise RuntimeError("Zstandard replay support requires Python 3.14 or pip install backports.zstd") from exc
    return zstd


class _ChunkedWriter(io.RawIOBase):
    """Bound Python compressor output allocations without copying input arrays."""

    def __init__(self, stream):
        super().__init__()
        self._stream = stream

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        view = memoryview(data).cast("B")
        for offset in range(0, view.nbytes, _STREAM_CHUNK_BYTES):
            self._stream.write(view[offset:offset + _STREAM_CHUNK_BYTES])
        return view.nbytes

    def flush(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
        super().flush()

    def close(self) -> None:
        if not self.closed:
            self._stream.close()
        super().close()


def save_replay_buffer(model: Any, path: str | Path) -> None:
    """Stream unchanged SB3 pickle through checksummed Zstandard, then replace.

    Compression workers consume bounded libzstd jobs while this call retains
    ownership of the model/replay. It does not return until all jobs and fsync
    finish; no replay mutation or background snapshot race is introduced.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            zstd = _zstd()
            options = {
                zstd.CompressionParameter.compression_level: _COMPRESSION_LEVEL,
                zstd.CompressionParameter.checksum_flag: 1,
                zstd.CompressionParameter.nb_workers: 2,
            }
            writer = _ChunkedWriter(zstd.ZstdFile(raw, "wb", options=options))
            with io.BufferedWriter(writer, buffer_size=_STREAM_CHUNK_BYTES) as compressed:
                model.save_replay_buffer(compressed)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_replay_buffer(model: Any, path: str | Path) -> None:
    """Read Zstandard, historical gzip and plain SB3 pickle without migration."""
    source_path = Path(path)
    with source_path.open("rb") as source:
        magic = source.read(4)
        source.seek(0)
        if magic == _ZSTD_MAGIC:
            with io.BufferedReader(_zstd().ZstdFile(source, "rb"), buffer_size=1024 * 1024) as stream:
                model.load_replay_buffer(stream)
                # Pickle stops at STOP, before a codec's final checksum may
                # have been consumed. Validate the complete frame/trailer.
                while stream.read(1024 * 1024):
                    pass
            return
        if magic[:2] == _GZIP_MAGIC:
            with io.BufferedReader(gzip.GzipFile(fileobj=source, mode="rb"), buffer_size=1024 * 1024) as stream:
                model.load_replay_buffer(stream)
            return
    model.load_replay_buffer(str(source_path))


def prune_stale_checkpoint_replays(
    checkpoint_root: str | Path,
    *,
    keep: int = DEFAULT_CHECKPOINT_REPLAY_RETENTION,
) -> list[Path]:
    """Remove replay buffers from older numbered checkpoints, preserving models.

    Only direct ``step-N/replay.pkl`` children of a directory named
    ``checkpoints`` are eligible. Rejected checkpoints and other model slots
    are intentionally excluded.
    """
    root = Path(checkpoint_root)
    if root.name.lower() != "checkpoints":
        raise ValueError("stale replay cleanup requires a checkpoints directory")
    if keep < 0:
        raise ValueError("checkpoint replay retention cannot be negative")
    if not root.is_dir():
        return []

    checkpoints: list[tuple[int, Path, Path]] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        match = _CHECKPOINT_NAME.fullmatch(directory.name)
        if match is None:
            continue
        replay = directory / "replay.pkl"
        if replay.is_file():
            checkpoints.append((int(match.group(1)), directory, replay))

    checkpoints.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    removed: list[Path] = []
    for _, _, replay in checkpoints[keep:]:
        replay.unlink()
        removed.append(replay)
    return removed
