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
_COMPRESSION_LEVEL = 3
_CHECKPOINT_NAME = re.compile(r"^step-(\d+)$")
DEFAULT_CHECKPOINT_REPLAY_RETENTION = 5


class _GzipRawWriter(io.RawIOBase):
    def __init__(self, raw: io.BufferedWriter) -> None:
        super().__init__()
        self._stream = gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=_COMPRESSION_LEVEL, mtime=0
        )

    def writable(self) -> bool:
        return True

    def write(self, data: bytes | bytearray | memoryview) -> int:
        return self._stream.write(data)

    def flush(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
        super().flush()

    def close(self) -> None:
        if not self.closed:
            self._stream.close()
        super().close()


class _GzipRawReader(io.RawIOBase):
    def __init__(self, raw: io.BufferedReader) -> None:
        super().__init__()
        self._stream = gzip.GzipFile(fileobj=raw, mode="rb")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:
        data = self._stream.read(len(buffer))
        size = len(data)
        buffer[:size] = data
        return size

    def close(self) -> None:
        if not self.closed:
            self._stream.close()
        super().close()


def save_replay_buffer(model: Any, path: str | Path) -> None:
    """Save SB3 replay state as gzip-compressed pickle via an atomic replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with io.BufferedWriter(_GzipRawWriter(raw), buffer_size=1024 * 1024) as compressed:
                model.save_replay_buffer(compressed)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_replay_buffer(model: Any, path: str | Path) -> None:
    """Load compressed buffers and legacy uncompressed SB3 pickle buffers."""
    source_path = Path(path)
    with source_path.open("rb") as source:
        compressed = source.read(2) == _GZIP_MAGIC
        if compressed:
            source.seek(0)
            with io.BufferedReader(_GzipRawReader(source), buffer_size=1024 * 1024) as stream:
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
