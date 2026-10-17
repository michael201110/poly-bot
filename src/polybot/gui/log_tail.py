"""Bounded JSONL reads for the UI thread, with byte-exact partial-line retry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_event_batch(
    path: Path, position: int, *, max_events: int = 256, max_bytes: int = 1024 * 1024,
) -> tuple[list[dict[str, Any]], int, bool]:
    """Return complete events, the committed byte offset, and a catch-up hint.

    Never commit a partial/invalid line. A single large complete event is
    allowed through the byte budget so a log cannot get permanently stuck.
    """
    if max_events < 1 or max_bytes < 1:
        raise ValueError("log read budgets must be positive")
    events = []
    with path.open("rb") as stream:
        stream.seek(position)
        while len(events) < max_events and stream.tell() - position < max_bytes:
            before = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                stream.seek(before)
                break
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                stream.seek(before)
                break
            events.append(event)
        offset = stream.tell()
    limited = len(events) == max_events or offset - position >= max_bytes
    return events, offset, limited
