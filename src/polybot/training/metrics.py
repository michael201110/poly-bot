"""Small structured JSONL event sink shared by CLI and GUI."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class EventSink:
    def __init__(self, path: Path, listener: Callable[[dict[str, Any]], None] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.listener = listener
        self._stream = path.open("w", encoding="utf-8")

    def emit(self, event: dict[str, Any]) -> None:
        event = {"at": datetime.now(UTC).isoformat(), **event}
        self._stream.write(json.dumps(event, allow_nan=False) + "\n")
        self._stream.flush()
        if self.listener is not None:
            self.listener(event)

    def close(self) -> None:
        self._stream.close()
