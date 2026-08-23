"""Cross-platform recovery and curriculum orchestration."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from polybot.training.config import CurriculumConfig, TrainingConfig
from polybot.training.trainer import TrainingService


class TrainingManager:
    def __init__(
        self, config: TrainingConfig, status: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        self.config = config
        self.status = status
        self._stop = threading.Event()
        self.active: TrainingService | None = None

    def stop(self) -> None:
        self._stop.set()
        if self.active:
            self.active.stop()

    def phases(self) -> list[CurriculumConfig]:
        if self.config.curriculum.mode == "quarters":
            return [CurriculumConfig("section", i / 4, (i + 1) / 4) for i in range(4)] + [
                CurriculumConfig()
            ]
        return [self.config.curriculum]

    def run(
        self,
        *,
        repeat: bool = False,
        resume: str | Path | None = None,
        transport_factory: Callable[[], Any] | None = None,
    ) -> Path | None:
        latest: Path | None = Path(resume) if resume else None
        for phase in self.phases():
            self.config.curriculum = phase
            while not self._stop.is_set():
                try:
                    self.active = TrainingService(self.config, self.status)
                    latest = self.active.run(
                        resume=latest, transport=transport_factory() if transport_factory else None
                    )
                    break
                except Exception as exc:
                    if self.status:
                        self.status({"type": "error", "message": str(exc)})
                    if not repeat:
                        raise
            if self._stop.is_set():
                break
        return latest
