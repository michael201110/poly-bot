"""Readable live view of a running PolyBot JSONL log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QTextEdit, QVBoxLayout, QWidget

from polybot.gui.events import format_event


class LiveLogWindow(QWidget):
    def __init__(self, path: Path, *, follow_newest: bool = False) -> None:
        super().__init__()
        self.source = path
        self.path = path
        self.follow_newest = follow_newest
        self.position = 0
        self.setWindowTitle("PolyBot · readable training log")
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        self.status = QLabel(f"Waiting for {path}")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.events = QTextEdit()
        self.events.setReadOnly(True)
        self.events.document().setMaximumBlockCount(400)
        layout.addWidget(self.events)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()

    def refresh(self) -> None:
        if self.follow_newest:
            candidates = list(self.source.parent.glob(self.source.name))
            if candidates:
                newest = max(candidates, key=lambda item: (item.stat().st_mtime, item.name))
                if newest != self.path:
                    self.path = newest
                    self.position = 0
                    self.events.clear()
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as stream:
            stream.seek(self.position)
            while True:
                before = stream.tell()
                line = stream.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    stream.seek(before)
                    break
                summary = format_event(event)
                if summary:
                    self.events.append(summary)
                if event.get("type") == "progress":
                    learning = ""
                    if "actor_unlocked" in event:
                        learning = (
                            "\nPolicy learning"
                            if event["actor_unlocked"]
                            else f"\nPolicy frozen · critic warmup {event.get('critic_warmup_updates', 0):,} updates"
                        )
                    self.status.setText(
                        f"Step {event['timesteps']:,} · {event['steps_per_second']:.1f} TPS · "
                        f"attempt {event['progress']:.1%} · best seen {event['run_max_progress']:.1%} · "
                        f"updates {event.get('updates', 0):,}{learning}\nFull detail: {self.path}"
                    )
            self.position = stream.tell()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a running PolyBot training log in readable form")
    parser.add_argument("log", type=Path, help="JSONL file written by PolyBot training")
    parser.add_argument("--follow-newest", action="store_true",
                        help="treat log as a glob and switch to the newest matching run")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = LiveLogWindow(args.log, follow_newest=args.follow_newest)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
