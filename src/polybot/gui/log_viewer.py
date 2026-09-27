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
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
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
                    self.status.setText(
                        f"Step {event['timesteps']:,} · {event['steps_per_second']:.1f} TPS · "
                        f"attempt {event['progress']:.1%} · best seen {event['run_max_progress']:.1%} · "
                        f"updates {event.get('updates', 0):,}\nFull detail: {self.path}"
                    )
            self.position = stream.tell()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a running PolyBot training log in readable form")
    parser.add_argument("log", type=Path, help="JSONL file written by PolyBot training")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = LiveLogWindow(args.log)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
