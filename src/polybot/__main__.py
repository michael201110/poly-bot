"""Run installed PolyBot commands through python -m polybot."""

from __future__ import annotations

import sys

from polybot.cli import drive_main, evaluate_main, smoke_main, train_main


def main() -> int:
    commands = {
        "train": train_main, "evaluate": evaluate_main,
        "drive": drive_main, "smoke": smoke_main,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("usage: python -m polybot {train,evaluate,drive,smoke} [options]", file=sys.stderr)
        return 2
    return commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
