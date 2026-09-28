"""Promote a complete, evaluated checkpoint without partially replacing a champion."""

from __future__ import annotations

import os
import time
from pathlib import Path


def promote_directory(
    staging: Path, champion: Path, *, require_replay: bool = False,
) -> Path | None:
    """Swap sibling directories, retaining the old champion as a recovery backup."""
    staging = staging.resolve()
    champion = champion.resolve()
    if staging.parent != champion.parent or not staging.is_dir():
        raise ValueError("staging must be a complete sibling directory")
    for filename in ("policy.zip", "metadata.json"):
        if not (staging / filename).is_file():
            raise FileNotFoundError(staging / filename)
    if require_replay and not (staging / "replay.pkl").is_file():
        raise FileNotFoundError(staging / "replay.pkl")
    backup = None
    if champion.exists():
        index = 1
        while (champion.parent / f"{champion.name}-backup-{index}").exists():
            index += 1
        backup = champion.parent / f"{champion.name}-backup-{index}"
        os.replace(champion, backup)
    try:
        # Windows virus scanners can briefly hold a freshly written replay file.
        # Retry only the transient access error while the old checkpoint remains
        # safe in backup; never copy individual files over the live champion.
        for attempt in range(8):
            try:
                os.replace(staging, champion)
                break
            except PermissionError:
                if attempt == 7 or champion.exists():
                    raise
                time.sleep(0.25 * (attempt + 1))
    except Exception:
        if backup is not None:
            os.replace(backup, champion)
        raise
    return backup
