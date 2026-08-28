"""Persistent named reward configurations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from polybot.env import (
    RewardConfig,
    summer_1_bootstrap_reward_config,
    summer_1_pace_reward_config,
    summer_1_reward_config,
)

BUILTIN_REWARD_PROFILES = {
    "Summer 1 - balanced": summer_1_reward_config,
    "Summer 1 - full bootstrap": summer_1_bootstrap_reward_config,
    "Summer 1 - pace": summer_1_pace_reward_config,
}


def profile_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("profile name must contain letters or numbers")
    return slug


class RewardProfileStore:
    def __init__(self, root: str | Path = "profiles/rewards") -> None:
        self.root = Path(root)

    def names(self) -> list[str]:
        custom = []
        if self.root.exists():
            for path in self.root.glob("*.json"):
                try:
                    custom.append(str(json.loads(path.read_text(encoding="utf-8"))["name"]))
                except (KeyError, TypeError, json.JSONDecodeError, OSError):
                    continue
        return [*BUILTIN_REWARD_PROFILES, *sorted(set(custom) - BUILTIN_REWARD_PROFILES.keys())]

    def load(self, name: str) -> RewardConfig:
        path = self.root / f"{profile_slug(name)}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return RewardConfig(**payload["rewards"])
        try:
            return BUILTIN_REWARD_PROFILES[name]()
        except KeyError as exc:
            raise FileNotFoundError(f"unknown reward profile: {name}") from exc

    def save(self, name: str, rewards: RewardConfig) -> Path:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("profile name cannot be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{profile_slug(clean_name)}.json"
        path.write_text(
            json.dumps({"name": clean_name, "rewards": asdict(rewards)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return path
