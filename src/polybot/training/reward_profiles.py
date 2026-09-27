"""Persistent named reward configurations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from polybot.environment.rewards import (
    RewardConfig,
    summer_1_bootstrap_pace_reward_config,
    summer_1_bootstrap_reward_config,
    summer_1_ghost_learning_reward_config,
    summer_1_pace_reward_config,
    summer_1_recovery_reward_config,
    summer_1_reward_config,
)


def balanced_reward_config() -> RewardConfig:
    """Small, readable default for a first run on either algorithm."""
    return RewardConfig(
        progress_per_m=2.0, elapsed_cost_per_s=-0.1,
        on_track_speed_per_m=0.3, ground_brake_penalty_per_s=-0.5,
        imitation_bonus_per_s=12, guidance_reward_scale=0,
        low_speed_penalty_per_s=-2.0, unsafe_speed_penalty_per_m=-0.2,
        barrier_collision_impulse_threshold=1e9,
        checkpoint_bonus=25, finish_bonus=1200,
        crash_penalty=-150, stall_penalty=-100, off_track_penalty=-100,
        action_change_penalty=-0.002,
    )


def learning_reward_config() -> RewardConfig:
    from dataclasses import replace

    return replace(balanced_reward_config(), progress_per_m=3.0,
                   checkpoint_bonus=50, curriculum_section_bonus=100)


def pace_reward_config() -> RewardConfig:
    from dataclasses import replace

    return replace(balanced_reward_config(), elapsed_cost_per_s=-0.5,
                   speed_pace_reward_per_m_per_mps=0.04,
                   finish_fast_bonus=1200, finish_target_s=24)

BUILTIN_REWARD_PROFILES = {
    "Balanced": balanced_reward_config,
    "Learning": learning_reward_config,
    "Pace": pace_reward_config,
    "Summer 1 - balanced": summer_1_reward_config,
    "Summer 1 - full bootstrap": summer_1_bootstrap_reward_config,
    "Summer 1 - bootstrap pace": summer_1_bootstrap_pace_reward_config,
    "Summer 1 - ghost learning": summer_1_ghost_learning_reward_config,
    "Summer 1 - recovery": summer_1_recovery_reward_config,
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
            json.dumps({
                "name": clean_name, "description": "Custom PolyBot reward profile",
                "track": None, "level": "advanced", "algorithm_neutral": True,
                "rewards": asdict(rewards),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def duplicate(self, source: str, target: str) -> Path:
        return self.save(target, self.load(source))

    def compare(self, left: str, right: str) -> dict[str, tuple[float | int, float | int]]:
        before = asdict(self.load(left))
        after = asdict(self.load(right))
        return {key: (value, after[key]) for key, value in before.items()
                if value != after[key]}
