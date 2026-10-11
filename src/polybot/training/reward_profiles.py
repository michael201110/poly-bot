"""Persistent named reward configurations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, replace
from pathlib import Path

from polybot.environment.rewards import RewardConfig


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
    return replace(balanced_reward_config(), progress_per_m=3.0,
                   checkpoint_bonus=50, curriculum_section_bonus=100)


def scratch_completion_reward_config() -> RewardConfig:
    """Completion-first rewards from the documented independent GRTQC scratch run."""
    return replace(
        balanced_reward_config(),
        progress_per_m=2.0,
        elapsed_cost_per_s=-10.0,
        on_track_speed_per_m=0.0,
        speed_pace_reward_per_m_per_mps=0.01,
        speed_pace_limit_mps=55.0,
        ground_brake_penalty_per_s=-1.5,
        imitation_bonus_per_s=0.0,
        expert_action_bonus_per_s=0.0,
        ghost_speed_bonus_per_s=0.0,
        guidance_reward_scale=0.0,
        low_speed_penalty_per_s=-2.0,
        unsafe_speed_penalty_per_m=0.0,
        barrier_contact_penalty=-200.0,
        barrier_collision_impulse_threshold=250.0,
        failure_progress_clawback_per_m=-1.0,
        airborne_roll_timeout_s=1.0,
        airborne_roll_failure_penalty=-400.0,
        checkpoint_bonus=50.0,
        finish_bonus=2_000.0,
        finish_fast_bonus=15_000.0,
        finish_target_s=20.0,
        finish_pace_decay_per_s=0.15,
        curriculum_section_bonus=0.0,
        crash_penalty=-400.0,
        stall_penalty=-400.0,
        off_track_penalty=-400.0,
        early_off_track_penalty=-400.0,
    )


def pace_reward_config() -> RewardConfig:
    """Reward quick progress while retaining functional contact and failure signals."""
    return replace(
        balanced_reward_config(),
        elapsed_cost_per_s=-2.0,
        speed_pace_reward_per_m_per_mps=0.04,
        finish_fast_bonus=5_000.0,
        finish_target_s=30.0,
        finish_pace_decay_per_s=0.15,
        ground_brake_penalty_per_s=0.0,
        imitation_bonus_per_s=0.0,
        barrier_contact_penalty=-100.0,
        barrier_speed_loss_penalty_per_mps=-50.0,
        barrier_collision_impulse_threshold=250.0,
        early_off_track_penalty=-100.0,
    )


def pace_failure_x3_reward_config() -> RewardConfig:
    """Pace rewards with threefold contact and terminal failure costs."""
    pace = pace_reward_config()
    return replace(
        pace,
        barrier_contact_penalty=-300.0,
        barrier_speed_loss_penalty_per_mps=pace.barrier_speed_loss_penalty_per_mps * 3.0,
        crash_penalty=pace.crash_penalty * 3.0,
        stall_penalty=pace.stall_penalty * 3.0,
        off_track_penalty=pace.off_track_penalty * 3.0,
        early_off_track_penalty=pace.early_off_track_penalty * 3.0,
    )


def pure_pace_reward_config() -> RewardConfig:
    """Time-only rewards for an existing finisher, with terminal failure costs."""
    return RewardConfig(
        elapsed_cost_per_s=-10.0,
        imitation_bonus_per_s=0.0,
        guidance_reward_scale=0.0,
        low_speed_penalty_per_s=0.0,
        barrier_contact_penalty=0.0,
        barrier_collision_impulse_threshold=250.0,
        finish_bonus=2_000.0,
        finish_fast_bonus=5_000.0,
        finish_target_s=32.0,
        finish_pace_decay_per_s=0.15,
        crash_penalty=-1_000.0,
        stall_penalty=-1_000.0,
        off_track_penalty=-1_000.0,
        early_off_track_penalty=-1_000.0,
    )


BUILTIN_REWARD_PROFILES = {
    "Balanced": balanced_reward_config,
    "Learning": learning_reward_config,
    "Scratch completion": scratch_completion_reward_config,
    "Pace": pace_reward_config,
    "Pace + failure x3": pace_failure_x3_reward_config,
    "Pure Pace": pure_pace_reward_config,
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
