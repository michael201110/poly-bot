"""Track-scoped model registry and compatibility metadata."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OBSERVATION_SCHEMA = "polybot.telemetry.v1"


def track_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("track name must contain letters or numbers")
    return slug


@dataclass(slots=True)
class ModelMetadata:
    track_name: str
    track_id: str
    architecture: str
    parameter_count: int
    algorithm: str = "PPO"
    observation_schema: str = OBSERVATION_SCHEMA
    lookahead_count: int = 12
    action_schema: str = "pwm-multidiscrete-v1"
    pwm_enabled: bool = True
    pwm_resolution: int = 41
    frame_skip: int = 30
    training_timesteps: int = 0
    training_episodes: int | None = None
    best_lap_time_s: float | None = None
    training_date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    seed: int = 0
    reward_settings: dict[str, Any] = field(default_factory=dict)
    ppo_hyperparameters: dict[str, Any] = field(default_factory=dict)
    polybot_version: str = "unknown"
    git_commit: str = "unknown"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModelMetadata:
        return cls(**value)


class IncompatibleModelError(ValueError):
    pass


class ModelRegistry:
    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)

    def track_dir(self, track_name: str) -> Path:
        return self.root / track_slug(track_name)

    def initialise_track(self, track_name: str) -> Path:
        directory = self.track_dir(track_name)
        (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        return directory

    def metadata_path(self, track_name: str, name: str = "best") -> Path:
        directory = self.track_dir(track_name)
        return directory / ("metadata.json" if name == "best" else f"{name}.metadata.json")

    def write_metadata(self, metadata: ModelMetadata, name: str = "best") -> Path:
        self.initialise_track(metadata.track_name)
        path = self.metadata_path(metadata.track_name, name)
        path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
        return path

    def read_metadata(self, track_name: str, name: str = "best") -> ModelMetadata:
        return ModelMetadata.from_dict(
            json.loads(self.metadata_path(track_name, name).read_text(encoding="utf-8"))
        )

    def list_models(self, track_name: str) -> list[Path]:
        directory = self.track_dir(track_name)
        return sorted(directory.glob("*.zip")) if directory.exists() else []

    def assert_compatible(
        self,
        metadata: ModelMetadata,
        *,
        track_name: str,
        observation_schema: str = OBSERVATION_SCHEMA,
        action_schema: str,
        architecture: str | None = None,
        allow_track_override: bool = False,
    ) -> None:
        if track_slug(metadata.track_name) != track_slug(track_name) and not allow_track_override:
            message = (
                f"model is for {metadata.track_name!r}, not {track_name!r}; "
                "explicit override required"
            )
            raise IncompatibleModelError(message)
        mismatches = []
        if metadata.observation_schema != observation_schema:
            mismatches.append("observation schema")
        if metadata.action_schema != action_schema:
            mismatches.append("action schema")
        if architecture and metadata.architecture.lower() != architecture.lower():
            mismatches.append("architecture")
        if mismatches:
            raise IncompatibleModelError("incompatible " + ", ".join(mismatches))

    def promote(self, candidate: str | Path, metadata: ModelMetadata) -> Path:
        candidate = Path(candidate)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        directory = self.initialise_track(metadata.track_name)
        target = directory / "best.zip"
        shutil.copy2(candidate, target)
        self.write_metadata(metadata)
        return target


def git_commit(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_ppo(path: str | Path, *, env: Any = None, device: str = "auto") -> Any:
    """Load portably; SB3 remaps tensors to the explicitly selected device."""
    from stable_baselines3 import PPO

    return PPO.load(str(path), env=env, device=device)
