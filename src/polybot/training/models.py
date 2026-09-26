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

OBSERVATION_SCHEMA = "polybot.telemetry.v2"


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
    tqc_hyperparameters: dict[str, Any] = field(default_factory=dict)
    reward_profile: str | None = None
    simulator_ticks: int = 0
    wall_clock_seconds: float = 0.0
    finishes: int = 0
    crashes: int = 0
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

    def model_dir(self, track_name: str, algorithm: str) -> Path:
        if algorithm.lower() not in {"ppo", "tqc"}:
            raise ValueError("algorithm must be ppo or tqc")
        return self.track_dir(track_name) / algorithm.lower()

    def initialise_track(self, track_name: str, algorithm: str | None = None) -> Path:
        directory = (
            self.model_dir(track_name, algorithm) if algorithm else self.track_dir(track_name)
        )
        (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
        return directory

    def metadata_path(
        self, track_name: str, name: str = "best", algorithm: str | None = None
    ) -> Path:
        directory = (
            self.model_dir(track_name, algorithm) if algorithm else self.track_dir(track_name)
        )
        return directory / ("metadata.json" if name == "best" else f"{name}.metadata.json")

    def write_metadata(
        self, metadata: ModelMetadata, name: str = "best", algorithm: str | None = None
    ) -> Path:
        self.initialise_track(metadata.track_name, algorithm)
        path = self.metadata_path(metadata.track_name, name, algorithm)
        path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
        return path

    def read_metadata(
        self, track_name: str, name: str = "best", algorithm: str | None = None
    ) -> ModelMetadata:
        return ModelMetadata.from_dict(
            json.loads(self.metadata_path(track_name, name, algorithm).read_text(encoding="utf-8"))
        )

    def list_models(self, track_name: str, algorithm: str | None = None) -> list[Path]:
        directory = self.track_dir(track_name)
        if not directory.exists():
            return []
        legacy = list(directory.glob("*.zip")) if algorithm in {None, "ppo"} else []
        if algorithm is None:
            scoped = [*directory.glob("ppo/*.zip"), *directory.glob("tqc/*.zip")]
        else:
            scoped = list(self.model_dir(track_name, algorithm).glob("*.zip"))
        return sorted(scoped) + sorted(legacy) if algorithm else sorted([*legacy, *scoped])

    @staticmethod
    def metadata_for_archive(path: str | Path) -> ModelMetadata:
        archive = Path(path)
        metadata = archive.with_name(
            "metadata.json" if archive.stem == "best" else f"{archive.stem}.metadata.json"
        )
        return ModelMetadata.from_dict(json.loads(metadata.read_text(encoding="utf-8")))

    def assert_compatible(
        self,
        metadata: ModelMetadata,
        *,
        track_name: str,
        observation_schema: str = OBSERVATION_SCHEMA,
        action_schema: str,
        algorithm: str = "ppo",
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
        if metadata.algorithm.lower() != algorithm.lower():
            mismatches.append("algorithm")
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
        directory = self.initialise_track(metadata.track_name, metadata.algorithm)
        target = directory / "best.zip"
        if metadata.algorithm.lower() == "tqc":
            replay = candidate.with_suffix(".replay.pkl")
            if not replay.is_file():
                raise FileNotFoundError(f"TQC replay buffer is missing: {replay}")
        shutil.copy2(candidate, target)
        if metadata.algorithm.lower() == "tqc":
            shutil.copy2(replay, target.with_suffix(".replay.pkl"))
        self.write_metadata(metadata, algorithm=metadata.algorithm)
        return target

    def archive_latest(self, track_name: str, algorithm: str | None = None) -> Path | None:
        """Preserve the current latest pair before intentionally starting fresh."""

        directory = (
            self.model_dir(track_name, algorithm) if algorithm else self.track_dir(track_name)
        )
        latest = directory / "latest.zip"
        if not latest.exists():
            return None
        checkpoint_dir = self.initialise_track(track_name, algorithm) / "checkpoints"
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = checkpoint_dir / f"pre-fresh-{stamp}.zip"
        shutil.copy2(latest, target)
        metadata = directory / "latest.metadata.json"
        if metadata.exists():
            shutil.copy2(metadata, target.with_suffix(".metadata.json"))
        replay = latest.with_suffix(".replay.pkl")
        if replay.exists():
            shutil.copy2(replay, target.with_suffix(".replay.pkl"))
        return target


def git_commit(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_ppo(
    path: str | Path,
    *,
    env: Any = None,
    device: str = "auto",
    custom_objects: dict[str, Any] | None = None,
) -> Any:
    """Load portably; SB3 remaps tensors to the explicitly selected device."""
    from stable_baselines3 import PPO

    return PPO.load(str(path), env=env, device=device, custom_objects=custom_objects or {})
