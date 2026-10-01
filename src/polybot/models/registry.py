"""v2-only model layout and strict metadata compatibility."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polybot.environment.observations import SCHEMA as OBSERVATION_SCHEMA

MODEL_SCHEMA = "polybot.model.v2"
POLYBOT_VERSION = "2.3.0"
REWARD_SEMANTICS = "executed-controls-v1"
PPO_ACTION_SEMANTICS = "steering_signed_longitudinal_v1"


class IncompatibleModelError(ValueError):
    pass


def track_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("track name needs letters or digits")
    return slug


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


@dataclass(slots=True)
class ModelMetadata:
    algorithm: str
    architecture: str
    actor_parameters: int
    critic_parameters: int
    total_trainable_parameters: int
    observation_schema: str
    action_schema: str
    track_name: str
    track_id: str
    lookahead_count: int
    reward_profile: str | None
    curriculum: dict[str, Any]
    training_config: dict[str, Any]
    training_timesteps: int
    simulator_ticks: int
    wall_seconds: float
    seed: int
    device: str
    finishes: int
    crashes: int
    evaluation: dict[str, Any] | None = None
    implementation: str | None = None
    reward_semantics: str | None = None
    critic_adaptation_required: bool = False
    adaptation_stage: str | None = None
    adaptation_rollback_count: int = 0
    policy_overlays: list[dict[str, Any]] = field(default_factory=list)
    speed_bias_schedule: list[list[float]] = field(default_factory=list)
    action_semantics: str | None = None
    schema: str = MODEL_SCHEMA
    polybot_version: str = POLYBOT_VERSION
    git_commit: str = field(default_factory=git_commit)
    saved_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if self.schema != MODEL_SCHEMA or not self.polybot_version.startswith("2."):
            raise IncompatibleModelError("only v2 models are supported")


class ModelRegistry:
    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)

    def algorithm_dir(self, track_name: str, algorithm: str) -> Path:
        from polybot.algorithms.registry import backend_for

        backend_for(algorithm)
        return self.root / track_slug(track_name) / algorithm

    def slot(self, track_name: str, algorithm: str, name: str) -> Path:
        if name not in {"initialization", "latest", "champion"} and not name.startswith("checkpoints/step-"):
            raise ValueError("unknown v2 model slot")
        return self.algorithm_dir(track_name, algorithm) / name

    def write_metadata(self, directory: Path, metadata: ModelMetadata) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metadata.json").write_text(
            json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8"
        )

    def read_metadata(self, directory: Path) -> ModelMetadata:
        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if payload.get("schema") != MODEL_SCHEMA:
            raise IncompatibleModelError("only v2 model metadata is supported")
        return ModelMetadata(**payload)

    def validate(self, metadata: ModelMetadata, config: Any, action_schema: str) -> None:
        if metadata.algorithm == "ppo" and metadata.action_schema == "pwm-multidiscrete-v2":
            raise IncompatibleModelError(
                "legacy PPO checkpoint uses discrete PWM actions and cannot resume as continuous PPO; "
                "start a new continuous model or distill a compatible TQC teacher"
            )
        if metadata.algorithm == "ppo" and metadata.action_semantics != PPO_ACTION_SEMANTICS:
            raise IncompatibleModelError(
                "PPO checkpoint does not declare continuous steering and signed longitudinal actions; "
                "start a fresh model or use the TQC-to-PPO teacher pipeline"
            )
        mismatches = []
        for name, actual, expected in (
            ("algorithm", metadata.algorithm, config.algorithm),
            ("track", metadata.track_id, config.track_id),
            ("observation", metadata.observation_schema, OBSERVATION_SCHEMA),
            ("action", metadata.action_schema, action_schema),
            ("lookahead", metadata.lookahead_count, config.lookahead_count),
        ):
            if actual != expected:
                mismatches.append(name)
        if mismatches:
            raise IncompatibleModelError("incompatible " + ", ".join(mismatches))
