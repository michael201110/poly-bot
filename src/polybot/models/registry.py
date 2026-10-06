"""v2-only model layout and strict metadata compatibility."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from polybot.environment.observations import schema_for
from polybot.tracks import TrackDefinition, TrackWorkspace
from polybot.tracks import track_slug as canonical_track_slug

MODEL_SCHEMA = "polybot.model.v2"
POLYBOT_VERSION = "2.3.0"
REWARD_SEMANTICS = "nonterminal-contact-v3-deadline"
PPO_ACTION_SEMANTICS = "steering_signed_longitudinal_v1"
track_slug = canonical_track_slug


class IncompatibleModelError(ValueError):
    pass


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
    best_training_lap_s: float | None = None
    action_semantics: str | None = None
    track_slug: str | None = None
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

    def algorithm_dir(
        self,
        track_name: str | TrackDefinition,
        algorithm: str,
        *,
        track_slug: str | None = None,
    ) -> Path:
        from polybot.algorithms.registry import backend_for

        backend_for(algorithm)
        if isinstance(track_name, TrackDefinition):
            track = track_name
        else:
            slug = track_slug or canonical_track_slug(track_name)
            track = TrackDefinition(track_name, slug)
        return TrackWorkspace(track, self.root).algorithm_models(algorithm)

    def slot(
        self,
        track_name: str | TrackDefinition,
        algorithm: str,
        name: str,
        *,
        track_slug: str | None = None,
    ) -> Path:
        if name not in {"initialization", "latest", "champion", "contact-candidate"} and not name.startswith(
            "checkpoints/step-"
        ):
            raise ValueError("unknown v2 model slot")
        slot_path = Path(name)
        if slot_path.is_absolute() or ".." in slot_path.parts:
            raise ValueError("model slot cannot escape the track workspace")
        return self.algorithm_dir(track_name, algorithm, track_slug=track_slug) / slot_path

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
        if metadata.algorithm == "grtqc" and (
            metadata.training_config.get("grtqc", {}).get("training_origin", "transfer")
            != config.grtqc.training_origin
        ):
            raise IncompatibleModelError("scratch and transferred GRTQC experiments cannot share model/replay")
        if getattr(config.grtqc, "critic_raw_actions", False) and metadata.action_semantics != "grtqc.raw-policy.v1":
            raise IncompatibleModelError("GRTQC raw-policy learning requires matching replay action semantics")
        mismatches = []
        for name, actual, expected in (
            ("algorithm", metadata.algorithm, config.algorithm),
            ("track", metadata.track_id, config.track_id),
            (
                "track identity",
                metadata.track_slug or canonical_track_slug(metadata.track_name),
                getattr(config, "track_slug", None) or canonical_track_slug(config.track_name),
            ),
            ("observation", metadata.observation_schema, schema_for(config)),
            ("action", metadata.action_schema, action_schema),
            ("lookahead", metadata.lookahead_count, config.lookahead_count),
        ):
            if actual != expected:
                mismatches.append(name)
        if mismatches:
            raise IncompatibleModelError("incompatible " + ", ".join(mismatches))

    def list_algorithms(self, track: str | TrackDefinition, *, track_slug: str | None = None) -> list[str]:
        base = self.algorithm_dir(track, "grtqc", track_slug=track_slug).parent
        return sorted(
            algorithm for algorithm in ("grtqc", "tqc", "ppo")
            if (base / algorithm).is_dir()
        )

    def list_model_slots(
        self,
        track: str | TrackDefinition,
        algorithm: str,
        *,
        track_slug: str | None = None,
    ) -> list[tuple[str, Path, ModelMetadata | None]]:
        base = self.algorithm_dir(track, algorithm, track_slug=track_slug)
        result: list[tuple[str, Path, ModelMetadata | None]] = []
        for name in ("champion", "latest", "initialization", "contact-candidate"):
            directory = base / name
            if directory.is_dir():
                metadata = self.read_metadata(directory) if (directory / "metadata.json").is_file() else None
                result.append((name, directory, metadata))
        checkpoints = base / "checkpoints"
        if checkpoints.is_dir():
            for directory in sorted(checkpoints.glob("step-*")):
                if directory.is_dir():
                    metadata = self.read_metadata(directory) if (directory / "metadata.json").is_file() else None
                    result.append((f"checkpoints/{directory.name}", directory, metadata))
        return result
