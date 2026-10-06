"""Central path and discovery APIs for track-scoped PolyBot data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from polybot.tracks.registry import TrackDefinition, _validate_slug, track_slug

ALGORITHMS = frozenset({"grtqc", "tqc", "ppo"})


@dataclass(frozen=True, slots=True)
class ReplayRunInfo:
    run_id: str
    timestamp: str | None
    algorithm: str
    directory: Path
    episode_count: int
    minimum_step: int | None
    maximum_step: int | None


@dataclass(frozen=True, slots=True)
class TrackWorkspace:
    track: TrackDefinition
    models_root: Path = Path("models")
    logs_root: Path = Path("logs")

    def __post_init__(self) -> None:
        _validate_slug(self.track.slug)
        object.__setattr__(self, "models_root", Path(self.models_root))
        object.__setattr__(self, "logs_root", Path(self.logs_root))

    @property
    def track_dir(self) -> Path:
        return self.models_root / self.track.slug

    def algorithm_models(self, algorithm: str) -> Path:
        self._validate_algorithm(algorithm)
        return self.track_dir / algorithm

    def visual_replays(self, algorithm: str) -> Path:
        return self.algorithm_models(algorithm) / "visual_replays"

    def replay_run(self, algorithm: str, run_id: str) -> Path:
        if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
            raise ValueError("replay run ID must be a simple directory name")
        return self.visual_replays(algorithm) / run_id

    def algorithm_logs(self, algorithm: str) -> Path:
        self._validate_algorithm(algorithm)
        return self.logs_root / self.track.slug / algorithm

    def log_file(self, algorithm: str, filename: str) -> Path:
        if not filename or filename in {".", ".."} or Path(filename).name != filename:
            raise ValueError("log filename must be a simple file name")
        return self.algorithm_logs(algorithm) / filename

    def list_log_files(self, algorithm: str) -> list[Path]:
        canonical = self.algorithm_logs(algorithm)
        current = list(canonical.glob("*.jsonl")) if canonical.is_dir() else []
        legacy = list(self.logs_root.glob(f"{self.track.slug}-{algorithm}-*.jsonl"))
        return sorted(set(current + legacy), key=lambda path: path.stat().st_mtime, reverse=True)

    def list_replay_runs(self, algorithm: str) -> list[ReplayRunInfo]:
        root = self.visual_replays(algorithm)
        if not root.is_dir():
            return []
        runs: list[ReplayRunInfo] = []
        for index_path in root.glob("*/index.json"):
            directory = index_path.parent
            from polybot.training.visual_replays import ReplayFormatError, load_replay_index

            entries = load_replay_index(directory)
            try:
                run_metadata = json.loads(index_path.read_text(encoding="utf-8")).get("run", {})
            except (OSError, json.JSONDecodeError) as exc:
                raise ReplayFormatError(f"cannot read replay run metadata {index_path}") from exc
            timestamp = run_metadata.get("created_at") if isinstance(run_metadata, dict) else None
            if any(
                entry.get("track_slug", track_slug(entry["track_name"])) != self.track.slug
                for entry in entries
            ):
                raise ReplayFormatError(
                    f"replay index {index_path} contains episodes from another track"
                )
            ages = [int(entry["training_step_start"]) for entry in entries]
            runs.append(
                ReplayRunInfo(
                    run_id=directory.name,
                    timestamp=timestamp,
                    algorithm=algorithm,
                    directory=directory,
                    episode_count=len(entries),
                    minimum_step=min(ages) if ages else None,
                    maximum_step=max(ages) if ages else None,
                )
            )
        return sorted(runs, key=lambda item: item.run_id, reverse=True)

    def list_algorithms(self) -> list[str]:
        return sorted(
            algorithm for algorithm in ALGORITHMS
            if self.algorithm_models(algorithm).is_dir()
        )

    @staticmethod
    def _validate_algorithm(algorithm: str) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unsupported algorithm: {algorithm}")
