"""Persistent catalogue for PolyBot track identities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRACK_REGISTRY_SCHEMA = "polybot.tracks.v1"
_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def track_slug(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("track name must be text")
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise ValueError("track name needs letters or digits")
    return slug


def _validate_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("track name must be text")
    value = name.strip()
    if not value or len(value) > 100 or any(ord(character) < 32 for character in value):
        raise ValueError("track name must contain 1 to 100 printable characters")
    if "/" in value or "\\" in value:
        raise ValueError("track name cannot contain path separators")
    track_slug(value)
    return value


def _validate_slug(slug: str) -> str:
    if not isinstance(slug, str) or not _SLUG_PATTERN.fullmatch(slug):
        raise ValueError("track slug must contain lowercase letters, digits, and single hyphens")
    return slug


@dataclass(frozen=True, slots=True)
class TrackDefinition:
    name: str
    slug: str
    simulator_track_id: str = "current"
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name))
        _validate_slug(self.slug)
        if not isinstance(self.simulator_track_id, str) or not self.simulator_track_id.strip():
            raise ValueError("simulator track ID must be non-empty text")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(UTC).isoformat())


class TrackNotFoundError(ValueError):
    """Raised when a track name/slug is not registered."""


class TrackRegistry:
    """Load, discover, and persist the catalogue without moving user data."""

    def __init__(
        self,
        path: str | Path = Path("config") / "tracks.json",
        *,
        models_root: str | Path = "models",
    ) -> None:
        self.path = Path(path)
        self.models_root = Path(models_root)
        self._tracks = self._load_or_discover()
        if not self.path.exists():
            self._save()

    def _load_or_discover(self) -> list[TrackDefinition]:
        if self.path.is_file():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read track registry {self.path}") from exc
            if not isinstance(payload, dict) or payload.get("schema") != TRACK_REGISTRY_SCHEMA:
                raise ValueError(f"unsupported track registry schema in {self.path}")
            entries = payload.get("tracks")
            if not isinstance(entries, list):
                raise ValueError(f"track registry {self.path} must contain a tracks list")
            return self._validate_entries(entries)
        return self._discover_legacy_tracks()

    @staticmethod
    def _validate_entries(entries: list[Any]) -> list[TrackDefinition]:
        tracks: list[TrackDefinition] = []
        seen_slugs: set[str] = set()
        seen_names: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("track registry entries must be objects")
            track = TrackDefinition(**entry)
            if track.slug in seen_slugs or track.name.casefold() in seen_names:
                raise ValueError("track registry contains duplicate names or slugs")
            tracks.append(track)
            seen_slugs.add(track.slug)
            seen_names.add(track.name.casefold())
        return tracks

    def _discover_legacy_tracks(self) -> list[TrackDefinition]:
        discovered: dict[str, tuple[str, str]] = {}
        if self.models_root.is_dir():
            for metadata_path in self.models_root.glob("*/*/*/metadata.json"):
                try:
                    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                folder_slug = metadata_path.parents[2].name
                if not _SLUG_PATTERN.fullmatch(folder_slug):
                    continue
                name = payload.get("track_name") if isinstance(payload, dict) else None
                simulator_id = payload.get("track_id", "current") if isinstance(payload, dict) else "current"
                if not isinstance(name, str) or not name.strip():
                    name = folder_slug.replace("-", " ").title()
                if not isinstance(simulator_id, str) or not simulator_id:
                    simulator_id = "current"
                discovered.setdefault(folder_slug, (name.strip(), simulator_id))

            for index_path in self.models_root.glob("*/*/visual_replays/*/index.json"):
                try:
                    payload = json.loads(index_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                folder_slug = index_path.parents[3].name
                entries = payload.get("episodes", []) if isinstance(payload, dict) else []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("track_name")
                    simulator_id = entry.get("track_id", "current")
                    if isinstance(name, str) and name.strip() and _SLUG_PATTERN.fullmatch(folder_slug):
                        discovered.setdefault(folder_slug, (name.strip(), simulator_id))
                        break

            for directory in self.models_root.iterdir():
                if directory.is_dir() and _SLUG_PATTERN.fullmatch(directory.name):
                    discovered.setdefault(
                        directory.name,
                        (directory.name.replace("-", " ").title(), "current"),
                    )

        tracks = [
            TrackDefinition(name=name, slug=slug, simulator_track_id=simulator_id)
            for slug, (name, simulator_id) in sorted(discovered.items())
        ]
        for default in (
            TrackDefinition("Summer 1", "summer-1", "current"),
            TrackDefinition("Mock straight", "mock-straight", "mock/straight"),
        ):
            if not any(item.slug == default.slug for item in tracks):
                tracks.append(default)
        return tracks

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {"schema": TRACK_REGISTRY_SCHEMA, "tracks": [asdict(track) for track in self._tracks]},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def list_tracks(self) -> list[TrackDefinition]:
        if not self._tracks:
            self._tracks = self._discover_legacy_tracks()
        if not self.path.exists() or not self._tracks:
            self._save()
        return list(self._tracks)

    def resolve(self, name_or_slug: str) -> TrackDefinition:
        key = name_or_slug.strip().casefold()
        if not key:
            raise TrackNotFoundError("track name or slug cannot be empty")
        for track in self._tracks:
            if track.slug.casefold() == key or track.name.casefold() == key:
                return track
        raise TrackNotFoundError(f"track {name_or_slug!r} is not registered")

    def add(
        self,
        name: str,
        *,
        simulator_track_id: str = "current",
        slug: str | None = None,
    ) -> TrackDefinition:
        clean_name = _validate_name(name)
        canonical_slug = _validate_slug(slug) if slug is not None else track_slug(clean_name)
        if any(
            track.slug.casefold() == canonical_slug.casefold()
            or track.name.casefold() == clean_name.casefold()
            for track in self._tracks
        ):
            raise ValueError(f"track name or slug already exists: {clean_name}")
        track = TrackDefinition(clean_name, canonical_slug, simulator_track_id)
        self._tracks.append(track)
        self._save()
        return track

    def register_legacy(
        self,
        name: str,
        *,
        simulator_track_id: str = "current",
        slug: str | None = None,
    ) -> TrackDefinition:
        """Register an identity from an old saved config without changing its paths."""
        canonical_slug = _validate_slug(slug) if slug is not None else track_slug(name)
        for track in self._tracks:
            if track.slug == canonical_slug:
                return track
            if track.name.casefold() == name.casefold():
                return track
        track = TrackDefinition(name, canonical_slug, simulator_track_id)
        self._tracks.append(track)
        self._save()
        return track

    def rename(self, name_or_slug: str, new_name: str) -> TrackDefinition:
        old = self.resolve(name_or_slug)
        clean_name = _validate_name(new_name)
        if any(
            item.slug != old.slug and item.name.casefold() == clean_name.casefold()
            for item in self._tracks
        ):
            raise ValueError(f"track name already exists: {clean_name}")
        updated = TrackDefinition(clean_name, old.slug, old.simulator_track_id, old.created_at)
        self._tracks = [updated if item.slug == old.slug else item for item in self._tracks]
        self._save()
        return updated

    def remove(self, name_or_slug: str) -> TrackDefinition:
        track = self.resolve(name_or_slug)
        self._tracks = [item for item in self._tracks if item.slug != track.slug]
        self._save()
        return track
