"""CLI algorithm selection and metadata checks."""

from __future__ import annotations

import pytest

from polybot.cli import _model_algorithm, _validate_playback_metadata, evaluate_main, train_main
from polybot.training.models import IncompatibleModelError, ModelMetadata, ModelRegistry


def test_cli_tqc_training_uses_scoped_registry(tmp_path) -> None:
    assert train_main([
        "--algorithm", "tqc", "--backend", "mock", "--track", "mock/straight",
        "--timesteps", "8", "--max-steps", "4", "--frame-skip", "4",
        "--tqc-architecture", "compact", "--tqc-buffer-size", "128",
        "--tqc-learning-starts", "1000", "--output-root", str(tmp_path),
    ]) == 0
    path = tmp_path / "mock-straight" / "tqc" / "latest.zip"
    assert path.is_file()
    assert path.with_suffix(".replay.pkl").is_file()
    assert ModelRegistry.metadata_for_archive(path).algorithm == "TQC"
    assert _model_algorithm(path, None)[0] == "tqc"
    assert evaluate_main([
        str(path), "--backend", "mock", "--track", "mock/straight",
        "--episodes", "1", "--max-steps", "4", "--frame-skip", "4",
    ]) in (0, 1)
    with pytest.raises(ValueError, match="metadata says tqc"):
        _model_algorithm(path, "ppo")


def test_missing_metadata_requires_explicit_algorithm(tmp_path) -> None:
    archive = tmp_path / "legacy.zip"
    archive.write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="pass --algorithm"):
        _model_algorithm(archive, None)
    assert _model_algorithm(archive, "ppo") == ("ppo", None)


def test_playback_rejects_wrong_track_or_action_schema() -> None:
    metadata = ModelMetadata(
        "Summer 1", "current", "standard", 100, algorithm="TQC",
        action_schema="continuous-pwm-v1",
    )
    _validate_playback_metadata(metadata, "tqc", "current", 12)
    with pytest.raises(IncompatibleModelError, match="track ID"):
        _validate_playback_metadata(metadata, "tqc", "mock/straight", 12)
    _validate_playback_metadata(
        metadata, "tqc", "mock/straight", 12, allow_track_override=True
    )
    with pytest.raises(IncompatibleModelError, match="action schema"):
        _validate_playback_metadata(metadata, "ppo", "current", 12)


@pytest.mark.parametrize("flags", [
    ["--algorithm", "tqc", "--gae-lambda", "0.9"],
    ["--algorithm", "ppo", "--tqc-buffer-size", "128"],
])
def test_cli_rejects_settings_for_other_algorithm(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        train_main(flags)
    assert exc.value.code == 2
