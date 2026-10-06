from __future__ import annotations

import json

import pytest

from polybot.algorithms.registry import backend_for
from polybot.cli import _algorithm_options, _common, _config_from_args, tracks_main
from polybot.models.registry import (
    IncompatibleModelError,
    ModelMetadata,
    ModelRegistry,
)
from polybot.tracks.registry import TRACK_REGISTRY_SCHEMA, TrackDefinition, TrackRegistry
from polybot.tracks.workspace import TrackWorkspace
from polybot.training.config import PPOConfig, TrainingConfig


def test_registry_persists_slug_rename_and_non_destructive_remove(tmp_path) -> None:
    path = tmp_path / "config" / "tracks.json"
    models = tmp_path / "models"
    registry = TrackRegistry(path, models_root=models)
    assert {track.name for track in registry.list_tracks()} >= {"Summer 1", "Mock straight"}
    created = registry.add("Autumn 2")
    assert created.slug == "autumn-2"
    with pytest.raises(ValueError, match="already exists"):
        registry.add("autumn-2")
    reloaded = TrackRegistry(path, models_root=models)
    assert reloaded.resolve("AUTUMN-2") == created
    renamed = reloaded.rename("autumn-2", "Autumn Two")
    assert renamed.slug == "autumn-2"
    assert reloaded.resolve("Autumn Two").slug == "autumn-2"

    user_data = models / renamed.slug / "grtqc" / "latest"
    user_data.mkdir(parents=True)
    assert reloaded.remove("Autumn Two") == renamed
    assert user_data.is_dir()
    with pytest.raises(ValueError, match="not registered"):
        reloaded.resolve("autumn-2")


@pytest.mark.parametrize("name", ["", "   ", "../escape", "name/with/slash", "\x00bad"])
def test_registry_rejects_invalid_names(tmp_path, name) -> None:
    registry = TrackRegistry(tmp_path / "tracks.json", models_root=tmp_path / "models")
    with pytest.raises(ValueError):
        registry.add(name)


def test_registry_discovers_existing_model_metadata_without_moving_it(tmp_path) -> None:
    models = tmp_path / "models"
    metadata_path = models / "autumn-2" / "tqc" / "champion" / "metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps({"track_name": "Autumn 2", "track_id": "current"}),
        encoding="utf-8",
    )
    registry = TrackRegistry(tmp_path / "tracks.json", models_root=models)
    discovered = registry.resolve("Autumn 2")
    assert discovered.slug == "autumn-2"
    assert metadata_path.is_file()
    document = json.loads((tmp_path / "tracks.json").read_text(encoding="utf-8"))
    assert document["schema"] == TRACK_REGISTRY_SCHEMA


def test_workspace_resolves_isolated_model_replay_and_log_paths(tmp_path) -> None:
    track = TrackDefinition("Autumn Two", "autumn-2")
    workspace = TrackWorkspace(track, tmp_path / "models", tmp_path / "logs")
    assert workspace.algorithm_models("grtqc") == tmp_path / "models" / "autumn-2" / "grtqc"
    assert workspace.replay_run("grtqc", "run-1") == (
        tmp_path / "models" / "autumn-2" / "grtqc" / "visual_replays" / "run-1"
    )
    assert workspace.algorithm_logs("ppo") == tmp_path / "logs" / "autumn-2" / "ppo"
    with pytest.raises(ValueError, match="simple directory"):
        workspace.replay_run("tqc", "../escape")
    with pytest.raises(ValueError, match="unsupported algorithm"):
        workspace.algorithm_models("../escape")


def _ppo_metadata(config: TrainingConfig, *, legacy: bool = False) -> ModelMetadata:
    backend = backend_for("ppo")
    return ModelMetadata(
        algorithm="ppo",
        architecture="tiny",
        actor_parameters=1,
        critic_parameters=1,
        total_trainable_parameters=2,
        observation_schema="polybot.observation.v2",
        action_schema=backend.action_adapter(config).schema,
        track_name="Summer 1",
        track_id="current",
        track_slug=None if legacy else "summer-1",
        lookahead_count=config.lookahead_count,
        reward_profile=None,
        curriculum={},
        training_config=config.to_dict(),
        training_timesteps=10,
        simulator_ticks=300,
        wall_seconds=1.0,
        seed=0,
        device="cpu",
        finishes=0,
        crashes=0,
        action_semantics="steering_signed_longitudinal_v1",
    )


def test_model_track_compatibility_uses_slug_even_when_simulator_id_matches(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    summer = TrainingConfig(
        algorithm="ppo",
        ppo=PPOConfig(architecture="tiny"),
        backend="websocket",
        track_name="Summer 1",
        track_id="current",
        track_slug="summer-1",
    )
    registry.validate(_ppo_metadata(summer), summer, backend_for("ppo").action_adapter(summer).schema)

    autumn = TrainingConfig(
        algorithm="ppo",
        ppo=PPOConfig(architecture="tiny"),
        backend="websocket",
        track_name="Autumn 2",
        track_id="current",
        track_slug="autumn-2",
    )
    with pytest.raises(IncompatibleModelError, match="track identity"):
        registry.validate(
            _ppo_metadata(summer),
            autumn,
            backend_for("ppo").action_adapter(autumn).schema,
        )


def test_legacy_model_metadata_without_track_slug_remains_compatible(tmp_path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    config = TrainingConfig(
        algorithm="ppo",
        ppo=PPOConfig(architecture="tiny"),
        backend="websocket",
        track_name="Summer One",
        track_id="current",
        track_slug="summer-1",
    )
    registry.validate(
        _ppo_metadata(config, legacy=True),
        config,
        backend_for("ppo").action_adapter(config).schema,
    )


def test_cli_track_management_and_training_resolution(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert tracks_main(["add", "Autumn 2"]) == 0
    assert "autumn-2" in capsys.readouterr().out
    parser = __import__("argparse").ArgumentParser()
    _common(parser)
    _algorithm_options(parser)
    args = parser.parse_args(["--algorithm", "grtqc", "--backend", "websocket", "--track", "autumn-2"])
    config = _config_from_args(args, parser)
    assert config.track_name == "Autumn 2"
    assert config.track_slug == "autumn-2"
    assert config.track_id == "current"
    assert tracks_main(["list"]) == 0
    assert "Autumn 2" in capsys.readouterr().out
