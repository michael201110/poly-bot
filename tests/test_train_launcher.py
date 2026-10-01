from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from polybot.protocol import ProtocolViolation
from polybot.training.config import CurriculumConfig, CurriculumPhaseConfig, TrainingConfig
from scripts import train_with_stop_file as launcher


def test_reconnect_keeps_remaining_budget_and_current_curriculum() -> None:
    config = TrainingConfig(timesteps=100, curriculum=CurriculumConfig("custom", phases=(
        CurriculumPhaseConfig("full", 15),
        CurriculumPhaseConfig("section", 20, 0.07, 0.12),
        CurriculumPhaseConfig("full", 65),
    )))
    remaining = launcher.remaining_config(config, 25)
    assert remaining.timesteps == 75
    assert [(phase.mode, phase.steps) for phase in remaining.curriculum.phases] == [
        ("section", 10), ("full", 65),
    ]
    assert remaining.curriculum.phases[0].start_ratio == 0.07
    assert launcher.remaining_config(config, 100) is None


@pytest.mark.parametrize("verified", [False, True])
def test_reconnect_resumes_replay_or_repeats_interrupted_transfer_gate(tmp_path, monkeypatch, verified) -> None:
    config = TrainingConfig(output_root=tmp_path / "models", timesteps=100)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    registry = launcher.ModelRegistry(config.output_root)
    latest = registry.slot(config.track_name, "grtqc", "latest")
    initialization = registry.slot(config.track_name, "grtqc", "initialization")
    latest.mkdir(parents=True)
    (latest / "metadata.json").write_text("{}", encoding="utf-8")
    metadata = {
        latest: SimpleNamespace(training_timesteps=25),
        initialization: SimpleNamespace(evaluation={} if verified else None, training_timesteps=0),
    }
    monkeypatch.setattr(launcher.ModelRegistry, "read_metadata", lambda self, path: metadata[path])
    calls = []
    budgets = []

    class Runner:
        def __init__(self, cfg, status):
            budgets.append(cfg.timesteps)

        def run(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ProtocolViolation("stale_episode: simulator reset")
            return latest

    monkeypatch.setattr(launcher, "TrainingRunner", Runner)
    monkeypatch.setattr(launcher, "RECONNECT_DELAY_S", 0)
    monkeypatch.setattr("sys.argv", [
        "train", "--config", str(config_path), "--stop-file", str(tmp_path / "stop"),
        "--resume", str(initialization), "--retry-transport",
    ])
    launcher.main()
    assert calls[1]["resume"] == (latest if verified else initialization)
    assert calls[1]["fresh_replay"] is not verified
    assert budgets == [100, 75 if verified else 100]
