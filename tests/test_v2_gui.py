from __future__ import annotations

import json
import os
from dataclasses import fields

import pytest

from polybot.gui.events import format_event
from polybot.gui.log_viewer import LiveLogWindow
from polybot.gui.main import PolyBotWindow
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    DQNConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.parameters import (
    CURRICULUM_INFO,
    DQN_INFO,
    EVALUATION_INFO,
    GENERAL_INFO,
    PPO_INFO,
    REWARD_INFO,
    TQC_INFO,
    validate_metadata,
)
from polybot.training.presets import algorithm_presets, configuration_warnings


@pytest.fixture
def window(qt_app):
    widget = PolyBotWindow()
    yield widget
    widget.close()


@pytest.fixture
def qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_every_training_field_has_plain_language_help(window) -> None:
    validate_metadata()
    for mapping in (GENERAL_INFO, PPO_INFO, DQN_INFO, TQC_INFO, CURRICULUM_INFO,
                    EVALUATION_INFO, REWARD_INFO):
        assert all(info.description and len(info.description) > len(info.label)
                   for info in mapping.values())
    for collection in (window.general, window.ppo_form.widgets, window.dqn_form.widgets,
                       window.tqc_form.widgets,
                       window.reward_advanced.widgets, window.curriculum_form.widgets,
                       window.evaluation_form.widgets):
        assert all(widget.toolTip() for widget in collection.values())
    assert window.custom_phases.toolTip()
    assert all(field.name in PPO_INFO for field in fields(PPOConfig))
    assert all(field.name in DQN_INFO for field in fields(DQNConfig))
    assert all(field.name in TQC_INFO for field in fields(TQCConfig))
    assert all(name in GENERAL_INFO["algorithm"].description for name in ("PPO", "DQN", "TQC"))


def test_algorithm_switch_and_progressive_disclosure(window) -> None:
    assert window.algorithm.currentText() == "tqc"
    assert window.algorithm_stack.currentWidget() is window.tqc_form
    assert window.ppo_form.widgets["gamma"].isHidden()
    window.algorithm.setCurrentText("ppo")
    assert window.algorithm_stack.currentWidget() is window.ppo_form
    assert "on-policy" in window.algorithm_explanation.text()
    window.advanced.setChecked(True)
    assert not window.ppo_form.widgets["gamma"].isHidden()
    assert not window.reward_scroll.isHidden()
    assert window.configuration().tqc is None
    window.advanced.setChecked(False)
    assert window.ppo_form.widgets["gamma"].isHidden()
    window.algorithm.setCurrentText("dqn")
    assert window.algorithm_stack.currentWidget() is window.dqn_form
    assert window.ppo_form.isHidden() and window.tqc_form.isHidden()
    assert "quantile network" in window.algorithm_explanation.text()
    assert window.dqn_form.widgets["target_update_interval"].isHidden()
    window.advanced.setChecked(True)
    assert not window.dqn_form.widgets["target_update_interval"].isHidden()
    assert "pwm_levels" not in window.dqn_form.widgets
    assert "tau" not in window.dqn_form.widgets
    assert window.configuration().dqn is not None
    assert window.configuration().ppo is None and window.configuration().tqc is None


def test_gui_exact_config_roundtrip_and_presets(window) -> None:
    cfg = TrainingConfig(algorithm="ppo", ppo=PPOConfig(), reward_profile=None)
    window.load_configuration(cfg)
    assert window.configuration().to_dict() == cfg.to_dict()
    window.preset.setCurrentText("Fast training")
    assert window.configuration().ppo == algorithm_presets("ppo")["Fast training"]
    window.algorithm.setCurrentText("tqc")
    assert window.configuration().tqc == algorithm_presets("tqc")["Balanced"]
    dqn = TrainingConfig(algorithm="dqn", dqn=DQNConfig(architecture="standard"))
    window.load_configuration(dqn)
    assert window.configuration().to_dict() == dqn.to_dict()
    no_brake = TrainingConfig(algorithm="dqn", dqn=DQNConfig(
        architecture="yosh_2020", action_set="no_brake",
    ))
    window.load_configuration(no_brake)
    assert window.configuration().to_dict() == no_brake.to_dict()
    window.preset.setCurrentText("Stable")
    assert window.configuration().dqn == algorithm_presets("dqn")["Stable"]
    window._event({
        "type": "started", "algorithm": "dqn", "parameters": {
            "actor": 0, "critic": 12345, "total": 12345,
        }, "gpu_name": None, "device": "cpu", "log": "run.jsonl",
    })
    assert "Q-network 12,345" in window.parameter_label.text()
    assert "Actor" not in window.parameter_label.text()
    custom = TrainingConfig(algorithm="tqc", tqc=TQCConfig(), timesteps=100,
                            curriculum=CurriculumConfig("custom", phases=(
                                CurriculumPhaseConfig("section", 50, .75, 1.0),
                                CurriculumPhaseConfig("full", 50),
                            )))
    window.load_configuration(custom)
    assert window.configuration().to_dict() == custom.to_dict()


def test_warnings_explain_unusual_values() -> None:
    cfg = TrainingConfig(algorithm="tqc", tqc=TQCConfig(
        architecture="standard", learning_rate=.002, train_frequency=1,
        gradient_steps=8, replay_capacity=3_000_000,
    ))
    warnings = configuration_warnings(cfg, "NVIDIA T500")
    assert len(warnings) >= 4
    assert any("TPS" in warning for warning in warnings)
    dqn = TrainingConfig(algorithm="dqn", dqn=DQNConfig(
        architecture="standard", learning_rate=.002, replay_capacity=3_000_000,
        gradient_steps=8, target_update_interval=50, exploration_fraction=.01,
        exploration_final_eps=.5,
    ))
    dqn_warnings = configuration_warnings(dqn, "NVIDIA T500")
    assert len(dqn_warnings) >= 6
    assert any("epsilon" in warning for warning in dqn_warnings)


def test_episode_event_is_readable_without_raw_reward_dump(window) -> None:
    episode = {
        "type": "episode", "episode": 123, "timesteps": 35809,
        "events": ["airborne_roll_failure"], "progress": .352588,
        "reward": 94.567, "elapsed_s": 12.12,
        "reward_terms": {"progress": 0, "action_change": -0.00001},
    }
    text = format_event(episode)
    assert "Episode 123" in text
    assert "35.3%" in text
    assert "unstable landing" in text
    assert "+94.6" in text
    assert "reward_terms" not in text
    window._event(episode)
    assert text in window.log.toPlainText()
    assert "reward_terms" not in window.log.toPlainText()


def test_live_log_viewer_follows_appended_events(qt_app, tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({
        "type": "progress", "timesteps": 100, "steps_per_second": 50,
        "progress": .2, "run_max_progress": .4, "updates": 10,
    }) + "\n", encoding="utf-8")
    viewer = LiveLogWindow(path)
    try:
        assert "Step 100" in viewer.status.text()
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "type": "episode", "episode": 2, "timesteps": 110,
                "progress": .35, "reward": -20, "events": ["off_track"],
            }) + "\n")
        viewer.refresh()
        assert "Episode 2" in viewer.events.toPlainText()
        assert "off track" in viewer.events.toPlainText()
    finally:
        viewer.close()

    watcher = LiveLogWindow(tmp_path / "run*.jsonl", follow_newest=True)
    try:
        assert watcher.path == path
        newer = tmp_path / "run-new.jsonl"
        newer.write_text(json.dumps({
            "type": "episode", "episode": 3, "timesteps": 200,
            "progress": .5, "reward": 10, "events": ["stalled"],
        }) + "\n", encoding="utf-8")
        previous_time = path.stat().st_mtime
        os.utime(newer, (previous_time + 10, previous_time + 10))
        watcher.refresh()
        assert watcher.path == newer
        assert "Episode 3" in watcher.events.toPlainText()
        assert "Episode 2" not in watcher.events.toPlainText()
    finally:
        watcher.close()
