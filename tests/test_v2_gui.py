from __future__ import annotations

from dataclasses import fields

import pytest

from polybot.gui.main import PolyBotWindow
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.parameters import (
    CURRICULUM_INFO,
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
    for mapping in (GENERAL_INFO, PPO_INFO, TQC_INFO, CURRICULUM_INFO,
                    EVALUATION_INFO, REWARD_INFO):
        assert all(info.description and len(info.description) > len(info.label)
                   for info in mapping.values())
    for collection in (window.general, window.ppo_form.widgets, window.tqc_form.widgets,
                       window.reward_advanced.widgets, window.curriculum_form.widgets,
                       window.evaluation_form.widgets):
        assert all(widget.toolTip() for widget in collection.values())
    assert window.custom_phases.toolTip()
    assert all(field.name in PPO_INFO for field in fields(PPOConfig))
    assert all(field.name in TQC_INFO for field in fields(TQCConfig))


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


def test_gui_exact_config_roundtrip_and_presets(window) -> None:
    cfg = TrainingConfig(algorithm="ppo", ppo=PPOConfig(), reward_profile=None)
    window.load_configuration(cfg)
    assert window.configuration().to_dict() == cfg.to_dict()
    window.preset.setCurrentText("Fast training")
    assert window.configuration().ppo == algorithm_presets("ppo")["Fast training"]
    window.algorithm.setCurrentText("tqc")
    assert window.configuration().tqc == algorithm_presets("tqc")["Balanced"]
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
