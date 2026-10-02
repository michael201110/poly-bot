from __future__ import annotations

import json
import os
from dataclasses import fields

import pytest
from PySide6.QtWidgets import QLabel, QPushButton

from polybot.gui.events import format_event
from polybot.gui.log_viewer import LiveLogWindow
from polybot.gui.main import PolyBotWindow
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    GRTQCConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.parameters import (
    CURRICULUM_INFO,
    EVALUATION_INFO,
    GENERAL_INFO,
    GRTQC_INFO,
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
    for mapping in (GENERAL_INFO, PPO_INFO, GRTQC_INFO, TQC_INFO, CURRICULUM_INFO,
                    EVALUATION_INFO, REWARD_INFO):
        assert all(info.description and len(info.description) > len(info.label)
                   for info in mapping.values())
    for collection in (window.general, window.ppo_form.widgets, window.grtqc_form.widgets,
                       window.tqc_form.widgets,
                       window.reward_advanced.widgets, window.curriculum_form.widgets,
                       window.evaluation_form.widgets):
        assert all(widget.toolTip() for widget in collection.values())
    assert window.custom_phases.toolTip()
    assert all(field.name in PPO_INFO for field in fields(PPOConfig))
    assert all(field.name in GRTQC_INFO for field in fields(GRTQCConfig))
    assert all(field.name in TQC_INFO for field in fields(TQCConfig))
    assert all(name in GENERAL_INFO["algorithm"].description for name in ("PPO", "GRTQC", "TQC"))


def test_algorithm_switch_and_progressive_disclosure(window) -> None:
    assert window.algorithm.currentText() == "grtqc"
    assert window.configuration().rewards.barrier_collision_impulse_threshold == 250.0
    assert window.algorithm_stack.currentWidget() is window.grtqc_form
    assert window.ppo_form.widgets["gamma"].isHidden()
    window.algorithm.setCurrentText("ppo")
    assert window.algorithm_stack.currentWidget() is window.ppo_form
    assert "on-policy" in window.algorithm_explanation.text()
    window.advanced.setChecked(True)
    assert not window.ppo_form.widgets["gamma"].isHidden()
    assert not window.reward_scroll.isHidden()
    assert not window.pace_polish_section.isHidden()
    assert not window.adaptation_section.isHidden()
    assert not window.distillation_section.isHidden()
    assert not window.teacher_student_section.isHidden()
    assert "below 22.000s" in window.teacher_student_section.findChild(QLabel).text()
    assert window.dagger_rounds.value() == 3
    assert window.dagger_episodes.value() == 8
    assert window.dagger_nominal_weight.value() == pytest.approx(0.6)
    assert window.dagger_recovery_weight.value() == pytest.approx(0.4)
    assert any(button.text() == "Run DAgger cycle" for button in
               window.teacher_student_section.findChildren(QPushButton))
    assert any(button.text() == "Stop after current DAgger round" for button in
               window.teacher_student_section.findChildren(QPushButton))
    assert not window.wr_search_section.isHidden()
    assert not window.section_optimizer_section.isHidden()
    assert {"Start 1 hour", "Start 4 hours", "Run until stopped", "Resume saved search", "Pause and save",
            "Stop safely", "Skip section", "Force refine"} <= {
                button.text() for button in window.section_optimizer_section.findChildren(QPushButton)
            }
    assert window.wr_target.value() == pytest.approx(22.262)
    window._load_wr_profile()
    assert window.wr_trials.value() == 12
    assert window.configuration().tqc is None
    window.advanced.setChecked(False)
    assert window.ppo_form.widgets["gamma"].isHidden()
    assert window.pace_polish_section.isHidden()
    assert window.adaptation_section.isHidden()
    assert window.wr_search_section.isHidden()
    assert window.section_optimizer_section.isHidden()
    assert window.distillation_section.isHidden()
    assert window.teacher_student_section.isHidden()
    window.algorithm.setCurrentText("grtqc")
    assert window.algorithm_stack.currentWidget() is window.grtqc_form
    assert window.ppo_form.isHidden() and window.tqc_form.isHidden()
    assert "quantile critics" in window.algorithm_explanation.text()
    assert window.grtqc_form.widgets["critic_warmup_updates"].isHidden()
    window.advanced.setChecked(True)
    assert not window.grtqc_form.widgets["critic_warmup_updates"].isHidden()
    assert "pwm_levels" not in window.grtqc_form.widgets
    assert "tau" in window.grtqc_form.widgets
    assert window.configuration().grtqc is not None
    assert window.configuration().ppo is None and window.configuration().tqc is None


def test_distillation_summary_surfaces_live_bake_metrics() -> None:
    summary = PolyBotWindow._distillation_summary("full", {
        "collection": {"samples": 12000},
        "training": {
            "best_validation_loss": 0.0012,
            "action_metrics": {"max_action_error": 0.08},
            "bakeable_overlay_count": 3,
            "retained_overlay_count": 1,
        },
        "validation": {
            "accepted": True,
            "lap_delta_s": 0.01,
            "teacher": {"median_lap_s": 24.2},
            "student": {"median_lap_s": 24.21},
        },
    })
    assert "passes; staged for bake" in summary
    assert "24.2s" in summary and "24.21s" in summary
    assert "12,000 samples" in summary
    assert "overlays baked 3, retained 1" in summary


def test_adaptation_gui_preset_and_explicit_stage_controls(window) -> None:
    window._load_adaptation_preset()
    config = window.configuration()
    assert config.algorithm == "tqc"
    assert config.tqc.actor_learning_rate == pytest.approx(1e-6)
    assert config.tqc.critic_learning_rate == pytest.approx(5e-5)
    assert config.tqc.adaptation_noise_probability == pytest.approx(0.0001)
    labels = {button.text() for button in window.findChildren(QPushButton)}
    assert "Collect local replay" in labels
    assert "Adapt critics" in labels
    assert "Validate & promote candidate" in labels
    assert "Experimental actor-gradient polish" in labels
    assert "Run full cycle" in labels


def test_distillation_gui_has_explicit_snapshot_and_promotion_controls(window) -> None:
    window.advanced.setChecked(True)
    labels = {button.text() for button in window.distillation_section.findChildren(QPushButton)}
    assert {
        "Snapshot champion", "Collect teacher data", "Train actor student",
        "Validate student", "Bake / promote", "Rollback bake", "Run full workflow",
    } <= labels
    assert window.distillation_episodes.value() == 20
    assert window.distillation_validation_episodes.value() == 5
    assert window.distillation_tolerance.value() == pytest.approx(0.02)


def test_gui_exact_config_roundtrip_and_presets(window) -> None:
    cfg = TrainingConfig(algorithm="ppo", ppo=PPOConfig(), reward_profile=None)
    window.load_configuration(cfg)
    assert window.configuration().to_dict() == cfg.to_dict()
    low_noise = TrainingConfig(algorithm="ppo", ppo=PPOConfig(action_std=0.05))
    window.load_configuration(low_noise)
    assert window.configuration().ppo.action_std == pytest.approx(0.05)
    window.preset.setCurrentText("Fast training")
    assert window.configuration().ppo == algorithm_presets("ppo")["Fast training"]
    window.algorithm.setCurrentText("tqc")
    assert window.configuration().tqc == algorithm_presets("tqc")["Balanced"]
    window.preset.setCurrentText("Summer 1 - TQC Safe Polish")
    assert window.configuration().tqc.learning_rate == 1e-5
    grtqc = TrainingConfig(algorithm="grtqc", grtqc=GRTQCConfig(architecture="standard"))
    window.load_configuration(grtqc)
    assert window.configuration().to_dict() == grtqc.to_dict()
    grtqc.grtqc.screen_actor_evaluations = True
    grtqc.grtqc.actor_evaluation_interval_steps = 512
    grtqc.grtqc.recovery_weak_evaluations = 5
    grtqc.grtqc.finish_episode_before_actor_eval = True
    grtqc.grtqc.critic_controller_state = True
    grtqc.grtqc.actor_controller_state = True
    grtqc.grtqc.controller_adapter_only = True
    window.load_configuration(grtqc)
    assert window.configuration().to_dict() == grtqc.to_dict()
    window.preset.setCurrentText("Summer 1 - Transferred Champion")
    assert window.configuration().grtqc == algorithm_presets("grtqc")["Summer 1 - Transferred Champion"]
    window._event({
        "type": "started", "algorithm": "grtqc", "parameters": {
            "actor": 123, "critic": 12345, "total": 12468,
        }, "gpu_name": None, "device": "cpu", "log": "run.jsonl",
    })
    assert "Actor 123" in window.parameter_label.text()
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
    grtqc = TrainingConfig(algorithm="grtqc", grtqc=GRTQCConfig(
        architecture="standard", learning_rate=.002, replay_capacity=3_000_000,
        gradient_steps=8, train_frequency=1,
    ))
    assert len(configuration_warnings(grtqc, "NVIDIA T500")) >= 4


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


def test_speed_search_events_are_readable_in_status(window, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    log = tmp_path / "logs" / "summer-1-tqc-speed-search-test.jsonl"
    log.parent.mkdir()
    events = [
        {"type": "started", "champion_lap_s": 28.384, "target_s": 25.0},
        {"type": "trial", "trial": 3, "evaluation": {
            "finish_rate": 1.0, "median_progress": 1.0, "median_lap_s": 26.7,
        }},
        {"type": "champion", "trial": 3, "lap_s": 26.7},
    ]
    log.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    window._poll_speed_search_log()
    assert "26.700 s" in window.metrics.text()
    assert "5/5 laps" in window.log.toPlainText()
    assert "Speed trial 3" in window.log.toPlainText()
    assert window.speed_search_target.value() == 25.0


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
