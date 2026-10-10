from __future__ import annotations

import json
import os
import threading
from dataclasses import fields
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QLabel, QPushButton, QScrollArea

import polybot.gui.main as gui_main
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
from polybot.training.visual_replays import INDEX_SCHEMA


@pytest.fixture
def window(qt_app, tmp_path, monkeypatch):
    registry_type = gui_main.TrackRegistry
    settings_store_type = gui_main.AIOverlaySettingsStore
    monkeypatch.setattr(
        gui_main,
        "TrackRegistry",
        lambda: registry_type(tmp_path / "config" / "tracks.json", models_root=tmp_path / "models"),
    )
    monkeypatch.setattr(
        gui_main,
        "AIOverlaySettingsStore",
        lambda: settings_store_type(tmp_path / "config" / "ai-overlay.json"),
    )
    monkeypatch.setattr(
        PolyBotWindow,
        "_selected_track_path",
        staticmethod(lambda: tmp_path / "config" / "selected-track.json"),
    )
    widget = PolyBotWindow()
    widget.general["output_root"].setText(str(tmp_path / "models"))
    widget.general["log_root"].setText(str(tmp_path / "logs"))
    widget._refresh_replay_runs()
    widget.replay_swarm_step_min.setValue(0)
    widget.replay_swarm_step_max.setValue(1_000_000)
    widget.replay_swarm_color_min.setValue(0)
    widget.replay_swarm_color_max.setValue(1_000_000)
    yield widget
    widget.close()


def test_ai_hud_display_settings_persist_and_apply_to_active_runner(window, qt_app) -> None:
    class ActiveRunner:
        def __init__(self) -> None:
            self.settings = None

        def set_ai_overlay_settings(self, settings) -> None:
            self.settings = settings

    runner = ActiveRunner()
    window.runner = runner
    window.ai_overlay_widgets["preset"].setCurrentText("full")
    window.ai_overlay_widgets["scale"].setValue(1.3)
    window.ai_overlay_widgets["lookahead_points"].setValue(6)

    assert window._save_ai_overlay_settings()
    saved = window.ai_overlay_store.load()
    assert saved.enabled is True
    assert saved.preset == "full"
    assert saved.scale == pytest.approx(1.3)
    assert saved.lookahead_points == 6
    assert runner.settings == saved


def test_config_file_actions_use_native_file_picker(window, tmp_path, monkeypatch) -> None:
    path = tmp_path / "training.json"
    monkeypatch.setattr(gui_main.QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    window._save_config()
    saved = TrainingConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert saved.to_dict() == window.configuration().to_dict()

    expected = TrainingConfig(
        algorithm="ppo",
        reward_profile=None,
        ppo=PPOConfig(architecture="tiny"),
    )
    path.write_text(json.dumps(expected.to_dict()), encoding="utf-8")
    monkeypatch.setattr(gui_main.QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    window._load_config_dialog()
    assert window.configuration().to_dict() == expected.to_dict()


def test_editing_reward_values_marks_profile_custom(window) -> None:
    assert window.reward_profile.currentText() == "Balanced"
    window.reward_basic["failure_multiplier"].setValue(1.5)
    assert window.reward_profile.currentText() == "Custom"
    assert window.configuration().rewards.crash_penalty == pytest.approx(
        window._base_rewards.crash_penalty * 1.5
    )


@pytest.mark.parametrize("saved_semantics", [gui_main.REWARD_SEMANTICS, "executed-controls-v1"])
def test_resume_loads_saved_reward_settings_before_replay_validation(
    window, tmp_path, monkeypatch, saved_semantics,
) -> None:
    slot = tmp_path / "latest"
    slot.mkdir()
    (slot / "replay.pkl").write_bytes(b"saved replay placeholder")
    saved_rewards = window._reward_values.copy()
    saved_rewards["finish_bonus"] += 123.0
    saved_config = window.configuration().to_dict()
    saved_config["rewards"] = saved_rewards
    saved_config["reward_profile"] = None
    metadata = SimpleNamespace(reward_semantics=saved_semantics, training_config=saved_config)

    class Registry:
        def __init__(self, root) -> None:
            pass

        def slot(self, *args, **kwargs):
            return slot

        def read_metadata(self, path):
            return metadata

        def validate(self, *args):
            return None

        def list_model_slots(self, *args):
            return []

    class Thread:
        def __init__(self, *, target, args, daemon) -> None:
            self.args = args

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(gui_main, "ModelRegistry", Registry)
    monkeypatch.setattr(gui_main.threading, "Thread", Thread)
    monkeypatch.setattr(window, "_external_speed_search_running", lambda config: False)

    window._start(True)

    assert window.runner is not None
    assert window.runner.config.to_dict()["rewards"] == saved_rewards
    assert window.configuration().to_dict()["rewards"] == saved_rewards
    assert window.reward_profile.currentText() == "Custom"
    assert window.worker.args == (slot, False, False, False)
    assert "Loaded the checkpoint's policy, critic, curriculum, and reward settings" in window.log.toPlainText()
    if saved_semantics != gui_main.REWARD_SEMANTICS:
        assert "replay is preserved" in window.log.toPlainText()
        assert "mixed reward versions" in window.warnings.text()


@pytest.mark.parametrize("gui_origin", ["transfer", "scratch"])
def test_continue_from_best_skips_a_faster_but_incompatible_transfer_latest(
    window, tmp_path, monkeypatch, gui_origin,
) -> None:
    champion = tmp_path / "champion"
    latest = tmp_path / "latest"
    champion.mkdir()
    latest.mkdir()
    (champion / "metadata.json").touch()
    (latest / "metadata.json").touch()
    config = window.configuration()
    config.output_root = tmp_path
    config.grtqc.training_origin = gui_origin
    metadata_configs = {}
    for path, origin, env_state in (
        (champion, "scratch", True), (latest, "transfer", False),
    ):
        saved = TrainingConfig.from_dict(config.to_dict())
        saved.grtqc.training_origin = origin
        saved.grtqc.critic_controller_state = env_state
        saved.grtqc.critic_environment_state = env_state
        metadata_configs[path] = saved.to_dict()

    class Registry:
        def __init__(self, root) -> None:
            pass

        def slot(self, _track, _algorithm, name, **_kwargs):
            return champion if name == "champion" else latest

        def read_metadata(self, path):
            lap = 22.0 if path == latest else 22.595
            return SimpleNamespace(
                evaluation=gui_main.EvaluationResult(
                    episodes=5, finish_rate=1.0, median_progress=1.0, mean_progress=1.0,
                    best_lap_s=lap, median_lap_s=lap, crash_rate=0.0,
                    off_track_rate=0.0, stall_rate=0.0,
                ).to_dict(),
                training_config=metadata_configs[path],
            )

        def validate(self, metadata, _config, _schema):
            if metadata.training_config["grtqc"]["training_origin"] != "scratch":
                raise ValueError("scratch and transferred GRTQC experiments cannot share model/replay")

    monkeypatch.setattr(gui_main, "ModelRegistry", Registry)
    assert window._best_resume_slot(config) == champion


def test_resume_configuration_restores_saved_observation_shape_and_keeps_budget(window, tmp_path) -> None:
    current = window.configuration()
    current.output_root = tmp_path / "models"
    current.log_root = tmp_path / "logs"
    current.timesteps = 75_000
    current.grtqc.training_origin = "scratch"
    saved = TrainingConfig.from_dict(current.to_dict())
    saved.grtqc.critic_controller_state = True
    saved.grtqc.critic_environment_state = True
    metadata = SimpleNamespace(training_config=saved.to_dict())

    resumed = window._resume_configuration(current, metadata)

    assert resumed.grtqc.critic_environment_state is True
    assert resumed.grtqc.critic_controller_state is True
    assert resumed.timesteps == 75_000
    assert resumed.output_root == current.output_root
    assert resumed.log_root == current.log_root


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
    assert window.configuration().rewards.barrier_collision_impulse_threshold == 1_000_000_000.0
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
    assert "selected track's frozen TQC champion" in window.teacher_student_section.findChild(QLabel).text()
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


def test_advanced_toggle_restores_basic_settings_in_shown_window(window, qt_app) -> None:
    window.show()
    qt_app.processEvents()

    window.tabs.setCurrentWidget(window.algorithm_page)
    window.advanced.setChecked(True)
    qt_app.processEvents()
    window.advanced.setChecked(False)
    qt_app.processEvents()

    assert window.grtqc_form.widgets["architecture"].isVisible()
    assert window.grtqc_form.widgets["learning_rate"].isVisible()
    assert window.grtqc_form.widgets["train_frequency"].isVisible()
    assert window.grtqc_form.widgets["critic_warmup_updates"].isHidden()
    assert window.grtqc_form.widgets["architecture"].height() > 0

    window.tabs.setCurrentWidget(window.general_page)
    qt_app.processEvents()
    assert window.general["backend"].isVisible()
    assert window.general["timesteps"].isVisible()
    assert window.general["visual_replay_enabled"].isHidden()


def test_main_tabs_are_scrollable_and_setup_summary_tracks_choices(window, qt_app) -> None:
    assert window.general_page.findChild(QScrollArea) is not None
    assert window.algorithm_page.findChild(QScrollArea) is not None
    assert window.replay_swarm_page.findChild(QScrollArea) is not None
    assert "Summer 1" in window.run_summary.text()
    assert "GRTQC" in window.run_summary.text()

    window.general["timesteps"].setValue(250_000)
    window.general["backend"].setCurrentText("mock")
    qt_app.processEvents()
    assert "250,000 decisions" in window.run_summary.text()
    assert "mock" in window.run_summary.text()


def test_replay_swarm_explains_empty_run_list(window) -> None:
    assert window.replay_swarm_run.count() == 0
    assert "No saved replay runs found" in window.replay_swarm_runs_status.text()
    assert "WebSocket training" in window.replay_swarm_runs_status.text()
    assert "Each completed attempt" in window.replay_swarm_runs_status.text()
    assert window.replay_recording.currentText() == "Automatic for live training"
    assert window.replay_advanced_content.isHidden()


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
    window.track_selector.setCurrentIndex(window.track_selector.findData("summer-1"))
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
    window.track_selector.setCurrentIndex(window.track_selector.findData("summer-1"))
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


def test_replay_swarm_tab_defaults_and_input_validation(window) -> None:
    labels = {button.text() for button in window.replay_swarm_action_buttons}
    assert {
        "Watch selected attempt", "Compare selected attempts", "Pause", "Resume",
        "Restart", "Clear replays", "Jump", "Inspect selection", "Set full run range",
        "Load paused", "Apply settings", "Bridge status",
    } <= labels
    assert window.replay_swarm_step_min.value() == 0
    assert window.replay_swarm_step_max.value() == 1_000_000
    assert window.replay_swarm_max_cars.value() == 100
    assert window.replay_swarm_color_max.value() == 2_000_000
    with pytest.raises(ValueError, match="Choose a replay run"):
        window._replay_swarm_config()

    window.replay_swarm_path.setText("replays")
    window.replay_swarm_external.setChecked(True)
    window.replay_swarm_episode_min.setText("3")
    with pytest.raises(ValueError, match="both episode ID"):
        window._replay_swarm_config()
    window.replay_swarm_episode_max.setText("2")
    with pytest.raises(ValueError, match="non-negative and ordered"):
        window._replay_swarm_config()
    window.replay_swarm_episode_max.setText("4")
    window.replay_swarm_color_stops.setText("0:#ff0000, 100:#00ff00")
    assert window._replay_swarm_config()["color_scale"].hex_color(50) == "#808000"
    window.replay_swarm_color_stops.setText("broken")
    with pytest.raises(ValueError):
        window._replay_swarm_config()


def test_replay_recording_choice_updates_saved_training_configuration(window) -> None:
    window.replay_recording.setCurrentText("Always record")
    assert window.configuration().visual_replay_enabled is True
    window.replay_recording.setCurrentText("Do not record")
    assert window.configuration().visual_replay_enabled is False
    config = window.configuration()
    config.visual_replay_enabled = None
    window.load_configuration(config)
    assert window.replay_recording.currentText() == "Automatic for live training"


def test_replay_swarm_inspect_uses_indexes_only(window, tmp_path, monkeypatch) -> None:
    replay_run = tmp_path / "run"
    replay_run.mkdir()
    episodes = [
        {
            "run_id": "run", "episode_id": f"episode-{number:06d}",
            "algorithm": "tqc", "track_id": "current", "track_name": "Summer 1",
            "track_slug": "summer-1",
            "training_step": step, "training_step_start": step, "training_step_end": step + 1,
            "episode_length_decisions": 1, "episode_length_ticks": 30,
            "status": status, "final_progress_m": 1.0, "frame_skip": 30,
            "sample_count": 1, "file": f"episode-{number:06d}.npz",
        }
        for number, step, status in ((1, 10, "finished"), (2, 20, "failed"))
    ]
    (replay_run / "index.json").write_text(
        json.dumps({"schema": INDEX_SCHEMA, "run": {"run_id": "run"}, "episodes": episodes}),
        encoding="utf-8",
    )

    def unexpected_transport(*args, **kwargs):
        raise AssertionError("index-only inspection must not connect to the bridge")

    monkeypatch.setattr(gui_main, "WebSocketServerTransport", unexpected_transport)
    window.replay_swarm_path.setText(str(replay_run))
    window.replay_swarm_external.setChecked(True)
    window.replay_swarm_step_min.setValue(10)
    window.replay_swarm_step_max.setValue(10)
    report = window._run_replay_swarm(window._replay_swarm_config() | {"action": "inspect"})
    assert report["total_indexed_episodes"] == 2
    assert report["episodes_matching_filters"] == 1
    assert report["selected_episode_count"] == 1
    assert report["selected_step_range"] == [10, 10]
    assert report["finish_count"] == 1
    assert report["failure_count"] == 0


def test_replay_swarm_bridge_controls_do_not_require_replay_path(window, monkeypatch) -> None:
    calls = []

    class StubTransport:
        def __init__(self, **kwargs):
            calls.append(("transport", kwargs))

        def close(self):
            calls.append(("close",))

    def send(transport, **kwargs):
        calls.append(("send", kwargs))
        return {"ok": True}

    monkeypatch.setattr(gui_main, "WebSocketServerTransport", StubTransport)
    monkeypatch.setattr(gui_main, "send_replay_swarm", send)
    result = window._run_replay_swarm({"action": "status", "port": 8765})
    assert result["result"] == {"ok": True}
    assert calls[1][0] == "send"
    assert calls[1][1]["action"] == "status"
    assert calls[-1] == ("close",)

    window._run_replay_swarm({
        "action": "configure", "port": 8765, "speed": 2.0, "opacity": 0.4,
        "end_behavior": "fade", "fade_duration_s": 1.25,
    })
    assert calls[-2][1]["update_settings"] == (
        "speed", "opacity", "end_behavior", "fade_duration",
    )


def test_replay_swarm_operations_run_off_gui_thread(window, monkeypatch) -> None:
    worker_thread_ids = []
    gui_thread_id = threading.get_ident()

    def operation(config):
        worker_thread_ids.append(threading.get_ident())
        return {"action": config["action"], "ok": True}

    monkeypatch.setattr(window, "_run_replay_swarm", operation)
    window._submit_replay_swarm("status")
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: loop.quit() if window.replay_swarm_worker is None else None)
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timer.start()
    timeout.start(3000)
    loop.exec()
    timer.stop()
    assert worker_thread_ids and worker_thread_ids[0] != gui_thread_id
    assert window.replay_swarm_worker is None
    assert '"ok": true' in window.replay_swarm_output.toPlainText()


def test_global_track_switch_scopes_models_replays_and_training_config(
    qt_app, tmp_path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    window = PolyBotWindow()
    try:
        assert window.track_selector.currentData() == "summer-1"
        autumn = window.track_registry.add("Autumn 2")
        autumn_model = tmp_path / "models" / autumn.slug / "grtqc" / "champion"
        summer_model = tmp_path / "models" / "summer-1" / "grtqc" / "champion"
        autumn_model.mkdir(parents=True)
        summer_model.mkdir(parents=True)

        replay_dir = (
            tmp_path / "models" / autumn.slug / "grtqc" / "visual_replays" / "run-new"
        )
        replay_dir.mkdir(parents=True)
        episode = {
            "run_id": "run-new", "episode_id": "episode-000001",
            "algorithm": "grtqc", "track_id": "current", "track_name": autumn.name,
            "track_slug": autumn.slug, "training_step": 10, "training_step_start": 10,
            "training_step_end": 11, "episode_length_decisions": 1,
            "episode_length_ticks": 30, "status": "failed", "final_progress_m": 1.0,
            "frame_skip": 30, "sample_count": 1, "file": "episode-000001.npz",
        }
        (replay_dir / "index.json").write_text(
            json.dumps({
                "schema": "polybot.visual-replay-index.v1",
                "run": {"run_id": "run-new", "created_at": "2026-10-01T00:00:00Z"},
                "episodes": [episode],
            }),
            encoding="utf-8",
        )

        window._populate_track_selector(autumn.slug)
        window._track_changed()
        assert window.configuration().track_name == "Autumn 2"
        assert window.configuration().track_slug == "autumn-2"
        assert "autumn-2" in window.models_inventory.toPlainText()
        assert "summer-1" not in window.models_inventory.toPlainText()
        assert window.replay_swarm_run.count() == 2
        assert window.replay_swarm_run.currentData() == str(replay_dir.resolve())
        assert window.replay_swarm_path.text() == str(replay_dir.resolve())
        assert window.replay_episode_list.count() == 1
        assert "Attempt 000001" in window.replay_episode_list.item(0).text()
        assert "Failed" in window.replay_episode_list.item(0).text()
        assert "Autumn 2" in window.tabs.tabText(window.tabs.indexOf(window.models_page))
        assert "Autumn 2" in window.tabs.tabText(window.tabs.indexOf(window.replay_swarm_page))

        window.track_selector.setCurrentIndex(window.track_selector.findData("summer-1"))
        assert "summer-1" in window.models_inventory.toPlainText()
        assert "autumn-2" not in window.models_inventory.toPlainText()
        assert window.replay_swarm_run.count() == 0
        assert window.replay_episode_list.count() == 0
    finally:
        window.close()


def test_custom_model_root_refreshes_models_and_replay_runs(window, tmp_path) -> None:
    track = window._selected_track()
    model_root = tmp_path / "custom-models"
    (model_root / track.slug / "grtqc" / "champion").mkdir(parents=True)
    replay_dir = model_root / track.slug / "grtqc" / "visual_replays" / "run-custom"
    replay_dir.mkdir(parents=True)
    episode = {
        "run_id": "run-custom", "episode_id": "episode-000001",
        "algorithm": "grtqc", "track_id": track.simulator_track_id,
        "track_name": track.name, "track_slug": track.slug,
        "training_step": 10, "training_step_start": 10, "training_step_end": 11,
        "episode_length_decisions": 1, "episode_length_ticks": 30,
        "status": "failed", "final_progress_m": 0.1, "frame_skip": 30,
        "sample_count": 1, "file": "episode-000001.npz",
    }
    (replay_dir / "index.json").write_text(
        json.dumps({
            "schema": "polybot.visual-replay-index.v1",
            "run": {"run_id": "run-custom"},
            "episodes": [episode],
        }),
        encoding="utf-8",
    )

    root_field = window.general["output_root"]
    root_field.setText(str(model_root))
    root_field.editingFinished.emit()

    assert "champion" in window.models_inventory.toPlainText()
    assert window.replay_swarm_run.count() == 2
    assert window.replay_swarm_run.currentData() == str(replay_dir.resolve())


def test_legacy_saved_configuration_registers_and_selects_its_track(
    window, tmp_path, monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    config = TrainingConfig(
        algorithm="ppo",
        backend="websocket",
        track_name="Imported Circuit",
        track_id="current",
        ppo=PPOConfig(architecture="tiny"),
    )
    window.load_configuration(config)
    assert window.track_selector.currentText() == "Imported Circuit"
    resolved = window.configuration()
    assert resolved.track_slug == "imported-circuit"
    assert resolved.track_id == "current"


def test_resume_configuration_resizes_custom_curriculum_to_session_budget(window):
    from polybot.environment.curriculum import build_plan
    current = window.configuration()
    current.timesteps = 100_000
    saved = TrainingConfig.from_dict(current.to_dict())
    saved.timesteps = 2_000_000
    saved.curriculum = CurriculumConfig("custom", phases=(
        CurriculumPhaseConfig("section", 500_000, start_ratio=0, end_ratio=.25),
        CurriculumPhaseConfig("full", 1_500_000),
    ))
    resumed = window._resume_configuration(current, SimpleNamespace(training_config=saved.to_dict()))
    plan = build_plan(resumed.curriculum, resumed.timesteps)
    assert plan.total_steps == 100_000
    assert plan.phases[0].mode == "section"
    assert plan.phases[0].steps == 25_000
    assert plan.phases[1].steps == 75_000


def test_auto_resume_keeps_the_champions_verified_device(window):
    current = window.configuration()
    current.device = "auto"
    saved = TrainingConfig.from_dict(current.to_dict())
    saved.device = "cpu"
    metadata = SimpleNamespace(training_config=saved.to_dict(), device="cpu")
    assert window._resume_configuration(current, metadata).device == "cpu"
