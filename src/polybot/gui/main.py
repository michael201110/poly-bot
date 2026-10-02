"""Beginner friendly v2 trainer with full advanced access."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from polybot.environment.curriculum import build_plan
from polybot.environment.rewards import RewardConfig
from polybot.gui.events import format_event
from polybot.models.registry import REWARD_SEMANTICS, ModelRegistry
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    EvaluationConfig,
    GRTQCConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.evaluation import EvaluationResult
from polybot.training.parameters import (
    CURRICULUM_INFO,
    EVALUATION_INFO,
    GENERAL_INFO,
    GRTQC_INFO,
    METRIC_INFO,
    PPO_INFO,
    REWARD_INFO,
    TQC_INFO,
    ParameterInfo,
)
from polybot.training.presets import PresetStore, algorithm_presets, configuration_warnings
from polybot.training.reward_profiles import RewardProfileStore
from polybot.training.runner import TrainingRunner


def _editor(value: Any, help_text: str, choices: tuple[str, ...] = ()) -> QWidget:
    if choices:
        widget = QComboBox()
        widget.addItems(choices)
        widget.setCurrentText(str(value))
    elif isinstance(value, bool):
        widget = QCheckBox()
        widget.setChecked(value)
    elif isinstance(value, int):
        widget = QSpinBox()
        widget.setRange(-2_000_000_000, 2_000_000_000)
        widget.setValue(value)
    elif isinstance(value, float):
        widget = QDoubleSpinBox()
        widget.setRange(-1e12, 1e12)
        widget.setDecimals(8)
        widget.setSingleStep(0.001)
        widget.setValue(value)
    else:
        widget = QLineEdit("" if value is None else str(value))
    widget.setToolTip(help_text)
    return widget


def _value(widget: QWidget) -> Any:
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    assert isinstance(widget, QLineEdit)
    return widget.text().strip() or None


def _set(widget: QWidget, value: Any) -> None:
    previous = widget.blockSignals(True)
    try:
        if isinstance(widget, QComboBox):
            widget.setCurrentText(str(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(value)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setValue(value)
        else:
            assert isinstance(widget, QLineEdit)
            widget.setText("" if value is None else str(value))
    finally:
        widget.blockSignals(previous)


class ParameterForm(QWidget):
    def __init__(
        self, instance: Any, info: dict[str, ParameterInfo],
        basic: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.basic = basic
        self.widgets: dict[str, QWidget] = {}
        self.labels: dict[str, QLabel] = {}
        layout = QFormLayout(self)
        for field in fields(instance):
            if field.name == "phases":
                continue
            metadata = info[field.name]
            current = getattr(instance, field.name)
            if field.name == "architecture":
                if isinstance(instance, PPOConfig):
                    choices = (
                        "tiny", "compact", "standard", "tqc_compatible", "tqc_residual",
                    )
                else:
                    choices = ("tiny", "compact", "standard")
            else:
                choices = ()
            widget = _editor(current, metadata.description, choices)
            label = QLabel(metadata.label)
            label.setToolTip(metadata.description)
            layout.addRow(label, widget)
            self.widgets[field.name] = widget
            self.labels[field.name] = label
        self.set_advanced(False)

    def set_advanced(self, enabled: bool) -> None:
        for name, widget in self.widgets.items():
            visible = self.basic is None or enabled or name in self.basic
            widget.setVisible(visible)
            self.labels[name].setVisible(visible)

    def values(self) -> dict[str, Any]:
        result = {name: _value(widget) for name, widget in self.widgets.items()}
        for name in ("start_ratio", "end_ratio", "start_s", "end_s", "action_std"):
            if name in result and result[name] is not None:
                result[name] = float(result[name])
        return result

    def load(self, instance: Any) -> None:
        for name, widget in self.widgets.items():
            _set(widget, getattr(instance, name))


class EventBridge(QObject):
    event = Signal(dict)
    failed = Signal(str)


class PolyBotWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolyBot Training")
        self.resize(900, 760)
        self.runner: TrainingRunner | None = None
        self.worker: threading.Thread | None = None
        self.speed_search_process: QProcess | None = None
        self.adaptation_process: QProcess | None = None
        self.adaptation_stdout_buffer = ""
        self.distillation_process: QProcess | None = None
        self.distillation_stdout_buffer = ""
        self.distillation_command: str | None = None
        self.teacher_student_process: QProcess | None = None
        self.teacher_student_stop_file: Path | None = None
        self.wr_search_process: QProcess | None = None
        self.section_optimizer_process: QProcess | None = None
        self.section_optimizer_stop_file: Path | None = None
        self.section_optimizer_skip_file: Path | None = None
        self.section_optimizer_refine_file: Path | None = None
        self.section_optimizer_stdout = ""
        self.speed_search_stop_file: Path | None = None
        self.wr_search_stop_file: Path | None = None
        self.wr_stdout_buffer = ""
        self.speed_search_best: float | None = None
        self.speed_search_log_path: Path | None = None
        self.speed_search_log_position = 0
        self.bridge = EventBridge()
        self.bridge.event.connect(self._event)
        self.bridge.failed.connect(self._error)
        self.profiles = RewardProfileStore()
        self.presets = PresetStore()
        self._base_rewards = self.profiles.load("Balanced")
        self._reward_values = asdict(self._base_rewards)

        root = QVBoxLayout(self)
        intro = QLabel("Train a driving policy. Start with a preset; open Advanced for every setting.")
        root.addWidget(intro)
        self.advanced = QCheckBox("Advanced settings")
        self.advanced.setToolTip("Reveal all algorithm and reward numbers for custom experiments.")
        self.advanced.toggled.connect(self._toggle_advanced)
        root.addWidget(self.advanced)
        quick_actions = QHBoxLayout()
        guided = QPushButton("Guided new run")
        guided.setToolTip("Choose a track, algorithm, goal, hardware and preset before starting.")
        guided.clicked.connect(self._guided_new_run)
        quick_actions.addWidget(guided)
        start = QPushButton("Start with these settings")
        start.setToolTip("Validate the exact settings shown, then start a fresh model.")
        start.clicked.connect(lambda: self._start(False))
        quick_actions.addWidget(start)
        continue_best = QPushButton("Continue best model")
        continue_best.setToolTip("Continue the best evaluated checkpoint for these settings.")
        continue_best.clicked.connect(lambda: self._start(True, best=True))
        quick_actions.addWidget(continue_best)
        root.addLayout(quick_actions)
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self._general_tab()
        self._algorithm_tab()
        self._reward_tab()
        self._curriculum_tab()
        self._evaluation_tab()
        self._models_tab()
        self._status_tab()
        self.algorithm.currentTextChanged.connect(self._algorithm_changed)
        self._algorithm_changed(self.algorithm.currentText())
        self._toggle_advanced(False)
        contact_profile = Path("profiles/training/summer-1-grtqc-controller-adapter-30.json")
        if contact_profile.is_file():
            self.load_configuration(TrainingConfig.from_dict(
                json.loads(contact_profile.read_text(encoding="utf-8"))
            ))
        else:
            initialization = ModelRegistry().algorithm_dir("Summer 1", "grtqc") / "initialization"
            if (initialization / "metadata.json").is_file():
                saved = ModelRegistry().read_metadata(initialization)
                self.load_configuration(TrainingConfig.from_dict(saved.training_config))
        self.speed_search_watch = QTimer(self)
        self.speed_search_watch.timeout.connect(self._poll_speed_search_log)
        self.speed_search_watch.start(2000)

    def _page(self, name: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tabs.addTab(page, name)
        return page, layout

    def _add_field(
        self, layout: QFormLayout, name: str, value: Any,
        info: ParameterInfo, choices: tuple[str, ...] = (),
    ) -> QWidget:
        widget = _editor(value, info.description, choices)
        label = QLabel(info.label)
        label.setToolTip(info.description)
        layout.addRow(label, widget)
        return widget

    def _general_tab(self) -> None:
        _, page = self._page("General")
        form = QFormLayout()
        page.addLayout(form)
        self.general: dict[str, QWidget] = {}
        values = {
            "track_name": "Summer 1", "track_id": "current", "backend": "websocket",
            "device": "auto", "seed": 0, "frame_skip": 30, "timesteps": 100_000,
            "max_episode_seconds": 60.0, "max_episode_steps": 30_000,
            "lookahead_count": 12, "reward_scale": 0.01,
            "checkpoint_interval": 10_000,
            "output_root": "models", "log_root": "logs",
        }
        for name, value in values.items():
            choices = ("mock", "websocket") if name == "backend" else (
                ("auto", "cpu", "cuda") if name == "device" else ()
            )
            self.general[name] = self._add_field(form, name, value, GENERAL_INFO[name], choices)
        self.algorithm = self._add_field(
            form, "algorithm", "grtqc", GENERAL_INFO["algorithm"], ("grtqc", "tqc", "ppo")
        )
        self.general["backend"].currentTextChanged.connect(self._backend_changed)
        self.general_advanced = {
            name for name in values if name not in {
                "track_name", "backend", "device", "seed", "frame_skip",
                "timesteps", "max_episode_seconds",
            }
        }
        self.general_labels = {
            name: form.labelForField(widget) for name, widget in self.general.items()
        }
        basics = QPushButton("What do these training words mean?")
        basics.setToolTip("Plain-language explanation of the terms used in GRTQC, TQC and PPO training.")
        basics.clicked.connect(self._show_glossary)
        page.addWidget(basics)

    def _algorithm_tab(self) -> None:
        _, page = self._page("Algorithm")
        self.algorithm_explanation = QLabel()
        self.algorithm_explanation.setWordWrap(True)
        page.addWidget(self.algorithm_explanation)
        row = QHBoxLayout()
        page.addLayout(row)
        self.preset = QComboBox()
        self.preset.setToolTip("Choose explicit parameter values; every resolved value is shown below.")
        self.preset.currentTextChanged.connect(self._preset_changed)
        row.addWidget(self.preset)
        for label, handler, tip in (
            ("Save preset", self._save_preset, "Save current algorithm values as a custom v2 preset."),
            ("Duplicate preset", self._duplicate_preset, "Copy current values under a new name."),
            ("Reset preset", self._reset_preset, "Restore the selected built-in preset values."),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(handler)
            row.addWidget(button)
        self.algorithm_stack = QStackedWidget()
        page.addWidget(self.algorithm_stack)
        self.ppo_form = ParameterForm(
            PPOConfig(), PPO_INFO, {"architecture", "learning_rate", "rollout_steps"}
        )
        self.grtqc_form = ParameterForm(
            GRTQCConfig(), GRTQC_INFO,
            {"architecture", "learning_rate", "train_frequency", "disagreement_coefficient"},
        )
        self.tqc_form = ParameterForm(
            TQCConfig(), TQC_INFO, {"architecture", "learning_rate", "train_frequency"}
        )
        self.algorithm_stack.addWidget(self.ppo_form)
        self.algorithm_stack.addWidget(self.grtqc_form)
        self.algorithm_stack.addWidget(self.tqc_form)
        self.parameter_label = QLabel("Network and total parameter counts appear when training starts.")
        self.parameter_label.setToolTip(
            "GRTQC, TQC and PPO report actor and critic counts. Larger networks train slower."
        )
        page.addWidget(self.parameter_label)

    def _reward_tab(self) -> None:
        _, page = self._page("Rewards")
        row = QHBoxLayout()
        page.addLayout(row)
        self.reward_profile = QComboBox()
        self.reward_profile.addItems(self.profiles.names())
        self.reward_profile.addItem("Custom")
        self.reward_profile.setCurrentText("Balanced")
        self.reward_profile.setToolTip(GENERAL_INFO["reward_profile"].description)
        self.reward_profile.currentTextChanged.connect(self._load_reward_profile)
        row.addWidget(self.reward_profile)
        for label, handler, tip in (
            ("Save profile", self._save_profile, "Save all 69 resolved reward values to a named profile."),
            ("Duplicate", self._duplicate_profile, "Copy this profile under a new name."),
            ("Compare", self._compare_profiles, "Show exact differences between two profiles."),
            ("Reset", self._reset_profile, "Restore the selected profile's original values."),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(handler)
            row.addWidget(button)
        form = QFormLayout()
        page.addLayout(form)
        self.reward_basic: dict[str, QWidget] = {}
        for name, label in (
            ("progress_per_m", "Progress importance"),
            ("on_track_speed_per_m", "Speed importance"),
            ("guidance_reward_scale", "Ghost guidance"),
            ("action_change_penalty", "Smooth driving"),
        ):
            widget = _editor(self._reward_values[name], REWARD_INFO[name].description)
            text = QLabel(label)
            text.setToolTip(REWARD_INFO[name].description)
            form.addRow(text, widget)
            self.reward_basic[name] = widget
            widget.valueChanged.connect(self._basic_reward_changed)
        for name, label, description in (
            ("failure_multiplier", "Failure severity",
             "Multiplies crash, stall, off-track and barrier penalties from the selected profile."),
            ("finish_multiplier", "Finish importance",
             "Multiplies finish and fast-finish bonuses from the selected profile."),
        ):
            widget = _editor(1.0, description)
            form.addRow(QLabel(label), widget)
            self.reward_basic[name] = widget
            widget.valueChanged.connect(self._basic_reward_changed)
        affected = QLabel(
            "Failure changes: crash, stall, off-track, early off-track, barrier. "
            "Finish changes: finish bonus and fast finish bonus."
        )
        affected.setWordWrap(True)
        affected.setToolTip("These are the exact coefficients changed by the two group controls.")
        page.addWidget(affected)
        self.reward_advanced = ParameterForm(RewardConfig(), REWARD_INFO)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.reward_advanced)
        scroll.setMinimumHeight(290)
        page.addWidget(scroll)
        self.reward_scroll = scroll
        scroll.hide()
        for name, widget in self.reward_advanced.widgets.items():
            widget.valueChanged.connect(
                lambda value, field_name=name: self._advanced_reward_changed(field_name, value)
            )
        self.reward_preview = QTextEdit()
        self.reward_preview.setReadOnly(True)
        self.reward_preview.setToolTip("Exact reward coefficients sent to training and saved in metadata.")
        self.reward_preview.setMaximumHeight(120)
        page.addWidget(self.reward_preview)
        self._refresh_reward_view()

    def _curriculum_tab(self) -> None:
        _, page = self._page("Curriculum")
        self.curriculum_form = ParameterForm(
            CurriculumConfig(), CURRICULUM_INFO, {"mode"}
        )
        mode = self.curriculum_form.widgets["mode"]
        assert isinstance(mode, QLineEdit)
        replacement = QComboBox()
        replacement.addItems(("full", "section", "quarters", "quarters-randomised", "q4-full", "timed", "custom"))
        replacement.setToolTip(CURRICULUM_INFO["mode"].description)
        form = self.curriculum_form.layout()
        assert isinstance(form, QFormLayout)
        form.replaceWidget(mode, replacement)
        mode.deleteLater()
        self.curriculum_form.widgets["mode"] = replacement
        page.addWidget(self.curriculum_form)
        self.custom_phases = QTextEdit()
        self.custom_phases.setPlaceholderText(
            '[{"mode": "section", "steps": 50000, "start_ratio": 0.75, "end_ratio": 1.0}, '
            '{"mode": "full", "steps": 50000}]'
        )
        self.custom_phases.setToolTip(CURRICULUM_INFO["phases"].description)
        self.custom_phases.setMaximumHeight(100)
        page.addWidget(self.custom_phases)
        self.custom_phases.textChanged.connect(self._update_plan)
        replacement.currentTextChanged.connect(
            lambda mode: self.custom_phases.setVisible(mode == "custom")
        )
        self.custom_phases.setVisible(False)
        self.plan_label = QLabel("The total budget is split across phases; each phase does not get a full budget.")
        self.plan_label.setWordWrap(True)
        self.plan_label.setToolTip("Sequential quarters use five phases; Q4 then full uses two.")
        page.addWidget(self.plan_label)
        replacement.currentTextChanged.connect(self._update_plan)
        self.general["timesteps"].valueChanged.connect(self._update_plan)
        for name in ("start_ratio", "end_ratio", "start_s", "end_s"):
            self.curriculum_form.widgets[name].editingFinished.connect(self._update_plan)
        self._update_plan()

    def _update_plan(self) -> None:
        try:
            curriculum = self._curriculum_configuration()
            plan = build_plan(curriculum, int(_value(self.general["timesteps"])))
            phases = ", ".join(
                f"{phase.mode} {phase.steps:,}"
                + (f" (spawn {phase.spawn_ratio:.0%}; target {phase.start_ratio:.0%}–{phase.end_ratio:.0%})"
                   if phase.spawn_ratio is not None and phase.start_ratio is not None
                   and phase.end_ratio is not None else "")
                for phase in plan.phases
            )
            self.plan_label.setText(f"Total planned steps: {plan.total_steps:,}. Phases: {phases}.")
        except (ValueError, TypeError):
            self.plan_label.setText("Enter valid section bounds to see the training plan.")

    def _curriculum_configuration(self) -> CurriculumConfig:
        values = self.curriculum_form.values()
        if values["mode"] == "custom":
            phases = json.loads(self.custom_phases.toPlainText())
            if not isinstance(phases, list):
                raise ValueError("custom phases must be a JSON list")
            values["phases"] = tuple(CurriculumPhaseConfig(**phase) for phase in phases)
        return CurriculumConfig(**values)

    def _evaluation_tab(self) -> None:
        _, page = self._page("Evaluation")
        page.addWidget(QLabel(
            "A frozen policy drives seeded full laps. Only these results can replace the champion."
        ))
        self.evaluation_form = ParameterForm(EvaluationConfig(episodes=5), EVALUATION_INFO)
        page.addWidget(self.evaluation_form)
        note = QLabel("A lucky finish during training never promotes a model by itself.")
        note.setToolTip("Champion ranking uses finish rate, progress, then completed lap time.")
        page.addWidget(note)

    def _models_tab(self) -> None:
        _, page = self._page("Models")
        search_form = QFormLayout()
        self.speed_search_target = QDoubleSpinBox()
        self.speed_search_target.setRange(1.0, 600.0)
        self.speed_search_target.setDecimals(3)
        self.speed_search_target.setValue(25.0)
        self.speed_search_target.setToolTip("Stop once a five-lap-confirmed TQC champion meets this median lap time.")
        search_form.addRow("Speed target (s)", self.speed_search_target)
        self.speed_search_trials = QSpinBox()
        self.speed_search_trials.setRange(1, 1_000_000)
        self.speed_search_trials.setValue(1000)
        self.speed_search_trials.setToolTip("Maximum live simulator candidates to test before stopping.")
        search_form.addRow("Search trials", self.speed_search_trials)
        self.speed_search_mode = QComboBox()
        self.speed_search_mode.addItem("Whole-lap actor search", "global")
        self.speed_search_mode.addItem("Section speed search", "section")
        self.speed_search_mode.setToolTip(
            "Whole-lap search adjusts actor outputs; section search changes forward control in one window."
        )
        search_form.addRow("Search method", self.speed_search_mode)
        page.addLayout(search_form)
        speed_button = QPushButton("Optimize TQC champion speed")
        speed_button.setToolTip(
            "Test small actor-output changes on live laps. Save faster champions only after full confirmation."
        )
        speed_button.clicked.connect(self._start_speed_search)
        page.addWidget(speed_button)
        self.adaptation_section = QWidget()
        adaptation_layout = QVBoxLayout(self.adaptation_section)
        adaptation_layout.addWidget(QLabel("Tuned champion adaptation (advanced)"))
        preset = QPushButton("Load tuned champion adaptation preset")
        preset.clicked.connect(self._load_adaptation_preset)
        adaptation_layout.addWidget(preset)
        adaptation_actions = QHBoxLayout()
        adaptation_layout.addLayout(adaptation_actions)
        for label, stage in (("Collect local replay", "collect"),
                              ("Validate local replay", "validate"),
                              ("Adapt critics", "critics"),
                              ("Validate & promote candidate", "promote"),
                              ("Experimental actor-gradient polish", "polish"),
                              ("Run full cycle", "full"),
                              ("Roll back snapshot", "rollback")):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, selected=stage: self._start_adaptation(selected))
            adaptation_actions.addWidget(button)
        page.addWidget(self.adaptation_section)
        self.distillation_section = QWidget()
        distill_layout = QVBoxLayout(self.distillation_section)
        distill_layout.addWidget(QLabel("Bake proven TQC policy overlays into the actor (advanced)"))
        distill_form = QFormLayout()
        self.distillation_run_dir = QLineEdit()
        self.distillation_run_dir.setPlaceholderText("Snapshot a champion to choose a run folder")
        self.distillation_run_dir.setToolTip(
            "Private snapshot and staged student directory; never edits the champion until Bake."
        )
        distill_form.addRow("Distillation run", self.distillation_run_dir)
        self.distillation_episodes = QSpinBox()
        self.distillation_episodes.setRange(2, 1000)
        self.distillation_episodes.setValue(20)
        self.distillation_episodes.setToolTip(
            "Deterministic teacher laps recorded as final post-overlay action targets."
        )
        distill_form.addRow("Teacher laps", self.distillation_episodes)
        self.distillation_validation_episodes = QSpinBox()
        self.distillation_validation_episodes.setRange(5, 100)
        self.distillation_validation_episodes.setValue(5)
        distill_form.addRow("Validation laps", self.distillation_validation_episodes)
        self.distillation_tolerance = QDoubleSpinBox()
        self.distillation_tolerance.setRange(0.0, 1.0)
        self.distillation_tolerance.setDecimals(3)
        self.distillation_tolerance.setValue(0.02)
        self.distillation_tolerance.setToolTip(
            "Student must remain within this many seconds of the snapshotted teacher."
        )
        distill_form.addRow("Maximum lap loss (s)", self.distillation_tolerance)
        self.distillation_bake_kinds = QLineEdit()
        self.distillation_bake_kinds.setPlaceholderText("All smooth overlays and speed schedule")
        self.distillation_bake_kinds.setToolTip(
            "Optional comma-separated subset: steer_bias, steer_gain, drive_bias, drive_gain, "
            "speed_bias_schedule. Air-brake handling is always retained."
        )
        distill_form.addRow("Bake kinds (optional)", self.distillation_bake_kinds)
        distill_layout.addLayout(distill_form)
        distill_actions = QHBoxLayout()
        for label, command in (
            ("Snapshot champion", "snapshot"), ("Collect teacher data", "collect"),
            ("Train actor student", "train"), ("Validate student", "validate"),
            ("Bake / promote", "bake"), ("Rollback bake", "rollback"),
            ("Run full workflow", "full"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, selected=command: self._start_distillation(selected))
            distill_actions.addWidget(button)
        distill_layout.addLayout(distill_actions)
        self.distillation_status = QLabel("Air-brake controls remain low-level and are retained through baking.")
        self.distillation_status.setWordWrap(True)
        distill_layout.addWidget(self.distillation_status)
        page.addWidget(self.distillation_section)
        self.teacher_student_section = QWidget()
        teacher_student_layout = QVBoxLayout(self.teacher_student_section)
        teacher_student_layout.addWidget(QLabel(
            "Train continuous PPO from the frozen 24.263s TQC champion; target a confirmed lap below 22.000s."
        ))
        teacher_student_form = QFormLayout()
        self.teacher_student_teacher = QLineEdit(
            "models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion"
        )
        self.teacher_student_dataset = QLineEdit("runs/teacher-student/summer-1-teacher.npz")
        self.teacher_student_laps = QSpinBox()
        self.teacher_student_laps.setRange(2, 100)
        self.teacher_student_laps.setValue(20)
        self.teacher_student_timesteps = QSpinBox()
        self.teacher_student_timesteps.setRange(5_000, 50_000_000)
        self.teacher_student_timesteps.setSingleStep(100_000)
        self.teacher_student_timesteps.setValue(1_000_000)
        self.dagger_rounds = QSpinBox()
        self.dagger_rounds.setRange(1, 20)
        self.dagger_rounds.setValue(3)
        self.dagger_episodes = QSpinBox()
        self.dagger_episodes.setRange(2, 100)
        self.dagger_episodes.setValue(8)
        self.dagger_nominal_weight = QDoubleSpinBox()
        self.dagger_nominal_weight.setRange(0.0, 1.0)
        self.dagger_nominal_weight.setSingleStep(0.05)
        self.dagger_nominal_weight.setValue(0.6)
        self.dagger_recovery_weight = QDoubleSpinBox()
        self.dagger_recovery_weight.setRange(0.0, 1.0)
        self.dagger_recovery_weight.setSingleStep(0.05)
        self.dagger_recovery_weight.setValue(0.4)
        self.dagger_until_finishing = QCheckBox("Repeat rounds until PPO finishes 5/5")
        self.dagger_continue_rl = QCheckBox("After 5/5, value warmup then fine-tune to <22s")
        teacher_student_form.addRow("Frozen TQC champion", self.teacher_student_teacher)
        teacher_student_form.addRow("Teacher dataset", self.teacher_student_dataset)
        teacher_student_form.addRow("Successful teacher laps", self.teacher_student_laps)
        teacher_student_form.addRow("Fine-tuning block", self.teacher_student_timesteps)
        teacher_student_form.addRow("DAgger rounds", self.dagger_rounds)
        teacher_student_form.addRow("PPO episodes per round", self.dagger_episodes)
        teacher_student_form.addRow("Nominal data weight", self.dagger_nominal_weight)
        teacher_student_form.addRow("Recovery data weight", self.dagger_recovery_weight)
        teacher_student_layout.addLayout(teacher_student_form)
        teacher_student_layout.addWidget(self.dagger_until_finishing)
        teacher_student_layout.addWidget(self.dagger_continue_rl)
        teacher_student_actions = QHBoxLayout()
        for label, stage in (
            ("Collect", "collect"), ("Pretrain actor", "pretrain"),
            ("Validate (5 laps)", "validate"), ("Value warmup", "value_warmup"),
            ("Fine-tune block", "finetune"), ("Run DAgger cycle", "dagger"),
            ("Full transfer pipeline", "full"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, selected=stage: self._start_teacher_student(selected)
            )
            teacher_student_actions.addWidget(button)
        teacher_student_layout.addLayout(teacher_student_actions)
        stop_dagger_button = QPushButton("Stop after current DAgger round")
        stop_dagger_button.clicked.connect(self._stop_teacher_student_after_round)
        teacher_student_layout.addWidget(stop_dagger_button)
        self.teacher_student_status = QLabel("TQC teacher remains frozen; PPO is evaluated and saved separately.")
        self.teacher_student_status.setWordWrap(True)
        teacher_student_layout.addWidget(self.teacher_student_status)
        self.teacher_student_log = QTextEdit()
        self.teacher_student_log.setReadOnly(True)
        self.teacher_student_log.setMaximumHeight(100)
        teacher_student_layout.addWidget(self.teacher_student_log)
        page.addWidget(self.teacher_student_section)
        self.wr_search_section = QWidget()
        wr_layout = QVBoxLayout(self.wr_search_section)
        wr_layout.addWidget(QLabel("WR Pace Optimizer · frozen TQC policy · live lap-time search"))
        wr_form = QFormLayout()
        self.wr_target = QDoubleSpinBox()
        self.wr_target.setRange(1.0, 600.0)
        self.wr_target.setDecimals(3)
        self.wr_target.setValue(22.262)
        self.wr_target.setToolTip("Summer 1 no-fancy-cut world record target. It is configurable.")
        wr_form.addRow("Target lap (s)", self.wr_target)
        self.wr_trials = QSpinBox()
        self.wr_trials.setRange(1, 2000)
        self.wr_trials.setValue(12)
        wr_form.addRow("Maximum candidates", self.wr_trials)
        self.wr_resolution = QComboBox()
        for label, value in (("5% coarse", 0.05), ("2% medium", 0.02), ("1% fine", 0.01)):
            self.wr_resolution.addItem(label, value)
        self.wr_resolution.setToolTip("Progress resolution used to find and compare useful regions.")
        wr_form.addRow("Sector resolution", self.wr_resolution)
        self.wr_micro_gain = QDoubleSpinBox()
        self.wr_micro_gain.setRange(0.0, 5.0)
        self.wr_micro_gain.setDecimals(4)
        self.wr_micro_gain.setValue(0.01)
        self.wr_micro_gain.setToolTip("Gains below this size require the extra confirmation count.")
        wr_form.addRow("Micro-gain threshold (s)", self.wr_micro_gain)
        self.wr_micro_confirm = QSpinBox()
        self.wr_micro_confirm.setRange(5, 100)
        self.wr_micro_confirm.setValue(10)
        wr_form.addRow("Micro-gain confirmations", self.wr_micro_confirm)
        self.wr_family = QComboBox()
        for label, value in (("All parameters", "all"), ("Steering", "steering"),
                             ("Drive", "drive"), ("Air brake", "air_brake")):
            self.wr_family.addItem(label, value)
        wr_form.addRow("Parameter family", self.wr_family)
        self.wr_region_enabled = QCheckBox("Search only selected progress region")
        wr_layout.addWidget(self.wr_region_enabled)
        self.wr_region_start = QDoubleSpinBox()
        self.wr_region_start.setRange(0.0, 0.99)
        self.wr_region_start.setDecimals(3)
        self.wr_region_start.setSingleStep(0.01)
        self.wr_region_start.setValue(0.50)
        self.wr_region_end = QDoubleSpinBox()
        self.wr_region_end.setRange(0.01, 1.0)
        self.wr_region_end.setDecimals(3)
        self.wr_region_end.setSingleStep(0.01)
        self.wr_region_end.setValue(0.55)
        region_row = QHBoxLayout()
        region_row.addWidget(QLabel("From"))
        region_row.addWidget(self.wr_region_start)
        region_row.addWidget(QLabel("to"))
        region_row.addWidget(self.wr_region_end)
        wr_layout.addLayout(region_row)
        wr_layout.addLayout(wr_form)
        wr_actions = QHBoxLayout()
        load_wr_profile = QPushButton("Load Summer 1 WR profile")
        load_wr_profile.clicked.connect(self._load_wr_profile)
        wr_actions.addWidget(load_wr_profile)
        analyze = QPushButton("Analyze champion lap")
        analyze.clicked.connect(lambda: self._start_wr_search(analyze_only=True))
        wr_actions.addWidget(analyze)
        start_wr = QPushButton("Run coordinate descent")
        start_wr.clicked.connect(self._start_wr_search)
        wr_actions.addWidget(start_wr)
        stop_wr = QPushButton("Stop WR search safely")
        stop_wr.clicked.connect(self._stop_wr_search)
        wr_actions.addWidget(stop_wr)
        wr_layout.addLayout(wr_actions)
        self.wr_summary = QLabel("Champion split analysis has not run yet.")
        self.wr_summary.setWordWrap(True)
        wr_layout.addWidget(self.wr_summary)
        self.wr_sector_table = QTableWidget(0, 4)
        self.wr_sector_table.setHorizontalHeaderLabels(("Region", "Time (s)", "Speed (m/s)", "Status"))
        self.wr_sector_table.setSortingEnabled(True)
        self.wr_sector_table.setMaximumHeight(190)
        wr_layout.addWidget(self.wr_sector_table)
        self.section_optimizer_section = QWidget()
        section_layout = QVBoxLayout(self.section_optimizer_section)
        section_layout.addWidget(QLabel(
            "Autonomous Section Optimizer · sequential 10% sweep, then promising-section refinement"
        ))
        section_actions = QHBoxLayout()
        for label, hours in (("Start 1 hour", 1), ("Start 4 hours", 4), ("Run until stopped", None)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, budget=hours: self._start_section_optimizer(budget))
            section_actions.addWidget(button)
        resume = QPushButton("Resume saved search")
        resume.clicked.connect(lambda: self._start_section_optimizer(None))
        section_actions.addWidget(resume)
        pause = QPushButton("Pause and save")
        pause.clicked.connect(self._stop_section_optimizer)
        section_actions.addWidget(pause)
        stop = QPushButton("Stop safely")
        stop.clicked.connect(self._stop_section_optimizer)
        section_actions.addWidget(stop)
        skip = QPushButton("Skip section")
        skip.clicked.connect(self._skip_optimizer_section)
        section_actions.addWidget(skip)
        refine = QPushButton("Force refine")
        refine.clicked.connect(self._force_optimizer_refine)
        section_actions.addWidget(refine)
        section_layout.addLayout(section_actions)
        self.section_optimizer_status = QLabel("No section optimizer run active. Progress resumes from saved state.")
        self.section_optimizer_status.setWordWrap(True)
        section_layout.addWidget(self.section_optimizer_status)
        wr_layout.addWidget(self.section_optimizer_section)
        page.addWidget(self.wr_search_section)
        self.pace_polish_section = QWidget()
        polish_layout = QVBoxLayout(self.pace_polish_section)
        polish_layout.addWidget(QLabel("Pace polishing: conservative TQC gradients from champion"))
        polish_actions = QHBoxLayout()
        polish_layout.addLayout(polish_actions)
        for label, steps in (("Polish champion 25k", 25_000), ("Polish champion 50k", 50_000)):
            button = QPushButton(label)
            button.setToolTip("Load Summer 1 safe-polish settings and resume the evaluated champion.")
            button.clicked.connect(lambda _checked=False, budget=steps: self._start_polish(budget))
            polish_actions.addWidget(button)
        search_actions = QHBoxLayout()
        polish_layout.addLayout(search_actions)
        for label, mode in (("Search global pace", "global"), ("Search section pace", "section")):
            button = QPushButton(label)
            button.setToolTip("Screen candidates, then confirm faster laps before saving champion.")
            button.clicked.connect(lambda _checked=False, selected=mode: self._start_pace_search(selected))
            search_actions.addWidget(button)
        page.addWidget(self.pace_polish_section)
        for label, handler, description in (
            ("Guided new run", self._guided_new_run,
             "Choose a track, algorithm, goal, device and preset, then review exact values."),
            ("New training run", lambda: self._start(False),
             "Start fresh with the exact settings shown. Existing champion remains until evaluation improves it."),
            ("Continue from best", lambda: self._start(True, best=True),
             "Continue the best evaluated policy. Older champions refill replay before learning."),
            ("Resume latest (advanced)", lambda: self._start(True),
             "Continue the most recent policy, even if its evaluation regressed."),
            ("Evaluate latest", lambda: self._model_command("evaluate", "latest"),
             "Test latest deterministically without updating it."),
            ("Evaluate champion", lambda: self._model_command("evaluate", "champion"),
             "Test the best proven policy deterministically."),
            ("Play champion", lambda: self._model_command("drive", "champion"),
             "Drive one real-time lap using the champion policy."),
            ("Play latest", lambda: self._model_command("drive", "latest"),
             "Drive one real-time lap using the most recent policy."),
            ("Save configuration", self._save_config,
             "Write the exact resolved v2 training configuration to JSON."),
            ("Load configuration", self._load_config_dialog,
             "Load a v2 configuration and populate every visible and advanced field."),
        ):
            button = QPushButton(label)
            button.setToolTip(description)
            button.clicked.connect(handler)
            page.addWidget(button)
        self.stop_button = QPushButton("Stop cleanly")
        self.stop_button.setToolTip("Ask training to stop after this step and save latest state.")
        self.stop_button.clicked.connect(self._stop)
        page.addWidget(self.stop_button)

    def _status_tab(self) -> None:
        _, page = self._page("Status")
        self.warnings = QLabel("Warnings will appear here; unusual settings are suggestions, not blocks.")
        self.warnings.setWordWrap(True)
        self.warnings.setToolTip("These messages explain possible speed or stability tradeoffs.")
        page.addWidget(self.warnings)
        self.metrics = QLabel("No run active")
        self.metrics.setWordWrap(True)
        self.metrics.setToolTip(
            "TPS is environment decisions per wall second; progress is the current attempt's fraction of track."
        )
        page.addWidget(self.metrics)
        self.pace_status = QLabel("Pace: awaiting evaluation")
        self.pace_status.setWordWrap(True)
        page.addWidget(self.pace_status)
        self._pace_champion_lap: float | None = None
        metric_form = QFormLayout()
        page.addLayout(metric_form)
        self.metric_widgets: dict[str, QLabel] = {}
        for name, info in METRIC_INFO.items():
            label = QLabel(info.label)
            label.setToolTip(info.description)
            value = QLabel("—")
            value.setToolTip(info.description)
            metric_form.addRow(label, value)
            self.metric_widgets[name] = value
        self.metric_form = metric_form
        self.log_location = QLabel("Recent events · full detail is saved to a JSONL log")
        self.log_location.setWordWrap(True)
        page.addWidget(self.log_location)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(400)
        self.log.setToolTip(
            "Short summaries of training events. The JSONL file keeps every metric and reward term."
        )
        page.addWidget(self.log)

    def _toggle_advanced(self, enabled: bool) -> None:
        for form in (self.ppo_form, self.grtqc_form, self.tqc_form, self.curriculum_form):
            form.set_advanced(enabled)
        self.reward_scroll.setVisible(enabled)
        self.pace_polish_section.setVisible(enabled)
        self.adaptation_section.setVisible(enabled)
        self.distillation_section.setVisible(enabled)
        self.teacher_student_section.setVisible(enabled)
        self.wr_search_section.setVisible(enabled)
        self.section_optimizer_section.setVisible(enabled)
        for name in self.general_advanced:
            self.general[name].setVisible(enabled)
            self.general_labels[name].setVisible(enabled)

    def _backend_changed(self, backend: str) -> None:
        if backend == "mock":
            _set(self.general["track_name"], "Mock straight")
            _set(self.general["track_id"], "mock/straight")
            _set(self.general["frame_skip"], 4)
        else:
            _set(self.general["track_name"], "Summer 1")
            _set(self.general["track_id"], "current")
            _set(self.general["frame_skip"], 30)

    def _algorithm_changed(self, algorithm: str) -> None:
        forms = {"ppo": self.ppo_form, "grtqc": self.grtqc_form, "tqc": self.tqc_form}
        self.algorithm_stack.setCurrentWidget(forms[algorithm])
        explanations = {
            "ppo": (
                "PPO: on-policy. It learns from fresh rollouts, then discards them. "
                "It directly outputs continuous steering and signed throttle/brake demand."
            ),
            "grtqc": (
                "GRTQC: the primary off-policy learner. It starts from the proven TQC actor, "
                "warms gated quantile critics, then fine-tunes continuous controls."
            ),
            "tqc": (
                "TQC: off-policy. It reuses replay and learns continuous steering and pedal demand "
                "with an actor and quantile critics."
            ),
        }
        self.algorithm_explanation.setText(explanations[algorithm])
        ppo_metrics = {"policy_loss", "value_loss", "entropy", "explained_variance", "kl", "clip_fraction"}
        shared_replay = {"replay_size", "updates"}
        tqc_metrics = {"entropy_coefficient", "actor_loss", "critic_loss"} | shared_replay
        grtqc_metrics = tqc_metrics | {
            "critic_disagreement", "disagreement_penalty", "quantile_mean", "target_mean",
            "critic_warmup_updates", "actor_unlocked",
        }
        algorithm_metrics = {"ppo": ppo_metrics, "grtqc": grtqc_metrics, "tqc": tqc_metrics}
        specific_metrics = ppo_metrics | grtqc_metrics
        for name, value in self.metric_widgets.items():
            visible = name not in specific_metrics or name in algorithm_metrics[algorithm]
            value.setVisible(visible)
            self.metric_form.labelForField(value).setVisible(visible)
        current = self.preset.blockSignals(True)
        self.preset.clear()
        self.preset.addItems(algorithm_presets(algorithm))
        self.preset.addItems(self.presets.list(algorithm))
        self.preset.setCurrentText("Balanced")
        self.preset.blockSignals(current)
        self._preset_changed("Balanced")

    def _preset_changed(self, name: str) -> None:
        if not name:
            return
        preset = algorithm_presets(self.algorithm.currentText()).get(name)
        if preset is None:
            path = self.presets.list(self.algorithm.currentText()).get(name)
            if path is None:
                return
            preset = self.presets.load(path)
        {"ppo": self.ppo_form, "grtqc": self.grtqc_form, "tqc": self.tqc_form}[
            self.algorithm.currentText()
        ].load(preset)

    def _current_algorithm_settings(self) -> PPOConfig | GRTQCConfig | TQCConfig:
        forms = {"ppo": self.ppo_form, "grtqc": self.grtqc_form, "tqc": self.tqc_form}
        types = {"ppo": PPOConfig, "grtqc": GRTQCConfig, "tqc": TQCConfig}
        algorithm = self.algorithm.currentText()
        values = forms[algorithm].values()
        if algorithm in {"tqc", "grtqc"}:
            for name in ("actor_learning_rate", "critic_learning_rate"):
                if values[name] is not None:
                    values[name] = float(values[name])
        return types[algorithm](**values)

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name")
        if ok and name.strip():
            path = self.presets.save(self.algorithm.currentText(), name, self._current_algorithm_settings())
            if self.preset.findText(name) < 0:
                self.preset.addItem(name)
            self.preset.setCurrentText(name)
            self.log.append(f"Saved preset: {path}")

    def _duplicate_preset(self) -> None:
        self._save_preset()

    def _reset_preset(self) -> None:
        self._preset_changed(self.preset.currentText())

    def _show_glossary(self) -> None:
        glossary = (
            "Policy / actor: a network that chooses steering and pedals in PPO or TQC.\n"
            "Q-value: mean predicted future reward for one digital action.\n"
            "Quantiles: critics' estimates of low-to-high possible future returns.\n"
            "Critic: a network estimating how useful actions or states may be.\n"
            "Environment step: one driving decision. Physics tick: one fixed simulator update.\n"
            "Frame skip: ticks between decisions; larger is faster but reacts slower.\n"
            "Rollout: fresh PPO experiences, discarded after an update (on-policy).\n"
            "Replay buffer: reusable driving history for TQC and GRTQC (off-policy).\n"
            "Target network: slowly updated critics that steady value targets.\n"
            "Batch: experiences processed in one gradient update.\n"
            "Learning rate: size of a gradient update. Gamma: weight on future reward.\n"
            "Entropy / exploration: encouragement to try different actions.\n"
            "Curriculum: shorter practice sections before or alongside full laps.\n"
            "Teacher / ghost: an optional reference; the ghost also defines the route.\n"
            "Evaluation: frozen, repeatable full-track test. Champion: best evaluated policy."
        )
        QMessageBox.information(self, "Training basics", glossary)

    def _guided_new_run(self) -> None:
        wizard = QWizard(self)
        wizard.setWindowTitle("New PolyBot training run")

        def page(title: str) -> tuple[QWizardPage, QFormLayout]:
            item = QWizardPage()
            item.setTitle(title)
            layout = QFormLayout(item)
            wizard.addPage(item)
            return item, layout

        _, track_layout = page("1. Choose a track")
        track = QLineEdit(str(_value(self.general["track_name"])))
        track.setToolTip(GENERAL_INFO["track_name"].description)
        backend = _editor("websocket", GENERAL_INFO["backend"].description, ("websocket", "mock"))
        track_layout.addRow("Track name", track)
        track_layout.addRow("Simulator", backend)

        _, algorithm_layout = page("2. Choose an algorithm")
        algorithm = _editor("grtqc", GENERAL_INFO["algorithm"].description, ("grtqc", "tqc", "ppo"))
        algorithm_layout.addRow("Algorithm", algorithm)
        explanation = QLabel(
            "PPO uses continuous steering and signed longitudinal control with fresh rollouts. "
            "GRTQC uses gated continuous controls and replay. "
            "TQC is the reference algorithm."
        )
        explanation.setWordWrap(True)
        algorithm_layout.addRow(explanation)

        _, goal_layout = page("3. Choose a training goal")
        goal = _editor("Learn the track", "Selects a visible reward profile; exact values appear in review.",
                       ("Learn the track", "Improve consistency", "Improve lap time", "Experiment"))
        goal_layout.addRow("Goal", goal)

        _, device_layout = page("4. Choose hardware")
        device = _editor("auto", GENERAL_INFO["device"].description, ("auto", "cpu", "cuda"))
        device_layout.addRow("Device", device)

        _, preset_layout = page("5. Choose a parameter preset")
        preset = QComboBox()
        preset.setToolTip("Every preset resolves to exact algorithm values shown on the next page.")
        preset_layout.addRow("Preset", preset)

        _, review_layout = page("6. Review exact configuration")
        review = QTextEdit()
        review.setReadOnly(True)
        review.setToolTip("Exact v2 config that will be saved with the model.")
        review_layout.addRow(review)
        chosen: dict[str, TrainingConfig] = {}

        def update(current: int) -> None:
            if current == 4:
                preset.clear()
                preset.addItems(algorithm_presets(_value(algorithm)))
            if current == 5:
                data = self.configuration().to_dict()
                selected_algorithm = _value(algorithm)
                selected_backend = _value(backend)
                selected_goal = _value(goal)
                profile = {
                    "Learn the track": "Learning", "Improve consistency": "Balanced",
                    "Improve lap time": "Pace", "Experiment": "Balanced",
                }[selected_goal]
                data.update({
                    "algorithm": selected_algorithm, "backend": selected_backend,
                    "track_name": track.text().strip(),
                    "track_id": "current" if selected_backend == "websocket" else "mock/straight",
                    "frame_skip": 30 if selected_backend == "websocket" else 4,
                    "device": _value(device), "reward_profile": profile,
                    "rewards": asdict(self.profiles.load(profile)),
                    "ppo": None, "grtqc": None, "tqc": None,
                })
                data[selected_algorithm] = asdict(
                    algorithm_presets(selected_algorithm)[_value(preset)]
                )
                try:
                    resolved = TrainingConfig.from_dict(data)
                    chosen["config"] = resolved
                    review.setPlainText(json.dumps(data, indent=2) + "\n\n" +
                                        "Warnings:\n" + "\n".join(configuration_warnings(resolved)))
                except (ValueError, KeyError) as exc:
                    review.setPlainText(str(exc))

        wizard.currentIdChanged.connect(update)
        if wizard.exec() and "config" in chosen:
            self.load_configuration(chosen["config"])
            self._start(False)

    def _load_reward_profile(self, name: str) -> None:
        if not name or name == "Custom":
            return
        self._base_rewards = self.profiles.load(name)
        self._reward_values = asdict(self._base_rewards)
        for key, widget in self.reward_basic.items():
            _set(widget, 1.0 if key.endswith("_multiplier") else self._reward_values[key])
        self._refresh_reward_view()

    def _basic_reward_changed(self) -> None:
        for name in ("progress_per_m", "on_track_speed_per_m", "guidance_reward_scale", "action_change_penalty"):
            self._reward_values[name] = _value(self.reward_basic[name])
        base = asdict(self._base_rewards)
        failure = _value(self.reward_basic["failure_multiplier"])
        finish = _value(self.reward_basic["finish_multiplier"])
        for name in ("crash_penalty", "stall_penalty", "off_track_penalty",
                     "early_off_track_penalty", "barrier_contact_penalty"):
            self._reward_values[name] = base[name] * failure
        for name in ("finish_bonus", "finish_fast_bonus"):
            self._reward_values[name] = base[name] * finish
        self._refresh_reward_view()

    def _advanced_reward_changed(self, name: str, value: float) -> None:
        self._reward_values[name] = value
        self._refresh_reward_view(update_form=False)

    def _refresh_reward_view(self, *, update_form: bool = True) -> None:
        if update_form:
            self.reward_advanced.load(RewardConfig(**self._reward_values))
        self.reward_preview.setPlainText(json.dumps(self._reward_values, indent=2))

    def _save_profile(self) -> None:
        name, ok = QInputDialog.getText(self, "Save profile", "Profile name")
        if ok and name.strip():
            path = self.profiles.save(name, RewardConfig(**self._reward_values))
            self.reward_profile.addItem(name)
            self.reward_profile.setCurrentText(name)
            self.log.append(f"Saved reward profile: {path}")

    def _duplicate_profile(self) -> None:
        self._save_profile()

    def _compare_profiles(self) -> None:
        other, ok = QInputDialog.getItem(
            self, "Compare profiles", "Other profile", self.profiles.names(), 0, False
        )
        if ok:
            before = self._reward_values
            after = asdict(self.profiles.load(other))
            difference = {
                key: (value, after[key]) for key, value in before.items()
                if value != after[key]
            }
            QMessageBox.information(self, "Profile differences", json.dumps(difference, indent=2))

    def _reset_profile(self) -> None:
        self._load_reward_profile(self.reward_profile.currentText())

    def configuration(self) -> TrainingConfig:
        values = {name: _value(widget) for name, widget in self.general.items()}
        values["output_root"] = Path(values["output_root"])
        values["log_root"] = Path(values["log_root"])
        values["algorithm"] = self.algorithm.currentText()
        selected_profile = self.reward_profile.currentText()
        values["reward_profile"] = None if selected_profile == "Custom" else selected_profile
        values["curriculum"] = self._curriculum_configuration()
        values["evaluation"] = EvaluationConfig(**self.evaluation_form.values())
        values["rewards"] = RewardConfig(**self._reward_values)
        values[values["algorithm"]] = self._current_algorithm_settings()
        return TrainingConfig(**values)

    def load_configuration(self, config: TrainingConfig) -> None:
        _set(self.algorithm, config.algorithm)
        self._algorithm_changed(config.algorithm)
        self.general["backend"].blockSignals(True)
        for name, widget in self.general.items():
            _set(widget, getattr(config, name))
        self.general["backend"].blockSignals(False)
        self.curriculum_form.load(config.curriculum)
        self.custom_phases.setPlainText(json.dumps([asdict(phase) for phase in config.curriculum.phases], indent=2))
        self.custom_phases.setVisible(config.curriculum.mode == "custom")
        self.evaluation_form.load(config.evaluation)
        self.ppo_form.load(config.ppo or PPOConfig())
        self.grtqc_form.load(config.grtqc or GRTQCConfig())
        self.tqc_form.load(config.tqc or TQCConfig())
        self._reward_values = asdict(config.rewards)
        self._base_rewards = config.rewards
        self._refresh_reward_view()
        for key, widget in self.reward_basic.items():
            _set(widget, 1.0 if key.endswith("_multiplier") else self._reward_values[key])
        self.reward_profile.blockSignals(True)
        self.reward_profile.setCurrentText(config.reward_profile or "Custom")
        self.reward_profile.blockSignals(False)
        self._update_plan()

    def _save_config(self) -> None:
        try:
            cfg = self.configuration()
            name, ok = QInputDialog.getText(self, "Save configuration", "JSON path")
            if ok and name.strip():
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
                self.log.append(f"Saved configuration: {path}")
        except (ValueError, OSError) as exc:
            self._error(str(exc))

    def _load_config_dialog(self) -> None:
        name, ok = QInputDialog.getText(self, "Load configuration", "JSON path")
        if ok and name.strip():
            try:
                cfg = TrainingConfig.from_dict(json.loads(Path(name).read_text(encoding="utf-8")))
                self.load_configuration(cfg)
            except (ValueError, OSError, KeyError) as exc:
                self._error(str(exc))

    def _best_resume_slot(self, cfg: TrainingConfig) -> Path:
        registry = ModelRegistry(cfg.output_root)
        latest = registry.slot(cfg.track_name, cfg.algorithm, "latest")
        champion = registry.slot(cfg.track_name, cfg.algorithm, "champion")
        if not (champion / "metadata.json").is_file():
            return latest
        if not (latest / "metadata.json").is_file():
            return champion
        best_eval = registry.read_metadata(champion).evaluation
        latest_eval = registry.read_metadata(latest).evaluation
        if best_eval is not None and (
            latest_eval is None
            or EvaluationResult(**best_eval).rank() >= EvaluationResult(**latest_eval).rank()
        ):
            return champion
        return latest

    def _start_polish(self, steps: int) -> None:
        profile = Path("profiles/training/summer-1-tqc-safe-polish.json")
        try:
            config = TrainingConfig.from_dict(json.loads(profile.read_text(encoding="utf-8")))
            config.timesteps = steps
            self.load_configuration(config)
            self._start(True, pace_polish=True)
        except (OSError, ValueError, KeyError) as exc:
            self._error(str(exc))

    def _start_pace_search(self, mode: str) -> None:
        self.speed_search_mode.setCurrentIndex(self.speed_search_mode.findData(mode))
        self._start_speed_search()

    def _load_adaptation_preset(self) -> None:
        path = Path("profiles/training/summer-1-tqc-tuned-adaptation.json")
        try:
            self.load_configuration(TrainingConfig.from_dict(
                json.loads(path.read_text(encoding="utf-8-sig"))
            ))
            self.log.append("Loaded tuned champion adaptation preset (actor 1e-6, critic 5e-5).")
        except (OSError, ValueError, KeyError) as exc:
            self._error(str(exc))

    def _start_adaptation(self, stage: str) -> None:
        if self.distillation_process is not None and self.distillation_process.state() != QProcess.NotRunning:
            self._error("Wait for distillation to finish before starting champion adaptation.")
            return
        if self.section_optimizer_process is not None and self.section_optimizer_process.state() != QProcess.NotRunning:
            self._error("Stop the section optimizer before champion adaptation.")
            return
        if self.worker is not None and self.worker.is_alive():
            self._error("Stop gradient training before champion adaptation.")
            return
        if self.adaptation_process is not None and self.adaptation_process.state() != QProcess.NotRunning:
            self._error("Champion adaptation is already running.")
            return
        if self.speed_search_process is not None and self.speed_search_process.state() != QProcess.NotRunning:
            self._error("Stop speed search before champion adaptation.")
            return
        if self.wr_search_process is not None and self.wr_search_process.state() != QProcess.NotRunning:
            self._error("Stop WR pace search before champion adaptation.")
            return
        try:
            cfg = self.configuration()
            if self._external_speed_search_running(cfg):
                raise RuntimeError("A live speed search is already using the simulator")
            if cfg.algorithm != "tqc":
                raise ValueError("Tuned champion adaptation is available only for TQC")
            config_path = cfg.log_root / f"tuned-adaptation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
            process = QProcess(self)
            process.setProgram(sys.executable)
            process.setArguments(["-m", "polybot.training.adaptation", "--config", str(config_path), "--stage", stage])
            process.readyReadStandardOutput.connect(self._adaptation_output)
            process.readyReadStandardError.connect(self._adaptation_error)
            process.finished.connect(self._adaptation_finished)
            self.adaptation_process = process
            self.adaptation_stdout_buffer = ""
            process.start()
            if not process.waitForStarted(5000):
                raise RuntimeError("Could not start the champion adaptation process")
            self.log.append(f"Tuned champion adaptation started: {stage}.")
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
        except (OSError, ValueError, RuntimeError) as exc:
            self._error(str(exc))

    def _adaptation_output(self) -> None:
        if self.adaptation_process is None:
            return
        data = bytes(self.adaptation_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.adaptation_stdout_buffer += data
        while "\n" in self.adaptation_stdout_buffer:
            line, self.adaptation_stdout_buffer = self.adaptation_stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "adaptation_critic_progress":
                    self.metrics.setText(
                        f"Critic adaptation · {event['updates']:,}/{event['total_updates']:,} updates · "
                        f"loss {event.get('critic_loss', float('nan')):.4g} · "
                        f"Q disagreement {event.get('critic_disagreement', float('nan')):.4g}"
                    )
                elif event.get("type", "").startswith("adaptation_"):
                    self.log.append(format_event(event))
            except (json.JSONDecodeError, TypeError, ValueError):
                self.log.append(line)

    def _adaptation_error(self) -> None:
        if self.adaptation_process is not None:
            data = bytes(self.adaptation_process.readAllStandardError()).decode("utf-8", errors="replace")
            if data.strip():
                self.log.append(data.strip())

    def _adaptation_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._adaptation_output()
        self._adaptation_error()
        if self.adaptation_stdout_buffer.strip():
            self.log.append(self.adaptation_stdout_buffer.strip())
            self.adaptation_stdout_buffer = ""
        self.log.append("Champion adaptation finished." if exit_code == 0
                        else f"Champion adaptation failed (exit {exit_code}).")
        self.adaptation_process = None

    def _start_distillation(self, command: str) -> None:
        from polybot.training.distillation import simulator_service_active

        if self.distillation_process is not None and self.distillation_process.state() != QProcess.NotRunning:
            self._error("A distillation command is already running.")
            return
        cfg = None
        try:
            cfg = self.configuration()
            if cfg.algorithm != "tqc":
                raise ValueError("Overlay distillation currently supports TQC champions only")
            if command != "snapshot":
                if self.worker is not None and self.worker.is_alive():
                    raise RuntimeError("Wait for gradient training to finish before distillation")
                if any(process is not None and process.state() != QProcess.NotRunning for process in (
                    self.section_optimizer_process, self.wr_search_process, self.speed_search_process,
                    self.adaptation_process,
                )) or simulator_service_active():
                    raise RuntimeError("The live search owns the simulator; snapshot now and collect after it stops")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            cfg.log_root.mkdir(parents=True, exist_ok=True)
            config_path = cfg.log_root / f"distillation-config-{stamp}.json"
            config_path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
            args = ["-m", "polybot.training.distillation", command]
            if command in {"snapshot", "full"}:
                args.extend(("--config", str(config_path)))
            else:
                run_text = self.distillation_run_dir.text().strip()
                if not run_text:
                    raise ValueError("Snapshot a champion first and select its distillation run folder")
                args.extend(("--run-dir", run_text))
            if command == "snapshot":
                args.extend(("--run-id", f"gui-{stamp}"))
            elif command == "collect":
                args.extend(("--episodes", str(self.distillation_episodes.value())))
            elif command == "validate":
                args.extend(("--episodes", str(self.distillation_validation_episodes.value()),
                             "--tolerance", str(self.distillation_tolerance.value())))
            elif command == "train" and self.distillation_bake_kinds.text().strip():
                kinds = [kind.strip() for kind in self.distillation_bake_kinds.text().split(",") if kind.strip()]
                args.extend(("--bake-kinds", *kinds))
            elif command == "bake":
                args.extend(("--tolerance", str(self.distillation_tolerance.value())))
            elif command == "full":
                args.extend(("--episodes", str(self.distillation_episodes.value()),
                             "--validation-episodes", str(self.distillation_validation_episodes.value()),
                             "--tolerance", str(self.distillation_tolerance.value())))
            process = QProcess(self)
            process.setProgram(sys.executable)
            process.setArguments(args)
            process.setWorkingDirectory(str(Path.cwd()))
            process.readyReadStandardOutput.connect(self._distillation_output)
            process.readyReadStandardError.connect(self._distillation_error)
            process.finished.connect(self._distillation_finished)
            self.distillation_process = process
            self.distillation_command = command
            self.distillation_stdout_buffer = ""
            self.distillation_status.setText(f"Distillation {command} running…")
            process.start()
            if not process.waitForStarted(3000):
                raise RuntimeError(process.errorString())
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
        except (ValueError, OSError, RuntimeError, FileNotFoundError) as exc:
            self.distillation_process = None
            self._error(str(exc))

    def _distillation_output(self) -> None:
        if self.distillation_process is None:
            return
        data = bytes(self.distillation_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.distillation_stdout_buffer += data
        for line in data.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "distillation_progress":
                self.distillation_status.setText(event.get("message", "Distillation running…"))

    def _distillation_error(self) -> None:
        if self.distillation_process is None:
            return
        error = bytes(self.distillation_process.readAllStandardError()).decode("utf-8", errors="replace")
        if error.strip():
            self.log.append(f"Distillation: {error[-1600:]}")

    def _distillation_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._distillation_output()
        self._distillation_error()
        command = self.distillation_command
        try:
            result = json.loads(self.distillation_stdout_buffer.splitlines()[-1])
        except json.JSONDecodeError:
            result = None
        if exit_code == 0 and isinstance(result, dict):
            if command == "snapshot" and result.get("run_dir"):
                self.distillation_run_dir.setText(result["run_dir"])
            if command == "full" and result.get("run_dir"):
                self.distillation_run_dir.setText(result["run_dir"])
            if command == "validate":
                accepted = bool(result.get("accepted"))
                delta = result.get("lap_delta_s")
                self.distillation_status.setText(
                    f"Student {'passes' if accepted else 'fails'} lap gate; delta {delta!s}s."
                )
            else:
                self.distillation_status.setText(self._distillation_summary(command or "", result))
            self.log.append(f"Distillation {command}: {json.dumps(result, separators=(',', ':'))[:1600]}")
        else:
            self.distillation_status.setText(f"Distillation {command} failed (exit {exit_code}); see log.")
            if self.distillation_stdout_buffer.strip():
                self.log.append(self.distillation_stdout_buffer[-1600:])
        self.distillation_process = None

    def _start_teacher_student(self, stage: str) -> None:
        if self.teacher_student_process is not None and self.teacher_student_process.state() != QProcess.NotRunning:
            self._error("The PPO teacher-student pipeline is already running.")
            return
        if self.worker is not None and self.worker.is_alive():
            self._error("Stop the active trainer before using the simulator for teacher-student transfer.")
            return
        if any(process is not None and process.state() != QProcess.NotRunning for process in (
            self.adaptation_process, self.distillation_process, self.section_optimizer_process,
            self.wr_search_process, self.speed_search_process,
        )):
            self._error("Stop the active model or search process before starting teacher-student transfer.")
            return
        teacher = self.teacher_student_teacher.text().strip()
        dataset = self.teacher_student_dataset.text().strip()
        if not teacher or not dataset:
            self._error("Choose the frozen TQC champion and teacher dataset paths.")
            return
        args = [
            "-m", "polybot.training.teacher_student", "--stage", stage,
            "--teacher", teacher, "--dataset", dataset,
            "--laps", str(self.teacher_student_laps.value()),
            "--timesteps", str(self.teacher_student_timesteps.value()),
            "--device", str(_value(self.general["device"])),
        ]
        if stage == "dagger":
            stop_path = Path("runs/teacher-student/dagger.stop")
            stop_path.parent.mkdir(parents=True, exist_ok=True)
            stop_path.unlink(missing_ok=True)
            self.teacher_student_stop_file = stop_path
            args.extend((
                "--rounds", str(self.dagger_rounds.value()),
                "--episodes-per-round", str(self.dagger_episodes.value()),
                "--nominal-weight", str(self.dagger_nominal_weight.value()),
                "--recovery-weight", str(self.dagger_recovery_weight.value()),
                "--stop-file", str(stop_path),
            ))
            if self.dagger_until_finishing.isChecked():
                args.append("--until-finishing")
            if self.dagger_continue_rl.isChecked():
                args.append("--continue-to-rl")
        if stage == "full":
            args.extend(("--max-rounds", "0"))
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(args)
        process.setWorkingDirectory(str(Path.cwd()))
        process.readyReadStandardOutput.connect(self._teacher_student_output)
        process.readyReadStandardError.connect(self._teacher_student_error)
        process.finished.connect(self._teacher_student_finished)
        self.teacher_student_process = process
        self.teacher_student_status.setText(f"TQC → PPO {stage} stage is starting.")
        process.start()
        if not process.waitForStarted(5000):
            self.teacher_student_process = None
            self._error("Could not start the teacher-student process: " + process.errorString())
            return
        self.tabs.setCurrentWidget(self.teacher_student_section.parentWidget())

    def _stop_teacher_student_after_round(self) -> None:
        process = self.teacher_student_process
        if process is None or process.state() == QProcess.NotRunning:
            self._error("No DAgger run is active.")
            return
        if self.teacher_student_stop_file is None:
            self._error("The active teacher-student stage has no round-safe stop file.")
            return
        self.teacher_student_stop_file.parent.mkdir(parents=True, exist_ok=True)
        self.teacher_student_stop_file.write_text("stop after this round\n", encoding="utf-8")
        self.teacher_student_status.setText(
            "Stop requested; DAgger will finish the current round and keep its checkpoint."
        )

    def _teacher_student_output(self) -> None:
        if self.teacher_student_process is None:
            return
        data = bytes(self.teacher_student_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        if data:
            self.teacher_student_log.moveCursor(QTextCursor.MoveOperation.End)
            self.teacher_student_log.insertPlainText(data)
            self.teacher_student_log.ensureCursorVisible()
            for line in data.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "best_lap_s" in event:
                    self.teacher_student_status.setText(
                        f"PPO best lap {event.get('best_lap_s')}s · "
                        f"finish rate {event.get('finish_rate')} · target <22.000s"
                    )
                elif event.get("target_reached"):
                    self.teacher_student_status.setText(
                        f"PPO confirmed {event.get('lap_s')}s at {event.get('timesteps')} steps."
                    )
                elif "dagger_round" in event:
                    error = event.get("action_error", {})
                    divergence = error.get("first_major_divergence_progress")
                    if divergence is not None:
                        self.teacher_student_log.append(
                            f"First measured PPO/TQC divergence: {divergence} progress"
                        )
                    self.teacher_student_status.setText(
                        f"DAgger round {event['dagger_round']} · "
                        f"{event.get('samples', 0)} recovery states · "
                        f"finish rate {event.get('finish_rate', 0):.0%} · "
                        f"median progress {event.get('median_progress', 0):.1%}"
                    )
                    failures = error.get("failure_progress", [])
                    first_failure = failures[0].get("progress") if failures else None
                    self.teacher_student_log.append(
                        "Action error steering/longitudinal: "
                        f"{error.get('mean_steering_error', 0):.4f}/"
                        f"{error.get('mean_longitudinal_error', 0):.4f}; "
                        f"first failed progress: {first_failure if first_failure is not None else 'none'}"
                    )
                elif event.get("dagger_stopped"):
                    self.teacher_student_status.setText(
                        f"DAgger stopped cleanly; ready to resume at round {event.get('next_round')}."
                    )
                elif event.get("reliability_gate_blocked"):
                    self.teacher_student_status.setText(
                        f"PPO finish rate {event.get('finish_rate', 0):.0%}; "
                        "run DAgger until it reliably finishes before PPO fine-tuning."
                    )

    def _teacher_student_error(self) -> None:
        if self.teacher_student_process is None:
            return
        data = bytes(self.teacher_student_process.readAllStandardError()).decode(
            "utf-8", errors="replace"
        )
        if data.strip():
            self.teacher_student_log.append(data.strip())

    def _teacher_student_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._teacher_student_output()
        self._teacher_student_error()
        if exit_code == 0:
            self.teacher_student_status.setText("Teacher-student stage completed; see its saved report above.")
        else:
            self.teacher_student_status.setText(f"Teacher-student stage failed (exit {exit_code}).")
        self.teacher_student_process = None
        self.teacher_student_stop_file = None
        self.distillation_command = None
        self.distillation_stdout_buffer = ""

    @staticmethod
    def _distillation_summary(command: str, result: dict[str, Any]) -> str:
        """Keep the advanced tab useful by surfacing bake metrics at a glance."""
        if command == "full":
            collection = result.get("collection", {})
            training = result.get("training", {})
            validation = result.get("validation", {})
            metrics = training.get("action_metrics", {})
            teacher = validation.get("teacher", {})
            student = validation.get("student", {})
            delta = validation.get("lap_delta_s")
            return (
                f"Full workflow {'passes; staged for bake' if validation.get('accepted') else 'staged; gate failed'} · "
                f"teacher {teacher.get('median_lap_s')}s, student {student.get('median_lap_s')}s "
                f"(Δ {delta}s) · {collection.get('samples', 0):,} samples · "
                f"val MSE {training.get('best_validation_loss')} · "
                f"max action error {metrics.get('max_action_error')} · "
                f"overlays baked {training.get('bakeable_overlay_count', 0)}, "
                f"retained {training.get('retained_overlay_count', 0)}."
            )
        if command == "snapshot":
            teacher = result.get("teacher", {})
            return (
                f"Teacher snapshotted at {teacher.get('champion_lap_s')}s · "
                f"bakeable overlays {len(teacher.get('bakeable_overlays', []))} · "
                f"retained {len(teacher.get('retained_overlays', []))}."
            )
        if command == "collect":
            return (
                f"Teacher data collected · {result.get('episodes', 0)} laps · "
                f"{result.get('samples', 0):,} samples · "
                f"{result.get('overlay_modified_samples', 0):,} overlay-modified."
            )
        if command == "train":
            metrics = result.get("action_metrics", {})
            return (
                f"Actor student trained · val MSE {result.get('best_validation_loss')} · "
                f"max action error {metrics.get('max_action_error')} · "
                f"overlays baked {result.get('bakeable_overlay_count', 0)}, "
                f"retained {result.get('retained_overlay_count', 0)}."
            )
        return f"Distillation {command} finished successfully."

    def _start(self, resume: bool, *, best: bool = False, pace_polish: bool = False) -> None:
        if self.distillation_process is not None and self.distillation_process.state() != QProcess.NotRunning:
            self._error("Wait for the current distillation command to finish before training.")
            return
        if self.section_optimizer_process is not None and self.section_optimizer_process.state() != QProcess.NotRunning:
            self._error("Section optimizer is using the simulator. Stop it before starting training.")
            return
        if self.wr_search_process is not None and self.wr_search_process.state() != QProcess.NotRunning:
            self._error("WR pace search is using the simulator. Stop it before starting training.")
            return
        if self.speed_search_process is not None and self.speed_search_process.state() != QProcess.NotRunning:
            self._error("Speed search is using the simulator. Stop it before starting gradient training.")
            return
        if self.worker is not None and self.worker.is_alive():
            self._error("Training is already running.")
            return
        try:
            cfg = self.configuration()
            if self._external_speed_search_running(cfg):
                raise RuntimeError("A live speed search is already using the simulator; watch it in Status")
            registry = ModelRegistry(cfg.output_root)
            slot = (
                registry.slot(cfg.track_name, cfg.algorithm, "champion") if pace_polish
                else self._best_resume_slot(cfg) if best
                else registry.slot(cfg.track_name, cfg.algorithm, "latest")
            )
            if cfg.algorithm == "grtqc" and not resume:
                slot = registry.algorithm_dir(cfg.track_name, "grtqc") / "initialization"
                resume = True
            warnings = configuration_warnings(cfg)
            self.warnings.setText("\n".join(warnings) if warnings else "Settings look reasonable.")
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            fresh_replay = (
                resume and slot.name == "champion" and cfg.algorithm in {"grtqc", "tqc"}
                and (
                    not (slot / "replay.pkl").is_file()
                    or registry.read_metadata(slot).training_config["rewards"]
                    != cfg.to_dict()["rewards"]
                    or (cfg.algorithm in {"tqc", "grtqc"}
                        and registry.read_metadata(slot).reward_semantics != REWARD_SEMANTICS)
                )
            )
            self.runner = TrainingRunner(cfg, self.bridge.event.emit)
            self.worker = threading.Thread(
                target=self._run_worker,
                args=(slot if resume else None, fresh_replay, best or pace_polish, pace_polish),
                daemon=True,
            )
            self.worker.start()
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            self._error(str(exc))

    def _run_worker(
        self, resume: Path | None, fresh_replay: bool = False,
        rollback_to_champion: bool = False,
        pace_polish: bool = False,
    ) -> None:
        try:
            assert self.runner is not None
            self.runner.run(
                resume=resume, fresh_replay=fresh_replay,
                rollback_to_champion=rollback_to_champion,
                pace_polish=pace_polish,
            )
        except Exception as exc:
            self.bridge.failed.emit(f"Training failed: {exc}")

    def _stop(self) -> None:
        if self.wr_search_process is not None and self.wr_search_process.state() != QProcess.NotRunning:
            self._stop_wr_search()
            return
        if self.speed_search_process is not None and self.speed_search_process.state() != QProcess.NotRunning:
            assert self.speed_search_stop_file is not None
            self.speed_search_stop_file.write_text("stop\n", encoding="utf-8")
            self.log.append("Stopping speed search after the current candidate; champion remains saved.")
            return
        if self.runner is not None:
            self.runner.stop()
            self.log.append("Stopping after the current simulator step; latest will be saved.")

    def _start_wr_search(self, *, analyze_only: bool = False) -> None:
        if self.distillation_process is not None and self.distillation_process.state() != QProcess.NotRunning:
            self._error("Wait for distillation to finish before starting WR search.")
            return
        if self.worker is not None and self.worker.is_alive():
            self._error("Stop gradient training before starting WR pace search.")
            return
        if self.wr_search_process is not None and self.wr_search_process.state() != QProcess.NotRunning:
            self._error("WR pace search is already running.")
            return
        if self.speed_search_process is not None and self.speed_search_process.state() != QProcess.NotRunning:
            self._error("Stop the other live speed search before starting WR pace search.")
            return
        if self.section_optimizer_process is not None and self.section_optimizer_process.state() != QProcess.NotRunning:
            self._error("Stop the section optimizer before starting WR pace search.")
            return
        try:
            cfg = self.configuration()
            if cfg.algorithm != "tqc" or cfg.backend != "websocket":
                raise ValueError("WR pace optimization requires a live websocket TQC champion")
            if self.wr_region_enabled.isChecked() and self.wr_region_start.value() >= self.wr_region_end.value():
                raise ValueError("Selected progress region must have start < end")
            champion = ModelRegistry(cfg.output_root).slot(cfg.track_name, "tqc", "champion")
            if not (champion / "metadata.json").is_file():
                raise FileNotFoundError("No evaluated TQC champion is saved for this track")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            cfg.log_root.mkdir(parents=True, exist_ok=True)
            config_path = cfg.log_root / f"wr-pace-config-{stamp}.json"
            stop_path = cfg.log_root / f"wr-pace-stop-{stamp}.txt"
            config_path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
            arguments = [
                "-m", "polybot.training.wr_search", "--config", str(config_path),
                "--target", str(self.wr_target.value()),
                "--resolution", str(self.wr_resolution.currentData()),
                "--family", str(self.wr_family.currentData()),
                "--trials", "0" if analyze_only else str(self.wr_trials.value()),
                "--stop-file", str(stop_path),
                "--micro-gain", str(self.wr_micro_gain.value()),
                "--micro-confirm", str(self.wr_micro_confirm.value()),
            ]
            if self.wr_region_enabled.isChecked():
                arguments.extend(("--region-start", str(self.wr_region_start.value()),
                                  "--region-end", str(self.wr_region_end.value())))
            process = QProcess(self)
            process.setProgram(sys.executable)
            process.setArguments(arguments)
            process.setWorkingDirectory(str(Path.cwd()))
            process.readyReadStandardOutput.connect(self._wr_search_output)
            process.readyReadStandardError.connect(self._wr_search_error)
            process.finished.connect(self._wr_search_finished)
            self.wr_search_process = process
            self.wr_search_stop_file = stop_path
            self.wr_stdout_buffer = ""
            self.wr_summary.setText("Starting 10-lap deterministic champion baseline…")
            process.start()
            if not process.waitForStarted(3000):
                raise RuntimeError(process.errorString())
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            self.log_location.setText(f"WR search record: {champion.parent / 'wr-search-history.jsonl'}")
        except (ValueError, OSError, RuntimeError, FileNotFoundError) as exc:
            self._error(str(exc))

    def _start_section_optimizer(self, hours: int | None) -> None:
        if self.distillation_process is not None and self.distillation_process.state() != QProcess.NotRunning:
            self._error("Wait for distillation to finish before starting the section optimizer.")
            return
        if self.worker is not None and self.worker.is_alive():
            self._error("Stop gradient training before starting the section optimizer.")
            return
        if any(process is not None and process.state() != QProcess.NotRunning for process in
               (self.section_optimizer_process, self.wr_search_process, self.speed_search_process,
                self.adaptation_process, self.distillation_process)):
            self._error("Another live training or search process is using the simulator.")
            return
        try:
            cfg = self.configuration()
            if cfg.algorithm != "tqc" or cfg.backend != "websocket":
                raise ValueError("Section optimization needs a live websocket TQC champion")
            champion = ModelRegistry(cfg.output_root).slot(cfg.track_name, "tqc", "champion")
            if not (champion / "metadata.json").is_file():
                raise FileNotFoundError("No evaluated TQC champion is saved for this track")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            cfg.log_root.mkdir(parents=True, exist_ok=True)
            config_path = cfg.log_root / f"section-optimizer-config-{stamp}.json"
            stop_path = cfg.log_root / f"section-optimizer-stop-{stamp}.txt"
            skip_path = cfg.log_root / f"section-optimizer-skip-{stamp}.txt"
            refine_path = cfg.log_root / f"section-optimizer-refine-{stamp}.txt"
            config_path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
            args = ["-m", "polybot.training.section_optimizer", "--config", str(config_path),
                    "--target", str(self.wr_target.value()), "--stop-file", str(stop_path),
                    "--skip-file", str(skip_path), "--refine-file", str(refine_path)]
            if hours is not None:
                args.extend(("--hours", str(hours)))
            process = QProcess(self)
            process.setProgram(sys.executable)
            process.setArguments(args)
            process.setWorkingDirectory(str(Path.cwd()))
            process.readyReadStandardOutput.connect(self._section_optimizer_output)
            process.readyReadStandardError.connect(self._section_optimizer_error)
            process.finished.connect(self._section_optimizer_finished)
            self.section_optimizer_process = process
            self.section_optimizer_stop_file = stop_path
            self.section_optimizer_skip_file = skip_path
            self.section_optimizer_refine_file = refine_path
            self.section_optimizer_stdout = ""
            self.section_optimizer_status.setText("Starting; measuring a 10-lap deterministic timing floor…")
            process.start()
            if not process.waitForStarted(3000):
                raise RuntimeError(process.errorString())
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            self.log_location.setText(f"Section search record: {champion.parent / 'section-search-history.jsonl'}")
        except (ValueError, OSError, RuntimeError, FileNotFoundError) as exc:
            self._error(str(exc))

    def _stop_section_optimizer(self) -> None:
        if self.section_optimizer_process is not None and self.section_optimizer_process.state() != QProcess.NotRunning:
            assert self.section_optimizer_stop_file is not None
            self.section_optimizer_stop_file.write_text("stop\n", encoding="utf-8")
            self.section_optimizer_status.setText("Stop requested; current atomic evaluation will finish safely.")

    def _skip_optimizer_section(self) -> None:
        if self.section_optimizer_process is not None and self.section_optimizer_process.state() != QProcess.NotRunning:
            assert self.section_optimizer_skip_file is not None
            self.section_optimizer_skip_file.write_text("skip\n", encoding="utf-8")

    def _force_optimizer_refine(self) -> None:
        if self.section_optimizer_process is not None and self.section_optimizer_process.state() != QProcess.NotRunning:
            assert self.section_optimizer_refine_file is not None
            self.section_optimizer_refine_file.write_text("refine\n", encoding="utf-8")

    def _section_optimizer_output(self) -> None:
        if self.section_optimizer_process is None:
            return
        self.section_optimizer_stdout += bytes(self.section_optimizer_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        while "\n" in self.section_optimizer_stdout:
            line, self.section_optimizer_stdout = self.section_optimizer_stdout.split("\n", 1)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind in {"optimizer_started", "optimizer_resumed", "baseline_measured", "section_started",
                        "trial", "champion_promoted", "refinement_started", "target_reached",
                        "optimizer_stopped", "sweep_completed"}:
                section = event.get("section", [])
                window = f" · {section[0]:.0%}–{section[1]:.0%}" if len(section) == 2 else ""
                trial = event.get("total_trials", "—")
                lap = event.get("champion_lap_s", event.get("median_lap_s"))
                lap_text = f"{lap:.3f}s" if isinstance(lap, (int, float)) else "—"
                self.section_optimizer_status.setText(
                    f"{kind.replace('_', ' ').title()}{window} · champion {lap_text} · trial {trial} · "
                    f"runtime {event.get('runtime_seconds', 0):.0f}s"
                )
                if kind in {"trial", "champion_promoted", "target_reached", "optimizer_stopped"}:
                    self.log.append(f"Section optimizer: {kind}{window} · champion {lap_text}")

    def _section_optimizer_error(self) -> None:
        if self.section_optimizer_process is not None:
            error = bytes(self.section_optimizer_process.readAllStandardError()).decode("utf-8", errors="replace")
            if error.strip():
                self.log.append(f"Section optimizer: {error[-1200:]}")

    def _section_optimizer_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._section_optimizer_output()
        self._section_optimizer_error()
        if exit_code:
            self.section_optimizer_status.setText(
                f"Section optimizer exited with code {exit_code}; checkpoint is saved."
            )
        self.section_optimizer_process = None

    def _stop_wr_search(self) -> None:
        if self.wr_search_process is not None and self.wr_search_process.state() != QProcess.NotRunning:
            assert self.wr_search_stop_file is not None
            self.wr_search_stop_file.write_text("stop\n", encoding="utf-8")
            self.log.append("WR search will stop after the current candidate; the champion remains protected.")

    def _wr_search_output(self) -> None:
        if self.wr_search_process is None:
            return
        self.wr_stdout_buffer += bytes(
            self.wr_search_process.readAllStandardOutput()
        ).decode("utf-8", errors="replace")
        while "\n" in self.wr_stdout_buffer:
            line, self.wr_stdout_buffer = self.wr_stdout_buffer.split("\n", 1)
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "baseline":
                lap = float(event["champion_lap_s"])
                target = float(event["target_lap_s"])
                self.wr_summary.setText(
                    f"Champion {lap:.3f}s · target {target:.3f}s · gap {lap-target:+.3f}s · "
                    f"10/10 baseline · measurement range {event['measurement_floor_s']:.4f}s · "
                    f"{len(event.get('airborne_regions', []))} airborne regions"
                )
                self._show_wr_sectors(event.get("sectors", []))
            elif kind == "trial":
                label = "accepted" if event.get("accepted") else "rejected"
                self.log.append(
                    f"WR trial {event.get('trial_id')}: {label} · "
                    f"screen {event.get('screen_lap_s')}s · {event.get('parameter')}"
                )
                if event.get("sector_deltas"):
                    self._show_wr_sectors(event["sector_deltas"], deltas=True)
            elif kind == "completed":
                self.wr_summary.setText(
                    f"Search done · champion {event['champion_lap_s']:.3f}s · "
                    f"target gap {event['gap_s']:+.3f}s · "
                    f"{event['accepted']} accepted / {event['trials']} trials"
                )
            elif kind == "stopped":
                self.log.append("WR pace search stopped safely.")

    def _show_wr_sectors(
        self, sectors: list[dict[str, Any]], *, deltas: bool = False,
    ) -> None:
        self.wr_sector_table.setSortingEnabled(False)
        self.wr_sector_table.setRowCount(len(sectors))
        for row, sector in enumerate(sectors):
            values = (
                f"{sector['start']:.0%}–{sector['end']:.0%}",
                f"{sector['delta_s']:+.3f}" if deltas else f"{sector['candidate_s']:.3f}",
                f"{sector['speed_delta_mps']:+.2f}" if deltas else f"{sector['speed_mps']:.2f}",
                ("gain" if sector["delta_s"] < 0 else "loss")
                if deltas else ("slow focus" if sector.get("focus") else "baseline"),
            )
            for column, value in enumerate(values):
                self.wr_sector_table.setItem(row, column, QTableWidgetItem(value))
        self.wr_sector_table.setSortingEnabled(True)

    def _load_wr_profile(self) -> None:
        try:
            profile = json.loads(
                Path("profiles/training/summer-1-wr-pace.json").read_text(encoding="utf-8")
            )
            self.wr_target.setValue(float(profile["target_lap_s"]))
            self.wr_trials.setValue(int(profile["candidate_limit"]))
            self.wr_resolution.setCurrentIndex(
                self.wr_resolution.findData(float(profile["default_sector_resolution"]))
            )
            self.wr_micro_gain.setValue(float(profile["micro_gain_threshold_s"]))
            self.wr_micro_confirm.setValue(int(profile["micro_gain_confirmation_episodes"]))
            self.log.append(
                f"Loaded WR target: {profile['target_label']} ({profile['target_lap_s']:.3f}s)."
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._error(f"Could not load WR pace profile: {exc}")

    def _wr_search_error(self) -> None:
        if self.wr_search_process is not None:
            error = bytes(self.wr_search_process.readAllStandardError()).decode(
                "utf-8", errors="replace"
            )
            if error.strip():
                self.log.append(f"WR search: {error[-1200:]}")

    def _wr_search_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._wr_search_output()
        self._wr_search_error()
        if exit_code != 0:
            self.wr_summary.setText(f"WR pace search failed (exit {exit_code}); see log for details.")
        self.wr_search_process = None

    def _start_speed_search(self) -> None:
        if self.distillation_process is not None and self.distillation_process.state() != QProcess.NotRunning:
            self._error("Wait for distillation to finish before starting speed search.")
            return
        if self.section_optimizer_process is not None and self.section_optimizer_process.state() != QProcess.NotRunning:
            self._error("Stop the section optimizer before starting speed search.")
            return
        if self.worker is not None and self.worker.is_alive():
            self._error("Stop gradient training before starting speed search.")
            return
        if self.speed_search_process is not None and self.speed_search_process.state() != QProcess.NotRunning:
            self._error("Speed search is already running.")
            return
        try:
            cfg = self.configuration()
            if self._external_speed_search_running(cfg):
                raise RuntimeError("A live speed search is already running; watch it in Status")
            if cfg.algorithm != "tqc" or cfg.backend != "websocket":
                raise ValueError("Speed search needs a TQC model connected to the live simulator")
            champion = ModelRegistry(cfg.output_root).slot(cfg.track_name, "tqc", "champion")
            if not (champion / "metadata.json").is_file():
                raise FileNotFoundError("No evaluated TQC champion is saved for this track")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            cfg.log_root.mkdir(parents=True, exist_ok=True)
            config_path = cfg.log_root / f"speed-search-config-{stamp}.json"
            log_path = cfg.log_root / f"{cfg.track_name.lower().replace(' ', '-')}-tqc-speed-search-{stamp}.jsonl"
            self.speed_search_stop_file = cfg.log_root / f"speed-search-stop-{stamp}.txt"
            config_path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
            process = QProcess(self)
            process.setProgram(sys.executable)
            process.setArguments([
                "-m", "polybot.training.speed_search", "--config", str(config_path),
                "--log", str(log_path), "--trials", str(self.speed_search_trials.value()),
                "--target", str(self.speed_search_target.value()),
                "--stop-file", str(self.speed_search_stop_file),
                "--mode", str(self.speed_search_mode.currentData()),
            ])
            process.setWorkingDirectory(str(Path.cwd()))
            process.finished.connect(self._speed_search_finished)
            self.speed_search_process = process
            self.speed_search_best = None
            self.speed_search_log_path = log_path
            self.speed_search_log_position = 0
            process.start()
            if not process.waitForStarted(3000):
                raise RuntimeError(process.errorString())
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            self.log_location.setText(f"Speed search detail: {log_path}")
        except (ValueError, OSError, RuntimeError, FileNotFoundError) as exc:
            self._error(str(exc))

    def _external_speed_search_running(self, cfg: TrainingConfig) -> bool:
        candidates = list(cfg.log_root.glob("*-tqc-speed-search-*.jsonl"))
        if not candidates:
            return False
        latest = max(candidates, key=lambda item: item.stat().st_mtime)
        if time.time() - latest.stat().st_mtime > 30:
            return False
        lines = latest.read_text(encoding="utf-8").splitlines()
        if not lines:
            return False
        try:
            return json.loads(lines[-1]).get("type") != "completed"
        except json.JSONDecodeError:
            return True

    def _show_speed_search_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "started" and "champion_lap_s" in event:
            self.speed_search_best = float(event["champion_lap_s"])
        elif kind == "champion" and "lap_s" in event:
            self.speed_search_best = float(event["lap_s"])
        trial = event.get("trial")
        if self.speed_search_best is not None:
            detail = f"trial {trial:,} · " if isinstance(trial, int) else ""
            self.metrics.setText(f"TQC speed search · {detail}best {self.speed_search_best:.3f} s")
        summary = format_event(event)
        if summary:
            self.log.append(summary)
        if kind == "champion" and event.get("critic_adaptation_required"):
            self.log.append(
                "Speed candidate promoted. Run Tuned champion adaptation before resuming TQC gradients."
            )

    def _poll_speed_search_log(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if self.speed_search_process is not None and self.speed_search_process.state() != QProcess.NotRunning:
            path = self.speed_search_log_path
            if path is None or not path.is_file():
                return
        else:
            candidates = list(Path("logs").glob("*-tqc-speed-search-*.jsonl"))
            if not candidates:
                return
            path = max(candidates, key=lambda item: item.stat().st_mtime)
        if path != self.speed_search_log_path:
            self.speed_search_log_path = path
            self.speed_search_log_position = 0
            self.log_location.setText(f"Speed search detail: {path}")
        with path.open(encoding="utf-8") as stream:
            stream.seek(self.speed_search_log_position)
            while line := stream.readline():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    break
                self._show_speed_search_event(event)
                self.speed_search_log_position = stream.tell()

    def _speed_search_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._poll_speed_search_log()
        if self.speed_search_process is not None and exit_code != 0:
            error = bytes(self.speed_search_process.readAllStandardError()).decode("utf-8", errors="replace")
            self.log.append(f"Speed search exited with code {exit_code}: {error[-1200:]}")
        self.speed_search_process = None

    def _model_command(self, command: str, slot: str) -> None:
        try:
            cfg = self.configuration()
            args = [sys.executable, "-m", "polybot", command, "--algorithm", cfg.algorithm,
                    "--track-name", cfg.track_name, "--slot", slot,
                    "--output-root", str(cfg.output_root)]
            if command == "drive":
                args.append("--realtime")
            subprocess.Popen(args, creationflags=subprocess.CREATE_NO_WINDOW)
            self.log.append(f"Started {command} for {slot} model.")
        except (OSError, ValueError) as exc:
            self._error(str(exc))

    def _event(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "started":
            try:
                cfg = self.configuration()
                slot = ModelRegistry(cfg.output_root).slot(cfg.track_name, cfg.algorithm, "champion")
                saved = ModelRegistry(cfg.output_root).read_metadata(slot).evaluation
                self._pace_champion_lap = saved.get("median_lap_s") if saved else None
            except (OSError, ValueError, KeyError):
                self._pace_champion_lap = None
            counts = event["parameters"]
            self.parameter_label.setText(
                f"Actor {counts['actor']:,} · critic {counts['critic']:,} · total {counts['total']:,}"
            )
            self.warnings.setText("\n".join(
                configuration_warnings(self.configuration(), event.get("gpu_name"))
            ) or "Settings look reasonable.")
            self.log_location.setText(f"Recent events · full JSONL detail: {event['log']}")
        if kind == "progress":
            stage = event.get("curriculum_stage", "full track")
            section = event.get("section_progress")
            section_text = f"section {section:.1%}" if section is not None else "full track"
            self.metrics.setText(
                f"Step {event['timesteps']:,} · {event['steps_per_second']:.1f} TPS · "
                f"{stage} · lap {event['progress']:.1%} · {section_text} · reward {event['reward']:.1f}"
            )
            self.metrics.setToolTip(
                "TPS counts environment decisions per wall second. Progress is this attempt; "
                "reward includes all active components. A single loss value does not prove driving quality."
            )
            for name, widget in self.metric_widgets.items():
                if name in event and event[name] is not None:
                    value = event[name]
                    widget.setText(f"{value:.3f}" if isinstance(value, float) else str(value))
            self.pace_status.setText(
                f"Pace polish · LR {event.get('learning_rate') or 0:.1e} · "
                f"drift {event.get('anchor_action_drift') or 0:.2e} · "
                f"replay {event.get('replay_size') or 0:,} · "
                f"rollbacks {event.get('rollback_count') or 0} · "
                f"air bonus {event.get('air_brake_bonus_per_s') or 0:g} · "
                f"speed windows {len(event.get('speed_bias_schedule') or [])}"
            )
        if kind == "evaluation":
            lap = event.get("median_lap_s")
            air = event.get("air_brake_time_s")
            target_gap = (
                f" · target gap {lap - self.speed_search_target.value():+.3f}s"
                if lap is not None else ""
            )
            champion_gap = (
                f" · versus champion {lap - self._pace_champion_lap:+.3f}s"
                if lap is not None and self._pace_champion_lap is not None else ""
            )
            self.pace_status.setText(
                f"Evaluation median {lap:.3f}s · best {event.get('best_lap_s'):.3f}s · "
                f"air-brake duty time {air:.2f}s{target_gap}{champion_gap}"
                if lap is not None and event.get("best_lap_s") is not None and air is not None
                else "Evaluation did not produce a complete lap"
            )
        summary = format_event(event)
        if summary:
            self.log.append(summary)

    def _error(self, message: str) -> None:
        self.log.append(message)
        QMessageBox.warning(self, "PolyBot", message)

    def closeEvent(self, event: Any) -> None:
        if self.speed_search_process is not None and self.speed_search_process.state() != QProcess.NotRunning:
            self._stop()
            event.ignore()
            return
        if self.worker is not None and self.worker.is_alive():
            self._stop()
            event.ignore()
        else:
            event.accept()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PolyBot v2 training GUI")
    parser.add_argument("--check", action="store_true", help="start and close for a GUI smoke check")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = PolyBotWindow()
    window.show()
    if args.check:
        QTimer.singleShot(150, app.quit)
    return app.exec()
