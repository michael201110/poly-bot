"""Beginner friendly v2 trainer with full advanced access."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal
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
from polybot.models.registry import ModelRegistry
from polybot.training.config import (
    CurriculumConfig,
    CurriculumPhaseConfig,
    DQNConfig,
    EvaluationConfig,
    PPOConfig,
    TQCConfig,
    TrainingConfig,
)
from polybot.training.parameters import (
    CURRICULUM_INFO,
    DQN_INFO,
    EVALUATION_INFO,
    GENERAL_INFO,
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
    elif isinstance(value, int) and not isinstance(value, bool):
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
            choices = ("tiny", "compact", "standard") if field.name == "architecture" else ()
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
        for name in ("start_ratio", "end_ratio", "start_s", "end_s"):
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
            form, "algorithm", "tqc", GENERAL_INFO["algorithm"], ("ppo", "dqn", "tqc")
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
        basics.setToolTip("Plain-language explanation of the terms used in PPO, DQN and TQC training.")
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
        self.dqn_form = ParameterForm(
            DQNConfig(), DQN_INFO,
            {"architecture", "learning_rate", "replay_capacity", "exploration_fraction"},
        )
        self.tqc_form = ParameterForm(
            TQCConfig(), TQC_INFO, {"architecture", "learning_rate", "train_frequency"}
        )
        self.algorithm_stack.addWidget(self.ppo_form)
        self.algorithm_stack.addWidget(self.dqn_form)
        self.algorithm_stack.addWidget(self.tqc_form)
        self.parameter_label = QLabel("Network and total parameter counts appear when training starts.")
        self.parameter_label.setToolTip(
            "DQN uses one Q-network; PPO and TQC report actor and critic counts. Larger networks train slower."
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
            phases = ", ".join(f"{phase.mode} {phase.steps:,}" for phase in plan.phases)
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
        self.evaluation_form = ParameterForm(EvaluationConfig(), EVALUATION_INFO)
        page.addWidget(self.evaluation_form)
        note = QLabel("A lucky finish during training never promotes a model by itself.")
        note.setToolTip("Champion ranking uses finish rate, progress, then completed lap time.")
        page.addWidget(note)

    def _models_tab(self) -> None:
        _, page = self._page("Models")
        for label, handler, description in (
            ("Guided new run", self._guided_new_run,
             "Choose a track, algorithm, goal, device and preset, then review exact values."),
            ("New training run", lambda: self._start(False),
             "Start fresh with the exact settings shown. Existing champion remains until evaluation improves it."),
            ("Resume latest", lambda: self._start(True),
             "Continue from latest policy; TQC requires its replay buffer."),
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
        for form in (self.ppo_form, self.dqn_form, self.tqc_form, self.curriculum_form):
            form.set_advanced(enabled)
        self.reward_scroll.setVisible(enabled)
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
        forms = {"ppo": self.ppo_form, "dqn": self.dqn_form, "tqc": self.tqc_form}
        self.algorithm_stack.setCurrentWidget(forms[algorithm])
        explanations = {
            "ppo": (
                "PPO: on-policy. It learns from fresh rollouts, then discards them. "
                "Steering uses discrete PWM pulses."
            ),
            "dqn": (
                "DQN: off-policy. Its Q-network estimates the value of nine native digital keyboard-style "
                "actions. It reuses replay and sometimes chooses a random action through epsilon-greedy exploration. "
                "It never uses PWM."
            ),
            "tqc": (
                "TQC: off-policy. It reuses replay and learns continuous steering and pedal demand "
                "with an actor and quantile critics."
            ),
        }
        self.algorithm_explanation.setText(explanations[algorithm])
        ppo_metrics = {"policy_loss", "value_loss", "entropy", "explained_variance", "kl", "clip_fraction"}
        shared_replay = {"replay_size", "updates"}
        dqn_metrics = {"loss", "exploration_rate"} | shared_replay
        tqc_metrics = {"entropy_coefficient", "actor_loss", "critic_loss"} | shared_replay
        algorithm_metrics = {"ppo": ppo_metrics, "dqn": dqn_metrics, "tqc": tqc_metrics}
        specific_metrics = ppo_metrics | dqn_metrics | tqc_metrics
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
        {"ppo": self.ppo_form, "dqn": self.dqn_form, "tqc": self.tqc_form}[
            self.algorithm.currentText()
        ].load(preset)

    def _current_algorithm_settings(self) -> PPOConfig | DQNConfig | TQCConfig:
        forms = {"ppo": self.ppo_form, "dqn": self.dqn_form, "tqc": self.tqc_form}
        types = {"ppo": PPOConfig, "dqn": DQNConfig, "tqc": TQCConfig}
        algorithm = self.algorithm.currentText()
        return types[algorithm](**forms[algorithm].values())

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
            "Q-value: DQN's estimate of future reward for one digital action.\n"
            "Q-network: DQN's network that predicts all nine action values.\n"
            "Critic: a network estimating how useful actions or states may be.\n"
            "Environment step: one driving decision. Physics tick: one fixed simulator update.\n"
            "Frame skip: ticks between decisions; larger is faster but reacts slower.\n"
            "Rollout: fresh PPO experiences, discarded after an update (on-policy).\n"
            "Replay buffer: reusable driving history for DQN and TQC (off-policy).\n"
            "Epsilon: DQN's chance of choosing a random action. Epsilon-greedy: choose randomly "
            "with that chance, otherwise choose the highest-Q action.\n"
            "Target network: a slowly refreshed DQN Q-network that steadies learning targets.\n"
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
        algorithm = _editor("tqc", GENERAL_INFO["algorithm"].description, ("ppo", "dqn", "tqc"))
        algorithm_layout.addRow("Algorithm", algorithm)
        explanation = QLabel(
            "PPO uses PWM and fresh rollouts. DQN uses nine native digital actions and replay. "
            "TQC uses continuous controls and replay."
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
                    "ppo": None, "dqn": None, "tqc": None,
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
        self.dqn_form.load(config.dqn or DQNConfig())
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

    def _start(self, resume: bool) -> None:
        if self.worker is not None and self.worker.is_alive():
            self._error("Training is already running.")
            return
        try:
            cfg = self.configuration()
            warnings = configuration_warnings(cfg)
            self.warnings.setText("\n".join(warnings) if warnings else "Settings look reasonable.")
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            slot = ModelRegistry(cfg.output_root).slot(cfg.track_name, cfg.algorithm, "latest")
            self.runner = TrainingRunner(cfg, self.bridge.event.emit)
            self.worker = threading.Thread(
                target=self._run_worker, args=(slot if resume else None,), daemon=True
            )
            self.worker.start()
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            self._error(str(exc))

    def _run_worker(self, resume: Path | None) -> None:
        try:
            assert self.runner is not None
            self.runner.run(resume=resume)
        except Exception as exc:
            self.bridge.failed.emit(f"Training failed: {exc}")

    def _stop(self) -> None:
        if self.runner is not None:
            self.runner.stop()
            self.log.append("Stopping after the current simulator step; latest will be saved.")

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
            counts = event["parameters"]
            if event["algorithm"] == "dqn":
                self.parameter_label.setText(
                    f"Q-network {counts['critic']:,} · total trainable {counts['total']:,}"
                )
            else:
                self.parameter_label.setText(
                    f"Actor {counts['actor']:,} · critic {counts['critic']:,} · total {counts['total']:,}"
                )
            self.warnings.setText("\n".join(
                configuration_warnings(self.configuration(), event.get("gpu_name"))
            ) or "Settings look reasonable.")
            self.log_location.setText(f"Recent events · full JSONL detail: {event['log']}")
        if kind == "progress":
            self.metrics.setText(
                f"Step {event['timesteps']:,} · {event['steps_per_second']:.1f} TPS · "
                f"attempt progress {event['progress']:.1%} · reward {event['reward']:.1f}"
            )
            self.metrics.setToolTip(
                "TPS counts environment decisions per wall second. Progress is this attempt; "
                "reward includes all active components. A single loss value does not prove driving quality."
            )
            for name, widget in self.metric_widgets.items():
                if name in event and event[name] is not None:
                    value = event[name]
                    widget.setText(f"{value:.3f}" if isinstance(value, float) else str(value))
        summary = format_event(event)
        if summary:
            self.log.append(summary)

    def _error(self, message: str) -> None:
        self.log.append(message)
        QMessageBox.warning(self, "PolyBot", message)

    def closeEvent(self, event: Any) -> None:
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
