"""Beginner friendly v2 trainer with full advanced access."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from math import floor, log10
from dataclasses import asdict, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QProcess, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from polybot.algorithms.registry import backend_for
from polybot.ai_overlay import AIOverlaySettings, AIOverlaySettingsStore, HUD_MODES
from polybot.environment.curriculum import build_plan
from polybot.environment.rewards import RewardConfig
from polybot.gui.events import format_event
from polybot.gui.presentation import ActionGrid, CompactDoubleSpinBox, ExpandableSection
from polybot.models.registry import REWARD_SEMANTICS, ModelRegistry
from polybot.replay_swarm import (
    MAX_REPLAY_GHOSTS,
    MAX_REPLAY_TOTAL_SAMPLES,
    ColorScale,
    ReplayPlaybackOptions,
    ReplaySelection,
    replay_camera_rank,
    SelectedReplayPayload,
    filter_replays,
    load_replay_episode,
    load_replay_index,
    parse_color_stops,
    resolve_replay_directories,
    select_replays,
    selection_report,
    swarm_colors,
    send_replay_swarm,
    send_replay_playback,
)
from polybot.tracks.registry import TrackDefinition, TrackRegistry, track_slug
from polybot.tracks.workspace import TrackWorkspace
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
from polybot.transport import WebSocketServerTransport


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
        widget = CompactDoubleSpinBox()
        widget.setRange(-1e12, 1e12)
        widget.setDecimals(8)
        widget.setSingleStep(10 ** floor(log10(abs(value))) if 0 < abs(value) < 0.01 else 0.001)
        widget.setValue(value)
    else:
        widget = QLineEdit("" if value is None else str(value))
    widget.setToolTip(help_text)
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setKeyboardTracking(False)
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
            if isinstance(widget, CompactDoubleSpinBox) and isinstance(value, (int, float)) and 0 < abs(value) < 0.01:
                widget.setSingleStep(10 ** (floor(log10(abs(value))) - 1))
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
            widget.setObjectName(field.name)
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
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.updateGeometry()

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


class ReplaySwarmTask(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, operation: Any) -> None:
        super().__init__()
        self.operation = operation

    def run(self) -> None:
        try:
            self.completed.emit(self.operation())
        except Exception as exc:
            self.failed.emit(str(exc))


class PolyBotWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PolyBot Training")
        self.resize(1160, 820)
        self.setMinimumSize(940, 640)
        self.workflow_sections: list[ExpandableSection] = []
        self._last_activity = ""
        self._stop_requested = False
        self._session_start_step: int | None = None
        self._close_when_idle = False
        self.runner: TrainingRunner | None = None
        self.worker: threading.Thread | None = None
        self.replay_swarm_worker: ReplaySwarmTask | None = None
        self.speed_search_process: QProcess | None = None
        self.model_command_processes: set[QProcess] = set()
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
        self.track_registry = TrackRegistry()
        self.track_registry.list_tracks()
        self.presets = PresetStore()
        self.ai_overlay_store = AIOverlaySettingsStore()
        self._ai_overlay_settings_error: str | None = None
        try:
            self.ai_overlay_settings = self.ai_overlay_store.load()
        except ValueError as exc:
            self.ai_overlay_settings = AIOverlaySettings()
            self._ai_overlay_settings_error = str(exc)
        self._base_rewards = self.profiles.load("Balanced")
        self._session_reward_profiles: dict[str, RewardConfig] = {}
        self._last_algorithm_preset: dict[str, str] = {}
        self._reward_values = asdict(self._base_rewards)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 12)
        root.setSpacing(12)
        track_row = QHBoxLayout()
        brand = QLabel("PolyBot")
        brand.setObjectName("brand")
        track_row.addWidget(brand)
        track_row.addSpacing(20)
        track_row.addWidget(QLabel("Track"))
        self.track_selector = QComboBox()
        self._populate_track_selector(self._saved_track_slug())
        track_row.addWidget(self.track_selector, 1)
        track_row.addWidget(QLabel("Algorithm"))
        self.header_algorithm = QComboBox()
        self.header_algorithm.addItems(("grtqc", "tqc", "ppo"))
        self.header_algorithm.setMinimumWidth(95)
        self.header_algorithm.setToolTip("Choose which algorithm's models to train, evaluate, and drive.")
        track_row.addWidget(self.header_algorithm)
        self.track_context = QLabel()
        self.track_context.hide()
        add_track = QPushButton("+ Add Track")
        add_track.clicked.connect(self._add_track_dialog)
        track_row.addWidget(add_track)
        manage_tracks = QPushButton("Manage")
        manage_tracks.clicked.connect(self._manage_tracks_dialog)
        track_row.addWidget(manage_tracks)
        root.addLayout(track_row)
        self.run_summary = QLabel()
        self.run_summary.setWordWrap(True)
        self.run_summary.setStyleSheet("font-weight: 600; color: palette(highlight);")
        root.addWidget(self.run_summary)
        self.feedback_bar = QFrame()
        self.feedback_bar.setObjectName("feedbackBar")
        feedback = QHBoxLayout(self.feedback_bar)
        self.feedback_label = QLabel()
        self.feedback_label.setWordWrap(True)
        self.feedback_label.setTextFormat(Qt.TextFormat.PlainText)
        self.feedback_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        feedback.addWidget(self.feedback_label, 1)
        dismiss = QPushButton("Dismiss")
        self.dismiss_feedback_button = dismiss
        dismiss.clicked.connect(self._dismiss_feedback)
        feedback.addWidget(dismiss)
        root.addWidget(self.feedback_bar)
        self.feedback_bar.hide()
        self.advanced = QCheckBox("Show advanced settings on all tabs")
        self.advanced.setToolTip(
            "Reveal secondary training fields and advanced workflows. Turn this off to return "
            "to the recommended basic controls."
        )
        self.advanced.toggled.connect(self._toggle_advanced)
        body = QHBoxLayout()
        body.setSpacing(16)
        root.addLayout(body, 1)
        sidebar = QWidget()
        sidebar.setFixedWidth(180)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar_layout.addWidget(self.navigation, 1)
        self.advanced.setText("Advanced settings")
        sidebar_layout.addWidget(self.advanced)
        for label, handler in (("Load configuration…", self._load_config_dialog),
                               ("Save configuration…", self._save_config)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            sidebar_layout.addWidget(button)
        body.addWidget(sidebar)
        content = QVBoxLayout()
        content.setSpacing(8)
        body.addLayout(content, 1)
        self.setting_search = QLineEdit()
        self.setting_search.setPlaceholderText("Find a setting or action…  Ctrl+K")
        self.setting_search.setClearButtonEnabled(True)
        self.setting_search.setToolTip("Search all pages, including hidden advanced settings.")
        content.addWidget(self.setting_search)
        self.search_results = QListWidget()
        self.search_results.setMaximumHeight(210)
        self.search_results.hide()
        content.addWidget(self.search_results)
        self.page_description = QLabel()
        self.page_description.setWordWrap(True)
        content.addWidget(self.page_description)
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.tabBar().hide()
        content.addWidget(self.tabs, 1)

        self._overview_tab()
        self._general_tab()
        self._algorithm_tab()
        self._reward_tab()
        self._curriculum_tab()
        self._evaluation_tab()
        self._models_tab()
        self._replay_swarm_tab()
        self._ai_overlay_tab()
        self._status_tab()
        self._finish_workspace(root)
        self.algorithm.currentTextChanged.connect(self._algorithm_changed)
        self.header_algorithm.currentTextChanged.connect(self.algorithm.setCurrentText)
        self._algorithm_changed(self.algorithm.currentText())
        self._toggle_advanced(False)
        self.track_selector.currentIndexChanged.connect(self._track_changed)
        self.algorithm.currentTextChanged.connect(self._refresh_run_summary)
        self.general["backend"].currentTextChanged.connect(self._refresh_run_summary)
        self.general["timesteps"].valueChanged.connect(self._refresh_run_summary)
        self.general["visual_replay_enabled"].currentTextChanged.connect(self._refresh_run_summary)
        self.reward_profile.currentTextChanged.connect(self._refresh_run_summary)
        self._track_changed()
        self._refresh_run_summary()
        for name in ("output_root", "log_root"):
            widget = self.general[name]
            if isinstance(widget, QLineEdit):
                widget.editingFinished.connect(self._workspace_roots_changed)
        self.speed_search_watch = QTimer(self)
        self.speed_search_watch.timeout.connect(self._poll_speed_search_log)
        self.speed_search_watch.start(2000)
        self.activity_watch = QTimer(self)
        self.activity_watch.timeout.connect(self._refresh_activity)
        self.activity_watch.start(500)
        self._refresh_activity()

    def _page(self, name: str, *, scrollable: bool = True) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        if scrollable:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            content = QWidget()
            layout = QVBoxLayout(content)
            layout.setContentsMargins(18, 16, 18, 16)
            layout.setSpacing(12)
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            scroll.setWidget(content)
            page_layout.addWidget(scroll)
        else:
            layout = QVBoxLayout()
            page_layout.addLayout(layout)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tabs.addTab(page, name)
        return page, layout

    def _overview_tab(self) -> None:
        self.overview_page, page = self._page("Overview")
        heading = QLabel("What would you like to do?")
        heading.setObjectName("pageHeading")
        page.addWidget(heading)
        cards = QHBoxLayout()
        page.addLayout(cards)
        self.checkpoint_cards: dict[str, QLabel] = {}
        for slot, title in (("champion", "Champion · best evaluated"),
                            ("latest", "Latest · most recent training")):
            card = QGroupBox(title)
            layout = QVBoxLayout(card)
            value = QLabel("No checkpoint saved yet")
            value.setWordWrap(True)
            value.setTextFormat(Qt.TextFormat.PlainText)
            value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(value)
            self.checkpoint_cards[slot] = value
            cards.addWidget(card, 1)
        actions = QGroupBox("Use your model")
        grid = ActionGrid()
        actions.setLayout(grid)
        page.addWidget(actions)
        self.overview_model_buttons: list[QPushButton] = []
        for label, handler, hint in (
            ("Continue best model", lambda: self._start(True, best=True),
             "Restore the best compatible checkpoint and its saved training settings."),
            ("Play champion", lambda: self._model_command("drive", "champion"),
             "Watch the champion drive a live lap in PolyTrack."),
            ("Evaluate champion", lambda: self._model_command("evaluate", "champion"),
             "Run a deterministic evaluation and save the recorded attempts."),
        ):
            button = QPushButton(label)
            button.setToolTip(hint)
            button.clicked.connect(handler)
            grid.addWidget(button)
            self.overview_model_buttons.append(button)
        self.overview_model_buttons[0].setObjectName("primaryAction")
        explore = QGroupBox("Set up and explore")
        grid = ActionGrid()
        explore.setLayout(grid)
        page.addWidget(explore)
        for label, handler in (
            ("Guided setup", self._guided_new_run),
            ("Configure a new run", lambda: self.tabs.setCurrentWidget(self.general_page)),
            ("Watch saved replays", lambda: self.tabs.setCurrentWidget(self.replay_swarm_page)),
        ):
            button = QPushButton(label)
            button.clicked.connect(handler)
            grid.addWidget(button)
        self.overview_setup = QLabel()
        self.overview_setup.setWordWrap(True)
        page.addWidget(self.overview_setup)
        self.overview_activity = QLabel("Ready when you are.")
        self.overview_activity.setWordWrap(True)
        page.addWidget(self.overview_activity)

    def _finish_workspace(self, root: QVBoxLayout) -> None:
        self._page_hints = {
            "Overview": "Your selected track, saved models, and next steps.",
            "General": "Run setup · Choose the algorithm, simulator, device, and training budget.",
            "Algorithm": "Learning settings · Presets provide a starting point; every value stays editable.",
            "Rewards": "Rewards · Decide what the agent learns to value.",
            "Curriculum": "Curriculum · Choose full laps or a progression of training sections.",
            "Evaluation": "Evaluation · Deterministic full laps decide whether a candidate becomes champion.",
            "Models": "Models · Train, evaluate, and drive saved checkpoints. Expand advanced workflows below.",
            "Replay": "Replays · Watch one saved attempt or compare a group in a swarm.",
            "AI HUD": "In-game display · Choose which controls, observations, and rewards to show.",
            "Status": "Activity · Follow training metrics, evaluations, and detailed event messages.",
        }
        self._navigation_names = {
            "General": "Run setup", "Algorithm": "Learning settings",
            "AI HUD": "In-game display", "Status": "Activity & logs",
        }
        for index in range(self.tabs.count()):
            title = self.tabs.tabText(index)
            item = QListWidgetItem(self._navigation_names.get(title, title))
            item.setToolTip(self._page_hints.get(title, title))
            item.setSizeHint(QSize(166, 36))
            self.navigation.addItem(item)
        self.navigation.currentRowChanged.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self._page_changed)
        self.navigation.setCurrentRow(0)
        self._page_changed(0)
        self.setting_search.textChanged.connect(self._search_settings)
        self.setting_search.returnPressed.connect(self._open_search_result)
        self.search_results.itemActivated.connect(lambda _item: self._open_search_result())
        shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        shortcut.activated.connect(self.setting_search.setFocus)
        escape = QShortcut(QKeySequence("Escape"), self.setting_search)
        escape.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        escape.activated.connect(self.setting_search.clear)
        down = QShortcut(QKeySequence("Down"), self.setting_search)
        down.setContext(Qt.ShortcutContext.WidgetShortcut)
        down.activated.connect(lambda: self.search_results.setFocus() if self.search_results.isVisible() else None)
        for sequence, handler in (("Ctrl+O", self._load_config_dialog),
                                   ("Ctrl+Shift+S", self._save_config)):
            QShortcut(QKeySequence(sequence), self).activated.connect(handler)
        footer = QHBoxLayout()
        self.activity_label = QLabel("Ready")
        self.activity_label.setWordWrap(True)
        footer.addWidget(self.activity_label, 1)
        self.session_progress = QProgressBar()
        self.session_progress.setFixedWidth(170)
        self.session_progress.setRange(0, 1000)
        self.session_progress.setFormat("Session %p%")
        self.session_progress.hide()
        footer.addWidget(self.session_progress)
        details = QPushButton("View activity")
        details.clicked.connect(lambda: self.tabs.setCurrentIndex(self.tabs.count() - 1))
        footer.addWidget(details)
        footer.addWidget(self.stop_button)
        root.addLayout(footer)
        dark = self.palette().window().color().lightness() < 128
        border, surface, accent, muted = (
            ("#424750", "#282d35", "#65b8ff", "#aab4c2") if dark else
            ("#d7dee7", "#f2f6fb", "#1769aa", "#596a7e")
        )
        self.setStyleSheet(f"""
            QWidget {{ font-size: 13px; }}
            QLabel#brand {{ font-size: 23px; font-weight: 700; }}
            QLabel#pageHeading {{ font-size: 22px; font-weight: 600; }}
            QTabWidget::pane {{ border: 1px solid {border}; border-radius: 8px; }}
            QListWidget#navigation {{ border: none; background: transparent; outline: 0; }}
            QListWidget#navigation::item {{ padding: 6px 8px; margin: 1px 0; border-radius: 6px; }}
            QListWidget#navigation::item:selected {{ background: {surface}; color: {accent}; font-weight: 600; }}
            QPushButton {{ padding: 7px 10px; min-height: 20px; }}
            QPushButton#primaryAction {{ font-weight: 600; border: 1px solid {accent}; border-radius: 5px; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ min-height: 26px; padding: 2px 5px; }}
            QGroupBox {{ border: 1px solid {border}; border-radius: 7px; margin-top: 14px; padding: 16px 12px 12px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 5px; font-weight: 600; }}
            QToolButton#sectionToggle {{ padding: 9px; background: {surface}; border: 1px solid {border}; border-radius: 5px; }}
            QToolTip {{ padding: 5px; }}
            QFrame#feedbackBar {{ border: 1px solid #b58035; border-radius: 6px; }}
        """)
        self.page_description.setStyleSheet(f"color: {muted}; padding: 4px 0;")

    def _page_changed(self, index: int) -> None:
        if index < 0:
            return
        self.navigation.setCurrentRow(index)
        key = self.tabs.tabText(index).split(" — ", 1)[0]
        self.page_description.setText(self._page_hints.get(key, key))

    def _search_settings(self, query: str) -> None:
        self.search_results.clear()
        words = query.casefold().split()
        self.search_results.setVisible(bool(words))
        if not words:
            return
        matches = []
        for index in range(self.tabs.count()):
            page = self.tabs.widget(index)
            title = self.tabs.tabText(index).split(" — ", 1)[0]
            # The inactive algorithm forms are not relevant to the chosen model.
            inactive = [form for form in (self.ppo_form, self.grtqc_form, self.tqc_form)
                        if form is not self.algorithm_stack.currentWidget()]
            candidates: list[tuple[str, QWidget]] = []
            for layout in page.findChildren(QFormLayout):
                for row in range(layout.rowCount()):
                    label_item = layout.itemAt(row, QFormLayout.ItemRole.LabelRole)
                    field_item = layout.itemAt(row, QFormLayout.ItemRole.FieldRole)
                    label = label_item.widget() if label_item else None
                    field = field_item.widget() if field_item else None
                    if isinstance(label, QLabel) and field is not None:
                        candidates.append((label.text(), field))
            candidates.extend((button.text().replace("&", ""), button)
                              for button in page.findChildren(QPushButton))
            for label, target in candidates:
                if any(form.isAncestorOf(target) for form in inactive):
                    continue
                haystack = f"{title} {label} {target.objectName().replace('_', ' ')} {target.toolTip()}".casefold()
                if not all(word in haystack for word in words):
                    continue
                score = sum(word in label.casefold() for word in words)
                matches.append((score, title, label, index, target))
        for _score, title, label, index, target in sorted(matches, key=lambda match: -match[0])[:60]:
            item = QListWidgetItem(f"{title}  ›  {label}")
            item.setToolTip(target.toolTip())
            item.setData(Qt.ItemDataRole.UserRole, (index, target))
            self.search_results.addItem(item)
        if self.search_results.count():
            self.search_results.setCurrentRow(0)
        else:
            item = QListWidgetItem("No matches. Try ‘learning rate’, ‘replay’, or ‘port’.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.search_results.addItem(item)

    def _open_search_result(self) -> None:
        item = self.search_results.currentItem()
        data = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not data:
            return
        index, target = data
        self.tabs.setCurrentIndex(index)
        if not target.isVisibleTo(self.tabs.widget(index)):
            self.advanced.setChecked(True)
        parent = target.parentWidget()
        scrolls = []
        while parent is not None and parent is not self:
            if isinstance(parent, ExpandableSection):
                parent.expand()
            if parent is self.replay_advanced_content:
                self.replay_advanced_toggle.setChecked(True)
            if isinstance(parent, QScrollArea):
                scrolls.append(parent)
            parent = parent.parentWidget()
        self.setting_search.clear()
        target.setFocus(Qt.FocusReason.ShortcutFocusReason)
        if isinstance(target, (QLineEdit, QSpinBox, QDoubleSpinBox)):
            target.selectAll()
        QTimer.singleShot(0, lambda: [scroll.ensureWidgetVisible(target, 20, 50) for scroll in scrolls])

    def _active_operations(self) -> list[str]:
        active = []
        if self.worker is not None and self.worker.is_alive():
            active.append("Training")
        for attribute, label in (
            ("speed_search_process", "Speed search"), ("wr_search_process", "WR pace search"),
            ("section_optimizer_process", "Section optimizer"), ("adaptation_process", "Champion adaptation"),
            ("distillation_process", "Policy distillation"), ("teacher_student_process", "Teacher-student transfer"),
        ):
            process = getattr(self, attribute)
            if process is not None and process.state() != QProcess.ProcessState.NotRunning:
                active.append(label)
        for process in self.model_command_processes:
            if process.state() != QProcess.ProcessState.NotRunning:
                verb = "Driving" if process.property("polybot_command") == "drive" else "Evaluating"
                active.append(f"{verb} {process.property('polybot_slot')}")
        if self.replay_swarm_worker is not None and self.replay_swarm_worker.isRunning():
            active.append("Replay request")
        return active

    def _require_idle(self) -> bool:
        active = self._active_operations()
        if active:
            self._error(f"{', '.join(active)} is active. Finish or stop it before starting another simulator task.")
            return False
        return True

    def _refresh_activity(self) -> None:
        active = self._active_operations()
        text = " · ".join(active)
        if text:
            message = f"Stopping safely · {text}" if self._stop_requested else f"Active · {text}"
            self.activity_label.setText(message)
            self.overview_activity.setText(message + ". Open Activity & logs for details.")
        else:
            self.activity_label.setText("Ready · no task running")
            self.overview_activity.setText("Ready when you are. Choose an action above.")
            self._stop_requested = False
        # Replay requests have a bounded connection timeout, rather than a stop API.
        stoppable = any(label in active for label in (
            "Training", "Speed search", "WR pace search", "Section optimizer",
        )) or bool(self.model_command_processes) or (
            "Teacher-student transfer" in active and self.teacher_student_stop_file is not None
        )
        self.stop_button.setEnabled(stoppable and not self._stop_requested)
        self.stop_button.setToolTip(
            "Stop training or search safely; cancel a live drive or evaluation."
            if stoppable else "No cancellable task. An active replay request or offline command will finish on its own."
        )
        for button in self.overview_model_buttons:
            button.setEnabled(not active)
        training = "Training" in active
        self.session_progress.setVisible(training)
        self.track_selector.setEnabled(not active)
        self.header_algorithm.setEnabled(not active)
        self.algorithm.setEnabled(not active)
        if text != self._last_activity:
            if not text and self._last_activity:
                self._refresh_models()
                self._refresh_replay_runs()
            self._last_activity = text
        if not active and self._close_when_idle:
            self.close()

    @staticmethod
    def _selected_track_path() -> Path:
        return Path("config") / "selected-track.json"

    def _saved_track_slug(self) -> str | None:
        path = self._selected_track_path()
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload["slug"]
            return value if isinstance(value, str) else None
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"cannot read selected track from {path}") from exc

    def _populate_track_selector(self, selected_slug: str | None = None) -> None:
        tracks = self.track_registry.list_tracks()
        previous = self.track_selector.blockSignals(True)
        self.track_selector.clear()
        for track in tracks:
            self.track_selector.addItem(track.name, track.slug)
        match = self.track_selector.findData(selected_slug) if selected_slug else -1
        if match < 0:
            match = self.track_selector.findData("summer-1")
        self.track_selector.setCurrentIndex(match if match >= 0 else 0)
        self.track_selector.blockSignals(previous)

    def _selected_track(self) -> TrackDefinition:
        selected_slug = self.track_selector.currentData()
        if not isinstance(selected_slug, str):
            raise ValueError("select a registered track")
        return self.track_registry.resolve(selected_slug)

    def _track_changed(self, _index: int = -1) -> None:
        track = self._selected_track()
        self.setWindowTitle(f"PolyBot Training — {track.name}")
        path = self._selected_track_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"slug": track.slug}, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        self.tabs.setTabText(self.tabs.indexOf(self.models_page), f"Models — {track.name}")
        self.tabs.setTabText(self.tabs.indexOf(self.replay_swarm_page), f"Replay — {track.name}")
        if hasattr(self, "navigation"):
            self.navigation.item(self.tabs.indexOf(self.models_page)).setToolTip(f"Models for {track.name}")
            self.navigation.item(self.tabs.indexOf(self.replay_swarm_page)).setToolTip(f"Replays for {track.name}")
        if (
            hasattr(self, "preset")
            and self.preset.currentText().startswith("Summer 1 -")
            and track.slug != "summer-1"
        ):
            self.preset.setCurrentText("Balanced")
        self._refresh_models()
        self._refresh_replay_runs()
        self.track_context.setText(f"Current workspace: {track.name} ({track.slug})")
        if hasattr(self, "teacher_student_teacher"):
            workspace = self._workspace(track)
            self.teacher_student_teacher.setText(
                str(workspace.algorithm_models("tqc") / "champion")
            )
            self.teacher_student_dataset.setText(
                str(Path("runs") / "teacher-student" / track.slug / "teacher.npz")
            )
        if hasattr(self, "run_summary"):
            self._refresh_run_summary()

    def _refresh_run_summary(self, *_args: Any) -> None:
        if not hasattr(self, "run_summary") or not hasattr(self, "general"):
            return
        track_name = self.track_selector.currentText() or "No track selected"
        algorithm = self.algorithm.currentText().upper()
        backend = self.general["backend"].currentText()
        steps = int(_value(self.general["timesteps"]))
        recording = self.general["visual_replay_enabled"].currentText()
        recording_text = {
            "automatic": "recording automatic for WebSocket",
            "enabled": "replay recording on",
            "disabled": "replay recording off",
        }.get(recording, f"replay recording {recording}")
        self.run_summary.setText(
            f"{track_name} · {algorithm} · {'Live simulator' if backend == 'websocket' else 'Local mock simulator'} · "
            f"{steps:,} decisions · {recording_text}"
        )

    def _workspace(self, track: TrackDefinition | None = None) -> TrackWorkspace:
        selected = track or self._selected_track()
        model_root = Path(str(_value(self.general.get("output_root")) or "models"))
        log_root = Path(str(_value(self.general.get("log_root")) or "logs"))
        return TrackWorkspace(selected, model_root, log_root)

    def _workspace_roots_changed(self) -> None:
        self._refresh_models()
        self._refresh_replay_runs()

    @staticmethod
    def _workspace_for_config(config: TrainingConfig) -> TrackWorkspace:
        track = TrackDefinition(config.track_name, config.track_slug, config.track_id)
        return TrackWorkspace(track, config.output_root, config.log_root)

    def _refresh_models(self) -> None:
        if not hasattr(self, "models_inventory"):
            return
        track = self._selected_track()
        self.models_heading.setText(f"Models — {track.name}")
        registry = ModelRegistry(self._workspace(track).models_root)
        _set(self.header_algorithm, self.algorithm.currentText())
        self.model_actions_group.setTitle(f"{self.algorithm.currentText().upper()} actions · {track.name}")
        lines: list[str] = []
        selected_slots = {}
        inventory_errors = {}
        self.models_table.setRowCount(0)
        for algorithm in ("grtqc", "tqc", "ppo"):
            try:
                slots = registry.list_model_slots(track, algorithm)
            except (OSError, ValueError, TypeError) as exc:
                inventory_errors[algorithm] = str(exc)
                lines.append(f"{algorithm.upper()}: could not read inventory: {exc}")
                continue
            if not self.show_archived_checkpoints.isChecked():
                slots = [slot for slot in slots if not slot[0].startswith("checkpoints/")]
            if not slots:
                continue
            lines.append(f"{algorithm.upper()}")
            for name, directory, metadata in slots:
                if algorithm == self.algorithm.currentText():
                    selected_slots[name] = (directory, metadata)
                lap_text, verification = self._checkpoint_result(metadata)
                if metadata is None:
                    details = "metadata missing"
                else:
                    details = f"{metadata.training_timesteps:,} training steps"
                    if metadata.evaluation:
                        evaluation = metadata.evaluation
                        details += (
                            f"; finish rate {evaluation.get('finish_rate', 0):.0%}"
                            if isinstance(evaluation.get("finish_rate"), (int, float))
                            else ""
                        )
                lines.append(f"  {name}: {details} ({directory})")
                row = self.models_table.rowCount()
                self.models_table.insertRow(row)
                for column, value in enumerate((algorithm.upper(), name.title(), lap_text, verification)):
                    item = QTableWidgetItem(value)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setToolTip(str(directory))
                    self.models_table.setItem(row, column, item)
        self.models_inventory.setPlainText(
            "\n".join(lines) if lines else f"No model slots saved for {track.name}."
        )
        self.models_table.resizeColumnsToContents()
        self.models_table.horizontalHeader().setStretchLastSection(True)
        for slot, label in self.checkpoint_cards.items():
            entry = selected_slots.get(slot)
            metadata = entry[1] if entry else None
            if entry:
                lap, verification = self._checkpoint_result(metadata)
                steps = f"{metadata.training_timesteps:,} training decisions" if metadata else "Metadata missing"
                label.setText(f"{self.algorithm.currentText().upper()} · {lap}\n{verification}\n{steps}")
                label.setToolTip(str(entry[0]))
            elif self.algorithm.currentText() in inventory_errors:
                label.setText("Checkpoint details could not be read. Open Models → Checkpoint details for the error.")
                label.setToolTip(inventory_errors[self.algorithm.currentText()])
            else:
                label.setText(f"No {self.algorithm.currentText().upper()} {slot} saved for {track.name}.")
                label.setToolTip("")
        has_champion = "champion" in selected_slots
        self.overview_setup.setText(
            "Open PolyTrack with the PolyBot bridge to drive or train. Saved models use their cached "
            "racing line; a ghost is needed when initializing a new model.\n"
            + ("Continue best restores the checkpoint's policy, curriculum, and reward settings. "
               "The session budget comes from Run setup." if has_champion else
               "No champion yet? Start with Guided setup, then train and evaluate your first model.")
        )

    @staticmethod
    def _checkpoint_result(metadata: Any) -> tuple[str, str]:
        evaluation = metadata.evaluation if metadata is not None else None
        if not evaluation:
            return "No evaluated lap", "Awaiting evaluation"
        lap = evaluation.get("median_lap_s")
        lap_text = f"{lap:.3f} s median" if isinstance(lap, (int, float)) else "No completed lap"
        episodes, rate = evaluation.get("episodes"), evaluation.get("finish_rate")
        if isinstance(episodes, int) and isinstance(rate, (int, float)):
            return lap_text, f"{round(episodes * rate)}/{episodes} laps finished"
        return lap_text, "Evaluation recorded"

    def _add_track_dialog(self) -> None:
        name, accepted = QInputDialog.getText(self, "Add Track", "Track name")
        if not accepted:
            return
        try:
            track = self.track_registry.add(name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Add Track", str(exc))
            return
        self._populate_track_selector(track.slug)
        self._track_changed()

    def _manage_tracks_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Tracks")
        layout = QVBoxLayout(dialog)
        listing = QListWidget()

        def refresh() -> None:
            listing.clear()
            for definition in self.track_registry.list_tracks():
                listing.addItem(f"{definition.name}  [{definition.slug}]")

        refresh()
        layout.addWidget(listing)
        actions = QHBoxLayout()
        rename = QPushButton("Rename")
        remove = QPushButton("Remove registration")
        actions.addWidget(rename)
        actions.addWidget(remove)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)

        def selected() -> TrackDefinition | None:
            row = listing.currentRow()
            tracks = self.track_registry.list_tracks()
            return tracks[row] if 0 <= row < len(tracks) else None

        def rename_selected() -> None:
            track = selected()
            if track is None:
                return
            name, accepted = QInputDialog.getText(
                dialog, "Rename Track", "Display name", text=track.name,
            )
            if accepted:
                try:
                    self.track_registry.rename(track.slug, name)
                    refresh()
                    self._populate_track_selector(track.slug)
                    self._track_changed()
                except (OSError, ValueError) as exc:
                    QMessageBox.warning(dialog, "Rename Track", str(exc))

        def remove_selected() -> None:
            track = selected()
            if track is None:
                return
            answer = QMessageBox.question(
                dialog, "Remove Track",
                f"Remove {track.name} from the registry? Workspace files will not be deleted.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            try:
                self.track_registry.remove(track.slug)
                refresh()
                self._populate_track_selector(
                    self.track_selector.currentData()
                    if self.track_selector.currentData() != track.slug else None
                )
                self._track_changed()
            except (OSError, ValueError) as exc:
                QMessageBox.warning(dialog, "Remove Track", str(exc))

        rename.clicked.connect(rename_selected)
        remove.clicked.connect(remove_selected)
        dialog.exec()

    def _add_field(
        self, layout: QFormLayout, name: str, value: Any,
        info: ParameterInfo, choices: tuple[str, ...] = (),
    ) -> QWidget:
        widget = _editor(value, info.description, choices)
        widget.setObjectName(name)
        label = QLabel(info.label)
        label.setToolTip(info.description)
        layout.addRow(label, widget)
        return widget

    def _general_tab(self) -> None:
        self.general_page, page = self._page("General")
        setup_note = QLabel(
            "For live training, open this track in PolyTrack with the PolyBot bridge and choose "
            "WebSocket. Existing models use their saved racing line. Load a ghost when initializing "
            "a new model. Mock mode runs locally without the game."
        )
        setup_note.setWordWrap(True)
        page.addWidget(setup_note)
        form = QFormLayout()
        page.addLayout(form)
        self.general: dict[str, QWidget] = {}
        values = {
            "backend": "websocket", "websocket_port": 8765,
            "device": "auto", "seed": 0, "frame_skip": 30, "timesteps": 100_000,
            "max_episode_seconds": 60.0, "max_episode_steps": 30_000,
            "lookahead_count": 12, "reward_scale": 0.01,
            "checkpoint_interval": 10_000,
            "output_root": "models", "log_root": "logs",
            "visual_replay_enabled": "automatic", "visual_replay_sample_hz": 20.0,
            "visual_replay_observations": True,
        }
        for name, value in values.items():
            choices = ("mock", "websocket") if name == "backend" else (
                ("auto", "cpu", "cuda") if name == "device" else ()
            )
            if name == "visual_replay_enabled":
                choices = ("automatic", "enabled", "disabled")
            self.general[name] = self._add_field(form, name, value, GENERAL_INFO[name], choices)
        self.algorithm = self._add_field(
            form, "algorithm", "grtqc", GENERAL_INFO["algorithm"], ("grtqc", "tqc", "ppo")
        )
        self.general["backend"].currentTextChanged.connect(self._backend_changed)
        self.general_advanced = {
            name for name in values if name not in {
                "backend", "device", "seed", "frame_skip",
                "timesteps", "max_episode_seconds",
            }
        }
        self.general_labels = {
            name: form.labelForField(widget) for name, widget in self.general.items()
        }
        for name, label in (
            ("backend", "Simulator connection"), ("device", "Compute device"),
            ("seed", "Random seed"), ("timesteps", "Training budget (decisions)"),
            ("max_episode_seconds", "Time limit per attempt (s)"),
            ("frame_skip", "Physics ticks per decision"),
        ):
            self.general_labels[name].setText(label)
        basics = QPushButton("What do these training words mean?")
        basics.setToolTip("Plain-language explanation of the terms used in GRTQC, TQC and PPO training.")
        basics.clicked.connect(self._show_glossary)
        page.addWidget(basics)
        actions = ActionGrid()
        page.addLayout(actions)
        for label, handler in (("Guided setup", self._guided_new_run),
                               ("Start a new run", lambda: self._start(False)),
                               ("Continue best model", lambda: self._start(True, best=True))):
            button = QPushButton(label)
            button.clicked.connect(handler)
            actions.addWidget(button)

    def _algorithm_tab(self) -> None:
        self.algorithm_page, page = self._page("Algorithm")
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
        for form in (self.ppo_form, self.grtqc_form, self.tqc_form):
            for widget in form.widgets.values():
                if isinstance(widget, QComboBox):
                    widget.currentTextChanged.connect(self._mark_algorithm_custom)
                elif isinstance(widget, QCheckBox):
                    widget.toggled.connect(self._mark_algorithm_custom)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.valueChanged.connect(self._mark_algorithm_custom)
                elif isinstance(widget, QLineEdit):
                    widget.textEdited.connect(self._mark_algorithm_custom)
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
            ("guidance_reward_scale", "Racing line guidance"),
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
        page.addWidget(ExpandableSection("Exact reward coefficients", self.reward_preview))
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
        self.models_page, page = self._page("Models")
        self.models_heading = QLabel()
        self.models_heading.setStyleSheet("font-weight: bold; font-size: 16px")
        page.addWidget(self.models_heading)
        inventory_options = QHBoxLayout()
        self.show_archived_checkpoints = QCheckBox("Show archived checkpoints")
        self.show_archived_checkpoints.setToolTip("Include saved and rejected training snapshots in the inventory.")
        self.show_archived_checkpoints.toggled.connect(self._refresh_models)
        inventory_options.addWidget(self.show_archived_checkpoints)
        inventory_options.addStretch()
        refresh_models = QPushButton("Refresh models")
        refresh_models.clicked.connect(self._refresh_models)
        inventory_options.addWidget(refresh_models)
        page.addLayout(inventory_options)
        self.models_table = QTableWidget(0, 4)
        self.models_table.setHorizontalHeaderLabels(("Algorithm", "Checkpoint", "Median lap", "Evaluation"))
        self.models_table.verticalHeader().hide()
        self.models_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.models_table.setMinimumHeight(160)
        self.models_table.setMaximumHeight(230)
        self.models_table.setToolTip("Champion is the best evaluated checkpoint. Latest may have regressed. Hover for its folder.")
        page.addWidget(self.models_table)
        self.models_inventory = QTextEdit()
        self.models_inventory.setReadOnly(True)
        self.models_inventory.setMaximumHeight(180)
        self.models_inventory.setPlaceholderText("No model slots have been saved for this track yet.")
        details = ExpandableSection("Checkpoint details and file locations", self.models_inventory)
        page.addWidget(details)
        self.model_actions_group = QGroupBox()
        model_actions = ActionGrid()
        self.model_actions_group.setLayout(model_actions)
        page.addWidget(self.model_actions_group)
        self.speed_search_section = QWidget()
        speed_layout = QVBoxLayout(self.speed_search_section)
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
        speed_layout.addLayout(search_form)
        speed_button = QPushButton("Optimize TQC champion speed")
        speed_button.setToolTip(
            "Test small actor-output changes on live laps. Save faster champions only after full confirmation."
        )
        speed_button.clicked.connect(self._start_speed_search)
        speed_layout.addWidget(speed_button)
        self._add_workflow(page, "TQC · Actor speed search", self.speed_search_section)
        self.adaptation_section = QWidget()
        adaptation_layout = QVBoxLayout(self.adaptation_section)
        adaptation_layout.addWidget(QLabel("Tuned champion adaptation (advanced)"))
        preset = QPushButton("Load tuned champion adaptation preset")
        preset.clicked.connect(self._load_adaptation_preset)
        adaptation_layout.addWidget(preset)
        adaptation_actions = ActionGrid(2)
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
        self._add_workflow(page, "TQC · Champion adaptation", self.adaptation_section)
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
        distill_actions = ActionGrid(2)
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
        self._add_workflow(page, "TQC · Policy distillation", self.distillation_section)
        self.teacher_student_section = QWidget()
        teacher_student_layout = QVBoxLayout(self.teacher_student_section)
        teacher_student_layout.addWidget(QLabel(
            "Train continuous PPO from the selected track's frozen TQC champion."
        ))
        teacher_student_form = QFormLayout()
        self.teacher_student_teacher = QLineEdit()
        self.teacher_student_dataset = QLineEdit()
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
        teacher_student_actions = ActionGrid(2)
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
        self._add_workflow(page, "PPO · Learn from a TQC teacher", self.teacher_student_section)
        self.wr_search_section = QWidget()
        wr_layout = QVBoxLayout(self.wr_search_section)
        wr_layout.addWidget(QLabel("WR Pace Optimizer · frozen TQC policy · live lap-time search"))
        wr_form = QFormLayout()
        self.wr_target = QDoubleSpinBox()
        self.wr_target.setRange(1.0, 600.0)
        self.wr_target.setDecimals(3)
        self.wr_target.setValue(22.262)
        self.wr_target.setToolTip("Target lap time in seconds. Tune this value for the selected track.")
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
        wr_actions = ActionGrid(2)
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
        section_actions = ActionGrid(2)
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
        self._add_workflow(page, "TQC · Pace and section optimization", self.wr_search_section)
        self.pace_polish_section = QWidget()
        polish_layout = QVBoxLayout(self.pace_polish_section)
        polish_layout.addWidget(QLabel("Pace polishing: conservative TQC gradients from champion"))
        polish_actions = QHBoxLayout()
        polish_layout.addLayout(polish_actions)
        for label, steps in (("Polish champion 25k", 25_000), ("Polish champion 50k", 50_000)):
            button = QPushButton(label)
            button.setToolTip("Load the saved Summer 1 safe-polish settings and resume its evaluated champion.")
            button.clicked.connect(lambda _checked=False, budget=steps: self._start_polish(budget))
            polish_actions.addWidget(button)
        search_actions = QHBoxLayout()
        polish_layout.addLayout(search_actions)
        for label, mode in (("Search global pace", "global"), ("Search section pace", "section")):
            button = QPushButton(label)
            button.setToolTip("Screen candidates, then confirm faster laps before saving champion.")
            button.clicked.connect(lambda _checked=False, selected=mode: self._start_pace_search(selected))
            search_actions.addWidget(button)
        self._add_workflow(page, "TQC · Conservative pace polishing", self.pace_polish_section)
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
            ("Cache initialization ghost", lambda: self._model_command("evaluate", "champion", bootstrap=True),
             "Verify the champion for five laps and save the loaded initialization ghost for future sessions."),
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
            model_actions.addWidget(button)
        self.stop_button = QPushButton("Stop cleanly")
        self.stop_button.setToolTip("Ask training to stop after this step and save latest state.")
        self.stop_button.clicked.connect(self._stop)

    def _add_workflow(self, layout: QVBoxLayout, title: str, content: QWidget) -> None:
        for label in content.findChildren(QLabel):
            label.setWordWrap(True)
        section = ExpandableSection(title, content)
        self.workflow_sections.append(section)
        layout.addWidget(section)

    def _replay_swarm_tab(self) -> None:
        self.replay_swarm_page, tab_layout = self._page("Replay", scrollable=False)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        page = QVBoxLayout(content)
        page.setContentsMargins(18, 16, 18, 16)
        page.setSpacing(10)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        page.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        intro = QLabel(
            "Replays are saved automatically during live training. Pick a run and watch one attempt, "
            "or tick several attempts and compare them together."
        )
        intro.setWordWrap(True)
        page.addWidget(intro)
        self.replay_recording = QComboBox()
        self.replay_recording.addItems(("Automatic for live training", "Always record", "Do not record"))
        self.replay_recording.setToolTip("Automatic records PolyTrack/WebSocket runs and skips mock runs.")
        self.replay_recording.currentIndexChanged.connect(self._replay_recording_changed)
        self.general["visual_replay_enabled"].currentTextChanged.connect(
            self._general_replay_recording_changed
        )
        recording_row = QHBoxLayout()
        recording_row.addWidget(QLabel("Save replays"))
        recording_row.addWidget(self.replay_recording)
        page.addLayout(recording_row)

        run_form = QFormLayout()
        self.replay_swarm_algorithm = QComboBox()
        self.replay_swarm_algorithm.addItems(("grtqc", "tqc", "ppo"))
        self.replay_swarm_algorithm.setCurrentText("grtqc")
        self.replay_swarm_algorithm.currentTextChanged.connect(self._refresh_replay_runs)
        run_form.addRow("Algorithm", self.replay_swarm_algorithm)
        self.replay_swarm_run = QComboBox()
        self.replay_swarm_run.currentIndexChanged.connect(self._replay_swarm_run_changed)
        run_form.addRow("Saved run", self.replay_swarm_run)
        refresh_runs = QPushButton("Refresh runs")
        refresh_runs.setToolTip("Find replays saved since this window opened.")
        refresh_runs.clicked.connect(self._refresh_replay_runs)
        run_form.addRow(refresh_runs)
        self.replay_swarm_runs_status = QLabel()
        self.replay_swarm_runs_status.setWordWrap(True)
        run_form.addRow(self.replay_swarm_runs_status)
        page.addLayout(run_form)

        page.addWidget(QLabel("Click an attempt to watch it. Tick its box to include it in a comparison:"))
        self.replay_episode_list = QListWidget()
        self.replay_episode_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.replay_episode_list.setMinimumHeight(160)
        self.replay_episode_list.itemChanged.connect(self._update_replay_group_count)
        page.addWidget(self.replay_episode_list)
        group_actions = QHBoxLayout()
        select_all = QPushButton("Select all attempts")
        select_all.clicked.connect(lambda: self._check_replay_group(True))
        uncheck_all = QPushButton("Clear selection")
        uncheck_all.clicked.connect(lambda: self._check_replay_group(False))
        self.replay_group_count = QLabel("No attempts selected; comparison uses all matching attempts")
        self.replay_group_count.setWordWrap(True)
        group_actions.addWidget(select_all)
        group_actions.addWidget(uncheck_all)
        group_actions.addWidget(self.replay_group_count)
        page.addLayout(group_actions)

        self.replay_swarm_action_buttons: list[QPushButton] = []
        action_row = QHBoxLayout()
        self.replay_single_play = QPushButton("Watch selected attempt")
        self.replay_single_play.setToolTip("Play the highlighted attempt in PolyTrack.")
        self.replay_single_play.clicked.connect(lambda: self._submit_replay_swarm("single_play"))
        action_row.addWidget(self.replay_single_play)
        self.replay_swarm_action_buttons.append(self.replay_single_play)
        compare_button = QPushButton("Compare selected attempts")
        compare_button.setToolTip(
            "Play checked attempts together. With none checked, compare all matching attempts in this run."
        )
        compare_button.clicked.connect(lambda: self._submit_replay_swarm("play"))
        action_row.addWidget(compare_button)
        self.replay_swarm_action_buttons.append(compare_button)
        page.addLayout(action_row)
        self.replay_swarm_loaded_ghosts = QCheckBox("Play alongside loaded ghosts")
        self.replay_swarm_loaded_ghosts.setToolTip(
            "Keep PolyTrack ghosts already loaded in the race visible during replay, and match their opacity to the replay."
        )
        self.replay_swarm_loaded_ghosts.toggled.connect(
            lambda _enabled: self._submit_replay_swarm("configure")
        )
        page.addWidget(self.replay_swarm_loaded_ghosts)

        controls = QHBoxLayout()
        for label, action in (("Pause", "pause"), ("Resume", "resume"),
                              ("Restart", "restart"), ("Clear replays", "clear")):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, selected=action: self._submit_replay_swarm(selected))
            controls.addWidget(button)
            self.replay_swarm_action_buttons.append(button)
        page.addLayout(controls)
        seek_row = QHBoxLayout()
        self.replay_swarm_seek = QDoubleSpinBox()
        self.replay_swarm_seek.setRange(0.0, 1_000_000.0)
        self.replay_swarm_seek.setDecimals(2)
        seek_row.addWidget(QLabel("Jump to time (seconds)"))
        seek_row.addWidget(self.replay_swarm_seek)
        jump = QPushButton("Jump")
        jump.clicked.connect(lambda: self._submit_replay_swarm("seek"))
        seek_row.addWidget(jump)
        self.replay_swarm_action_buttons.append(jump)
        page.addLayout(seek_row)

        advanced_toggle = QToolButton()
        self.replay_advanced_toggle = advanced_toggle
        advanced_toggle.setText("Advanced filters and playback settings")
        advanced_toggle.setCheckable(True)
        advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        page.addWidget(advanced_toggle)
        advanced_content = QWidget()
        self.replay_advanced_content = advanced_content
        advanced_layout = QVBoxLayout(advanced_content)
        advanced_content.setVisible(False)
        advanced_toggle.toggled.connect(advanced_content.setVisible)
        advanced_toggle.toggled.connect(
            lambda expanded: advanced_toggle.setArrowType(
                Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
            )
        )
        page.addWidget(advanced_content)

        advanced_form = QFormLayout()
        self.replay_swarm_external = QCheckBox("Browse an external replay folder")
        self.replay_swarm_external.toggled.connect(self._toggle_external_replay_path)
        advanced_form.addRow(self.replay_swarm_external)
        self.replay_swarm_path = QLineEdit()
        self.replay_swarm_path.setPlaceholderText("Replay run folder or parent folder")
        path_row = QHBoxLayout()
        path_row.addWidget(self.replay_swarm_path)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_replay_swarm_path)
        path_row.addWidget(browse)
        self.replay_swarm_external_row = QWidget()
        self.replay_swarm_external_row.setLayout(path_row)
        self.replay_swarm_external_row.setVisible(False)
        advanced_form.addRow("External replay folder", self.replay_swarm_external_row)

        self.replay_swarm_container = QGroupBox("Compare filters and appearance")
        swarm_layout = QVBoxLayout(self.replay_swarm_container)
        swarm_form = QFormLayout()
        self.replay_swarm_step_min = QSpinBox()
        self.replay_swarm_step_min.setRange(0, 2_000_000_000)
        self.replay_swarm_step_min.setValue(0)
        self.replay_swarm_step_max = QSpinBox()
        self.replay_swarm_step_max.setRange(0, 2_000_000_000)
        self.replay_swarm_step_max.setValue(1_000_000)
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("From"))
        steps_row.addWidget(self.replay_swarm_step_min)
        steps_row.addWidget(QLabel("to"))
        steps_row.addWidget(self.replay_swarm_step_max)
        swarm_form.addRow("Training-step range", steps_row)
        self.replay_swarm_episode_min = QLineEdit()
        self.replay_swarm_episode_min.setPlaceholderText("Any")
        self.replay_swarm_episode_max = QLineEdit()
        self.replay_swarm_episode_max.setPlaceholderText("Any")
        episodes_row = QHBoxLayout()
        episodes_row.addWidget(QLabel("From episode"))
        episodes_row.addWidget(self.replay_swarm_episode_min)
        episodes_row.addWidget(QLabel("to"))
        episodes_row.addWidget(self.replay_swarm_episode_max)
        swarm_form.addRow("Episode range", episodes_row)
        self.replay_swarm_finished = QCheckBox("Finished only")
        self.replay_swarm_failed = QCheckBox("Failed or timed out only")
        self.replay_swarm_finished.toggled.connect(
            lambda checked: self.replay_swarm_failed.setChecked(False) if checked else None
        )
        self.replay_swarm_failed.toggled.connect(
            lambda checked: self.replay_swarm_finished.setChecked(False) if checked else None
        )
        status_row = QHBoxLayout()
        status_row.addWidget(self.replay_swarm_finished)
        status_row.addWidget(self.replay_swarm_failed)
        swarm_form.addRow("Attempt status", status_row)
        self.replay_swarm_max_cars = QSpinBox()
        self.replay_swarm_max_cars.setRange(1, MAX_REPLAY_GHOSTS)
        self.replay_swarm_max_cars.setValue(100)
        self.replay_swarm_seed = QSpinBox()
        self.replay_swarm_seed.setRange(0, 2_000_000_000)
        selection_row = QHBoxLayout()
        selection_row.addWidget(QLabel("Maximum cars"))
        selection_row.addWidget(self.replay_swarm_max_cars)
        selection_row.addWidget(QLabel("Sampling seed"))
        selection_row.addWidget(self.replay_swarm_seed)
        swarm_form.addRow("Large groups", selection_row)
        self.replay_swarm_color_min = QSpinBox()
        self.replay_swarm_color_min.setRange(0, 0)
        self.replay_swarm_color_min.setValue(0)
        self.replay_swarm_color_min.setEnabled(False)
        self.replay_swarm_color_max = QSpinBox()
        self.replay_swarm_color_max.setRange(1, 2_000_000_000)
        self.replay_swarm_color_max.setValue(2_000_000)
        self.replay_swarm_color_max.setGroupSeparatorShown(True)
        self.replay_swarm_color_max.setToolTip(
            "Training step shown as green. Earlier steps blend from red through orange, yellow, "
            "and yellow-green. Later steps stay green. Changing runs keeps this setting."
        )
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Training age"))
        gradient = QLabel("0")
        gradient.setMinimumWidth(140)
        gradient.setStyleSheet(
            "color: black; padding: 4px; border-radius: 3px; "
            "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #ff0000,stop:0.25 #ff8000,stop:0.5 #ffff00,"
            "stop:0.75 #80ff00,stop:1 #00ff00);"
        )
        gradient.setToolTip("Red → orange → yellow → yellow-green → green, based on recorded training steps.")
        color_row.addWidget(gradient, 1)
        color_row.addWidget(QLabel("Green at step"))
        color_row.addWidget(self.replay_swarm_color_max)
        apply_colors = QPushButton("Apply colours")
        apply_colors.setToolTip("Update the loaded replay colours without restarting playback.")
        apply_colors.clicked.connect(lambda: self._submit_replay_swarm("configure"))
        color_row.addWidget(apply_colors)
        self.replay_swarm_action_buttons.append(apply_colors)
        page.insertLayout(page.indexOf(self.replay_episode_list), color_row)
        self.replay_swarm_color_stops = QLineEdit()
        self.replay_swarm_color_stops.setPlaceholderText("Optional, e.g. 0:#ff0000, 500000:#ffff00")
        swarm_form.addRow("Custom colors", self.replay_swarm_color_stops)
        self.replay_swarm_speed = QDoubleSpinBox()
        self.replay_swarm_speed.setRange(0.1, 8.0)
        self.replay_swarm_speed.setSingleStep(0.1)
        self.replay_swarm_speed.setValue(1.0)
        self.replay_swarm_opacity = QDoubleSpinBox()
        self.replay_swarm_opacity.setRange(0.0, 1.0)
        self.replay_swarm_opacity.setSingleStep(0.05)
        self.replay_swarm_opacity.setValue(1.0)
        self.replay_swarm_end = QComboBox()
        self.replay_swarm_end.addItems(("fade", "disappear"))
        self.replay_swarm_fade = QDoubleSpinBox()
        self.replay_swarm_fade.setRange(0.0, 10.0)
        self.replay_swarm_fade.setSingleStep(0.1)
        self.replay_swarm_fade.setValue(0.75)
        appearance_row = QHBoxLayout()
        for label, widget in (("Speed", self.replay_swarm_speed),
                              ("Opacity", self.replay_swarm_opacity),
                              ("When done", self.replay_swarm_end),
                              ("Fade seconds", self.replay_swarm_fade)):
            appearance_row.addWidget(QLabel(label))
            appearance_row.addWidget(widget)
        swarm_form.addRow("Appearance", appearance_row)
        self.replay_swarm_port = QSpinBox()
        self.replay_swarm_port.setRange(1, 65535)
        self.replay_swarm_port.setValue(8765)
        swarm_form.addRow("Bridge port", self.replay_swarm_port)
        swarm_layout.addLayout(swarm_form)
        advanced_layout.addLayout(advanced_form)
        advanced_layout.addWidget(self.replay_swarm_container)
        advanced_actions = QGridLayout()
        for index, (label, action) in enumerate((
            ("Load paused", "load"), ("Inspect selection", "inspect"),
            ("Set full run range", "range"), ("Apply settings", "configure"),
            ("Bridge status", "status"),
        )):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, selected=action: self._submit_replay_swarm(selected))
            advanced_actions.addWidget(button, index // 3, index % 3)
            self.replay_swarm_action_buttons.append(button)
        advanced_layout.addLayout(advanced_actions)

        self.replay_swarm_output = QTextEdit()
        self.replay_swarm_output.setReadOnly(True)
        self.replay_swarm_output.setPlaceholderText("Replay status and messages appear here.")
        self.replay_swarm_output.document().setMaximumBlockCount(500)
        self.replay_swarm_output.setMinimumHeight(95)
        page.addWidget(self.replay_swarm_output)
        self._refresh_replay_runs()

    def _replay_recording_changed(self, index: int) -> None:
        if not hasattr(self, "general"):
            return
        value = ("automatic", "enabled", "disabled")[index]
        widget = self.general["visual_replay_enabled"]
        widget.blockSignals(True)
        widget.setCurrentText(value)
        widget.blockSignals(False)
        self._refresh_run_summary()

    def _general_replay_recording_changed(self, value: str) -> None:
        if not hasattr(self, "replay_recording"):
            return
        text = {
            "automatic": "Automatic for live training",
            "enabled": "Always record",
            "disabled": "Do not record",
        }.get(value, "Automatic for live training")
        self.replay_recording.blockSignals(True)
        self.replay_recording.setCurrentText(text)
        self.replay_recording.blockSignals(False)

    def _toggle_external_replay_path(self, enabled: bool) -> None:
        self.replay_swarm_external_row.setVisible(enabled)
        self.replay_swarm_run.setEnabled(not enabled)
        self.replay_swarm_algorithm.setEnabled(not enabled)
        self._replay_swarm_run_changed()

    def _refresh_replay_runs(self, _algorithm: str = "") -> None:
        if not hasattr(self, "replay_swarm_run"):
            return
        workspace = self._workspace()
        algorithm = self.replay_swarm_algorithm.currentText()
        previous = self.replay_swarm_run.currentData()
        load_error: str | None = None
        self.replay_swarm_run.blockSignals(True)
        self.replay_swarm_run.clear()
        try:
            runs = workspace.list_replay_runs(algorithm)
            for run in runs:
                entries = load_replay_index(run.directory)
                finished = [entry for entry in entries if entry.get("lap_time_s") is not None]
                best = min((float(entry["lap_time_s"]) for entry in finished), default=None)
                timestamp = run.timestamp if isinstance(run.timestamp, str) else run.run_id
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone().strftime(
                        "%d %b %H:%M"
                    )
                except ValueError:
                    pass
                summary = f"{timestamp} · {run.episode_count} attempts"
                if best is not None:
                    summary += f" · best {best:.3f}s"
                self.replay_swarm_run.addItem(summary, str(run.directory.resolve()))
            if runs:
                self.replay_swarm_run.addItem(
                    f"All runs · {sum(run.episode_count for run in runs)} episodes",
                    str(workspace.visual_replays(algorithm).resolve()),
                )
        except (OSError, ValueError) as exc:
            load_error = str(exc)
            self.replay_swarm_output.setPlainText(f"Could not list replay runs: {exc}")
        selected = self.replay_swarm_run.findData(previous)
        self.replay_swarm_run.setCurrentIndex(selected if selected >= 0 else 0)
        self.replay_swarm_run.blockSignals(False)
        self._replay_swarm_run_changed()
        self._refresh_replay_episodes()
        run_count = self.replay_swarm_run.count()
        if load_error:
            self.replay_swarm_runs_status.setText(f"Could not list replay runs: {load_error}")
        elif run_count:
            self.replay_swarm_runs_status.setText(
                f"{run_count - 1} saved run(s) for {self._selected_track().name}. "
                "Refresh runs after training to see new attempts."
            )
        else:
            self.replay_swarm_runs_status.setText(
                f"No saved replay runs found for {self._selected_track().name} · {algorithm.upper()}. "
                "Start WebSocket training in PolyTrack with Save replays set to Automatic or Always record. "
                "Each completed attempt will appear here."
            )

    def _replay_swarm_run_changed(self, _index: int = -1) -> None:
        if not hasattr(self, "replay_swarm_path") or self.replay_swarm_external.isChecked():
            return
        run_path = self.replay_swarm_run.currentData()
        self.replay_swarm_path.setText(run_path if isinstance(run_path, str) else "")
        self._refresh_replay_episodes()
        steps = [
            self.replay_episode_list.item(index).data(Qt.ItemDataRole.UserRole)[1]["training_step_start"]
            for index in range(self.replay_episode_list.count())
        ]
        if steps:
            self.replay_swarm_step_min.setValue(min(steps))
            self.replay_swarm_step_max.setValue(max(steps))

    def _refresh_replay_episodes(self) -> None:
        if not hasattr(self, "replay_episode_list"):
            return
        self.replay_episode_list.clear()
        self._update_replay_group_count()
        run_path = self._replay_swarm_path_value()
        if not run_path:
            return
        try:
            directories = resolve_replay_directories(run_path)
            for directory in directories:
                for entry in load_replay_index(directory):
                    step = int(entry["training_step_start"])
                    lap = entry.get("lap_time_s")
                    status = str(entry["status"]).capitalize()
                    lap_text = f"{float(lap):.3f}s" if lap is not None else f"{status} · {float(entry.get('final_progress_ratio', 0.0)):.0%} complete"
                    label = f"Attempt {entry['episode_id'].rsplit('-', 1)[-1]} · {lap_text}"
                    tooltip = (
                        f"{status} · {float(entry.get('final_progress_m', 0.0)):.1f} m · "
                        f"{entry['sample_count']} recorded positions · training step {step:,}"
                    )
                    item = QListWidgetItem(label)
                    item.setToolTip(tooltip)
                    item.setData(Qt.ItemDataRole.UserRole, (str(directory), entry))
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    self.replay_episode_list.addItem(item)
            if self.replay_episode_list.count():
                self.replay_episode_list.setCurrentRow(0)
        except (OSError, ValueError, KeyError) as exc:
            self._append_replay_swarm_output(f"Could not list replay episodes: {exc}")

    def _check_replay_group(self, checked: bool) -> None:
        self.replay_episode_list.blockSignals(True)
        for index in range(self.replay_episode_list.count()):
            self.replay_episode_list.item(index).setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
            )
        self.replay_episode_list.blockSignals(False)
        self._update_replay_group_count()

    def _update_replay_group_count(self, *_args: Any) -> None:
        if not hasattr(self, "replay_group_count"):
            return
        count = sum(
            self.replay_episode_list.item(index).checkState() == Qt.CheckState.Checked
            for index in range(self.replay_episode_list.count())
        )
        self.replay_group_count.setText(
            f"{count} attempt(s) selected for comparison" if count
            else "No attempts selected; comparison uses all matching attempts",
        )

    def _replay_swarm_path_value(self) -> str:
        if self.replay_swarm_external.isChecked():
            return self.replay_swarm_path.text().strip()
        path = self.replay_swarm_run.currentData()
        return path if isinstance(path, str) else ""

    def _browse_replay_swarm_path(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose replay run or parent folder")
        if path:
            self.replay_swarm_path.setText(path)

    def _replay_swarm_config(self) -> dict[str, Any]:
        path = self._replay_swarm_path_value()
        if not path:
            raise ValueError(
                "Choose a replay run by selecting a recorded run or enabling external replay browsing."
            )
        steps = (self.replay_swarm_step_min.value(), self.replay_swarm_step_max.value())
        if steps[1] < steps[0]:
            raise ValueError("The maximum training step must be at least the minimum.")
        episode_values = (
            self.replay_swarm_episode_min.text().strip(),
            self.replay_swarm_episode_max.text().strip(),
        )
        if bool(episode_values[0]) != bool(episode_values[1]):
            raise ValueError("Enter both episode ID boundaries, or leave both empty.")
        episodes = None
        if episode_values[0]:
            try:
                episodes = (int(episode_values[0]), int(episode_values[1]))
            except ValueError as exc:
                raise ValueError("Episode ID boundaries must be integers.") from exc
            if episodes[0] < 0 or episodes[1] < episodes[0]:
                raise ValueError("Episode IDs must be non-negative and ordered.")

        color_scale = self._replay_color_scale()
        return {
            "path": path,
            "track_slug": self._selected_track().slug,
            "external_path": self.replay_swarm_external.isChecked(),
            "steps": steps,
            "episodes": episodes,
            "finished_only": self.replay_swarm_finished.isChecked(),
            "failed_only": self.replay_swarm_failed.isChecked(),
            "max_cars": self.replay_swarm_max_cars.value(),
            "seed": self.replay_swarm_seed.value(),
            "color_scale": color_scale,
            "speed": self.replay_swarm_speed.value(),
            "opacity": self.replay_swarm_opacity.value(),
            "play_alongside_loaded_ghosts": self.replay_swarm_loaded_ghosts.isChecked(),
            "end_behavior": self.replay_swarm_end.currentText(),
            "fade_duration_s": self.replay_swarm_fade.value(),
            "seek_seconds": self.replay_swarm_seek.value(),
            "port": self.replay_swarm_port.value(),
            "selected_episodes": [
                [item.data(Qt.ItemDataRole.UserRole)[1]["run_id"],
                 item.data(Qt.ItemDataRole.UserRole)[1]["episode_id"]]
                for index in range(self.replay_episode_list.count())
                if (item := self.replay_episode_list.item(index)).checkState() == Qt.CheckState.Checked
            ],
        }

    def _replay_color_scale(self) -> ColorScale:
        raw_stops = [
            stop.strip()
            for stop in self.replay_swarm_color_stops.text().replace(",", "\n").splitlines()
            if stop.strip()
        ]
        return ColorScale(0, self.replay_swarm_color_max.value(),
                          parse_color_stops(raw_stops) if raw_stops else None)

    @staticmethod
    def _replay_swarm_selection(config: dict[str, Any]) -> tuple[int, list[ReplaySelection], int]:
        directories = resolve_replay_directories(config["path"])
        entries = [
            metadata
            for directory in directories
            for metadata in load_replay_index(directory)
        ]
        if config.get("external_path"):
            expected_slug = config["track_slug"]
            mismatched = [
                entry for entry in entries
                if entry.get("track_slug", track_slug(entry["track_name"])) != expected_slug
            ]
            if mismatched:
                raise ValueError(
                    f"external replay contains episodes for another track; expected {expected_slug}"
                )
        selected_keys = {tuple(key) for key in config.get("selected_episodes", ())}
        if selected_keys:
            selected = [
                ReplaySelection(directory, metadata)
                for directory in directories
                for metadata in load_replay_index(directory)
                if (metadata["run_id"], metadata["episode_id"]) in selected_keys
            ]
            if len(selected) > config["max_cars"]:
                raise ValueError("The checked episode group exceeds Maximum cars.")
            return len(entries), sorted(selected, key=lambda item: (item.training_step, item.metadata["episode_id"])), len(selected)
        matching_count = len(
            filter_replays(
                entries,
                steps=config["steps"],
                episodes=config["episodes"],
                finished_only=config["finished_only"],
                failed_only=config["failed_only"],
            )
        )
        total_indexed, selected = select_replays(
            directories,
            steps=config["steps"],
            episodes=config["episodes"],
            finished_only=config["finished_only"],
            failed_only=config["failed_only"],
            max_cars=config["max_cars"],
            seed=config["seed"],
        )
        return total_indexed, selected, matching_count

    def _submit_replay_swarm(self, action: str) -> None:
        if action not in {"inspect", "range"} and not self._require_idle():
            return
        try:
            if action == "range":
                path = self._replay_swarm_path_value()
                if not path:
                    raise ValueError("Choose a replay run directory first.")
                config = {"path": path}
            elif action in {"pause", "resume", "restart", "clear", "status"}:
                config = {
                    "port": self.replay_swarm_port.value(),
                    "action": action,
                }
            elif action in {"seek", "configure"}:
                config = {
                    "port": self.replay_swarm_port.value(),
                    "action": action,
                    "seek_seconds": self.replay_swarm_seek.value(),
                    "speed": self.replay_swarm_speed.value(),
                    "opacity": self.replay_swarm_opacity.value(),
                    "play_alongside_loaded_ghosts": self.replay_swarm_loaded_ghosts.isChecked(),
                    "end_behavior": self.replay_swarm_end.currentText(),
                    "fade_duration_s": self.replay_swarm_fade.value(),
                    "color_scale": self._replay_color_scale() if action == "configure" else None,
                }
            else:
                if action in {"single_play", "single_load"}:
                    item = self.replay_episode_list.currentItem()
                    data = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
                    if not isinstance(data, tuple) or len(data) != 2:
                        raise ValueError("Select a replay episode first.")
                    config = self._replay_swarm_config()
                    config["single_selection"] = [data[0], data[1]]
                    config["action"] = action
                else:
                    config = self._replay_swarm_config()
                    config["action"] = action
        except (ValueError, OSError) as exc:
            self._append_replay_swarm_output(f"Validation error: {exc}")
            return

        if self.replay_swarm_worker is not None and self.replay_swarm_worker.isRunning():
            self._append_replay_swarm_output("A replay swarm operation is already running.")
            return
        for button in self.replay_swarm_action_buttons:
            button.setEnabled(False)
        self._append_replay_swarm_output(f"Running {action}...")
        worker = ReplaySwarmTask(lambda: self._run_replay_swarm(config))
        worker.completed.connect(self._replay_swarm_completed)
        worker.failed.connect(lambda message: self._replay_swarm_failed(action, message))
        worker.finished.connect(lambda: self._replay_swarm_worker_finished(worker))
        self.replay_swarm_worker = worker
        worker.start()

    def _run_replay_swarm(self, config: dict[str, Any]) -> dict[str, Any]:
        action = config["action"]
        if action == "range":
            directories = resolve_replay_directories(config["path"])
            entries = [
                metadata
                for directory in directories
                for metadata in load_replay_index(directory)
            ]
            if not entries:
                raise ValueError("No visual replay episodes were found in this folder.")
            steps = [int(metadata["training_step_start"]) for metadata in entries]
            return {"_set_step_range": [min(steps), max(steps)], "episode_count": len(entries)}

        if action in {"single_play", "single_load"}:
            directory_text, metadata = config["single_selection"]
            directory = Path(directory_text)
            selection = ReplaySelection(directory, metadata)
            payload = load_replay_episode(
                directory, metadata, include_optional=False, include_hud=True,
            )
            options = ReplayPlaybackOptions(
                speed=config.get("speed", 1.0), opacity=config.get("opacity", 1.0),
                play_alongside_loaded_ghosts=config.get("play_alongside_loaded_ghosts", False),
                color=config["color_scale"].hex_color(selection.training_step),
                end_behavior=config.get("end_behavior", "fade"),
                fade_duration_s=config.get("fade_duration_s", 0.75),
            )
            transport = WebSocketServerTransport(
                port=config["port"], connect_timeout_s=60.0, request_timeout_s=35.0,
            )
            try:
                result = send_replay_playback(
                    transport, action="play" if action == "single_play" else "load",
                    payload=payload, options=options,
                )
            finally:
                transport.close()
            return {"action": action, "episode": selection.metadata["episode_id"], "result": result}

        if action in {"inspect", "play", "load"}:
            total, selected, matching_count = self._replay_swarm_selection(config)
            if action == "inspect":
                report = selection_report(
                    total,
                    selected,
                    matching_count=matching_count,
                    color_scale=config["color_scale"],
                )
                report["step_filter"] = list(config["steps"])
                report["episode_filter"] = list(config["episodes"]) if config["episodes"] else None
                report["sample_seed"] = config["seed"]
                return report
            if not selected:
                raise ValueError(f"{action.capitalize()} requires at least one matching replay episode.")
            indexed_samples = sum(int(item.metadata["sample_count"]) for item in selected)
            if indexed_samples > MAX_REPLAY_TOTAL_SAMPLES:
                raise ValueError(
                    f"Selection exceeds the maximum total sample count ({MAX_REPLAY_TOTAL_SAMPLES})."
                )
            camera_selection = min(selected, key=replay_camera_rank)
            payloads = [
                SelectedReplayPayload(
                    item,
                    load_replay_episode(
                        item.replay_directory, item.metadata,
                        include_optional=False, include_hud=item is camera_selection,
                    ),
                    color,
                )
                for item, color in zip(selected, swarm_colors(selected, config["color_scale"]), strict=True)
            ]
            selection_info = {
                "total_indexed_episodes": total,
                "matching_episodes": matching_count,
                "selected_episode_count": len(payloads),
            }
        else:
            payloads = []
            selection_info = {}

        settings = ReplayPlaybackOptions(
            speed=config.get("speed", 1.0),
            opacity=config.get("opacity", 1.0),
            play_alongside_loaded_ghosts=config.get("play_alongside_loaded_ghosts", False),
            end_behavior=config.get("end_behavior", "fade"),
            fade_duration_s=config.get("fade_duration_s", 0.75),
        )
        update_settings = (
            ("speed", "opacity", "end_behavior", "fade_duration", "loaded_ghosts")
            if action in {"play", "load", "configure"}
            else ()
        )
        transport = WebSocketServerTransport(
            port=config["port"],
            connect_timeout_s=60.0,
            request_timeout_s=35.0,
        )
        try:
            result = send_replay_swarm(
                transport,
                action=action,
                payloads=payloads,
                options=settings,
                seek_seconds=config.get("seek_seconds") if action == "seek" else None,
                update_settings=update_settings,
                color_scale=config.get("color_scale") if action == "configure" else None,
            )
        finally:
            transport.close()
        return {"action": action, "selection": selection_info, "result": result}

    def _replay_swarm_completed(self, result: dict[str, Any]) -> None:
        if "_set_step_range" in result:
            minimum, maximum = result["_set_step_range"]
            self.replay_swarm_step_min.setValue(minimum)
            self.replay_swarm_step_max.setValue(maximum)
            self._append_replay_swarm_output(
                f"Set training-step window to {minimum}:{maximum} ({result['episode_count']} episodes indexed)."
            )
        else:
            outcome = result.get("result")
            if isinstance(outcome, dict) and "loaded_ghosts" in outcome:
                message = (
                    f"Loaded {outcome['loaded_ghosts']} replay car(s); "
                    f"{outcome['visible_ghosts']} visible at {outcome['playback_seconds']:.2f}s."
                )
                camera = outcome.get("leader_episode_id")
                if camera:
                    message += f" Camera follows the best run ({camera.rsplit(':', 1)[-1]})."
                self._append_replay_swarm_output(message)
            elif result.get("action") in {"single_play", "single_load"}:
                verb = "Playing" if result["action"] == "single_play" else "Loaded"
                self._append_replay_swarm_output(f"{verb} attempt {result['episode']}.")
            else:
                self._append_replay_swarm_output(json.dumps(result, indent=2, allow_nan=False))

    def _replay_swarm_failed(self, action: str, message: str) -> None:
        self._append_replay_swarm_output(f"{action.capitalize()} failed: {message}")

    def _replay_swarm_worker_finished(self, worker: ReplaySwarmTask) -> None:
        if self.replay_swarm_worker is worker:
            self.replay_swarm_worker = None
        for button in self.replay_swarm_action_buttons:
            button.setEnabled(True)

    def _append_replay_swarm_output(self, message: str) -> None:
        self.replay_swarm_output.append(message)
        self.replay_swarm_output.moveCursor(QTextCursor.End)

    def _ai_overlay_tab(self) -> None:
        _, page = self._page("AI HUD")
        description = QLabel(
            "Show the policy's actual inputs and outputs, applied controls, episode state, "
            "and reward breakdown inside PolyTrack. Updates follow policy decisions; "
            "simulator ticks and frame skip are shown separately. Requires WebSocket training "
            "and PolyBot bridge 0.1.41 or newer for PolyTrack 0.6.3."
        )
        description.setWordWrap(True)
        page.addWidget(description)
        form = QFormLayout()
        page.addLayout(form)
        settings = self.ai_overlay_settings
        self.ai_overlay_widgets: dict[str, QWidget] = {}
        options: tuple[tuple[str, str, Any, str, tuple[str, ...]], ...] = (
            ("display_mode", "HUD mode", settings.display_mode,
             "Choose a presentation for live driving or saved replay telemetry. Also switch modes in the game.", HUD_MODES),
            ("wr_target_s", "World record target (seconds)", settings.wr_target_s,
             "Set the verified WR target for WR Chase. Zero leaves the target unknown.", ()),
            ("preset", "Layout", settings.preset,
             "Full uses a wider input grid. Modes show the most important 60% of inputs.", ("compact", "full")),
            ("scale", "Scale", settings.scale,
             "Scale the non-interactive overlay from 0.5× to 1.5×.", ()),
            ("show_episode_status", "Episode status", settings.show_episode_status,
             "Show mode, track, algorithm, episode, progress, and timing.", ()),
            ("show_labels", "Feature labels", settings.show_labels,
             "Show names beside observation, control, and reward values.", ()),
            ("show_observations", "Policy observations", settings.show_observations,
             "Show the exact normalized feature values passed to the policy.", ()),
            ("show_controls", "Policy and applied controls", settings.show_controls,
             "Show model output, transformed output, adapter demand, and applied control fractions.", ()),
            ("show_reward_breakdown", "Reward breakdown", settings.show_reward_breakdown,
             "Show exact reward terms, groups, and episode return.", ()),
            ("show_event_popups", "Event popups", settings.show_event_popups,
             "Briefly surface finish, crash, timeout, and other episode events.", ()),
            ("lookahead_points", "Lookahead points", settings.lookahead_points,
             "Maximum route points shown in the upcoming-track diagram. The input list separately selects the most important 60%.", ()),
        )
        for name, label_text, value, help_text, choices in options:
            widget = _editor(value, help_text, choices)
            if name == "scale":
                assert isinstance(widget, QDoubleSpinBox)
                widget.setRange(0.5, 1.5)
                widget.setDecimals(1)
                widget.setSingleStep(0.1)
            elif name == "lookahead_points":
                assert isinstance(widget, QSpinBox)
                widget.setRange(0, 12)
            elif name == "wr_target_s":
                assert isinstance(widget, QDoubleSpinBox)
                widget.setRange(0, 3600)
                widget.setDecimals(3)
            form.addRow(QLabel(label_text), widget)
            self.ai_overlay_widgets[name] = widget
        save = QPushButton("Save / apply telemetry display settings")
        save.clicked.connect(self._save_ai_overlay_settings)
        page.addWidget(save)
        self.ai_overlay_status = QLabel()
        self.ai_overlay_status.setWordWrap(True)
        if self._ai_overlay_settings_error:
            self.ai_overlay_status.setText(
                f"Settings could not be loaded; default display settings will be used until saved: "
                f"{self._ai_overlay_settings_error}"
            )
        else:
            self.ai_overlay_status.setText(
                "Use the PolyBot HUD button in the game page to show or hide the HUD. "
                "These display settings affect incoming telemetry."
            )
        page.addWidget(self.ai_overlay_status)

    def _save_ai_overlay_settings(self) -> bool:
        values = {name: _value(widget) for name, widget in self.ai_overlay_widgets.items()}
        try:
            settings = AIOverlaySettings(**values)
            self.ai_overlay_store.save(settings)
        except (OSError, ValueError) as exc:
            self.ai_overlay_status.setText(f"Could not save AI HUD settings: {exc}")
            return False
        self.ai_overlay_settings = settings
        self._ai_overlay_settings_error = None
        runner = getattr(self, "runner", None)
        if runner is not None:
            runner.set_ai_overlay_settings(settings)
        self.ai_overlay_status.setText("HUD settings saved and applied.")
        return True

    def _status_tab(self) -> None:
        _, page = self._page("Status")
        self.warnings = QLabel("Configuration notes appear here when you start a run.")
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
        metric_content = QWidget()
        metric_form = QFormLayout(metric_content)
        self.metric_details = ExpandableSection("Detailed training metrics", metric_content)
        page.addWidget(self.metric_details)
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
        self.log.setMinimumHeight(230)
        self.log.setToolTip(
            "Short summaries of training events. The JSONL file keeps every metric and reward term."
        )
        page.addWidget(self.log, 1)

    def _toggle_advanced(self, enabled: bool) -> None:
        for form in (self.ppo_form, self.grtqc_form, self.tqc_form, self.curriculum_form):
            form.set_advanced(enabled)
        self.reward_scroll.setVisible(enabled)
        for section in self.workflow_sections:
            section.setVisible(enabled)
        self.section_optimizer_section.setVisible(enabled)
        for name in self.general_advanced:
            self.general[name].setVisible(enabled)
            self.general_labels[name].setVisible(enabled)
        layout = self.layout()
        if layout is not None:
            layout.invalidate()
            layout.activate()
        self.updateGeometry()

    def _backend_changed(self, backend: str) -> None:
        if backend == "mock":
            _set(self.general["frame_skip"], 4)
        else:
            _set(self.general["frame_skip"], 30)
        self._refresh_models()

    def _algorithm_changed(self, algorithm: str) -> None:
        forms = {"ppo": self.ppo_form, "grtqc": self.grtqc_form, "tqc": self.tqc_form}
        self.algorithm_stack.setCurrentWidget(forms[algorithm])
        explanations = {
            "ppo": (
                "PPO: on-policy. It learns from fresh rollouts, then discards them. "
                "It directly outputs continuous steering and signed throttle/brake demand."
            ),
            "grtqc": (
                "GRTQC: the primary off-policy learner. It learns continuous controls with gated "
                "quantile critics and a replay buffer. Its training origin can be scratch or transfer; "
                "continuing a checkpoint restores the origin saved with that model."
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
        self.preset.addItem("Custom")
        self.preset.setCurrentText("Balanced")
        self.preset.blockSignals(current)
        self._preset_changed("Balanced")
        self._refresh_models()

    def _preset_changed(self, name: str) -> None:
        if not name or name == "Custom":
            return
        if name.startswith("Summer 1 -") and self._selected_track().slug != "summer-1":
            self.preset.blockSignals(True)
            self.preset.setCurrentText("Balanced")
            self.preset.blockSignals(False)
            name = "Balanced"
        preset = algorithm_presets(self.algorithm.currentText()).get(name)
        if preset is None:
            path = self.presets.list(self.algorithm.currentText()).get(name)
            if path is None:
                return
            preset = self.presets.load(path)
        self._last_algorithm_preset[self.algorithm.currentText()] = name
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
        name = self.preset.currentText()
        if name == "Custom":
            name = self._last_algorithm_preset.get(self.algorithm.currentText(), "Balanced")
            _set(self.preset, name)
        self._preset_changed(name)

    def _mark_algorithm_custom(self, *_args: Any) -> None:
        _set(self.preset, "Custom")

    def _show_glossary(self) -> None:
        glossary = (
            "Policy / actor: a network that chooses steering and pedals in PPO or TQC.\n"
            "Q-value: predicted future reward for a chosen action.\n"
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
            "Racing line: saved route guidance used by the model. A ghost can initialize it.\n"
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
        track = QComboBox()
        for definition in self.track_registry.list_tracks():
            track.addItem(definition.name, definition.slug)
        track.setCurrentIndex(max(0, track.findData(self._selected_track().slug)))
        track_actions = QHBoxLayout()
        track_actions.addWidget(track, 1)
        add_track = QPushButton("+ Add new track")
        add_track.clicked.connect(lambda: self._add_track_to_combo(track))
        track_actions.addWidget(add_track)
        backend = _editor("websocket", GENERAL_INFO["backend"].description, ("websocket", "mock"))
        track_layout.addRow("Track", track_actions)
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
                track_definition = self.track_registry.resolve(track.currentData())
                profile = {
                    "Learn the track": "Learning", "Improve consistency": "Balanced",
                    "Improve lap time": "Pace", "Experiment": "Balanced",
                }[selected_goal]
                data.update({
                    "algorithm": selected_algorithm, "backend": selected_backend,
                    "track_name": track_definition.name,
                    "track_slug": track_definition.slug,
                    "track_id": (
                        track_definition.simulator_track_id
                        if selected_backend == "websocket" else "mock/straight"
                    ),
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

    def _add_track_to_combo(self, combo: QComboBox) -> None:
        name, accepted = QInputDialog.getText(self, "Add Track", "Track name")
        if not accepted:
            return
        try:
            track = self.track_registry.add(name)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Add Track", str(exc))
            return
        self._populate_track_selector(track.slug)
        self._track_changed()
        combo.addItem(track.name, track.slug)
        combo.setCurrentIndex(combo.findData(track.slug))

    def _load_reward_profile(self, name: str) -> None:
        if not name or name == "Custom":
            return
        self._base_rewards = self._session_reward_profiles.get(name) or self.profiles.load(name)
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
        self._mark_reward_profile_custom()
        self._refresh_reward_view()

    def _advanced_reward_changed(self, name: str, value: float) -> None:
        self._reward_values[name] = value
        self._mark_reward_profile_custom()
        self._refresh_reward_view(update_form=False)

    def _mark_reward_profile_custom(self) -> None:
        if self.reward_profile.currentText() != "Custom":
            self.reward_profile.setCurrentText("Custom")

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

    def _load_saved_replay_rewards(self, rewards: dict[str, object]) -> None:
        saved = RewardConfig(**rewards)
        self._base_rewards = saved
        self._reward_values = asdict(saved)
        for key, widget in self.reward_basic.items():
            _set(widget, 1.0 if key.endswith("_multiplier") else self._reward_values[key])
        self.reward_profile.blockSignals(True)
        self.reward_profile.setCurrentText("Custom")
        self.reward_profile.blockSignals(False)
        self._refresh_reward_view()

    def configuration(self) -> TrainingConfig:
        values = {name: _value(widget) for name, widget in self.general.items()}
        track = self._selected_track()
        values["visual_replay_enabled"] = {
            "automatic": None, "enabled": True, "disabled": False,
        }[values["visual_replay_enabled"]]
        values["output_root"] = Path(values["output_root"])
        values["log_root"] = Path(values["log_root"])
        values["algorithm"] = self.algorithm.currentText()
        values["track_name"] = track.name
        values["track_slug"] = track.slug
        values["track_id"] = (
            track.simulator_track_id
            if values["backend"] == "websocket" else "mock/straight"
        )
        selected_profile = self.reward_profile.currentText()
        values["reward_profile"] = None if selected_profile == "Custom" else selected_profile
        values["curriculum"] = self._curriculum_configuration()
        values["evaluation"] = EvaluationConfig(**self.evaluation_form.values())
        values["rewards"] = RewardConfig(**self._reward_values)
        values[values["algorithm"]] = self._current_algorithm_settings()
        return TrainingConfig(**values)

    def load_configuration(self, config: TrainingConfig) -> None:
        track = self.track_registry.register_legacy(
            config.track_name,
            simulator_track_id=config.track_id,
            slug=config.track_slug,
        )
        self._populate_track_selector(track.slug)
        if config.backend == "websocket":
            config.track_id = track.simulator_track_id
        config.track_name = track.name
        config.track_slug = track.slug
        _set(self.algorithm, config.algorithm)
        self._algorithm_changed(config.algorithm)
        self.general["backend"].blockSignals(True)
        for name, widget in self.general.items():
            value = getattr(config, name)
            if name == "visual_replay_enabled":
                value = "automatic" if value is None else "enabled" if value else "disabled"
            _set(widget, value)
        self.general["backend"].blockSignals(False)
        if hasattr(self, "replay_recording"):
            replay_index = {None: 0, True: 1, False: 2}[config.visual_replay_enabled]
            self.replay_recording.blockSignals(True)
            self.replay_recording.setCurrentIndex(replay_index)
            self.replay_recording.blockSignals(False)
        self.curriculum_form.load(config.curriculum)
        self.custom_phases.setPlainText(json.dumps([asdict(phase) for phase in config.curriculum.phases], indent=2))
        self.custom_phases.setVisible(config.curriculum.mode == "custom")
        self.evaluation_form.load(config.evaluation)
        self.ppo_form.load(config.ppo or PPOConfig())
        self.grtqc_form.load(config.grtqc or GRTQCConfig())
        self.tqc_form.load(config.tqc or TQCConfig())
        settings = {"ppo": config.ppo, "grtqc": config.grtqc, "tqc": config.tqc}[config.algorithm]
        matching_preset = next((name for name, preset in algorithm_presets(config.algorithm).items()
                                if preset == settings), "Custom")
        _set(self.preset, matching_preset)
        self._reward_values = asdict(config.rewards)
        self._base_rewards = config.rewards
        self._refresh_reward_view()
        for key, widget in self.reward_basic.items():
            _set(widget, 1.0 if key.endswith("_multiplier") else self._reward_values[key])
        self.reward_profile.blockSignals(True)
        if config.reward_profile and self.reward_profile.findText(config.reward_profile) < 0:
            self.reward_profile.addItem(config.reward_profile)
            self._session_reward_profiles[config.reward_profile] = config.rewards
        self.reward_profile.setCurrentText(config.reward_profile or "Custom")
        self.reward_profile.blockSignals(False)
        self._update_plan()
        self._track_changed()
        self._refresh_run_summary()

    def _save_config(self) -> None:
        try:
            cfg = self.configuration()
            name, _ = QFileDialog.getSaveFileName(
                self,
                "Save training configuration",
                str(Path("config") / "training-config.json"),
                "PolyBot configuration (*.json);;All files (*)",
            )
            if name:
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")
                self.log.append(f"Saved configuration: {path}")
        except (ValueError, OSError) as exc:
            self._error(str(exc))

    def _load_config_dialog(self) -> None:
        if not self._require_idle():
            return
        name, _ = QFileDialog.getOpenFileName(
            self,
            "Load training configuration",
            str(Path("config")),
            "PolyBot configuration (*.json);;All files (*)",
        )
        if name:
            try:
                cfg = TrainingConfig.from_dict(json.loads(Path(name).read_text(encoding="utf-8")))
                self.load_configuration(cfg)
            except (ValueError, OSError, KeyError) as exc:
                self._error(str(exc))

    def _best_resume_slot(self, cfg: TrainingConfig) -> Path:
        registry = ModelRegistry(cfg.output_root)
        latest = registry.slot(
            cfg.track_name, cfg.algorithm, "latest", track_slug=cfg.track_slug,
        )
        champion = registry.slot(
            cfg.track_name, cfg.algorithm, "champion", track_slug=cfg.track_slug,
        )
        compatible: list[tuple[Path, Any]] = []
        champion_origin = None
        if cfg.algorithm == "grtqc" and (champion / "metadata.json").is_file():
            champion_origin = TrainingConfig.from_dict(
                registry.read_metadata(champion).training_config,
            ).grtqc.training_origin
        for candidate in (champion, latest):
            if not (candidate / "metadata.json").is_file():
                continue
            metadata = registry.read_metadata(candidate)
            try:
                saved_config = TrainingConfig.from_dict(metadata.training_config)
                if saved_config.algorithm != cfg.algorithm or saved_config.track_slug != cfg.track_slug:
                    continue
                if cfg.algorithm == "grtqc" and (
                    champion_origin is not None
                    and saved_config.grtqc.training_origin != champion_origin
                ):
                    continue
                registry.validate(
                    metadata, saved_config,
                    backend_for(saved_config.algorithm).action_adapter(saved_config).schema,
                )
            except (TypeError, ValueError, KeyError):
                # Resume uses the saved model configuration, not GUI defaults.
                # Keep competing learners in the champion's experiment family.
                continue
            compatible.append((candidate, metadata.evaluation))
        if not compatible:
            raise ValueError("No compatible champion or latest checkpoint exists for this profile")
        compatible.sort(
            key=lambda item: EvaluationResult(**item[1]).rank() if item[1] else (0,),
            reverse=True,
        )
        return compatible[0][0]

    @staticmethod
    def _resume_configuration(cfg: TrainingConfig, metadata: Any) -> TrainingConfig:
        """Restore checkpoint architecture and learning settings before loading its weights."""
        saved = TrainingConfig.from_dict(metadata.training_config)
        if saved.algorithm != cfg.algorithm or saved.track_slug != cfg.track_slug:
            raise ValueError("selected checkpoint does not match the current algorithm and track")
        if saved.backend != cfg.backend:
            raise ValueError("select the checkpoint's simulator backend before continuing it")
        # The user may choose a new session budget, output/log destinations or
        # compute device. All policy and replay semantics come from the model.
        saved.timesteps = cfg.timesteps
        if saved.curriculum.mode == "custom":
            phases = saved.curriculum.phases
            if cfg.timesteps < len(phases):
                raise ValueError("training budget is smaller than the saved curriculum's phase count")
            total = sum(phase.steps for phase in phases)
            if total != cfg.timesteps:
                available = cfg.timesteps - len(phases)
                cumulative = previous = 0
                resized = []
                for phase in phases:
                    cumulative += phase.steps
                    allocated = available * cumulative // total
                    resized.append(replace(phase, steps=1 + allocated - previous))
                    previous = allocated
                saved.curriculum.phases = tuple(resized)
        saved.output_root = cfg.output_root
        saved.log_root = cfg.log_root
        # PWM decisions can diverge across CPU/CUDA floating-point kernels.
        # Auto continuation keeps the device on which this actor was verified.
        saved.device = getattr(metadata, "device", saved.device) if cfg.device == "auto" else cfg.device
        saved.visual_replay_enabled = cfg.visual_replay_enabled
        saved.visual_replay_sample_hz = cfg.visual_replay_sample_hz
        saved.visual_replay_observations = cfg.visual_replay_observations
        return saved

    def _start_polish(self, steps: int) -> None:
        if self._selected_track().slug != "summer-1":
            self._error("The safe-polish profile is specific to Summer 1; select Summer 1 to use it.")
            return
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
        if self._selected_track().slug != "summer-1":
            self._error("This saved adaptation preset is specific to Summer 1.")
            return
        path = Path("profiles/training/summer-1-tqc-tuned-adaptation.json")
        try:
            self.load_configuration(TrainingConfig.from_dict(
                json.loads(path.read_text(encoding="utf-8-sig"))
            ))
            self.log.append("Loaded tuned champion adaptation preset (actor 1e-6, critic 5e-5).")
        except (OSError, ValueError, KeyError) as exc:
            self._error(str(exc))

    def _start_adaptation(self, stage: str) -> None:
        if not self._require_idle():
            return
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
            log_directory = self._workspace_for_config(cfg).algorithm_logs(cfg.algorithm)
            config_path = log_directory / f"tuned-adaptation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.json"
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

        if command != "snapshot" and not self._require_idle():
            return
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
            log_directory = self._workspace_for_config(cfg).algorithm_logs(cfg.algorithm)
            log_directory.mkdir(parents=True, exist_ok=True)
            config_path = log_directory / f"distillation-config-{stamp}.json"
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
        if not self._require_idle():
            return
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
        self.teacher_student_stop_file = None
        args = [
            "-m", "polybot.training.teacher_student", "--stage", stage,
            "--teacher", teacher, "--dataset", dataset,
            "--laps", str(self.teacher_student_laps.value()),
            "--timesteps", str(self.teacher_student_timesteps.value()),
            "--device", str(_value(self.general["device"])),
        ]
        if stage == "dagger":
            stop_path = Path("runs") / "teacher-student" / self._selected_track().slug / "dagger.stop"
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
        self.tabs.setCurrentWidget(self.models_page)

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
        if not self._require_idle():
            return
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
                registry.slot(
                    cfg.track_name, cfg.algorithm, "champion", track_slug=cfg.track_slug,
                ) if pace_polish
                else self._best_resume_slot(cfg) if best
                else registry.slot(
                    cfg.track_name, cfg.algorithm, "latest", track_slug=cfg.track_slug,
                )
            )
            if cfg.algorithm == "grtqc" and not resume:
                slot = registry.algorithm_dir(
                    cfg.track_name, "grtqc", track_slug=cfg.track_slug,
                ) / "initialization"
                resume = True
            resume_metadata = None
            saved_replay = slot / "replay.pkl"
            fresh_replay = False
            resume_warnings: list[str] = []
            if resume and cfg.algorithm in {"grtqc", "tqc"}:
                resume_metadata = registry.read_metadata(slot)
                cfg = self._resume_configuration(cfg, resume_metadata)
                registry.validate(
                    resume_metadata, cfg,
                    backend_for(cfg.algorithm).action_adapter(cfg).schema,
                )
                self.load_configuration(cfg)
                self.log.append(
                    "Loaded the checkpoint's policy, critic, curriculum, and reward settings; kept the selected step budget."
                )
                saved_rewards = resume_metadata.training_config.get("rewards")
                if isinstance(saved_rewards, dict) and saved_rewards != cfg.to_dict()["rewards"]:
                    self._load_saved_replay_rewards(saved_rewards)
                    cfg = self.configuration()
                    self.log.append(
                        "Loaded the reward settings saved with this checkpoint."
                    )
                if not saved_replay.is_file():
                    fresh_replay = True
                    self.log.append(
                        "No replay buffer was saved with this checkpoint; a new buffer will be collected."
                    )
                elif resume_metadata.reward_semantics != REWARD_SEMANTICS:
                    warning = (
                        "The saved replay contains rewards from older semantics. Its saved reward settings "
                        "are loaded and the replay is preserved; newly collected rewards use current "
                        "semantics, so the buffer may contain mixed reward versions."
                    )
                    resume_warnings.append(warning)
                    self.log.append(f"WARNING: {warning}")
            warnings = [*configuration_warnings(cfg), *resume_warnings]
            self.warnings.setText("\n".join(warnings) if warnings else "Settings look reasonable.")
            self.tabs.setCurrentIndex(self.tabs.count() - 1)
            self.runner = TrainingRunner(cfg, self.bridge.event.emit)
            self.session_progress.setValue(0)
            self._session_start_step = None
            self.feedback_bar.hide()
            self._stop_requested = False
            self.runner.set_ai_overlay_settings(self.ai_overlay_settings)
            self.worker = threading.Thread(
                target=self._run_worker,
                args=(slot if resume else None, fresh_replay, best or pace_polish, pace_polish),
                daemon=True,
            )
            self.worker.start()
            self._refresh_activity()
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
        self._stop_requested = True
        if self.section_optimizer_process is not None and self.section_optimizer_process.state() != QProcess.NotRunning:
            self._stop_section_optimizer()
            self._refresh_activity()
            return
        if self.teacher_student_process is not None and self.teacher_student_process.state() != QProcess.NotRunning:
            self._stop_teacher_student_after_round()
            self._refresh_activity()
            return
        for process in tuple(self.model_command_processes):
            if process.state() != QProcess.NotRunning:
                process.setProperty("polybot_cancelled", True)
                process.terminate()
                self.log.append(f"Cancelling {process.property('polybot_command')}.")
        if self.wr_search_process is not None and self.wr_search_process.state() != QProcess.NotRunning:
            self._stop_wr_search()
            return
        if self.speed_search_process is not None and self.speed_search_process.state() != QProcess.NotRunning:
            assert self.speed_search_stop_file is not None
            self.speed_search_stop_file.write_text("stop\n", encoding="utf-8")
            self.log.append("Stopping speed search after the current candidate; champion remains saved.")
            return
        if self.runner is not None and self.worker is not None and self.worker.is_alive():
            self.runner.stop()
            self.log.append("Stopping after the current simulator step; latest will be saved.")
        self._refresh_activity()

    def _start_wr_search(self, *, analyze_only: bool = False) -> None:
        if not self._require_idle():
            return
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
            champion = ModelRegistry(cfg.output_root).slot(
                cfg.track_name, "tqc", "champion", track_slug=cfg.track_slug,
            )
            if not (champion / "metadata.json").is_file():
                raise FileNotFoundError("No evaluated TQC champion is saved for this track")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            log_directory = self._workspace_for_config(cfg).algorithm_logs(cfg.algorithm)
            log_directory.mkdir(parents=True, exist_ok=True)
            config_path = log_directory / f"wr-pace-config-{stamp}.json"
            stop_path = log_directory / f"wr-pace-stop-{stamp}.txt"
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
        if not self._require_idle():
            return
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
            champion = ModelRegistry(cfg.output_root).slot(
                cfg.track_name, "tqc", "champion", track_slug=cfg.track_slug,
            )
            if not (champion / "metadata.json").is_file():
                raise FileNotFoundError("No evaluated TQC champion is saved for this track")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            log_directory = self._workspace_for_config(cfg).algorithm_logs(cfg.algorithm)
            log_directory.mkdir(parents=True, exist_ok=True)
            config_path = log_directory / f"section-optimizer-config-{stamp}.json"
            stop_path = log_directory / f"section-optimizer-stop-{stamp}.txt"
            skip_path = log_directory / f"section-optimizer-skip-{stamp}.txt"
            refine_path = log_directory / f"section-optimizer-refine-{stamp}.txt"
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
        if self._selected_track().slug != "summer-1":
            self._error("The saved WR profile applies only to Summer 1.")
            return
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
        if not self._require_idle():
            return
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
            champion = ModelRegistry(cfg.output_root).slot(
                cfg.track_name, "tqc", "champion", track_slug=cfg.track_slug,
            )
            if not (champion / "metadata.json").is_file():
                raise FileNotFoundError("No evaluated TQC champion is saved for this track")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            log_directory = self._workspace_for_config(cfg).algorithm_logs(cfg.algorithm)
            log_directory.mkdir(parents=True, exist_ok=True)
            config_path = log_directory / f"speed-search-config-{stamp}.json"
            log_path = log_directory / f"speed-search-{stamp}.jsonl"
            self.speed_search_stop_file = log_directory / f"speed-search-stop-{stamp}.txt"
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
        candidates = [
            path for path in self._workspace_for_config(cfg).list_log_files("tqc")
            if "speed-search" in path.name
        ]
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
            candidates = [
                path for path in self._workspace().list_log_files("tqc")
                if "speed-search" in path.name
            ]
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

    def _model_command(self, command: str, slot: str, *, bootstrap: bool = False) -> None:
        if not self._require_idle():
            return
        try:
            cfg = self.configuration()
            directory = ModelRegistry(cfg.output_root).slot(
                cfg.track_name, cfg.algorithm, slot, track_slug=cfg.track_slug,
            )
            if not (directory / "metadata.json").is_file():
                raise ValueError(
                    f"No {cfg.algorithm.upper()} {slot} is saved for {cfg.track_name}. "
                    "Choose the matching algorithm in Run setup or train a model first."
                )
            args = [sys.executable, "-m", "polybot", command, "--algorithm", cfg.algorithm,
                    "--track-name", cfg.track_name, "--slot", slot,
                    "--output-root", str(cfg.output_root)]
            if command == "drive":
                args.append("--realtime")
            if bootstrap:
                args.extend(["--bootstrap-reference", "--episodes", "5"])
            if command == "evaluate":
                args.append("--record-replays")
            process = QProcess(self)
            process.setProgram(args[0])
            process.setArguments(args[1:])
            process.setWorkingDirectory(str(Path.cwd()))
            process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
            process.setProperty("polybot_command", command)
            process.setProperty("polybot_slot", slot)
            process.setProperty("polybot_stdout", "")
            process.setProperty("polybot_stderr", "")
            process.started.connect(lambda p=process: self._model_command_started(p))
            process.readyReadStandardOutput.connect(lambda p=process: self._model_command_output(p, False))
            process.readyReadStandardError.connect(lambda p=process: self._model_command_output(p, True))
            process.errorOccurred.connect(lambda error, p=process: self._model_command_error(p, error))
            process.finished.connect(
                lambda code, status, p=process: self._model_command_finished(p, code, status)
            )
            self.model_command_processes.add(process)
            self.feedback_bar.hide()
            process.start()
            self._refresh_activity()
        except (OSError, ValueError) as exc:
            self._error(str(exc))

    def _model_command_started(self, process: QProcess) -> None:
        command = process.property("polybot_command")
        slot = process.property("polybot_slot")
        message = f"Launched {command} for {slot} model."
        if command == "drive":
            message += " Waiting for the simulator bridge."
        self.log.append(message)
        self.tabs.setCurrentIndex(self.tabs.count() - 1)

    def _model_command_output(self, process: QProcess, stderr: bool) -> None:
        channel = process.readAllStandardError() if stderr else process.readAllStandardOutput()
        text = bytes(channel).decode("utf-8", errors="replace")
        key = "polybot_stderr" if stderr else "polybot_stdout"
        accumulated = str(process.property(key) or "") + text
        process.setProperty(key, accumulated[-16_000:])
        if text.strip():
            level = "error" if stderr else "output"
            self.log.append(
                f"{process.property('polybot_command')} {level}: {text.strip()[-1200:]}"
            )

    def _model_command_error(self, process: QProcess, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._error(f"Could not start {process.property('polybot_command')}: {process.errorString()}")
            self.model_command_processes.discard(process)
            process.deleteLater()

    def _model_command_finished(
        self, process: QProcess, exit_code: int, _status: QProcess.ExitStatus,
    ) -> None:
        self._model_command_output(process, False)
        self._model_command_output(process, True)
        command = process.property("polybot_command")
        if process.property("polybot_cancelled"):
            self.log.append(f"{command} cancelled.")
        elif exit_code:
            detail = str(process.property("polybot_stderr") or "").strip()
            if not detail:
                detail = str(process.property("polybot_stdout") or "").strip()
            self._error(f"{command} exited with code {exit_code}: {detail[-1600:]}")
        elif command != "drive":
            output = str(process.property("polybot_stdout") or "").strip()
            if output:
                self.log.append(output[-1600:])
        self.model_command_processes.discard(process)
        process.deleteLater()

    def _event(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "started":
            try:
                cfg = self.configuration()
                slot = ModelRegistry(cfg.output_root).slot(
                    cfg.track_name, cfg.algorithm, "champion", track_slug=cfg.track_slug,
                )
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
            budget = self.runner.config.timesteps if self.runner is not None else int(_value(self.general["timesteps"]))
            if self._session_start_step is None:
                # The first callback follows one decision; resumed counters include past sessions.
                self._session_start_step = max(0, int(event["timesteps"]) - 1)
            completed = max(0, int(event["timesteps"]) - self._session_start_step)
            self.session_progress.setValue(min(1000, int(completed * 1000 / max(1, budget))))
            self.session_progress.setToolTip(f"{completed:,} / {budget:,} decisions in this session")
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
        self.feedback_label.setText(f"Could not complete the action.\n{message}")
        self.feedback_bar.show()

    def _dismiss_feedback(self) -> None:
        self._close_when_idle = False
        self.dismiss_feedback_button.setText("Dismiss")
        self.feedback_bar.hide()

    def closeEvent(self, event: Any) -> None:
        if self._active_operations():
            self._close_when_idle = True
            if self.stop_button.isEnabled():
                self._stop()
            self.feedback_label.setText("Closing when the active task finishes. Training and searches stop safely first.")
            self.dismiss_feedback_button.setText("Keep open")
            self.feedback_bar.show()
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


if __name__ == "__main__":
    raise SystemExit(main())
