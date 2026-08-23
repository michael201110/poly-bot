"""Responsive PySide6 front end for :mod:`polybot.training`."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from polybot.env import RewardConfig
from polybot.protocol import Telemetry
from polybot.training.config import TrainingConfig, estimate_ppo_parameters
from polybot.training.devices import resolve_device
from polybot.training.manager import TrainingManager
from polybot.training.models import ModelRegistry


def main() -> int:
    try:
        from PySide6.QtCore import QObject, Signal
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print(
            "The GUI dependency is missing; install with: pip install -e '.[train,gui]'",
            file=sys.stderr,
        )
        return 2

    class Events(QObject):
        update = Signal(dict)

    class Window(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("PolyBot Training")
            self.events = Events()
            self.events.update.connect(self.on_status)
            self.manager = None
            root, form, buttons = QWidget(), QFormLayout(), QHBoxLayout()
            self.track = QComboBox()
            self.track.setEditable(True)
            self.track.addItems(["Summer 1"])
            self.backend = QComboBox()
            self.backend.addItems(["websocket", "mock"])
            self.model = QComboBox()
            self.model.setEditable(True)
            self.model.addItem("New model")
            self.arch = QComboBox()
            self.arch.addItems(["xl", "large", "medium", "legacy"])
            self.device = QComboBox()
            self.device.addItems(["auto", "cpu", "cuda"])
            self.pwm = QCheckBox()
            self.pwm.setChecked(True)
            self.levels = QSpinBox()
            self.levels.setRange(3, 201)
            self.levels.setSingleStep(2)
            self.levels.setValue(41)
            self.frame_skip = QSpinBox()
            self.frame_skip.setRange(1, 1000)
            self.frame_skip.setValue(30)
            self.timesteps = QSpinBox()
            self.timesteps.setRange(1, 2_000_000_000)
            self.timesteps.setValue(100_000)
            self.episodes = QSpinBox()
            self.episodes.setRange(0, 10_000_000)
            self.lr = QDoubleSpinBox()
            self.lr.setDecimals(7)
            self.lr.setRange(0.0000001, 1)
            self.lr.setValue(0.0005)
            self.gamma = QDoubleSpinBox()
            self.gamma.setDecimals(5)
            self.gamma.setRange(0, 1)
            self.gamma.setValue(0.999)
            self.gae = QDoubleSpinBox()
            self.gae.setDecimals(5)
            self.gae.setRange(0, 1)
            self.gae.setValue(0.98)
            self.entropy = QDoubleSpinBox()
            self.entropy.setDecimals(5)
            self.entropy.setRange(-0.1, 0.1)
            self.ghost_reward = QDoubleSpinBox()
            self.ghost_reward.setRange(0, 1_000_000)
            self.ghost_reward.setValue(18)
            self.barrier_penalty = QDoubleSpinBox()
            self.barrier_penalty.setRange(-1_000_000, 0)
            self.barrier_penalty.setValue(-50)
            self.finish_bonus = QDoubleSpinBox()
            self.finish_bonus.setRange(0, 1_000_000)
            self.finish_bonus.setValue(1000)
            self.finish_fast_bonus = QDoubleSpinBox()
            self.finish_fast_bonus.setRange(0, 1_000_000)
            self.finish_fast_bonus.setValue(2000)
            self.finish_target = QDoubleSpinBox()
            self.finish_target.setRange(0.001, 100_000)
            self.finish_target.setValue(22)
            self.finish_decay = QDoubleSpinBox()
            self.finish_decay.setRange(0.001, 1000)
            self.finish_decay.setValue(1.5)
            self.slip_penalty = QDoubleSpinBox()
            self.slip_penalty.setRange(-1_000_000, 0)
            self.slip_penalty.setValue(-1000)
            self.curriculum = QComboBox()
            self.curriculum.addItems(["full", "quarters", "timed"])
            self.checkpoint = QSpinBox()
            self.checkpoint.setRange(0, 100_000_000)
            self.checkpoint.setValue(10_000)
            self.output = QLineEdit("models")
            self.parameters = QLabel()
            self.runtime = QLabel("Idle")
            fields = [
                ("Track/profile", self.track),
                ("Simulator backend", self.backend),
                ("Model", self.model),
                ("Architecture", self.arch),
                ("Parameters", self.parameters),
                ("Device", self.device),
                ("PWM steering", self.pwm),
                ("PWM levels", self.levels),
                ("Frame skip", self.frame_skip),
                ("Timesteps", self.timesteps),
                ("Episodes (0=unlimited)", self.episodes),
                ("Learning rate", self.lr),
                ("Gamma", self.gamma),
                ("GAE lambda", self.gae),
                ("Entropy coefficient", self.entropy),
                ("Ghost pose reward/s", self.ghost_reward),
                ("Barrier penalty", self.barrier_penalty),
                ("Finish bonus", self.finish_bonus),
                ("Fast finish bonus", self.finish_fast_bonus),
                ("Finish target (s)", self.finish_target),
                ("Finish pace decay", self.finish_decay),
                ("Ground slip penalty", self.slip_penalty),
                ("Curriculum", self.curriculum),
                ("Checkpoint interval", self.checkpoint),
                ("Model location", self.output),
                ("Training status", self.runtime),
            ]
            for label, widget in fields:
                form.addRow(label, widget)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            start, stop, resume, browse = (
                QPushButton("Start training"),
                QPushButton("Stop cleanly"),
                QPushButton("Resume selected"),
                QPushButton("Browse…"),
            )
            for button in (start, stop, resume, browse):
                buttons.addWidget(button)
            layout = QVBoxLayout(root)
            layout.addLayout(form)
            layout.addLayout(buttons)
            layout.addWidget(self.log)
            self.setCentralWidget(root)
            self.resize(760, 760)
            start.clicked.connect(lambda: self.start(False))
            resume.clicked.connect(lambda: self.start(True))
            stop.clicked.connect(self.stop)
            browse.clicked.connect(self.browse)
            self.track.currentTextChanged.connect(self.refresh_models)
            self.arch.currentTextChanged.connect(self.refresh_parameters)
            self.pwm.toggled.connect(self.refresh_parameters)
            self.levels.valueChanged.connect(self.refresh_parameters)
            self.refresh_models()
            self.refresh_parameters()

        def refresh_parameters(self) -> None:
            dims = (self.levels.value() if self.pwm.isChecked() else 3, 2, 2)
            count = estimate_ppo_parameters(
                Telemetry.vector_size(12), dims, self.arch.currentText()
            )
            self.parameters.setText(f"{count:,} (pre-creation estimate)")

        def refresh_models(self) -> None:
            current = self.model.currentText()
            self.model.clear()
            self.model.addItem("New model")
            for path in ModelRegistry(self.output.text() or "models").list_models(
                self.track.currentText()
            ):
                self.model.addItem(str(path))
            if current:
                self.model.setCurrentText(current)

        def browse(self) -> None:
            value = QFileDialog.getExistingDirectory(
                self, "Model output directory", self.output.text()
            )
            if value:
                self.output.setText(value)
                self.refresh_models()

        def config(self) -> TrainingConfig:
            return TrainingConfig(
                backend=self.backend.currentText(),
                track_name=self.track.currentText(),
                architecture=self.arch.currentText(),
                device=self.device.currentText(),
                pwm_enabled=self.pwm.isChecked(),
                pwm_levels=self.levels.value(),
                frame_skip=self.frame_skip.value(),
                timesteps=self.timesteps.value(),
                max_episodes=self.episodes.value() or None,
                learning_rate=self.lr.value(),
                gamma=self.gamma.value(),
                gae_lambda=self.gae.value(),
                entropy_coefficient=self.entropy.value(),
                checkpoint_interval=self.checkpoint.value(),
                output_root=Path(self.output.text()),
                rewards=RewardConfig(
                    imitation_bonus_per_s=self.ghost_reward.value(),
                    barrier_contact_penalty=self.barrier_penalty.value(),
                    finish_bonus=self.finish_bonus.value(),
                    finish_fast_bonus=self.finish_fast_bonus.value(),
                    finish_target_s=self.finish_target.value(),
                    finish_pace_decay_per_s=self.finish_decay.value(),
                    ground_slip_penalty_per_rad_s=self.slip_penalty.value(),
                ),
            )

        def start(self, resume: bool) -> None:
            if self.manager:
                return
            try:
                resolve_device(self.device.currentText())
            except RuntimeError as exc:
                QMessageBox.critical(self, "Device unavailable", str(exc))
                return
            cfg = self.config()
            cfg.curriculum.mode = self.curriculum.currentText()
            self.manager = TrainingManager(cfg, self.events.update.emit)
            selected = (
                self.model.currentText()
                if resume and self.model.currentText() != "New model"
                else None
            )
            threading.Thread(target=self._run, args=(selected,), daemon=True).start()
            self.runtime.setText("Starting…")

        def _run(self, selected: str | None) -> None:
            try:
                self.manager.run(resume=selected)
            except Exception as exc:
                self.events.update.emit({"type": "error", "message": str(exc)})
            finally:
                self.events.update.emit({"type": "idle"})

        def stop(self) -> None:
            if self.manager:
                self.manager.stop()
                self.runtime.setText("Stopping after current PPO step…")

        def on_status(self, event: dict) -> None:
            self.log.appendPlainText(str(event))
            if event.get("type") == "started":
                device_label = f"{event['device']} {event.get('gpu_name') or ''}".strip()
                self.runtime.setText(f"{device_label} — {event['parameter_count']:,} parameters")
            elif event.get("type") == "progress":
                steps = event.get("timesteps", 0)
                elapsed = event.get("elapsed_s", 0)
                self.runtime.setText(
                    f"Step {steps:,}; lap {elapsed:.3f}s; events {event.get('events', ())}"
                )
            elif event.get("type") == "idle":
                self.manager = None
                self.runtime.setText("Idle")
                self.refresh_models()

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec()
