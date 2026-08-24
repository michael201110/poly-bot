"""Responsive PySide6 front end for :mod:`polybot.training`."""

from __future__ import annotations

import sys
import threading
from dataclasses import replace
from pathlib import Path

from polybot.env import summer_1_pace_reward_config, summer_1_reward_config
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
            QScrollArea,
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
            self.rollout_steps = QSpinBox()
            self.rollout_steps.setRange(2, 1_000_000)
            self.rollout_steps.setValue(2048)
            self.batch_size = QSpinBox()
            self.batch_size.setRange(2, 1_000_000)
            self.batch_size.setValue(256)
            self.ppo_epochs = QSpinBox()
            self.ppo_epochs.setRange(1, 100)
            self.ppo_epochs.setValue(5)
            self.reward_profile = QComboBox()
            self.reward_profile.addItems(["Summer 1 - balanced", "Summer 1 - pace"])
            self.ghost_reward = QDoubleSpinBox()
            self.ghost_reward.setRange(0, 1_000_000)
            self.ghost_reward.setValue(10)
            self.barrier_penalty = QDoubleSpinBox()
            self.barrier_penalty.setRange(-1_000_000, 0)
            self.barrier_penalty.setValue(-150)
            self.finish_bonus = QDoubleSpinBox()
            self.finish_bonus.setRange(0, 1_000_000)
            self.finish_bonus.setValue(1200)
            self.finish_fast_bonus = QDoubleSpinBox()
            self.finish_fast_bonus.setRange(0, 1_000_000)
            self.finish_fast_bonus.setValue(1800)
            self.finish_target = QDoubleSpinBox()
            self.finish_target.setRange(0.001, 100_000)
            self.finish_target.setValue(22)
            self.finish_decay = QDoubleSpinBox()
            self.finish_decay.setRange(0.001, 1000)
            self.finish_decay.setValue(0.35)
            self.slip_penalty = QDoubleSpinBox()
            self.slip_penalty.setRange(-1_000_000, 0)
            self.slip_penalty.setValue(-30)
            self.ground_brake_penalty = QDoubleSpinBox()
            self.ground_brake_penalty.setRange(-1000, 0)
            self.ground_brake_penalty.setValue(-2)
            self.speed_carry = QDoubleSpinBox()
            self.speed_carry.setRange(0, 1000)
            self.speed_carry.setValue(0)
            self.curriculum = QComboBox()
            self.curriculum.addItems(["full", "quarters", "quarters-randomised", "timed"])
            self.checkpoint = QSpinBox()
            self.checkpoint.setRange(0, 100_000_000)
            self.checkpoint.setValue(10_000)
            self.output = QLineEdit("models")
            self.parameters = QLabel()
            self.runtime = QLabel("Idle")
            self.overview = QLabel("No episode data yet")
            self.overview.setWordWrap(True)
            self.overview.setStyleSheet(
                "font-family: Consolas, monospace; padding: 8px; "
                "background: palette(alternate-base); border: 1px solid palette(mid);"
            )
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
                ("Rollout steps", self.rollout_steps),
                ("Batch size", self.batch_size),
                ("PPO epochs", self.ppo_epochs),
                ("Reward profile", self.reward_profile),
                ("Ghost pose reward/s", self.ghost_reward),
                ("Barrier penalty", self.barrier_penalty),
                ("Finish bonus", self.finish_bonus),
                ("Fast finish bonus", self.finish_fast_bonus),
                ("Finish target (s)", self.finish_target),
                ("Finish pace decay", self.finish_decay),
                ("Ground slip penalty", self.slip_penalty),
                ("Ground braking penalty / s", self.ground_brake_penalty),
                ("Checkpoint speed carry / m/s", self.speed_carry),
                ("Curriculum", self.curriculum),
                ("Checkpoint interval", self.checkpoint),
                ("Model location", self.output),
                ("Training status", self.runtime),
            ]
            for label, widget in fields:
                form.addRow(label, widget)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.document().setMaximumBlockCount(500)
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
            layout.addWidget(self.overview)
            layout.addWidget(self.log)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(root)
            self.setCentralWidget(scroll)
            self.resize(780, 800)
            start.clicked.connect(lambda: self.start(False))
            resume.clicked.connect(lambda: self.start(True))
            stop.clicked.connect(self.stop)
            browse.clicked.connect(self.browse)
            self.track.currentTextChanged.connect(self.refresh_models)
            self.arch.currentTextChanged.connect(self.refresh_parameters)
            self.pwm.toggled.connect(self.refresh_parameters)
            self.levels.valueChanged.connect(self.refresh_parameters)
            self.reward_profile.currentTextChanged.connect(self.apply_reward_profile)
            self.refresh_models()
            self.refresh_parameters()

        def selected_reward_profile(self):
            if self.reward_profile.currentText() == "Summer 1 - pace":
                return summer_1_pace_reward_config()
            return summer_1_reward_config()

        def apply_reward_profile(self) -> None:
            rewards = self.selected_reward_profile()
            self.ghost_reward.setValue(rewards.imitation_bonus_per_s)
            self.barrier_penalty.setValue(rewards.barrier_contact_penalty)
            self.finish_bonus.setValue(rewards.finish_bonus)
            self.finish_fast_bonus.setValue(rewards.finish_fast_bonus)
            self.finish_target.setValue(rewards.finish_target_s)
            self.finish_decay.setValue(rewards.finish_pace_decay_per_s)
            self.slip_penalty.setValue(rewards.ground_slip_penalty_per_rad_s)
            self.ground_brake_penalty.setValue(rewards.ground_brake_penalty_per_s)
            self.speed_carry.setValue(rewards.checkpoint_speed_bonus_per_mps)

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
                rollout_steps=self.rollout_steps.value(),
                batch_size=self.batch_size.value(),
                ppo_epochs=self.ppo_epochs.value(),
                checkpoint_interval=self.checkpoint.value(),
                output_root=Path(self.output.text()),
                rewards=replace(
                    self.selected_reward_profile(),
                    imitation_bonus_per_s=self.ghost_reward.value(),
                    barrier_contact_penalty=self.barrier_penalty.value(),
                    finish_bonus=self.finish_bonus.value(),
                    finish_fast_bonus=self.finish_fast_bonus.value(),
                    finish_target_s=self.finish_target.value(),
                    finish_pace_decay_per_s=self.finish_decay.value(),
                    ground_slip_penalty_per_rad_s=self.slip_penalty.value(),
                    ground_brake_penalty_per_s=self.ground_brake_penalty.value(),
                    checkpoint_speed_bonus_per_mps=self.speed_carry.value(),
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
            event_type = event.get("type")
            if event_type == "started":
                device_label = f"{event['device']} {event.get('gpu_name') or ''}".strip()
                self.runtime.setText(f"{device_label} — {event['parameter_count']:,} parameters")
                self.log.appendPlainText(
                    f"Started on {device_label}: {event['parameter_count']:,} parameters"
                )
            elif event_type == "progress":
                steps = event.get("timesteps", 0)
                elapsed = event.get("elapsed_s", 0)
                sps = event.get("steps_per_second", 0)
                self.runtime.setText(
                    f"Step {steps:,} | {sps:,.0f} steps/sec | Episode {event['episode']}"
                )
                best = event.get("best_lap_s")
                best_text = f"{best:.3f}s" if best is not None else "--"
                quarter = event.get("quarter")
                section_text = f"    Quarter: {quarter}" if quarter else ""
                self.overview.setText(
                    f"CURRENT EPISODE {event['episode']}\n"
                    f"Reward: {event['episode_reward']:+.1f}    "
                    f"Progress: {event['max_progress']:.1%}    "
                    f"Time: {elapsed:.2f}s    Speed: {event['speed_kmh']:.1f} km/h"
                    f"{section_text}\n"
                    f"Steps: {event['episode_steps']:,}    "
                    f"Finishes: {event['finishes']}    Crashes: {event['crashes']}    "
                    f"Best lap: {best_text}"
                )
            elif event_type == "episode":
                best = event.get("best_lap_s")
                best_text = f"{best:.3f}s" if best is not None else "--"
                quarter = f" Q{event['quarter']}" if event.get("quarter") else ""
                self.log.appendPlainText(
                    f"Episode {event['episode']:>4}{quarter}: {event['result']:<16} "
                    f"reward={event['reward']:+9.1f}  progress={event['progress']:>6.1%}  "
                    f"time={event['elapsed_s']:>7.2f}s  steps={event['steps']:>5}  "
                    f"best={best_text}"
                )
            elif event_type == "checkpoint":
                self.log.appendPlainText(f"Checkpoint saved at timestep {event['timesteps']:,}")
            elif event_type == "error":
                self.log.appendPlainText(f"ERROR: {event.get('message', 'unknown error')}")
            elif event_type in {"archived", "completed", "stopped"}:
                self.log.appendPlainText(str(event))
            elif event_type == "idle":
                self.manager = None
                self.runtime.setText("Idle")
                self.refresh_models()

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec()
