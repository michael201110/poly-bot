"""Responsive PySide6 front end for :mod:`polybot.training`."""

from __future__ import annotations

import argparse
import sys
import threading
from dataclasses import fields
from pathlib import Path

from polybot.env import RewardConfig
from polybot.protocol import Telemetry
from polybot.training.config import TrainingConfig, estimate_ppo_parameters
from polybot.training.devices import resolve_device
from polybot.training.manager import TrainingManager
from polybot.training.models import ModelRegistry
from polybot.training.reward_profiles import RewardProfileStore


def preferred_model(models: list[Path], current: str = "") -> Path | None:
    """Keep a valid selection, otherwise prefer the track's latest model."""

    current_path = Path(current) if current.strip() else None
    if current_path is not None and current_path in models:
        return current_path
    return next((path for path in models if path.name.casefold() == "latest.zip"), None)


def playback_model_path(output_root: str | Path, track_name: str, name: str) -> Path:
    """Resolve one of the two GUI playback slots for a track."""

    if name not in {"latest", "best"}:
        raise ValueError("playback model name must be latest or best")
    return ModelRegistry(output_root).track_dir(track_name) / f"{name}.zip"


def realtime_drive_arguments(
    model: str | Path,
    *,
    frame_skip: int,
    pwm_enabled: bool,
    pwm_levels: int,
    device: str,
) -> list[str]:
    """Build deterministic one-lap playback arguments paced at simulation time."""

    return [
        "--model",
        str(model),
        "--episodes",
        "1",
        "--frame-skip",
        str(frame_skip),
        "--pwm" if pwm_enabled else "--no-pwm",
        "--pwm-levels",
        str(pwm_levels),
        "--device",
        device,
    ]


def timed_curriculum_bounds(mode: str, start_s: float, end_s: float) -> tuple[float, float] | None:
    """Return validated timed bounds, or no bounds for another curriculum mode."""

    if mode != "timed":
        return None
    if not 0 <= start_s < end_s:
        raise ValueError("timed curriculum must satisfy 0 <= start < end")
    return start_s, end_s


def reward_breakdown(terms: dict[str, float]) -> str:
    """Group detailed reward terms into a compact episode summary."""

    groups = {
        "drive": {"progress", "on_track_speed", "airborne_speed", "takeoff_speed"},
        "ghost": {"ghost_imitation"},
        "milestone": {"checkpoint", "finish", "curriculum_section"},
        "control": {"elapsed", "action_change", "ground_brake", "airborne_brake"},
        "handling": {
            "unsafe_speed",
            "airborne_spin",
            "airborne_tilt",
            "ground_slip",
            "off_track_landing",
        },
        "terminal": {
            "barrier_contact",
            "failure_progress_clawback",
            "failure_early",
            "airborne_roll_failure",
            "crash",
            "stall",
            "off_track",
        },
    }
    totals = {
        label: sum(float(terms.get(name, 0.0)) for name in names)
        for label, names in groups.items()
    }
    return "  ".join(f"{label}={value:+.1f}" for label, value in totals.items())


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--track-name")
    parser.add_argument("--model")
    parser.add_argument("--architecture", choices=["legacy", "medium", "large", "xl"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--frame-skip", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--entropy-coefficient", type=float)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument("--teacher-model")
    parser.add_argument("--teacher-kl-coefficient", type=float)
    parser.add_argument("--reward-profile")
    parser.add_argument(
        "--curriculum",
        choices=["full", "quarters", "quarters-from-q4", "quarters-randomised", "timed"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    launch, qt_arguments = parser.parse_known_args(sys.argv[1:])
    try:
        from PySide6.QtCore import QObject, QProcess, QTimer, Signal
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
            QTableWidget,
            QTableWidgetItem,
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
            self.playback = None
            self.pending_playback: Path | None = None
            root, form, buttons, playback_buttons = (
                QWidget(),
                QFormLayout(),
                QHBoxLayout(),
                QHBoxLayout(),
            )
            self.track = QComboBox()
            self.track.setEditable(True)
            self.track.addItems(["Summer 1", "Winter 4"])
            if launch.track_name:
                self.track.setCurrentText(launch.track_name)
            self.backend = QComboBox()
            self.backend.addItems(["websocket", "mock"])
            self.model = QComboBox()
            self.model.setEditable(True)
            self.model.lineEdit().setPlaceholderText("No saved model found")
            self.arch = QComboBox()
            self.arch.addItems(["xl", "large", "medium", "legacy"])
            if launch.architecture:
                self.arch.setCurrentText(launch.architecture)
            self.device = QComboBox()
            self.device.addItems(["auto", "cpu", "cuda"])
            if launch.device:
                self.device.setCurrentText(launch.device)
            self.pwm = QCheckBox()
            self.pwm.setChecked(True)
            self.levels = QSpinBox()
            self.levels.setRange(3, 201)
            self.levels.setSingleStep(2)
            self.levels.setValue(41)
            self.frame_skip = QSpinBox()
            self.frame_skip.setRange(1, 1000)
            self.frame_skip.setValue(30)
            self.max_episode_seconds = QDoubleSpinBox()
            self.max_episode_seconds.setRange(1.0, 3600.0)
            self.max_episode_seconds.setValue(60.0)
            self.max_episode_seconds.setSuffix(" s")
            self.timesteps = QSpinBox()
            self.timesteps.setRange(1, 2_000_000_000)
            self.timesteps.setValue(100_000)
            if launch.timesteps is not None:
                self.timesteps.setValue(launch.timesteps)
            self.episodes = QSpinBox()
            self.episodes.setRange(0, 10_000_000)
            self.lr = QDoubleSpinBox()
            self.lr.setDecimals(7)
            self.lr.setRange(0.0000001, 1)
            self.lr.setValue(0.0001)
            self.gamma = QDoubleSpinBox()
            self.gamma.setDecimals(5)
            self.gamma.setRange(0, 1)
            self.gamma.setValue(0.9995)
            self.gae = QDoubleSpinBox()
            self.gae.setDecimals(5)
            self.gae.setRange(0, 1)
            self.gae.setValue(0.995)
            self.entropy = QDoubleSpinBox()
            self.entropy.setDecimals(5)
            self.entropy.setRange(-0.1, 0.1)
            self.entropy.setValue(0.001)
            self.rollout_steps = QSpinBox()
            self.rollout_steps.setRange(2, 1_000_000)
            self.rollout_steps.setValue(8192)
            if launch.rollout_steps is not None:
                self.rollout_steps.setValue(launch.rollout_steps)
            self.batch_size = QSpinBox()
            self.batch_size.setRange(2, 1_000_000)
            self.batch_size.setValue(1024)
            if launch.batch_size is not None:
                self.batch_size.setValue(launch.batch_size)
            self.ppo_epochs = QSpinBox()
            self.ppo_epochs.setRange(1, 100)
            self.ppo_epochs.setValue(3)
            if launch.ppo_epochs is not None:
                self.ppo_epochs.setValue(launch.ppo_epochs)
            self.teacher_model = QLineEdit(launch.teacher_model or "")
            self.teacher_model.setPlaceholderText("Optional fixed teacher .zip")
            self.teacher_kl = QDoubleSpinBox()
            self.teacher_kl.setDecimals(4)
            self.teacher_kl.setRange(0.0, 100.0)
            self.teacher_kl.setValue(
                0.0 if launch.teacher_kl_coefficient is None else launch.teacher_kl_coefficient
            )
            self.reward_profiles = RewardProfileStore()
            self.reward_profile = QComboBox()
            self.reward_profile.setEditable(True)
            self.reward_profile.addItems(self.reward_profiles.names())
            if launch.reward_profile:
                self.reward_profile.setCurrentText(launch.reward_profile)
            self.reward_table = QTableWidget(len(fields(RewardConfig)), 2)
            self.reward_table.setHorizontalHeaderLabels(["Reward parameter", "Value"])
            self.reward_table.verticalHeader().setVisible(False)
            self.reward_table.horizontalHeader().setStretchLastSection(True)
            self.reward_table.setMinimumHeight(420)
            self.reward_inputs = {}
            defaults = RewardConfig()
            for row, reward_field in enumerate(fields(RewardConfig)):
                name = reward_field.name
                self.reward_table.setItem(row, 0, QTableWidgetItem(name))
                default_value = getattr(defaults, name)
                if isinstance(default_value, int):
                    editor = QSpinBox()
                    editor.setRange(-1_000_000_000, 1_000_000_000)
                else:
                    editor = QDoubleSpinBox()
                    editor.setDecimals(6)
                    editor.setRange(-1_000_000_000, 1_000_000_000)
                self.reward_table.setCellWidget(row, 1, editor)
                self.reward_inputs[name] = editor
            self.curriculum = QComboBox()
            self.curriculum.addItems(
                ["full", "quarters", "quarters-from-q4", "quarters-randomised", "timed"]
            )
            if launch.curriculum:
                self.curriculum.setCurrentText(launch.curriculum)
            self.curriculum_start_s = QDoubleSpinBox()
            self.curriculum_start_s.setDecimals(2)
            self.curriculum_start_s.setRange(0.0, 3600.0)
            self.curriculum_start_s.setValue(0.0)
            self.curriculum_start_s.setSuffix(" s")
            self.curriculum_end_s = QDoubleSpinBox()
            self.curriculum_end_s.setDecimals(2)
            self.curriculum_end_s.setRange(0.01, 3600.0)
            self.curriculum_end_s.setValue(10.0)
            self.curriculum_end_s.setSuffix(" s")
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
            form_fields = [
                ("Track/profile", self.track),
                ("Simulator backend", self.backend),
                ("Model", self.model),
                ("Architecture", self.arch),
                ("Parameters", self.parameters),
                ("Device", self.device),
                ("PWM steering", self.pwm),
                ("PWM levels", self.levels),
                ("Frame skip", self.frame_skip),
                ("Maximum episode time", self.max_episode_seconds),
                ("Timesteps", self.timesteps),
                ("Episodes (0=unlimited)", self.episodes),
                ("Learning rate", self.lr),
                ("Gamma", self.gamma),
                ("GAE lambda", self.gae),
                ("Entropy coefficient", self.entropy),
                ("Rollout steps", self.rollout_steps),
                ("Batch size", self.batch_size),
                ("PPO epochs", self.ppo_epochs),
                ("Teacher model", self.teacher_model),
                ("Teacher KL coefficient", self.teacher_kl),
                ("Reward profile", self.reward_profile),
                ("All reward parameters", self.reward_table),
                ("Curriculum", self.curriculum),
                ("Timed section start", self.curriculum_start_s),
                ("Timed section end", self.curriculum_end_s),
                ("Checkpoint interval", self.checkpoint),
                ("Model location", self.output),
                ("Training status", self.runtime),
            ]
            for label, widget in form_fields:
                form.addRow(label, widget)
            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.document().setMaximumBlockCount(500)
            fresh, stop, resume, save_profile, browse = (
                QPushButton("Start a NEW model…"),
                QPushButton("Stop cleanly"),
                QPushButton("Resume latest / selected"),
                QPushButton("Save reward profile"),
                QPushButton("Browse…"),
            )
            resume.setDefault(True)
            fresh.setToolTip("Create a fresh policy instead of loading the selected model")
            for button in (resume, stop, fresh, save_profile, browse):
                buttons.addWidget(button)
            play_latest = QPushButton("Play Latest (1×)")
            play_best = QPushButton("Play Best (1×)")
            play_latest.setToolTip("Stop training cleanly, then drive one deterministic lap")
            play_best.setToolTip("Replay the policy snapshot that set the fastest logged lap")
            playback_buttons.addWidget(play_latest)
            playback_buttons.addWidget(play_best)
            layout = QVBoxLayout(root)
            layout.addLayout(form)
            layout.addLayout(buttons)
            layout.addLayout(playback_buttons)
            layout.addWidget(self.overview)
            layout.addWidget(self.log)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(root)
            self.setCentralWidget(scroll)
            self.resize(780, 800)
            fresh.clicked.connect(self.start_fresh)
            resume.clicked.connect(self.resume_selected)
            stop.clicked.connect(self.stop)
            save_profile.clicked.connect(self.save_reward_profile)
            browse.clicked.connect(self.browse)
            play_latest.clicked.connect(lambda: self.play_named_model("latest"))
            play_best.clicked.connect(lambda: self.play_named_model("best"))
            self.track.currentTextChanged.connect(self.refresh_models)
            self.arch.currentTextChanged.connect(self.refresh_parameters)
            self.pwm.toggled.connect(self.refresh_parameters)
            self.levels.valueChanged.connect(self.refresh_parameters)
            self.reward_profile.currentTextChanged.connect(self.apply_reward_profile)
            self.curriculum.currentTextChanged.connect(self.refresh_curriculum_controls)
            self.refresh_models()
            if launch.model:
                self.model.setCurrentText(launch.model)
            self.refresh_parameters()
            self.refresh_curriculum_controls()
            self.apply_reward_profile()
            if launch.frame_skip is not None:
                self.frame_skip.setValue(launch.frame_skip)
            if launch.learning_rate is not None:
                self.lr.setValue(launch.learning_rate)
            if launch.entropy_coefficient is not None:
                self.entropy.setValue(launch.entropy_coefficient)

        def apply_reward_profile(self) -> None:
            try:
                rewards = self.reward_profiles.load(self.reward_profile.currentText())
            except (FileNotFoundError, ValueError):
                return
            for reward_field in fields(RewardConfig):
                self.reward_inputs[reward_field.name].setValue(getattr(rewards, reward_field.name))

        def reward_config_from_table(self) -> RewardConfig:
            values = {
                reward_field.name: self.reward_inputs[reward_field.name].value()
                for reward_field in fields(RewardConfig)
            }
            return RewardConfig(**values)

        def save_reward_profile(self) -> None:
            name = self.reward_profile.currentText().strip()
            try:
                path = self.reward_profiles.save(name, self.reward_config_from_table())
            except (OSError, ValueError) as exc:
                QMessageBox.critical(self, "Could not save profile", str(exc))
                return
            current = name
            self.reward_profile.blockSignals(True)
            self.reward_profile.clear()
            self.reward_profile.addItems(self.reward_profiles.names())
            self.reward_profile.setCurrentText(current)
            self.reward_profile.blockSignals(False)
            self.log.appendPlainText(f"Saved reward profile: {path}")

        def refresh_parameters(self) -> None:
            dims = (self.levels.value() if self.pwm.isChecked() else 3, 2, 2)
            count = estimate_ppo_parameters(
                Telemetry.vector_size(12), dims, self.arch.currentText()
            )
            self.parameters.setText(f"{count:,} (pre-creation estimate)")

        def refresh_curriculum_controls(self) -> None:
            enabled = self.curriculum.currentText() == "timed"
            self.curriculum_start_s.setEnabled(enabled)
            self.curriculum_end_s.setEnabled(enabled)

        def refresh_models(self) -> None:
            current = self.model.currentText()
            models = ModelRegistry(self.output.text() or "models").list_models(
                self.track.currentText()
            )
            selected = preferred_model(models, current)
            self.model.clear()
            for path in models:
                self.model.addItem(str(path))
            if selected is not None:
                self.model.setCurrentText(str(selected))

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
                max_episode_seconds=self.max_episode_seconds.value(),
                timesteps=self.timesteps.value(),
                max_episodes=self.episodes.value() or None,
                learning_rate=self.lr.value(),
                gamma=self.gamma.value(),
                gae_lambda=self.gae.value(),
                entropy_coefficient=self.entropy.value(),
                rollout_steps=self.rollout_steps.value(),
                batch_size=self.batch_size.value(),
                ppo_epochs=self.ppo_epochs.value(),
                teacher_model=(
                    Path(self.teacher_model.text().strip())
                    if self.teacher_model.text().strip()
                    else None
                ),
                teacher_kl_coefficient=self.teacher_kl.value(),
                checkpoint_interval=self.checkpoint.value(),
                output_root=Path(self.output.text()),
                rewards=self.reward_config_from_table(),
            )

        def resume_selected(self) -> None:
            selected = self.model.currentText().strip()
            if not selected or not Path(selected).is_file():
                QMessageBox.critical(
                    self,
                    "No model selected",
                    "Select an existing .zip model before resuming. "
                    "Resume will never create a fresh model.",
                )
                return
            self.start(selected)

        def start_fresh(self) -> None:
            answer = QMessageBox.question(
                self,
                "Start a fresh model?",
                "This creates a brand-new policy and does not load the selected model.\n\n"
                "The existing latest model will be archived first. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.start(None)

        def play_named_model(self, name: str) -> None:
            target = playback_model_path(
                self.output.text() or "models", self.track.currentText(), name
            ).resolve()
            if not target.is_file():
                QMessageBox.critical(
                    self,
                    f"No {name} model",
                    f"No {name}.zip exists for {self.track.currentText()} yet.",
                )
                return
            if (
                self.playback is not None
                and self.playback.state() != QProcess.ProcessState.NotRunning
            ):
                QMessageBox.information(self, "Playback active", "A model is already driving.")
                return
            self.pending_playback = target
            if self.manager:
                self.log.appendPlainText(
                    f"Stopping training cleanly before realtime playback: {target.name}"
                )
                self.manager.stop()
                self.runtime.setText("Stopping training before playback…")
                return
            self._start_pending_playback()

        def _start_pending_playback(self) -> None:
            target = self.pending_playback
            if target is None or self.manager:
                return
            self.pending_playback = None
            try:
                device = resolve_device(self.device.currentText()).resolved
            except RuntimeError as exc:
                QMessageBox.critical(self, "Device unavailable", str(exc))
                return
            arguments = realtime_drive_arguments(
                target,
                frame_skip=self.frame_skip.value(),
                pwm_enabled=self.pwm.isChecked(),
                pwm_levels=self.levels.value(),
                device=device,
            )
            process = QProcess(self)
            process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
            process.readyReadStandardOutput.connect(self._read_playback_output)
            process.finished.connect(self._playback_finished)
            process.errorOccurred.connect(self._playback_error)
            self.playback = process
            self.runtime.setText(f"Playing {target.name} at 1× realtime…")
            self.log.appendPlainText(f"Realtime playback started: {target}")
            command = "from polybot.cli import drive_main; raise SystemExit(drive_main())"
            process.start(sys.executable, ["-c", command, *arguments])

        def _read_playback_output(self) -> None:
            if self.playback is None:
                return
            text = bytes(self.playback.readAllStandardOutput()).decode(errors="replace").rstrip()
            if text:
                self.log.appendPlainText(text)

        def _playback_finished(self, exit_code: int, _exit_status: object) -> None:
            self._read_playback_output()
            self.log.appendPlainText(f"Realtime playback ended with exit code {exit_code}")
            if self.playback is not None:
                self.playback.deleteLater()
            self.playback = None
            self.runtime.setText("Idle")
            self.refresh_models()

        def _playback_error(self, error: object) -> None:
            self.log.appendPlainText(f"PLAYBACK ERROR: {error}")

        def start(self, selected: str | None) -> None:
            if self.manager:
                return
            try:
                resolve_device(self.device.currentText())
            except RuntimeError as exc:
                QMessageBox.critical(self, "Device unavailable", str(exc))
                return
            cfg = self.config()
            cfg.curriculum.mode = self.curriculum.currentText()
            try:
                bounds = timed_curriculum_bounds(
                    cfg.curriculum.mode,
                    self.curriculum_start_s.value(),
                    self.curriculum_end_s.value(),
                )
            except ValueError as exc:
                QMessageBox.critical(self, "Invalid timed section", str(exc))
                return
            if bounds is not None:
                cfg.curriculum.start_s, cfg.curriculum.end_s = bounds
            self.manager = TrainingManager(cfg, self.events.update.emit)
            threading.Thread(target=self._run, args=(selected,), daemon=True).start()
            self.runtime.setText("Starting…")

        def _run(self, selected: str | None) -> None:
            try:
                self.manager.run(resume=selected, repeat=True)
            except Exception as exc:
                self.events.update.emit({"type": "error", "message": str(exc)})
            finally:
                self.events.update.emit({"type": "idle"})

        def stop(self) -> None:
            if self.manager:
                self.manager.stop()
                self.runtime.setText("Stopping after current PPO step…")
            elif (
                self.playback is not None
                and self.playback.state() != QProcess.ProcessState.NotRunning
            ):
                self.playback.terminate()
                self.runtime.setText("Stopping playback…")

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
                terms = event.get("reward_terms")
                if terms:
                    self.log.appendPlainText(f"             {reward_breakdown(terms)}")
            elif event_type == "checkpoint":
                self.log.appendPlainText(f"Checkpoint saved at timestep {event['timesteps']:,}")
            elif event_type == "best_model":
                self.log.appendPlainText(
                    f"New best model saved: {event['lap_s']:.3f}s — {event['path']}"
                )
            elif event_type == "recovery_checkpoint":
                self.log.appendPlainText(
                    f"Recovery saved at timestep {event['timesteps']:,}: {event['path']}"
                )
            elif event_type == "retrying":
                delay = event.get("delay_s", 0)
                attempt = event.get("attempt", 1)
                self.runtime.setText(f"Simulator disconnected — retrying in {delay:g}s")
                self.log.appendPlainText(
                    f"Retrying simulator connection in {delay:g}s (attempt {attempt})"
                )
            elif event_type == "error":
                self.log.appendPlainText(f"ERROR: {event.get('message', 'unknown error')}")
            elif event_type in {"archived", "completed", "stopped"}:
                self.log.appendPlainText(str(event))
            elif event_type == "idle":
                self.manager = None
                self.runtime.setText("Idle")
                self.refresh_models()
                if self.pending_playback is not None:
                    QTimer.singleShot(0, self._start_pending_playback)

    app = QApplication([sys.argv[0], *qt_arguments])
    window = Window()
    window.show()
    if launch.resume and launch.fresh:
        parser.error("--resume and --fresh cannot be used together")
    if launch.resume:
        QTimer.singleShot(0, window.resume_selected)
    elif launch.fresh:
        QTimer.singleShot(0, lambda: window.start(None))
    return app.exec()
