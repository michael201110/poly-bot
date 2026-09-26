"""Verify the first-run TQC settings in the real Qt form without opening a game."""

from __future__ import annotations

import sys

import pytest


def test_tiny_tqc_gui_launch_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widgets = pytest.importorskip("PySide6.QtWidgets")
    from polybot.gui.main import main

    monkeypatch.setattr(sys, "argv", [
        "polybot-gui", "--algorithm", "tqc", "--tqc-architecture", "tiny",
        "--track-name", "Summer 1", "--device", "cpu", "--frame-skip", "30",
        "--timesteps", "100000", "--reward-profile", "Summer 1 - full bootstrap",
        "--seed", "0", "--max-episode-seconds", "60", "--checkpoint-interval", "25000",
        "--tqc-learning-rate", "0.0003", "--tqc-buffer-size", "250000",
        "--tqc-learning-starts", "5000", "--tqc-batch-size", "256",
        "--tqc-gamma", "0.999", "--tqc-tau", "0.005",
        "--tqc-train-freq", "1", "--tqc-gradient-steps", "1", "--tqc-ent-coef", "auto",
        "--tqc-forward-warmup-fraction", "0.8",
        "--tqc-forward-warmup-steering-std", "0.18",
        "--tqc-initial-throttle-bias", "1.0",
    ])
    observed = {}

    def inspect(app):
        window = next(w for w in app.topLevelWidgets() if w.windowTitle() == "PolyBot Training")
        form = window.centralWidget().widget().layout().itemAt(0).layout()
        fields = {
            form.itemAt(row, widgets.QFormLayout.LabelRole).widget().text():
                form.itemAt(row, widgets.QFormLayout.FieldRole).widget()
            for row in range(form.rowCount())
        }
        observed.update({
            "algorithm": fields["Algorithm"].currentText(),
            "backend": fields["Simulator backend"].currentText(),
            "architecture": fields["TQC architecture"].currentText(),
            "parameters": fields["Parameters"].text(),
            "action": fields["Action mode"].text(),
            "track": fields["Track/profile"].currentText(),
            "device": fields["Device"].currentText(),
            "seed": fields["Seed"].value(),
            "frame_skip": fields["Frame skip"].value(),
            "episode_seconds": fields["Maximum episode time"].value(),
            "timesteps": fields["Timesteps"].value(),
            "reward_profile": fields["Reward profile"].currentText(),
            "checkpoint": fields["Checkpoint interval"].value(),
            "learning_rate": fields["TQC learning rate"].value(),
            "replay_buffer": fields["TQC replay buffer"].value(),
            "learning_starts": fields["TQC learning starts"].value(),
            "batch_size": fields["TQC batch size"].value(),
            "gamma": fields["TQC gamma"].value(),
            "tau": fields["TQC tau"].value(),
            "train_frequency": fields["TQC train frequency"].value(),
            "gradient_steps": fields["TQC gradient steps"].value(),
            "entropy": fields["TQC entropy"].text(),
            "forward_fraction": fields["TQC forward warmup fraction"].value(),
            "steering_std": fields["TQC warmup steering std"].value(),
            "throttle_bias": fields["TQC initial throttle bias"].value(),
            "tqc_enabled": fields["TQC replay buffer"].isEnabled(),
            "ppo_disabled": not fields["PPO epochs"].isEnabled(),
            "teacher_disabled": not fields["Teacher model"].isEnabled(),
            "kl_disabled": not fields["Teacher KL coefficient"].isEnabled(),
        })
        window.close()
        return 0

    monkeypatch.setattr(widgets.QApplication, "exec", inspect)
    assert main() == 0
    assert observed == {
        "algorithm": "TQC", "backend": "websocket", "architecture": "tiny",
        "parameters": "Actor: 11,204 | Training network total: 61,992",
        "action": "Continuous PWM: steering [-1, 1], longitudinal [-1, 1]",
        "track": "Summer 1", "device": "cpu", "seed": 0, "frame_skip": 30,
        "episode_seconds": 60.0, "timesteps": 100_000,
        "reward_profile": "Summer 1 - full bootstrap",
        "checkpoint": 25_000, "learning_rate": 0.0003,
        "replay_buffer": 250_000, "learning_starts": 5_000, "batch_size": 256,
        "gamma": 0.999, "tau": 0.005, "train_frequency": 1,
        "gradient_steps": 1, "entropy": "auto",
        "forward_fraction": 0.8, "steering_std": 0.18, "throttle_bias": 1.0,
        "tqc_enabled": True, "ppo_disabled": True,
        "teacher_disabled": True, "kl_disabled": True,
    }
