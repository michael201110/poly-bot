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
            "frame_skip": fields["Frame skip"].value(),
            "episode_seconds": fields["Maximum episode time"].value(),
            "timesteps": fields["Timesteps"].value(),
            "reward_profile": fields["Reward profile"].currentText(),
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
        "track": "Summer 1", "device": "cpu", "frame_skip": 30,
        "episode_seconds": 60.0, "timesteps": 100_000,
        "reward_profile": "Summer 1 - full bootstrap",
        "tqc_enabled": True, "ppo_disabled": True,
        "teacher_disabled": True, "kl_disabled": True,
    }
