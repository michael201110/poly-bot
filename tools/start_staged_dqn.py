"""Run Summer 1 DQN without brake, then expand and continue with brake in the GUI."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from polybot.algorithms.registry import backend_for
from polybot.environment.env import PolyTrackEnv
from polybot.gui.main import PolyBotWindow
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import ModelRegistry
from polybot.training.config import DQNConfig, EvaluationConfig, TrainingConfig
from polybot.training.dqn_brake_stage import expand_no_brake_checkpoint
from polybot.training.reward_profiles import RewardProfileStore

TOTAL_STEPS = 2_000_000
EARLY_MAX_STEPS = 600_000
UNLOCK_MIN_STEPS = 100_000
UNLOCK_PROGRESS = 0.15
ROOT = Path("models/v2-dqn-yosh-staged-20260927")


def main() -> int:
    app = QApplication(sys.argv)
    window = PolyBotWindow()
    early = TrainingConfig(
        algorithm="dqn", backend="websocket", track_name="Summer 1", track_id="current",
        device="cpu", seed=20260927, frame_skip=30, timesteps=EARLY_MAX_STEPS,
        max_episode_seconds=60, checkpoint_interval=50_000,
        evaluation=EvaluationConfig(interval_steps=20_000, episodes=3),
        reward_profile="Balanced", rewards=RewardProfileStore().load("Balanced"),
        dqn=DQNConfig(
            architecture="yosh_2020", action_set="no_brake", learning_rate=0.0001,
            replay_capacity=250_000, learning_starts=5_000, batch_size=128,
            gamma=0.995, train_frequency=4, gradient_steps=1,
            target_update_interval=10_000, exploration_fraction=0.25,
            exploration_initial_eps=1.0, exploration_final_eps=0.05,
        ),
        output_root=ROOT / "no-brake", log_root=Path("logs"),
    )
    window.load_configuration(early)
    window.show()
    stage = "early"
    unlock_requested = False
    early_finished = False

    def on_event(event: dict) -> None:
        nonlocal unlock_requested, early_finished
        if stage != "early":
            return
        if (
            event["type"] == "evaluation"
            and event["timesteps"] >= UNLOCK_MIN_STEPS
            and event["median_progress"] >= UNLOCK_PROGRESS
        ):
            unlock_requested = True
            assert window.runner is not None
            window.runner.stop()
        if event["type"] == "completed":
            unlock_requested = True
            early_finished = True
        elif event["type"] == "stopped":
            early_finished = True

    window.bridge.event.connect(on_event)

    def advance_stage() -> None:
        nonlocal stage
        if stage != "early" or not (unlock_requested and early_finished):
            return
        if window.worker is not None and window.worker.is_alive():
            return
        stage = "transferring"
        try:
            source = ModelRegistry(early.output_root).slot(early.track_name, "dqn", "latest")
            metadata = ModelRegistry().read_metadata(source)
            later = replace(
                early,
                timesteps=max(1, TOTAL_STEPS - metadata.training_timesteps),
                output_root=ROOT / "with-brake",
                dqn=replace(
                    early.dqn, action_set="full", exploration_initial_eps=0.30,
                    exploration_fraction=0.25,
                ),
            )
            env = PolyTrackEnv(
                MockSimulatorTransport(), track_id="mock/straight",
                lookahead_count=later.lookahead_count,
                action_adapter=backend_for("dqn").action_adapter(later),
            )
            try:
                expand_no_brake_checkpoint(source, later, env, "cpu")
            finally:
                env.close()
            window.load_configuration(later)
            stage = "full"
            window._start(True)
        except Exception as exc:
            window._error(f"Brake-stage transfer failed: {exc}")

    timer = QTimer(window)
    timer.timeout.connect(advance_stage)
    timer.start(1000)
    QTimer.singleShot(200, lambda: window._start(False))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
