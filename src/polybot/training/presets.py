"""Explicit beginner and expert parameter presets with advisory warnings."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from polybot.training.config import GRTQCConfig, PPOConfig, TQCConfig, TrainingConfig


def algorithm_presets(algorithm: str) -> dict[str, PPOConfig | TQCConfig | GRTQCConfig]:
    if algorithm == "ppo":
        balanced = PPOConfig()
        return {
            "Beginner": replace(balanced, architecture="tiny", rollout_steps=256,
                                batch_size=64),
            "Balanced": balanced,
            "Fast training": replace(balanced, architecture="tiny", rollout_steps=256,
                                     batch_size=64, epochs=3),
            "Advanced": replace(balanced, architecture="standard"),
            "Summer 1 - PPO from TQC Teacher": replace(
                balanced, architecture="compact", learning_rate=3e-5,
                rollout_steps=1024, batch_size=128, epochs=3,
                entropy_coefficient=0.0001, target_lap_s=22.0,
            ),
        }
    if algorithm == "grtqc":
        balanced = GRTQCConfig()
        return {
            "Summer 1 - Transferred Champion": replace(
                balanced, learning_rate=3e-5, actor_learning_rate=1e-7,
                critic_learning_rate=5e-5, train_frequency=2,
                learning_starts=3_000, critic_warmup_updates=10_000,
            ),
            "Balanced": balanced,
        }
    if algorithm == "tqc":
        balanced = TQCConfig()
        return {
            "Beginner / Stable": replace(balanced, architecture="tiny", train_frequency=2),
            "Balanced": balanced,
            "Fast / Lightweight": replace(balanced, architecture="tiny", batch_size=128,
                                          train_frequency=4),
            "Advanced": replace(balanced, architecture="standard"),
            "Summer 1 - TQC Safe Polish": replace(
                balanced, architecture="compact", learning_rate=1e-5,
                replay_capacity=250_000, learning_starts=20_000,
                batch_size=256, gamma=0.995, tau=0.005,
                train_frequency=4, gradient_steps=1, entropy="auto_0.1",
                champion_action_drift_limit=1e-5,
                champion_lap_tolerance_s=0.15,
            ),
        }
    raise ValueError("unknown algorithm")


class PresetStore:
    def __init__(self, root: Path = Path("profiles/algorithms")) -> None:
        self.root = root

    def save(self, algorithm: str, name: str, settings: PPOConfig | TQCConfig | GRTQCConfig) -> Path:
        from polybot.models.registry import track_slug

        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{algorithm}-{track_slug(name)}.json"
        path.write_text(json.dumps({
            "schema": "polybot.algorithm-preset.v2", "algorithm": algorithm,
            "name": name, "settings": asdict(settings),
        }, indent=2) + "\n", encoding="utf-8")
        return path

    def load(self, path: Path) -> PPOConfig | TQCConfig | GRTQCConfig:
        value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if value["schema"] != "polybot.algorithm-preset.v2":
            raise ValueError("only v2 presets are supported")
        config_type = {"ppo": PPOConfig, "tqc": TQCConfig, "grtqc": GRTQCConfig}.get(value["algorithm"])
        if config_type is None:
            raise ValueError("unknown algorithm in preset")
        return config_type(**value["settings"])

    def list(self, algorithm: str) -> dict[str, Path]:
        result = {}
        for path in self.root.glob(f"{algorithm}-*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value["schema"] == "polybot.algorithm-preset.v2":
                    result[value["name"]] = path
            except (OSError, ValueError, KeyError):
                continue
        return result


def configuration_warnings(config: TrainingConfig, gpu_name: str | None = None) -> list[str]:
    settings = getattr(config, config.algorithm)
    assert settings is not None
    warnings = []
    if settings.learning_rate > 1e-3:
        warnings.append("Learning rate above 0.001 can make policy updates unstable.")
    if settings.gamma < 0.9 or settings.gamma > 0.9999:
        warnings.append("Extreme gamma can make long-term credit assignment difficult.")
    if config.evaluation.interval_steps < 500:
        warnings.append("Very frequent evaluation may spend more time testing than training.")
    if config.timesteps > 10_000_000:
        warnings.append("This budget may run for many days; check expected TPS first.")
    if config.tqc is not None or config.grtqc is not None:
        p = config.tqc or config.grtqc
        if p.replay_capacity > 2_000_000:
            warnings.append("A very large replay buffer can consume substantial RAM.")
        if p.gradient_steps > 4:
            warnings.append("Many gradient steps per collection can sharply reduce TPS.")
        if p.architecture == "standard" and p.train_frequency == 1:
            warnings.append("256×256 TQC with train frequency 1 can be slow on modest GPUs.")
        if gpu_name and "T500" in gpu_name.upper() and p.architecture == "standard":
            warnings.append("A T500 may train standard TQC slowly; consider frequency 2–4.")
        if p.entropy.startswith("auto_") and float(p.entropy[5:]) > 1:
            warnings.append("High initial entropy may keep driving unusually random.")
    return warnings
