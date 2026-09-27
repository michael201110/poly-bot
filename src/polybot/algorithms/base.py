"""The only interface the training runner needs from an RL algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polybot.control.actions import ActionAdapter

if TYPE_CHECKING:
    from polybot.training.config import TrainingConfig


class AlgorithmBackend(ABC):
    name: str

    @abstractmethod
    def validate_config(self, config: TrainingConfig) -> None: ...

    @abstractmethod
    def action_adapter(self, config: TrainingConfig) -> ActionAdapter: ...

    @abstractmethod
    def create_model(self, config: TrainingConfig, env: Any, device: str) -> Any: ...

    @abstractmethod
    def load_model(
        self, path: Path, env: Any, device: str, *, resume: bool = False
    ) -> Any: ...

    def save_model(self, model: Any, directory: Path, *, resume: bool = False) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        model.save(str(directory / "policy.zip"))

    def configure_resume(
        self, model: Any, config: TrainingConfig, device: str, *, fresh_replay: bool = False
    ) -> None:
        """Restore optional training-only state after a model load."""
        return None

    @abstractmethod
    def parameter_counts(self, model: Any) -> dict[str, int]: ...

    @abstractmethod
    def metrics(self, model: Any) -> dict[str, float | int | None]: ...

    @abstractmethod
    def architecture(self, config: TrainingConfig) -> str: ...
