"""Resolve pace-search settings against the evaluated champion."""

from __future__ import annotations

from pathlib import Path

from polybot.models.registry import ModelMetadata
from polybot.training.config import TrainingConfig


def champion_evaluation_config(requested: TrainingConfig, metadata: ModelMetadata) -> TrainingConfig:
    """Use the champion's recorded simulator/reward semantics for comparisons.

    Pace searches do not train, so using a newly edited reward profile or
    termination threshold would change what counts as a valid lap. Keep the
    requested seed for reproducible follow-up batches, and take the rest of
    the evaluation settings from the policy being compared.
    """
    saved = TrainingConfig.from_dict(metadata.training_config)
    if requested.algorithm != saved.algorithm or requested.algorithm != "tqc":
        raise ValueError("pace search configuration does not match the TQC champion")
    if requested.track_name != saved.track_name or requested.track_id != saved.track_id:
        raise ValueError("pace search configuration targets a different track than the champion")
    if Path(requested.output_root).resolve() != Path(saved.output_root).resolve():
        raise ValueError("pace search configuration uses a different model root than the champion")
    saved.seed = requested.seed
    saved.device = "cpu"
    return saved
