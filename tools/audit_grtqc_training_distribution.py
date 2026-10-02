"""Drive an unchanged GRTQC actor with the exact stochastic training distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.logger import configure

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig
from polybot.training.evaluation import evaluate_model
from polybot.training.runner import TrainingRunner


class TrainingDistributionDriver:
    """Sample raw actor actions; retain the checkpoint's existing output transforms."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.log_probabilities: list[float] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)

    def _sample(self, observation: np.ndarray, *args: Any, **kwargs: Any) -> tuple[np.ndarray, None]:
        tensor, vectorized = self.model.policy.obs_to_tensor(observation)
        with th.no_grad():
            raw, log_prob = self.model._training_actions_log_prob(tensor)
        self.log_probabilities.extend(log_prob.cpu().reshape(-1).tolist())
        action = self.model.policy.unscale_action(raw.cpu().numpy())
        return action if vectorized else action.squeeze(axis=0), None

    def predict(self, observation: np.ndarray, *args: Any, **kwargs: Any) -> tuple[np.ndarray, Any]:
        original = self.model.policy.predict
        self.model.policy.predict = self._sample
        try:
            # This retains NumPy overlay thresholds and the actual touchdown
            # release request used by deterministic live evaluation.
            return self.model.predict(observation, *args, **kwargs)
        finally:
            self.model.policy.predict = original


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--limits", type=float, nargs="+", default=[0.02, 0.0005, 0.000001])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"preserve existing diagnostic: {args.output}")
    if any(not 0 < limit <= 1 for limit in args.limits):
        raise ValueError("training standard-deviation limits must be in (0, 1]")
    th.set_num_threads(1)
    metadata = ModelRegistry().read_metadata(args.checkpoint)
    config = TrainingConfig.from_dict(metadata.training_config)
    model = GRTQCBackend().load_model(args.checkpoint / "policy.zip", None, "cpu")
    model.set_logger(configure(None, []))
    runner = TrainingRunner(config)
    initial = {name: value.clone() for name, value in model.actor.state_dict().items()}
    records = []
    for limit in args.limits:
        model.policy_std_limit = limit
        th.manual_seed(config.seed + 1_000_000)
        driver = TrainingDistributionDriver(model)
        result = evaluate_model(driver, runner._environment, episodes=args.episodes, seed=config.seed + 1_000_000)
        record = {"std_limit": limit, "evaluation": result.to_dict(),
                  "mean_training_log_probability": float(np.mean(driver.log_probabilities))}
        records.append(record)
        if any(not th.equal(value, initial[name]) for name, value in model.actor.state_dict().items()):
            raise RuntimeError("read-only distribution audit changed actor weights")
        report = {"checkpoint": str(args.checkpoint), "actor_weights_unchanged": True,
                  "policy_sha256": hashlib.sha256((args.checkpoint / "policy.zip").read_bytes()).hexdigest(),
                  "results": records}
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
