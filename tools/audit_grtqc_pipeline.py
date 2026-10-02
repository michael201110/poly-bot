"""Collect an exact live reference lap and audit critic geometry on real states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.logger import configure

from polybot.algorithms.grtqc import GatedReLU, GRTQCBackend
from polybot.algorithms.tqc import TQCBackend
from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig
from polybot.training.evaluation import PrefixObservationReference
from polybot.training.runner import TrainingRunner


def collect_reference(reference: Path, config: TrainingConfig, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"preserve existing live reference: {output}")
    backend = TQCBackend() if ModelRegistry().read_metadata(reference).algorithm == "tqc" else GRTQCBackend()
    model = backend.load_model(reference / "policy.zip", None, "cpu")
    env = TrainingRunner(config)._environment()
    env.capture_tick_controls = True
    extra = env.observation_space.shape[0] - model.observation_space.shape[0]
    driver = PrefixObservationReference(model, extra_features=extra) if extra else model
    fields = {key: [] for key in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts")}
    telemetry = []
    try:
        obs, _ = env.reset(seed=config.seed + 1_000_000)
        while True:
            action, _ = driver.predict(obs, deterministic=True)
            if driver._air_brake_active:
                env.request_air_brake(driver._air_brake_base_action)
            recorded_action = action
            if config.grtqc.critic_raw_actions:
                recorded_action, _ = model.policy.predict(
                    obs[..., :model.observation_space.shape[0]], deterministic=True,
                )
            following, reward, terminated, truncated, info = env.step(action)
            for key, value in zip(fields, (obs, following, recorded_action, reward * config.reward_scale,
                                          terminated or truncated, truncated and not terminated), strict=True):
                fields[key].append(value)
            telemetry.append(info)
            obs = following
            if terminated or truncated:
                break
    finally:
        env.close()
    np.savez_compressed(output, **{key: np.asarray(value) for key, value in fields.items()})
    report = {"reference": str(reference), "config": config.to_dict(), "telemetry": telemetry,
              "lap_s": info["elapsed_s"], "finished": "finish" in info["events"],
              "transitions": len(telemetry)}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key != "telemetry"}


def audit(checkpoint: Path, dataset: Path, output: Path) -> dict:
    th.set_num_threads(1)
    model = GRTQCBackend().load_model(checkpoint / "policy.zip", None, "cpu")
    model.set_logger(configure(None, []))
    data = np.load(dataset)
    observations = th.as_tensor(data["observations"], dtype=th.float32)
    actions = th.as_tensor(data["actions"], dtype=th.float32).requires_grad_(True)
    if observations.shape[1:] != model.observation_space.shape:
        raise ValueError("live reference and critic observation schemas differ")
    values = model.critic(observations, actions)
    per_critic = values.mean(dim=2)
    gradients = th.stack([
        th.autograd.grad(per_critic[:, index].sum(), actions, retain_graph=True)[0]
        for index in range(model.critic.n_critics)
    ], dim=1)
    values = values.detach()
    returns = np.empty(len(observations), dtype=np.float32)
    future = 0.0
    for index in reversed(range(len(returns))):
        future = float(data["rewards"][index]) + model.gamma * future
        returns[index] = future
    with th.no_grad():
        following = th.as_tensor(data["next_observations"], dtype=th.float32)
        th.manual_seed(17)
        next_actions, next_log_prob = model._training_actions_log_prob(following)
        next_actions = model._critic_actions(next_actions, following)
        quantiles = model.critic_target(following, next_actions).flatten(start_dim=1).sort(dim=1).values
        keep = model.critic.quantiles_total - model.top_quantiles_to_drop_per_net * model.critic.n_critics
        alpha = model.log_ent_coef.detach().exp() if model.log_ent_coef is not None else model.ent_coef_tensor
        target = (th.as_tensor(data["rewards"])[:, None] + (1 - th.as_tensor(data["dones"], dtype=th.float32))[:, None]
                  * model.gamma * (quantiles[:, :keep] - alpha * next_log_prob[:, None]))
        q = per_critic.mean(dim=1)
        td = target.mean(dim=1) - q
    gate_stats = {}
    handles = []
    for name, module in model.policy.named_modules():
        if isinstance(module, GatedReLU):
            def capture(layer, args, result, name=name):
                gate = layer.gate(args[0]).sigmoid().detach()
                gate_stats[name] = {"mean": float(gate.mean()), "std": float(gate.std()),
                                    "saturated_fraction": float(((gate < .01) | (gate > .99)).float().mean())}
            handles.append(module.register_forward_hook(capture))
    model.actor.zero_grad()
    model.critic.zero_grad()
    chosen = model._critic_actions(model.actor(observations, deterministic=True), observations)
    actor_loss = -model.critic(observations, chosen).mean()
    actor_loss.backward()
    for handle in handles:
        handle.remove()
    layer_gradients = {name: float(parameter.grad.norm()) for name, parameter in model.actor.named_parameters()
                       if parameter.grad is not None}
    records = []
    for start, end in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=True):
        mask = (observations[:, 12] >= start) & (observations[:, 12] < end)
        if mask.any():
            records.append({"progress": [float(start), float(end)], "states": int(mask.sum()),
                            "q_mean": float(q[mask].mean()), "return_mean": float(th.as_tensor(returns)[mask].mean()),
                            "td_absolute_mean": float(td[mask].abs().mean()),
                            "quantile_disagreement": float(values[mask].var(dim=1, unbiased=False).mean()),
                            "action_gradient_norm": float(gradients[mask].norm(dim=2).mean())})
    report = {
        "checkpoint": str(checkpoint), "dataset": str(dataset), "states": len(returns),
        "q_mean": float(q.mean()), "driving_return_mean": float(returns.mean()),
        "return_mae": float((q - th.as_tensor(returns)).abs().mean()),
        "td_mae": float(td.abs().mean()), "disagreement": float(values.var(dim=1, unbiased=False).mean()),
        "critic_gradient_cosine_mean": float(
            th.nn.functional.cosine_similarity(gradients[:, 0], gradients[:, 1]).mean()),
        "actor_gradient_norm": float(np.linalg.norm(list(layer_gradients.values()))),
        "actor_layer_gradient_norms": layer_gradients, "gate_activations": gate_stats, "sections": records,
        "actor_optimizer_states": len(model.actor.optimizer.state),
        "actor_optimizer_steps": sorted(set(
            float(v["step"]) for v in model.actor.optimizer.state.values() if "step" in v)),
        "note": ("Recorded driving-return comparison excludes entropy/truncation; "
                 "gradients do not prove counterfactual lap quality."),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--collect", type=Path, help="collect this immutable/reference policy before auditing")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    th.set_num_threads(1)
    if args.collect:
        config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
        print(json.dumps(collect_reference(args.collect, config, args.dataset)), flush=True)
    if args.checkpoint:
        print(json.dumps(audit(args.checkpoint, args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
