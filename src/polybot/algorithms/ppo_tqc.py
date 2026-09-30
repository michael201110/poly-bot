"""PPO actor components compatible with a TQC squashed-Gaussian policy."""

from __future__ import annotations

from typing import Any

import torch as th
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class TQCSquashedActorCriticPolicy(ActorCriticPolicy):
    """PPO policy whose bounded action distribution matches TQC's tanh Gaussian."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if len(self.action_space.shape) != 1:
            raise ValueError("TQC-compatible PPO needs a flat continuous action space")
        self.action_dist = SquashedDiagGaussianDistribution(self.action_space.shape[0])


class TQCResidualActorCriticPolicy(TQCSquashedActorCriticPolicy):
    """Freeze a grafted TQC mean actor and learn a bounded PPO residual."""

    def __init__(
        self, *args: Any, residual_action_limit: float = 0.1,
        residual_progress_start: float = 0.0, residual_progress_end: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= residual_action_limit <= 1.0:
            raise ValueError("residual action limit must be in [0, 1]")
        if not 0.0 <= residual_progress_start <= residual_progress_end <= 1.0:
            raise ValueError("residual progress window must be within [0, 1]")
        self.residual_action_limit = float(residual_action_limit)
        self.residual_progress_start = float(residual_progress_start)
        self.residual_progress_end = float(residual_progress_end)
        self._residual_progress: th.Tensor | None = None
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()
        self.residual_action = nn.Linear(
            self.mlp_extractor.latent_dim_pi, self.action_space.shape[0],
        )
        nn.init.zeros_(self.residual_action.weight)
        nn.init.zeros_(self.residual_action.bias)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Any:
        correction = self.residual_action_limit * th.tanh(self.residual_action(latent_pi))
        log_std = self.log_std
        if self._residual_progress is not None:
            active = (
                (self._residual_progress >= self.residual_progress_start)
                & (self._residual_progress <= self.residual_progress_end)
            ).to(dtype=correction.dtype).unsqueeze(-1)
            correction = correction * active
            # A gated residual must leave the frozen teacher's full behavior
            # unchanged outside its window. Masking only the mean still lets
            # PPO's Gaussian exploration perturb the teacher everywhere.
            log_std = th.where(active.bool(), self.log_std, th.full_like(correction, -9.2103405))
        mean_actions = self.action_net(latent_pi) + correction
        return self.action_dist.proba_distribution(
            mean_actions, log_std,
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Any:
        self._residual_progress = obs[:, 12]
        try:
            return super().forward(obs, deterministic=deterministic)
        finally:
            self._residual_progress = None

    def get_distribution(self, obs: th.Tensor) -> Any:
        self._residual_progress = obs[:, 12]
        try:
            return super().get_distribution(obs)
        finally:
            self._residual_progress = None

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Any:
        self._residual_progress = obs[:, 12]
        try:
            return super().evaluate_actions(obs, actions)
        finally:
            self._residual_progress = None

    def set_residual_progress_window(self, start: float, end: float) -> None:
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError("residual progress window must be within [0, 1]")
        self.residual_progress_start = float(start)
        self.residual_progress_end = float(end)

    def freeze_base_actor(self) -> None:
        """Keep the grafted TQC mean fixed while PPO trains the residual."""
        for module in (
            self.features_extractor, self.mlp_extractor.policy_net, self.action_net,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def initialize_actor_from_tqc(ppo_model: Any, tqc_model: Any) -> dict[str, int]:
    """Copy an exactly shape-compatible TQC mean actor into a PPO policy.

    With ``TQCSquashedActorCriticPolicy`` this makes deterministic PPO actions
    equal to raw deterministic TQC actions, rather than fitting them by regression.
    The critic and PPO optimizer are deliberately left fresh.
    """
    source = getattr(tqc_model, "actor", None)
    target_policy = getattr(ppo_model, "policy", None)
    if source is None or target_policy is None:
        raise TypeError("actor graft requires a TQC actor and a PPO model")
    if not isinstance(target_policy, TQCSquashedActorCriticPolicy):
        raise TypeError("actor graft requires TQCSquashedActorCriticPolicy")
    source_extractor = getattr(source, "features_extractor", None)
    target_extractor = getattr(target_policy, "features_extractor", None)
    if source_extractor is None or target_extractor is None or (
        type(source_extractor) is not type(target_extractor)
    ):
        raise ValueError("TQC and PPO feature extractors do not match")
    source_features = source_extractor.state_dict()
    target_features = target_extractor.state_dict()
    if source_features.keys() != target_features.keys() or any(
        source_features[key].shape != target_features[key].shape for key in source_features
    ):
        raise ValueError("TQC and PPO feature extractor parameters do not match")
    source_layers = getattr(source, "latent_pi", None)
    target_layers = target_policy.mlp_extractor.policy_net
    if source_layers is None or [type(layer) for layer in source_layers] != [
        type(layer) for layer in target_layers
    ]:
        raise ValueError("TQC and PPO mean actor activations do not match")

    mappings = (
        (source_layers, target_layers),
        (getattr(source, "mu", None), target_policy.action_net),
    )
    copied: dict[str, int] = {"parameters": 0}
    with th.no_grad():
        target_extractor.load_state_dict(source_features, strict=True)
        copied["parameters"] += sum(int(tensor.numel()) for tensor in source_features.values())
        for source_module, target_module in mappings:
            if source_module is None:
                raise TypeError("TQC actor does not expose a deterministic mean network")
            source_state = source_module.state_dict()
            target_state = target_module.state_dict()
            if source_state.keys() != target_state.keys() or any(
                source_state[key].shape != target_state[key].shape for key in source_state
            ):
                raise ValueError("TQC and PPO mean actor shapes do not match")
            target_module.load_state_dict(source_state, strict=True)
            copied["parameters"] = copied.get("parameters", 0) + sum(
                int(tensor.numel()) for tensor in source_state.values()
            )
        if hasattr(target_policy, "residual_action"):
            nn.init.zeros_(target_policy.residual_action.weight)
            nn.init.zeros_(target_policy.residual_action.bias)
            target_policy.freeze_base_actor()
    return copied
