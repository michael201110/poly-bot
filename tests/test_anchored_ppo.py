from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.distributions import Normal, kl_divergence

from polybot.algorithms.ppo import PPOBackend
from polybot.algorithms.ppo_teacher import expert_action_loss


def test_continuous_teacher_kl_is_zero_for_identical_gaussians() -> None:
    distribution = Normal(torch.tensor([[0.2, -0.4]]), torch.tensor([[0.7, 0.3]]))
    torch.testing.assert_close(
        kl_divergence(distribution, distribution).sum(dim=-1).mean(), torch.tensor(0.0),
        atol=1e-7, rtol=0,
    )


def test_continuous_teacher_kl_penalises_mean_and_variance_drift() -> None:
    teacher = Normal(torch.zeros((1, 2)), torch.ones((1, 2)))
    student = Normal(torch.full((1, 2), 3.0), torch.full((1, 2), 0.2))
    assert kl_divergence(teacher, student).sum().item() > 5.0


def test_ppo_backend_exposes_teacher_anchor_kl_metric() -> None:
    model = SimpleNamespace(
        logger=SimpleNamespace(name_to_value={"train/teacher_kl": 0.125})
    )
    assert PPOBackend().metrics(model)["teacher_kl"] == 0.125


def test_expert_action_loss_uses_protocol_v2_ghost_controls() -> None:
    observations = torch.zeros((2, 105))
    observations[0, 39:42] = torch.tensor([-1.0, 1.0, 0.0])
    observations[1, 39:42] = torch.tensor([1.0, 0.0, 1.0])
    mean = torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
    loss = expert_action_loss(mean, observations)
    assert loss.item() < 1e-6


def test_expert_guidance_fades_when_far_from_reference() -> None:
    observations = torch.zeros((1, 105))
    observations[:, 40] = 1
    mean = torch.zeros((1, 2), requires_grad=True)
    near = expert_action_loss(mean, observations)
    observations[:, 34] = 0.2  # 10 metres from the reference line
    far = expert_action_loss(mean, observations)
    assert far.item() < near.item() * 0.001
    far.backward()
    assert torch.isfinite(mean.grad).all()


def test_expert_guidance_teaches_throttle_from_standstill() -> None:
    observations = torch.zeros((1, 105))
    observations[:, 40] = 1.0
    mean = torch.zeros((1, 2))
    aligned_at_rest = expert_action_loss(mean, observations)
    observations[:, 38] = 0.4
    slower_than_ghost = expert_action_loss(mean, observations)

    torch.testing.assert_close(slower_than_ghost, aligned_at_rest)


def test_guidance_confidence_weights_each_sample_independently() -> None:
    observations = torch.zeros((2, 105))
    observations[1, 34] = 1  # only this example should be suppressed
    mean = torch.zeros((2, 2))
    expected = expert_action_loss(mean[:1], observations[:1]) / 2
    torch.testing.assert_close(expert_action_loss(mean, observations), expected)
