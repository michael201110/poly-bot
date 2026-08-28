from __future__ import annotations

import torch

from polybot.training.anchored_ppo import categorical_teacher_kl, expert_action_loss


def test_teacher_kl_is_zero_for_identical_policy_logits() -> None:
    logits = torch.tensor([[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1]])
    loss = categorical_teacher_kl(logits, logits, (3, 2, 2))
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_teacher_kl_penalises_departure_from_teacher() -> None:
    teacher = torch.tensor([[5.0, -5.0, -5.0, 4.0, -4.0, 3.0, -3.0]])
    student = -teacher
    loss = categorical_teacher_kl(student, teacher, (3, 2, 2))
    assert loss.item() > 5.0


def test_expert_action_loss_uses_protocol_v2_ghost_controls() -> None:
    observations = torch.zeros((2, 105))
    observations[0, 39:42] = torch.tensor([-1.0, 1.0, 0.0])
    observations[1, 39:42] = torch.tensor([1.0, 0.0, 1.0])
    logits = torch.full((2, 15), -10.0)
    logits[0, [0, 12, 13]] = 10.0
    logits[1, [10, 11, 14]] = 10.0
    loss = expert_action_loss(logits, observations, (11, 2, 2))
    assert loss.item() < 1e-6
