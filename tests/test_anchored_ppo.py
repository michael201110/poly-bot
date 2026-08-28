from __future__ import annotations

import torch

from polybot.training.anchored_ppo import categorical_teacher_kl


def test_teacher_kl_is_zero_for_identical_policy_logits() -> None:
    logits = torch.tensor([[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.1]])
    loss = categorical_teacher_kl(logits, logits, (3, 2, 2))
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_teacher_kl_penalises_departure_from_teacher() -> None:
    teacher = torch.tensor([[5.0, -5.0, -5.0, 4.0, -4.0, 3.0, -3.0]])
    student = -teacher
    loss = categorical_teacher_kl(student, teacher, (3, 2, 2))
    assert loss.item() > 5.0
