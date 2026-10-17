from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from src.distillation.config import DistillationSettings
from src.distillation.trainer import DistillationTrainer, distillation_loss
from src.model.output import DecisionEngineOutput


@pytest.mark.parametrize("alpha", [0.0, 0.25, 1.0])
@pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
def test_loss_matches_formula(alpha: float, temperature: float) -> None:
    logits = torch.tensor([[1.2, -0.4, 99.0], [-0.2, 0.8, 0.5]],
                          requires_grad=True)
    labels = torch.tensor([[0.75, 0.25, 0], [0, 0, 1.0]])
    teacher = torch.tensor([[1.0, 0, 0], [0.1, 0.6, 0.3]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    expected = []
    for z, y, q, valid in zip(logits, labels, teacher, mask):
        z, y, q = z[valid], y[valid], q[valid]
        q = q.pow(1 / temperature)
        q = q / q.sum()
        log_p = torch.log_softmax(z / temperature, -1)
        kl = (torch.xlogy(q, q) - q * log_p).sum()
        expected.append(alpha * -(y * torch.log_softmax(z, -1)).sum()
                        + (1 - alpha) * temperature**2 * kl)
    settings = DistillationSettings("t", "s", alpha, temperature)
    loss, ce_term, kl_term = distillation_loss(
        logits, labels, teacher, mask, settings,
    )
    torch.testing.assert_close(loss, torch.stack(expected).mean())
    torch.testing.assert_close(loss, ce_term + kl_term)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 2] == 0
    assert torch.isfinite(logits.grad).all()


def test_loss_handles_mixed_precision() -> None:
    loss, _, _ = distillation_loss(
        torch.tensor([[2.0, -1.0]], dtype=torch.bfloat16),
        torch.tensor([[1.0, 0]]), torch.tensor([[0.7, 0.3]]),
        torch.ones(1, 2, dtype=torch.bool), DistillationSettings("t", "s"),
    )
    assert loss.dtype == torch.float32


class LogitModel(nn.Module):
    """Model that fails if teacher metadata leaks into its forward call."""

    accepts_loss_kwargs = False

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.5, -0.1]))

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor,
        option_mask: torch.Tensor, num_options: int,
    ) -> DecisionEngineOutput:
        return DecisionEngineOutput.from_logits(
            self.logits.expand(len(input_ids), -1), 1.0, labels,
        )


def test_trainer_keeps_eval_human_and_requires_training_targets(
    tmp_path: Path,
) -> None:
    model = LogitModel()
    trainer = DistillationTrainer(
        model=model, args=TrainingArguments(output_dir=str(tmp_path),
                                           use_cpu=True, report_to="none"),
        distillation=DistillationSettings("t", "s"),
    )
    inputs = {
        "input_ids": torch.ones(2, 1, dtype=torch.long),
        "labels": torch.tensor([[1.0, 0], [0, 1.0]]),
        "teacher_targets": torch.tensor([[0.2, 0.8], [0.1, 0.9]]),
        "option_mask": torch.ones(2, 2, dtype=torch.bool), "num_options": 2,
    }
    model.eval()
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    torch.testing.assert_close(loss, outputs.loss)
    assert "teacher_targets" in inputs
    model.train()
    loss = trainer.compute_loss(model, inputs)
    assert not torch.isclose(loss, outputs.loss)
    with pytest.raises(ValueError, match="teacher"):
        trainer.compute_loss(model, {k: v for k, v in inputs.items()
                                     if k != "teacher_targets"})
    assert trainer.model_accepts_loss_kwargs is False


def test_accumulation_scales_once(tmp_path: Path) -> None:
    gradients = []
    for accumulation in (1, 2):
        model = LogitModel()
        trainer = DistillationTrainer(
            model=model, args=TrainingArguments(
                output_dir=str(tmp_path / str(accumulation)), use_cpu=True,
                gradient_accumulation_steps=accumulation, report_to="none",
            ), distillation=DistillationSettings("t", "s"),
        )
        trainer.current_gradient_accumulation_steps = accumulation
        for _ in range(accumulation):
            trainer.training_step(model, {
                "input_ids": torch.ones(2, 1, dtype=torch.long),
                "labels": torch.tensor([[1.0, 0], [0, 1.0]]),
                "teacher_targets": torch.tensor([[0.2, 0.8], [0.1, 0.9]]),
                "option_mask": torch.ones(2, 2, dtype=torch.bool),
                "num_options": 2,
            })
        gradients.append(model.logits.grad)
    torch.testing.assert_close(*gradients)


def test_log_adds_the_loss_parts_of_the_training_steps(tmp_path: Path) -> None:
    model = LogitModel()
    trainer = DistillationTrainer(
        model=model, args=TrainingArguments(output_dir=str(tmp_path),
                                           use_cpu=True, report_to="none"),
        distillation=DistillationSettings("t", "s", alpha=0.4),
    )
    inputs = {
        "input_ids": torch.ones(2, 1, dtype=torch.long),
        "labels": torch.tensor([[1.0, 0], [0, 1.0]]),
        "teacher_targets": torch.tensor([[0.2, 0.8], [0.1, 0.9]]),
        "option_mask": torch.ones(2, 2, dtype=torch.bool), "num_options": 2,
    }
    model.train()
    loss = trainer.compute_loss(model, inputs)
    trainer.compute_loss(model, inputs)
    logs = {"loss": loss.item()}
    trainer.log(logs)
    assert logs["loss_ce"] + logs["loss_kl"] == pytest.approx(logs["loss"])
    assert logs["loss_ce"] > 0 and logs["loss_kl"] > 0
    # the parts restart after each log and evaluation logs are untouched
    trainer.compute_loss(model, inputs)
    trainer.log({"loss": loss.item()})
    eval_logs = {"eval_loss": 1.0}
    trainer.log(eval_logs)
    assert "loss_ce" not in eval_logs
    model.eval()
    trainer.compute_loss(model, inputs)
    later = {"loss": 1.0}
    trainer.log(later)
    assert "loss_ce" not in later
