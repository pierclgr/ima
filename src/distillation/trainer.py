"""Teacher-target collation and masked option-distribution distillation."""

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer

from src.common.prompt import PromptBuilder
from src.data.collator import DecisionEngineCollator
from src.distillation.config import DistillationSettings


def distillation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_targets: torch.Tensor,
    option_mask: torch.Tensor,
    settings: DistillationSettings,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the loss and its two weighted terms (loss = CE term + KL term).

    The terms are alpha times the mean human CE and (1 - alpha) times the
    T-squared forward KL over real options.
    Teacher probabilities include the teacher's fitted calibration already.
    Student calibration is excluded: both terms use raw student logits.
    """
    logits = logits.float()
    minimum = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~option_mask, minimum)
    ce = F.cross_entropy(logits, labels.float())
    if settings.alpha == 1:
        return ce, ce, torch.zeros_like(ce)
    temperature = settings.temperature
    # log-space softening avoids underflow when T is small
    teacher = teacher_targets.float().masked_fill(~option_mask, 0)
    log_teacher = torch.log(teacher) / temperature
    softened = torch.softmax(log_teacher, dim=-1)
    log_student = F.log_softmax(
        (logits / temperature).masked_fill(~option_mask, minimum), dim=-1,
    )
    kl = F.kl_div(log_student, softened, reduction="batchmean")
    ce_term = settings.alpha * ce
    kl_term = (1 - settings.alpha) * temperature**2 * kl
    return ce_term + kl_term, ce_term, kl_term


class DistillationCollator:
    """Adds aligned teacher distributions to the ordinary decision batch.

    Batches without teacher targets are allowed for human-only validation.
    Mixed batches are rejected rather than silently dropping supervision.
    """

    def __init__(self, prompt: PromptBuilder) -> None:
        self.collator = DecisionEngineCollator(prompt)

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Pads teacher targets with zero probability on missing options."""
        batch = self.collator(items)
        present = ["teacher_target" in item for item in items]
        if any(present):
            if not all(present):
                raise ValueError("batch contains missing teacher targets")
            teacher = torch.zeros_like(batch["labels"])
            for row, item in enumerate(items):
                teacher[row, :len(item["teacher_target"])] = torch.tensor(
                    item["teacher_target"], dtype=torch.float32,
                )
            batch["teacher_targets"] = teacher
        return batch


class DistillationTrainer(Trainer):
    """Uses cached targets during training and human CE during evaluation.

    Training logs also hold `loss_ce` and `loss_kl`, the two weighted terms
    of `loss` (their sum), averaged over the steps since the last log.
    """

    def __init__(self, *args: Any, distillation: DistillationSettings,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.distillation = distillation
        self.terms: torch.Tensor | None = None
        self.term_steps: int = 0
        # soft-label elements are options, not the question count
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model: nn.Module, inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Consumes teacher metadata and preserves the usual model outputs."""
        inputs = dict(inputs)
        teacher = inputs.pop("teacher_targets", None)
        if model.training and teacher is None:
            raise ValueError("distillation training requires teacher targets")
        outputs = model(**inputs)
        if model.training:
            loss, ce_term, kl_term = distillation_loss(
                outputs.logits, inputs["labels"], teacher,
                inputs["option_mask"], self.distillation,
            )
            terms = torch.stack([ce_term, kl_term]).detach()
            self.terms = terms if self.terms is None else self.terms + terms
            self.term_steps += 1
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def log(
        self, logs: dict[str, float], start_time: float | None = None,
    ) -> None:
        """Adds the loss terms to a training log and restarts their average."""
        if "loss" in logs and self.terms is not None:
            logs["loss_ce"], logs["loss_kl"] = (
                self.terms / self.term_steps
            ).tolist()
            self.terms, self.term_steps = None, 0
        super().log(logs, start_time)
