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
) -> torch.Tensor:
    """Returns mean human CE plus T-squared forward KL over real options.

    Teacher probabilities include the teacher's fitted calibration already.
    Student calibration is excluded: both terms use raw student logits.
    """
    logits = logits.float()
    minimum = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~option_mask, minimum)
    ce = F.cross_entropy(logits, labels.float())
    if settings.alpha == 1:
        return ce
    temperature = settings.temperature
    # log-space softening avoids underflow when T is small
    teacher = teacher_targets.float().masked_fill(~option_mask, 0)
    log_teacher = torch.log(teacher) / temperature
    softened = torch.softmax(log_teacher, dim=-1)
    log_student = F.log_softmax(
        (logits / temperature).masked_fill(~option_mask, minimum), dim=-1,
    )
    kl = F.kl_div(log_student, softened, reduction="batchmean")
    return settings.alpha * ce + (1 - settings.alpha) * temperature**2 * kl


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
    """Uses cached targets during training and human CE during evaluation."""

    def __init__(self, *args: Any, distillation: DistillationSettings,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.distillation = distillation
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
        loss = (
            distillation_loss(outputs.logits, inputs["labels"], teacher,
                              inputs["option_mask"], self.distillation)
            if model.training else outputs.loss
        )
        return (loss, outputs) if return_outputs else loss
