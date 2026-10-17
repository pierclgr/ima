from dataclasses import replace
from typing import Any

from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput

from src.common.prompt import PromptBuilder
from src.data.collator import DecisionEngineCollator
from src.data.dataset import DecisionEngineDataset
from src.evaluation.metrics import DecisionMetrics
from src.model.system_one import PreTrainedSystemOneModel
from src.model.system_two import SystemTwoModel


class DecisionEvaluator:
    """Runs a decision model on labelled records, without training.

    Used to test a model (System One or Two) and to fit its temperature.

    Args:
        model: The System One model, or a `SystemTwoModel`.
        prompt: Prompt builder of the model (System Two: `model.prompt`).
        training_args: `Trainer` arguments (device, eval batch size,
            precision). A copy is used, with `remove_unused_columns` False
            (the collator needs the raw items) and `eval_strategy` "no" (no
            eval dataset; the training's args may have evaluation on).
    """

    def __init__(
        self,
        model: PreTrainedSystemOneModel | SystemTwoModel,
        prompt: PromptBuilder,
        training_args: TrainingArguments,
    ) -> None:
        self.max_options: int = model.config.max_options
        self.trainer: Trainer = Trainer(
            model=model,
            args=replace(
                training_args, remove_unused_columns=False, eval_strategy="no"
            ),
            data_collator=DecisionEngineCollator(prompt),
            compute_metrics=DecisionMetrics(
                system_two=isinstance(model, SystemTwoModel)
            ),
        )

    def predict(self, records: Any) -> PredictionOutput:
        """Runs the model on the records (options in their original order).

        Returns:
            `predictions` (option logits, probabilities and, for System Two,
            errors), `label_ids` and `metrics` (`test_loss`,
            `test_accuracy`, `test_brier`, `test_ece`, System Two
            `test_errors`), plus evaluated and skipped question counts.
        """
        dataset = DecisionEngineDataset(
            records, shuffle_options=False, max_options=self.max_options
        )
        output = self.trainer.predict(dataset)
        output.metrics["test_evaluated_questions"] = len(dataset)
        output.metrics["test_skipped_questions"] = dataset.skipped
        return output
