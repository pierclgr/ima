from typing import Any

import numpy as np
import torch
from transformers import EvalPrediction
from transformers.trainer_utils import PredictionOutput

from src.constants import MAX_LETTER_OPTIONS
from src.data.dataset import DecisionEngineDataset
from src.evaluation.metrics import DecisionMetrics
from src.llamacpp.runtime import LlamaCppRuntime
from src.model.output import DecisionEngineOutput


class LlamaCppEvaluator:
    """Runs native System One inference with the shared dataset and metrics.

    Args:
        runtime: Persistent native model worker.
        temperature: Calibration override, or None for saved temperatures.

    Attributes:
        times: Native inference seconds in eligible question order.
    """

    def __init__(
        self, runtime: LlamaCppRuntime, temperature: float | None = None,
    ) -> None:
        self.runtime = runtime
        self.temperature = temperature
        self.times: list[float] = []

    def predict(self, records: Any) -> PredictionOutput:
        """Returns prediction arrays and metrics in the HF test format."""
        dataset = DecisionEngineDataset(records, shuffle_options=False)
        if not len(dataset):
            raise ValueError("no eligible questions to evaluate")
        shape = (len(dataset), MAX_LETTER_OPTIONS)
        logits = np.full(
            shape, torch.finfo(torch.float32).min, dtype=np.float32
        )
        probabilities = np.zeros(shape, dtype=np.float32)
        labels = np.zeros(shape, dtype=np.float32)
        losses: list[float] = []
        self.times = []
        for i in range(len(dataset)):
            item = dataset[i]
            result = self.runtime.predict(item)
            count = len(item["target"])
            raw = torch.tensor(
                result["logits"], dtype=torch.float32
            ).unsqueeze(0)
            if raw.shape != (1, count) or not torch.isfinite(raw).all():
                raise RuntimeError("native worker returned invalid logits")
            temperature = (
                self.temperature if self.temperature is not None
                else self.runtime.temperatures[item["type"]]
            )
            target = torch.tensor([item["target"]], dtype=torch.float32)
            output = DecisionEngineOutput.from_logits(raw, temperature, target)
            logits[i, :count] = output.logits[0].numpy()
            probabilities[i, :count] = output.probabilities[0].numpy()
            labels[i, :count] = target[0].numpy()
            losses.append(output.loss.item())
            self.times.append(float(result["seconds"]))
        predictions = (logits, probabilities)
        metrics = {f"test_{key}": value for key, value in DecisionMetrics()(
            EvalPrediction(predictions=predictions, label_ids=labels)
        ).items()}
        metrics.update(test_loss=float(np.mean(losses)),
                       test_evaluated_questions=len(dataset),
                       test_skipped_questions=dataset.skipped)
        return PredictionOutput(predictions, labels, metrics)
