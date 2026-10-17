import numpy as np
import pytest
from transformers import EvalPrediction

from src.evaluation.metrics import DecisionMetrics


def evaluate(probabilities: list[list[float]], labels: list[list[float]]) -> dict:
    """Runs the metrics as `Trainer` would: predictions are (logits, probs)."""
    probs = np.array(probabilities, dtype=np.float32)
    logits = np.zeros_like(probs)
    return DecisionMetrics()(EvalPrediction((logits, probs), np.array(labels)))


def test_perfect_predictions() -> None:
    metrics = evaluate([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert metrics == pytest.approx({"accuracy": 1.0, "brier": 0.0, "ece": 0.0})


def test_known_values() -> None:
    probabilities = [[0.8, 0.2], [0.6, 0.4]]
    labels = [[1.0, 0.0], [0.0, 1.0]]
    metrics = evaluate(probabilities, labels)
    assert metrics["accuracy"] == pytest.approx(0.5)
    # (0.2^2 + 0.2^2 + 0.6^2 + 0.6^2) / 2
    assert metrics["brier"] == pytest.approx(0.4)
    # bins: 0.8 right (gap 0.2), 0.6 wrong (gap 0.6), each half the data
    assert metrics["ece"] == pytest.approx(0.4)


def test_padding_from_trainer_is_ignored() -> None:
    padded = evaluate(
        [[0.7, 0.3, -100.0], [0.1, 0.2, 0.7]],
        [[1.0, 0.0, -100.0], [0.0, 0.0, 1.0]],
    )
    plain = evaluate(
        [[0.7, 0.3, 0.0], [0.1, 0.2, 0.7]], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    assert padded == pytest.approx(plain)
    assert padded["accuracy"] == pytest.approx(1.0)


def test_soft_labels_use_most_likely_option() -> None:
    metrics = evaluate([[0.2, 0.8]], [[0.3, 0.7]])
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier"] == pytest.approx(0.1**2 * 2)


def test_errors_count_wrong() -> None:
    # System Two: predictions are (logits, probabilities, errors); the
    # uniform error row ties on its label A, but is an error
    probs = np.array([[0.5, 0.5], [0.9, 0.1]], dtype=np.float32)
    errors = np.array([True, False])
    labels = np.array([[1.0, 0.0], [1.0, 0.0]])
    metrics = DecisionMetrics(system_two=True)(
        EvalPrediction((np.zeros_like(probs), probs, errors), labels)
    )
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["errors"] == pytest.approx(0.5)
    # bins: 0.5 wrong (gap 0.5), 0.9 right (gap 0.1)
    assert metrics["ece"] == pytest.approx(0.3)


def test_system_one_has_no_errors_metric() -> None:
    assert "errors" not in evaluate([[1.0, 0.0]], [[1.0, 0.0]])
