from abc import ABC, abstractmethod
from typing import Any

from src.common.types import Question


class QuestionType(ABC):
    """Behavior of a `/v1/systemone` question type.

    Attributes:
        shuffle: Whether the options can be shuffled in training (their order
            has no meaning).
    """

    shuffle: bool = False

    @abstractmethod
    def build(self, text: str, criteria: Any) -> Question:
        """Turns the question text and criteria into a lettered `Question`."""

    @abstractmethod
    def option_index(self, key: Any, question: Question) -> int:
        """Returns the option index of a label or target key.

        Raises:
            ValueError: If the key matches no option.
        """

    @abstractmethod
    def format_answer(
        self, question: Question, probabilities: list[float]
    ) -> dict[str, Any]:
        """Builds the `/v1/systemone` answer from the option probabilities."""


class ChoiceType(QuestionType):
    """`choice`: options are the criteria keys, labels are option names."""

    shuffle: bool = True

    def build(self, text: str, criteria: Any) -> Question:
        return Question(text, list(criteria), list(criteria.values()))

    def option_index(self, key: Any, question: Question) -> int:
        if key not in question.options:
            raise ValueError(f"unknown option {key!r}")
        return question.options.index(key)

    def format_answer(
        self, question: Question, probabilities: list[float]
    ) -> dict[str, Any]:
        k: int = len(probabilities)
        best: int = probabilities.index(max(probabilities))
        return {
            "type": "choice",
            "choice": question.options[best],
            "confidence": max(0.0, (probabilities[best] - 1 / k) / (1 - 1 / k)),
            "probabilities": dict(zip(question.options, probabilities)),
        }


class NoulType(QuestionType):
    """`noul`: options "Yes" and "No", labels true/false."""

    def build(self, text: str, criteria: Any) -> Question:
        criteria = criteria or {}
        return Question(
            text, ["Yes", "No"], [criteria.get("true"), criteria.get("false")]
        )

    def option_index(self, key: Any, question: Question) -> int:
        if key not in (True, False, "true", "false"):
            raise ValueError(f"unknown noul label {key!r}")
        return 0 if key in (True, "true") else 1

    def format_answer(
        self, question: Question, probabilities: list[float]
    ) -> dict[str, Any]:
        return {"type": "noul", "noul": probabilities[0]}


class ScoreType(QuestionType):
    """`score`: options are the levels, labels are level indices."""

    def build(self, text: str, criteria: Any) -> Question:
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria must be an array of at least 2 levels")
        return Question(text, list(criteria))

    def option_index(self, key: Any, question: Question) -> int:
        index: int = int(key)
        if not 0 <= index < len(question.options):
            raise ValueError(f"level {index} out of range")
        return index

    def format_answer(
        self, question: Question, probabilities: list[float]
    ) -> dict[str, Any]:
        k: int = len(probabilities)
        best: int = probabilities.index(max(probabilities))
        # llama.cpp: spread around the mode, uniform spread around the center
        spread: float = sum(p * abs(i - best) for i, p in enumerate(probabilities))
        uniform: float = sum(abs(i - (k - 1) / 2) for i in range(k)) / k
        return {
            "type": "score",
            "score": sum(i * p for i, p in enumerate(probabilities)),
            "confidence": max(0.0, 1 - spread / uniform),
            "legend": {str(i): level for i, level in enumerate(question.options)},
            "probabilities": {str(i): p for i, p in enumerate(probabilities)},
        }


QUESTION_TYPES: dict[str, QuestionType] = {
    "choice": ChoiceType(),
    "noul": NoulType(),
    "score": ScoreType(),
}


def question_type(kind: str) -> QuestionType:
    """Returns the question type named `kind`.

    Raises:
        ValueError: If the question type is unknown.
    """
    if kind not in QUESTION_TYPES:
        raise ValueError(f"unknown question type {kind!r}")
    return QUESTION_TYPES[kind]
