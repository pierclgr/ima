from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import Dataset

from src.common.question_types import QuestionType, question_type
from src.common.request import RequestParser
from src.common.types import Question
from src.constants import MAX_LETTER_OPTIONS


class DecisionEngineDataset(Dataset):
    """Training items from labelled `/v1/systemone` records.

    A record is a request whose questions also hold a `label` (choice: option
    name, noul: true/false, score: level index) and optionally a soft `target`
    (weights keyed like the label, e.g. vote counts; normalized here). Each
    question of each record is one item. Questions with more options than
    `max_options` are skipped and counted in `skipped`.

    Args:
        records: The labelled records.
        shuffle_options: Shuffle the options of the question types that
            allow it (`choice`) at each access (the target follows its
            option, together with any cached teacher target).
        max_options: Max options per question (one letter each).

    Raises:
        ValueError: If a question type is unknown.
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        shuffle_options: bool = True,
        max_options: int = MAX_LETTER_OPTIONS,
    ) -> None:
        self.records: Sequence[dict[str, Any]] = records
        self.shuffle_options: bool = shuffle_options
        self.index: list[tuple[int, str]] = []
        self.skipped: int = 0
        for i, record in enumerate(records):
            for question_id, spec in record["questions"].items():
                question: Question = RequestParser.build_question(spec)
                if len(question.options) > max_options:
                    self.skipped += 1
                else:
                    self.index.append((i, question_id))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, Any]:
        """Returns one question as an item.

        Returns:
            `type`, `state` (text), `images`, `question` (`Question`) and
            `target` (human probabilities per option), plus `teacher_target`
            when the record has cached teacher supervision.

        Raises:
            ValueError: If the label or a target key is unknown, or the target
                sums to 0.
        """
        record_index, question_id = self.index[i]
        record: dict[str, Any] = self.records[record_index]
        spec: dict[str, Any] = record["questions"][question_id]
        kind: QuestionType = question_type(spec["type"])
        question: Question = RequestParser.build_question(spec)
        target: list[float] = self._target(spec, kind, question)
        teacher: list[float] | None = (
            record["teacher_targets"][question_id]
            if "teacher_targets" in record else None
        )
        if self.shuffle_options and kind.shuffle:
            question, target, teacher = self._shuffle(question, target, teacher)
        item: dict[str, Any] = {
            "type": spec["type"],
            "state": RequestParser.render_text(record["state"]),
            "images": RequestParser.load_media(record.get("media", [])),
            "question": question,
            "target": target,
        }
        if teacher is not None:
            item["teacher_target"] = teacher
        return item

    @staticmethod
    def _target(
        spec: dict[str, Any], kind: QuestionType, question: Question
    ) -> list[float]:
        weights: list[float] = [0.0] * len(question.options)
        if "target" in spec:
            for key, weight in spec["target"].items():
                weights[kind.option_index(key, question)] = float(weight)
        else:
            weights[kind.option_index(spec["label"], question)] = 1.0
        total: float = sum(weights)
        if total <= 0:
            raise ValueError("target sums to 0")
        return [weight / total for weight in weights]

    @staticmethod
    def _shuffle(
        question: Question, target: list[float], teacher: list[float] | None
    ) -> tuple[Question, list[float], list[float] | None]:
        order: list[int] = torch.randperm(len(question.options)).tolist()
        descriptions = question.descriptions or [None] * len(question.options)
        shuffled = Question(
            question.text,
            [question.options[j] for j in order],
            [descriptions[j] for j in order],
        )
        return (shuffled, [target[j] for j in order],
                [teacher[j] for j in order] if teacher is not None else None)
