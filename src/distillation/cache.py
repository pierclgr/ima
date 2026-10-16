"""Small, checked teacher-target files alongside the original HF data."""

import json
import math
from pathlib import Path
from typing import Any

import datasets
import numpy as np

from src.common.request import RequestParser
from src.constants import MAX_LETTER_OPTIONS
from src.data.dataset import DecisionEngineDataset


class TeacherTargetCache:
    """Loads a complete cache and attaches aligned targets to source records.

    The manifest is the completion marker. Indices refer to the original
    split before any record selection or calibration carve-out.
    """

    VERSION: int = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        manifest = self.path / "manifest.json"
        if not manifest.is_file():
            raise ValueError("teacher cache manifest is missing or incomplete")
        self.manifest: dict[str, Any] = json.loads(manifest.read_text())
        if self.manifest.get("version") != self.VERSION:
            raise ValueError("unsupported teacher cache version")
        required = {
            "teacher", "temperature", "dataset", "max_options", "splits",
        }
        if not required <= self.manifest.keys():
            raise ValueError("teacher cache manifest is incomplete")
        temperature = self.manifest["temperature"]
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid teacher temperature in manifest")
        if not 2 <= self.manifest["max_options"] <= MAX_LETTER_OPTIONS:
            raise ValueError("invalid teacher option limit in manifest")
        splits = self.manifest["splits"]
        if "train" not in splits or set(splits) - {"train", "validation"}:
            raise ValueError("invalid teacher cache splits")
        for name, metadata in splits.items():
            required_split = {"split", "fingerprint", "records", "questions"}
            if not required_split <= metadata.keys():
                raise ValueError("teacher cache split metadata is incomplete")
            if not (self.path / f"{name}.jsonl").is_file():
                raise ValueError(f"teacher cache is missing {name} targets")

    @staticmethod
    def probabilities(values: Any, count: int) -> list[float]:
        """Checks a finite, normalized distribution over the real options."""
        probabilities = np.asarray(values, dtype=np.float64)
        if (probabilities.shape != (count,)
                or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or (probabilities > 1).any()
                or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-5)):
            raise ValueError("invalid teacher probability distribution")
        return probabilities.tolist()

    @classmethod
    def write(
        cls, path: str | Path, teacher: str, temperature: float,
        dataset: str, max_options: int,
        splits: dict[str, tuple[str, datasets.Dataset, np.ndarray]],
    ) -> None:
        """Writes split files atomically, then publishes the manifest last.

        Args:
            path: A fresh cache directory.
            teacher: Fine-tuned teacher path or HF id.
            temperature: Teacher's saved calibration temperature.
            dataset: Source HF dataset id.
            max_options: Teacher option limit.
            splits: train and optional validation: expression, records,
                and predictions in the dataset's canonical question order.
        """
        path = Path(path)
        if path.exists() and any(path.iterdir()):
            raise ValueError("target cache directory exists; use a new one")
        if "train" not in splits or set(splits) - {"train", "validation"}:
            raise ValueError("expected train and optional validation splits")
        path.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "version": cls.VERSION, "teacher": teacher,
            "temperature": temperature, "dataset": dataset,
            "max_options": max_options, "splits": {},
        }
        for name, (expression, records, predictions) in splits.items():
            questions = DecisionEngineDataset(
                records, shuffle_options=False, max_options=max_options,
            )
            if predictions.ndim != 2 or len(predictions) != len(questions):
                raise ValueError("teacher prediction count does not match")
            temporary = path / f"{name}.jsonl.tmp"
            with temporary.open("x") as output:
                for i, key in enumerate(questions.index):
                    record_index, question_id = key
                    options = RequestParser.build_question(
                        records[record_index]["questions"][question_id],
                    ).options
                    row = {
                        "record_index": record_index,
                        "question_id": question_id,
                        "options": options, "probabilities": cls.probabilities(
                            predictions[i, :len(options)], len(options),
                        ),
                    }
                    output.write(json.dumps(row, allow_nan=False) + "\n")
            temporary.replace(path / f"{name}.jsonl")
            manifest["splits"][name] = {
                "split": expression, "fingerprint": records._fingerprint,
                "records": len(records), "questions": len(questions),
            }
        temporary = path / "manifest.json.tmp"
        temporary.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        )
        temporary.replace(path / "manifest.json")

    def attach(
        self, records: datasets.Dataset, dataset: str, name: str,
        expression: str, max_options: int,
    ) -> datasets.Dataset:
        """Validates the source and adds targets before calibration splitting.

        Targets are a JSON string column to avoid Arrow struct fields for
        every possible question id. The Hub record transform decodes them.
        """
        metadata = self.manifest["splits"].get(name)
        if (self.manifest["dataset"] != dataset
                or self.manifest["max_options"] < max_options
                or metadata is None
                or metadata["split"] != expression
                or metadata["fingerprint"] != records._fingerprint
                or metadata["records"] != len(records)):
            raise ValueError("teacher cache does not match the source data")
        questions = DecisionEngineDataset(
            records, shuffle_options=False,
            max_options=self.manifest["max_options"],
        )
        expected = set(questions.index)
        targets: list[dict[str, list[float]]] = [
            {} for _ in range(len(records))
        ]
        seen: set[tuple[int, str]] = set()
        with (self.path / f"{name}.jsonl").open() as source:
            for line in source:
                row = json.loads(line)
                required_row = {
                    "record_index", "question_id", "options", "probabilities",
                }
                if not isinstance(row, dict) or not required_row <= row.keys():
                    raise ValueError("teacher cache row is incomplete")
                key = (row["record_index"], row["question_id"])
                if key not in expected or key in seen:
                    raise ValueError("unknown or duplicate teacher question")
                record_index, question_id = key
                options = RequestParser.build_question(
                    records[record_index]["questions"][question_id],
                ).options
                if row["options"] != options:
                    raise ValueError("teacher cache option order differs")
                targets[record_index][question_id] = self.probabilities(
                    row["probabilities"], len(options),
                )
                seen.add(key)
        if seen != expected or len(seen) != metadata["questions"]:
            raise ValueError("teacher cache has missing question targets")
        return records.add_column(
            "teacher_targets", [json.dumps(t) for t in targets],
        )
