import json
from pathlib import Path

import datasets
import numpy as np
import pytest
import torch

from src.calibration.calibrate import TemperatureCalibrator
from src.common.prompt import PromptBuilder
from src.data.collator import DecisionEngineCollator
from src.data.dataset import DecisionEngineDataset
from src.distillation.cache import TeacherTargetCache
from src.distillation.trainer import DistillationCollator
from conftest import TinyProcessor


def write_cache(path: Path, records: datasets.Dataset) -> TeacherTargetCache:
    """Writes mixed-option predictions, including Trainer padding."""
    probabilities = np.tile([[0.6, 0.3, 0.1], [0.8, 0.2, -100],
                             [0.4, 0.6, 0]], (len(records), 1))
    TeacherTargetCache.write(path, "teacher", 1.8, "dataset", 26,
                             {"train": ("train", records, probabilities)})
    return TeacherTargetCache(path)


def test_cache_alignment_split_shuffle_and_collation(
    tmp_path: Path, records: datasets.Dataset, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = write_cache(tmp_path, records)
    enriched = cache.attach(records, "dataset", "train", "train", 26)
    train, calibration = TemperatureCalibrator.split(enriched, 0.5, 42)
    assert len(train) + len(calibration) == len(records)
    assert "teacher_targets" in calibration[0]
    train_states = {record["state"]["id"] for record in train}
    calibration_states = {record["state"]["id"] for record in calibration}
    assert train_states.isdisjoint(calibration_states)
    _, original_calibration = TemperatureCalibrator.split(records, 0.5, 42)
    assert calibration_states == {
        record["state"]["id"] for record in original_calibration
    }
    monkeypatch.setattr(torch, "randperm", lambda n: torch.tensor([2, 0, 1]))
    dataset = DecisionEngineDataset(train)
    item = dataset[0]
    assert item["question"].options == ["green", "red", "blue"]
    assert item["target"] == [0, 0.75, 0.25]
    assert item["teacher_target"] == pytest.approx([0.1, 0.6, 0.3])
    binary, score = dataset[1], dataset[2]
    assert binary["teacher_target"] == pytest.approx([0.8, 0.2])
    assert score["teacher_target"] == pytest.approx([0.4, 0.6])
    prompt = PromptBuilder(TinyProcessor(), 26)
    batch = DistillationCollator(prompt)([item, binary])
    torch.testing.assert_close(batch["teacher_targets"],
                               torch.tensor([[0.1, 0.6, 0.3], [0.8, 0.2, 0]]))
    assert not batch["option_mask"][1, 2]
    assert "teacher_targets" not in DecisionEngineCollator(prompt)([item])
    with pytest.raises(ValueError, match="teacher"):
        DistillationCollator(prompt)([item, DecisionEngineDataset(records)[1]])


@pytest.mark.parametrize(
    "change", ["dataset", "split", "fingerprint", "options"],
)
def test_cache_rejects_wrong_source(
    tmp_path: Path, records: datasets.Dataset, change: str,
) -> None:
    cache = write_cache(tmp_path, records)
    dataset, split = "dataset", "train"
    if change == "dataset":
        dataset = "different"
    elif change == "split":
        split = "train[:2]"
    elif change == "fingerprint":
        records = records.shuffle(seed=7)
    else:
        target_file = tmp_path / "train.jsonl"
        rows = target_file.read_text().splitlines()
        row = json.loads(rows[0])
        row["options"].reverse()
        rows[0] = json.dumps(row)
        target_file.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        cache.attach(records, dataset, "train", split, 26)


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "negative", "nan", "sum", "schema"],
)
def test_cache_rejects_invalid_targets(
    tmp_path: Path, records: datasets.Dataset, change: str,
) -> None:
    cache = write_cache(tmp_path, records)
    target_file = tmp_path / "train.jsonl"
    rows = target_file.read_text().splitlines()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "schema":
        rows[0] = "{}"
    else:
        row = json.loads(rows[0])
        row["probabilities"] = {
            "negative": [-0.1, 0.8, 0.3], "nan": [float("nan"), 0.5, 0.5],
            "sum": [0.2, 0.2, 0.2],
        }[change]
        rows[0] = json.dumps(row)
    target_file.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        cache.attach(records, "dataset", "train", "train", 26)


def test_cache_requires_manifest_and_preserves_skip_rule(
    tmp_path: Path, records: datasets.Dataset,
) -> None:
    with pytest.raises(ValueError, match="manifest"):
        TeacherTargetCache(tmp_path)
    probabilities = np.tile([[0.8, 0.2], [0.4, 0.6]], (len(records), 1))
    TeacherTargetCache.write(tmp_path, "teacher", 1.8, "dataset", 2,
                             {"train": ("train", records, probabilities)})
    cache = TeacherTargetCache(tmp_path)
    enriched = cache.attach(records, "dataset", "train", "train", 2)
    dataset = DecisionEngineDataset(enriched, max_options=2)
    assert dataset.skipped == len(records)
    assert len(dataset) == 2 * len(records)
    with pytest.raises(ValueError):
        cache.attach(records, "dataset", "train", "train", 26)
    with pytest.raises(ValueError, match="exist"):
        TeacherTargetCache.write(tmp_path, "teacher", 1.8, "dataset", 2,
                                 {"train": ("train", records, probabilities)})


def test_failed_export_does_not_publish_manifest(
    tmp_path: Path, records: datasets.Dataset,
) -> None:
    invalid = np.zeros((12, 3))
    with pytest.raises(ValueError, match="probability"):
        TeacherTargetCache.write(
            tmp_path, "teacher", 1.8, "dataset", 26,
            {"train": ("train", records, invalid)},
        )
    assert not (tmp_path / "manifest.json").exists()
    with pytest.raises(ValueError, match="manifest"):
        TeacherTargetCache(tmp_path)
