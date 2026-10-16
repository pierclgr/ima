from pathlib import Path

import pytest

from src.common.config_parser import ConfigParser
from src.distillation.config import DistillationConfig, DistillationSettings
from src.training.config import TrainConfig


@pytest.mark.parametrize("alpha,temperature", [(-0.1, 1), (1.1, 1),
                                               (0.25, 0), (0.25, -1),
                                               (float("nan"), 1),
                                               (0.25, float("inf"))])
def test_invalid_loss_settings(alpha: float, temperature: float) -> None:
    with pytest.raises(ValueError):
        DistillationSettings("teacher", "labels", alpha, temperature)


@pytest.mark.parametrize("values", [
    {"soft_labels": "labels"}, {"teacher": "teacher"},
    {"teacher": "t", "soft_labels": "l", "generating": {"output_dir": "x"}},
])
def test_invalid_distillation_section(values: dict) -> None:
    with pytest.raises(ValueError):
        DistillationSettings(**values)


def test_nested_overrides(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text("distillation: {alpha: 0.5}\n")
    values = ConfigParser.load(
        path, ["--distillation.alpha", "0.1",
               "--distillation.generating.bf16", "true",
               "--training.num_epochs", "2"],
        keys=[], sections=["training", "distillation"],
    )
    assert values == {
        "distillation": {"alpha": "0.1", "generating": {"bf16": "true"}},
        "training": {"num_epochs": "2"},
    }


def test_training_config_distillation_section(tmp_path: Path) -> None:
    path = tmp_path / "student.yml"
    path.write_text(
        "backbone: student\nseed: 7\n"
        "data: {dataset: dataset, train: train}\n"
        "training: {output_dir: run, use_cpu: true, report_to: none}\n"
        "distillation: {teacher: teacher, soft_labels: labels}\n"
        "calibration: {temperature: fit, split: 0.1}\n",
    )
    config = TrainConfig.from_yaml(
        path, ["--distillation.temperature", "2",
               "--training.num_epochs", "3"],
    )
    assert config.distillation == DistillationSettings(
        "teacher", "labels", alpha=0.25, temperature=2,
    )
    assert config.training.num_train_epochs == 3
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(path, ["--distillation.unknown", "1"])
    with pytest.raises(ValueError, match="teacher"):
        TrainConfig.from_yaml(path, ["--distillation.teacher", ""])


def test_plain_training_config_has_no_distillation(tmp_path: Path) -> None:
    path = tmp_path / "train.yml"
    path.write_text(
        "backbone: student\ndata: {dataset: dataset, train: train}\n"
        "training: {output_dir: run, use_cpu: true, report_to: none}\n",
    )
    assert TrainConfig.from_yaml(path).distillation is None


def test_training_config_adapter_key(tmp_path: Path) -> None:
    path = tmp_path / "train.yml"
    path.write_text(
        "backbone: student\ndata: {dataset: dataset, train: train}\n"
        "training: {output_dir: run, use_cpu: true, report_to: none}\n",
    )
    assert TrainConfig.from_yaml(path).adapter is None
    config = TrainConfig.from_yaml(path, ["--adapter", "runs/x/checkpoint-1"])
    assert config.adapter == "runs/x/checkpoint-1"


def test_cache_only_config(tmp_path: Path) -> None:
    path = tmp_path / "cache.yml"
    path.write_text(
        "data: {dataset: dataset, train: train, validation: validation}\n"
        "distillation:\n  teacher: teacher\n  soft_labels: labels\n"
        "  generating: {per_device_eval_batch_size: 4}\n",
    )
    config = DistillationConfig.from_yaml(
        path, ["--distillation.generating.use_cpu", "true"],
    )
    assert config.data.validation == "validation"
    assert config.distillation.soft_labels == "labels"
    assert config.distillation.generating == {
        "per_device_eval_batch_size": 4, "use_cpu": "true",
    }
    with pytest.raises(ValueError):
        DistillationConfig.from_yaml(path, ["--lora.r", "8"])
    path.write_text("data: {dataset: dataset, train: train}\n")
    with pytest.raises(ValueError, match="distillation"):
        DistillationConfig.from_yaml(path)
