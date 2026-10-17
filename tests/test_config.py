from pathlib import Path

import pytest

from src.training.config import TrainConfig

YAML: str = """
backbone: Qwen/Qwen3.5-0.8B
lora:
  r: 8
  alpha: 16
data:
  dataset: Jacqkues/kev-vision-decisions-full
  train: train
  validation: validation
training:
  output_dir: runs/test
  learning_rate: 1e-4
  per_device_train_batch_size: 2
  report_to: [none]
"""


def write(tmp_path: Path, text: str = YAML) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(text)
    return path


def test_loads_typed_sections(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(write(tmp_path))
    assert config.backbone == "Qwen/Qwen3.5-0.8B"
    assert (config.lora.r, config.lora.alpha) == (8, 16)
    assert config.data.dataset == "Jacqkues/kev-vision-decisions-full"
    assert (config.data.train, config.data.validation) == ("train", "validation")
    assert config.training.output_dir == "runs/test"
    assert config.training.learning_rate == pytest.approx(1e-4)
    assert config.training.per_device_train_batch_size == 2
    assert config.training.report_to == []


def test_cli_overrides(tmp_path: Path) -> None:
    overrides = [
        "--backbone", "other/model",
        "--lora.r", "4",
        "--data.train", "train[:10]",
        "--training.learning_rate", "3e-4",
    ]
    config = TrainConfig.from_yaml(write(tmp_path), overrides)
    assert config.backbone == "other/model"
    assert config.lora.r == 4
    assert config.data.train == "train[:10]"
    assert config.training.learning_rate == pytest.approx(3e-4)


def test_num_epochs(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(write(tmp_path, YAML + "  num_epochs: 3\n"))
    assert config.training.num_train_epochs == 3
    overridden = TrainConfig.from_yaml(
        write(tmp_path), ["--training.num_epochs", "2"]
    )
    assert overridden.training.num_train_epochs == 2


def test_num_train_epochs_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(write(tmp_path, YAML + "  num_train_epochs: 3\n"))


def test_checkpoint(tmp_path: Path) -> None:
    assert TrainConfig.from_yaml(write(tmp_path)).checkpoint is None
    text = YAML + "checkpoint: runs/test/checkpoint-500\n"
    config = TrainConfig.from_yaml(write(tmp_path, text))
    assert config.checkpoint == "runs/test/checkpoint-500"
    overridden = TrainConfig.from_yaml(
        write(tmp_path), ["--checkpoint", "runs/test/checkpoint-9"]
    )
    assert overridden.checkpoint == "runs/test/checkpoint-9"


def test_logs_every_update_by_default(tmp_path: Path) -> None:
    assert TrainConfig.from_yaml(write(tmp_path)).training.logging_steps == 1
    overridden = TrainConfig.from_yaml(
        write(tmp_path), ["--training.logging_steps", "5"]
    )
    assert overridden.training.logging_steps == 5


def test_no_calibration_section(tmp_path: Path) -> None:
    assert TrainConfig.from_yaml(write(tmp_path)).calibration is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("calibration:\n  temperature: fit\n", "fit"),
        ("calibration:\n  temperature: 1.5\n", 1.5),
        ("calibration:\n", "fit"),
    ],
)
def test_calibration_section(tmp_path: Path, text: str, expected: object) -> None:
    config = TrainConfig.from_yaml(write(tmp_path, YAML + text))
    assert config.calibration.temperature == expected


def test_calibration_override(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(
        write(tmp_path), ["--calibration.temperature", "2.5"]
    )
    assert config.calibration.temperature == 2.5


@pytest.mark.parametrize(
    "text",
    [
        "calibration:\n  temperature: maybe\n",
        "calibration:\n  temperature: 0\n",
        "calibration:\n  bogus: 1\n",
    ],
)
def test_bad_calibration_raises(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(write(tmp_path, YAML + text))


def test_calibration_split(tmp_path: Path) -> None:
    default = TrainConfig.from_yaml(write(tmp_path, YAML + "calibration:\n"))
    assert default.calibration.split == 0.1
    text = YAML + "calibration:\n  temperature: fit\n  split: 0.2\n"
    assert TrainConfig.from_yaml(write(tmp_path, text)).calibration.split == 0.2
    overridden = TrainConfig.from_yaml(
        write(tmp_path, YAML + "calibration:\n"), ["--calibration.split", "0.05"]
    )
    assert overridden.calibration.split == 0.05


@pytest.mark.parametrize("split", ["0", "1", "1.5", "-0.1"])
def test_bad_calibration_split_raises(tmp_path: Path, split: str) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(
            write(tmp_path, YAML + f"calibration:\n  split: {split}\n")
        )


def test_fit_works_without_validation(tmp_path: Path) -> None:
    text = YAML.replace("validation: validation", "validation: null")
    config = TrainConfig.from_yaml(write(tmp_path, text + "calibration:\n"))
    assert config.calibration.temperature == "fit"


def test_defaults_and_null_validation(tmp_path: Path) -> None:
    text = (
        "backbone: b\ndata:\n  dataset: d\n  train: t\n  validation: null\n"
        "training:\n  output_dir: o\n"
    )
    config = TrainConfig.from_yaml(write(tmp_path, text))
    assert (config.lora.r, config.lora.alpha) == (16, 32)
    assert config.data.validation is None


@pytest.mark.parametrize(
    "overrides",
    [
        ["--lora.rank", "4"],
        ["--model.r", "4"],
        ["--lora", "4"],
        ["--lora.r"],
        ["lora.r", "4"],
    ],
)
def test_bad_overrides_raise(tmp_path: Path, overrides: list[str]) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(write(tmp_path), overrides)


def test_unknown_yaml_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(write(tmp_path, YAML + "  bogus_key: 1\n"))


@pytest.mark.parametrize("path", sorted(Path("configs/train").glob("*.yml")))
def test_shipped_configs_load(path: Path) -> None:
    TrainConfig.from_yaml(path, ["--training.use_cpu", "true"])


def test_modal_section_is_skipped(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(write(tmp_path, YAML + "modal:\n  gpu: H100\n"))
    assert config.backbone == "Qwen/Qwen3.5-0.8B"


def test_top_level_seed(tmp_path: Path) -> None:
    config = TrainConfig.from_yaml(write(tmp_path, YAML + "seed: 7\n"))
    assert config.seed == config.training.seed == 7
    overridden = TrainConfig.from_yaml(write(tmp_path), ["--seed", "3"])
    assert overridden.seed == overridden.training.seed == 3
    assert TrainConfig.from_yaml(write(tmp_path)).training.seed == 42


def test_training_seed_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TrainConfig.from_yaml(write(tmp_path, YAML + "  seed: 7\n"))
