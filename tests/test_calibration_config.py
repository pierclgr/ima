from pathlib import Path

import pytest

from src.calibration.config import CalibrationConfig

YAML: str = """
model: runs/x/final
seed: 7
data:
  dataset: Jacqkues/kev-vision-decisions-full
  train: train
calibration:
  temperature: fit
  split: 0.2
calibrating:
  output_dir: runs/x/calibration
  per_device_eval_batch_size: 8
  report_to: none
"""


def write(tmp_path: Path, text: str = YAML) -> Path:
    path = tmp_path / "calibration.yml"
    path.write_text(text)
    return path


def test_loads_typed_sections(tmp_path: Path) -> None:
    config = CalibrationConfig.from_yaml(write(tmp_path))
    assert config.model == "runs/x/final"
    assert config.seed == 7
    assert config.data.dataset == "Jacqkues/kev-vision-decisions-full"
    assert config.data.train == "train"
    assert config.calibration.temperature == "fit"
    assert config.calibration.split == pytest.approx(0.2)
    assert config.calibrating.output_dir == "runs/x/calibration"
    assert config.calibrating.per_device_eval_batch_size == 8


def test_defaults(tmp_path: Path) -> None:
    text = YAML.replace("seed: 7\n", "").replace(
        "calibration:\n  temperature: fit\n  split: 0.2\n", ""
    )
    config = CalibrationConfig.from_yaml(write(tmp_path, text))
    assert config.seed == 42
    assert config.calibration.temperature == "fit"
    assert config.calibration.split == pytest.approx(0.1)


def test_cli_overrides(tmp_path: Path) -> None:
    overrides = [
        "--model", "runs/y/final",
        "--seed", "3",
        "--calibration.temperature", "1.5",
        "--calibrating.per_device_eval_batch_size", "2",
    ]
    config = CalibrationConfig.from_yaml(write(tmp_path), overrides)
    assert config.model == "runs/y/final"
    assert config.seed == 3
    assert config.calibration.temperature == pytest.approx(1.5)
    assert config.calibrating.per_device_eval_batch_size == 2


@pytest.mark.parametrize(
    "overrides",
    [["--backbone", "x"], ["--training.seed", "1"], ["--calibration.foo", "1"]],
)
def test_bad_overrides_raise(tmp_path: Path, overrides: list[str]) -> None:
    with pytest.raises(ValueError):
        CalibrationConfig.from_yaml(write(tmp_path), overrides)


def test_unknown_section_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CalibrationConfig.from_yaml(write(tmp_path, YAML + "lora:\n  r: 8\n"))


def test_modal_section_is_skipped(tmp_path: Path) -> None:
    CalibrationConfig.from_yaml(write(tmp_path, YAML + "modal:\n  gpu: H100\n"))


@pytest.mark.parametrize(
    "path", sorted(Path("configs/calibration").glob("*.yml"))
)
def test_shipped_configs_load(path: Path) -> None:
    CalibrationConfig.from_yaml(path, ["--calibrating.use_cpu", "true"])


def test_shipped_configs_match_training() -> None:
    from src.training.config import TrainConfig

    paths = sorted(Path("configs/calibration").glob("*.yml"))
    assert paths
    for path in paths:
        calibration = CalibrationConfig.from_yaml(
            path, ["--calibrating.use_cpu", "true"],
        )
        training = TrainConfig.from_yaml(
            Path("configs/train") / path.name, ["--training.use_cpu", "true"],
        )
        # same carved set as the training run
        assert calibration.seed == training.seed
        assert calibration.data.dataset == training.data.dataset
        assert calibration.data.train == training.data.train
        assert calibration.calibration.split == training.calibration.split
        assert calibration.model == f"{training.training.output_dir}/final"
