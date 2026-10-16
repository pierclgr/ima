from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

from src.calibration.config import CalibrationSettings
from src.common.config_parser import ConfigParser
from src.data.config import DataSettings
from src.distillation.config import DistillationSettings


@dataclass
class LoraSettings:
    """LoRA settings.

    Attributes:
        r: LoRA rank.
        alpha: LoRA scaling.
    """

    r: int = 16
    alpha: int = 32


SECTIONS: dict[str, type] = {
    "lora": LoraSettings,
    "data": DataSettings,
    "training": TrainingArguments,
}


@dataclass
class TrainConfig:
    """Training run configuration, loaded from a YAML file.

    The YAML has a `backbone` key, optional `seed`, `checkpoint` and
    `adapter` keys and the sections `lora`, `data`, `training` (any HF `TrainingArguments` key,
    except `num_train_epochs`, which is named `num_epochs`, and `seed`, which
    is top-level; `logging_steps` defaults to 1, i.e. every weight update) and
    the optional `calibration` and `distillation` (offline distillation).
    Values are converted to the field types, so `1e-4` works.

    Attributes:
        backbone: HF id or local path of the multimodal backbone.
        lora: LoRA settings.
        data: Training data.
        training: `Trainer` arguments.
        calibration: Calibration settings, or None to skip calibration.
        seed: Random seed of the run (LoRA init, data order, calibration
            split); also set as `training.seed`.
        checkpoint: `Trainer` checkpoint to resume training from (e.g.
            `runs/<run>/checkpoint-500`), or None to start fresh.
        adapter: LoRA adapter (e.g. `runs/<run>/checkpoint-500`) whose
            weights start the training with a fresh optimizer and schedule
            (its own LoRA settings replace `lora`), or None for a new LoRA.
        distillation: Offline distillation settings, or None for ordinary
            supervised training.
    """

    backbone: str
    lora: LoraSettings
    data: DataSettings
    training: TrainingArguments
    calibration: CalibrationSettings | None = None
    seed: int = 42
    checkpoint: str | None = None
    adapter: str | None = None
    distillation: DistillationSettings | None = None

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: list[str] | None = None
    ) -> "TrainConfig":
        """Loads the configuration from a YAML file.

        Args:
            path: The YAML file.
            overrides: CLI overrides as `--key value` pairs, with dotted keys
                for sections, e.g. `["--lora.r", "8", "--backbone", "x"]`.

        Returns:
            The configuration.

        Raises:
            ValueError: If an override is malformed, or a section or key is
                unknown or missing.
        """
        values: dict[str, Any] = ConfigParser.load(
            path,
            overrides,
            keys=["backbone", "seed", "checkpoint", "adapter"],
            sections=[*SECTIONS, "calibration", "distillation"],
        )
        backbone: str = str(values.pop("backbone"))
        seed: int = int(values.pop("seed", cls.seed))
        checkpoint: Any = values.pop("checkpoint", None)
        adapter: Any = values.pop("adapter", None)
        training: dict[str, Any] = values.get("training") or {}
        values["training"] = training
        if "num_train_epochs" in training:
            raise ValueError("use training.num_epochs, not num_train_epochs")
        if "seed" in training:
            raise ValueError("use the top-level seed, not training.seed")
        training["seed"] = seed
        if "num_epochs" in training:
            training["num_train_epochs"] = training.pop("num_epochs")
        training.setdefault("logging_steps", 1)
        distillation: DistillationSettings | None = ConfigParser.settings(
            values, "distillation", DistillationSettings
        )
        calibration: CalibrationSettings | None = ConfigParser.settings(
            values, "calibration", CalibrationSettings
        )
        sections = {
            name: ConfigParser.section(values, name, dataclass_type)
            for name, dataclass_type in SECTIONS.items()
        }
        ConfigParser.check_empty(values)
        return cls(
            backbone=backbone,
            calibration=calibration,
            seed=seed,
            checkpoint=None if checkpoint is None else str(checkpoint),
            adapter=None if adapter is None else str(adapter),
            distillation=distillation,
            **sections,
        )
