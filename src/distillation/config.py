"""Distillation settings and the config that builds the soft labels."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.config_parser import ConfigParser
from src.data.config import DataSettings


@dataclass
class DistillationSettings:
    """Offline distillation: teacher, soft label cache and loss.

    Attributes:
        teacher: Teacher model (fine-tuned System One model or raw backbone),
            used only when the soft labels must be built.
        soft_labels: Directory of the teacher soft label cache.
        alpha: Human-target CE weight, in [0, 1].
        temperature: Positive softening temperature, after teacher calibration.
        generating: HF prediction arguments of the teacher run (no
            `output_dir`).
    """

    teacher: str = ""
    soft_labels: str = ""
    alpha: float = 0.25
    temperature: float = 1.0
    generating: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.teacher:
            raise ValueError("distillation teacher is required")
        if not self.soft_labels:
            raise ValueError("distillation soft_labels directory is required")
        if "output_dir" in self.generating:
            raise ValueError("use soft_labels, not generating.output_dir")
        self.alpha = float(self.alpha)
        self.temperature = float(self.temperature)
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("distillation alpha must be in [0, 1]")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("distillation temperature must be finite and > 0")


@dataclass
class DistillationConfig:
    """Config that only builds the soft labels (`src.modal.cache`).

    Attributes:
        distillation: Teacher, soft label directory and prediction arguments.
        data: Dataset and the splits to label.
    """

    distillation: DistillationSettings
    data: DataSettings

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: list[str] | None = None,
    ) -> "DistillationConfig":
        """Loads the YAML (`data`, `distillation`) and applies overrides.

        Raises:
            ValueError: If a section or key is unknown or missing.
        """
        values: dict[str, Any] = ConfigParser.load(
            path, overrides, keys=[], sections=["data", "distillation"],
        )
        data = ConfigParser.section(values, "data", DataSettings)
        distillation = ConfigParser.settings(
            values, "distillation", DistillationSettings,
        )
        ConfigParser.check_empty(values)
        if distillation is None:
            raise ValueError("distillation section is required")
        return cls(distillation, data)
