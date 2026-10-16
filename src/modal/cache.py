"""Caches training inputs or builds the teacher soft labels.

Downloads the backbone into the HF Hub cache and the dataset into the
datasets cache, and prepares its splits, so later runs start without
downloads. With a distillation config (`data` and `distillation` sections,
no `backbone`), it builds the missing teacher soft labels on a GPU instead.

Usage:
    uv run python -m src.modal.cache <train_or_distillation_config.yml> \\
        [--section.key value]
"""

import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from src.data.hub import HubRecordLoader
from src.distillation.config import DistillationConfig
from src.distillation.generate import ensure_targets
from src.training.config import TrainConfig


def cache_uses_gpu(path: str | Path) -> bool:
    """Returns whether the config only builds GPU-generated soft labels."""
    values = yaml.safe_load(Path(path).read_text()) or {}
    return "distillation" in values and "backbone" not in values


def main(argv: list[str] | None = None) -> None:
    """Downloads the model/data, or builds missing teacher soft labels.

    Args:
        argv: Command-line arguments without the program name (default:
            `sys.argv[1:]`).
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    if cache_uses_gpu(argv[0]):
        distillation = DistillationConfig.from_yaml(argv[0], argv[1:])
        ensure_targets(distillation.distillation, distillation.data)
        return
    # no GPU needed, skips the bf16 hardware check of `TrainingArguments`
    config = TrainConfig.from_yaml(argv[0], [*argv[1:], "--training.use_cpu", "true"])
    if not Path(config.backbone).is_dir():
        snapshot_download(repo_id=config.backbone)
    # loading one split downloads the dataset and prepares all its splits
    HubRecordLoader(config.data.dataset).load(config.data.train)


if __name__ == "__main__":
    main()
