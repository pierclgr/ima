"""LoRA training of the System One model with HF `Trainer`.

Example:
    uv run python -m src.training.train configs/train/qwen3_5_0_8b.yml \\
        --training.learning_rate 1e-4 --data.train "train[:1000]"
"""

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import Trainer, TrainingArguments, set_seed

from src.calibration.calibrate import TemperatureCalibrator
from src.common.prompt import PromptBuilder
from src.data.collator import DecisionEngineCollator
from src.data.dataset import DecisionEngineDataset
from src.data.hub import HubRecordLoader
from src.distillation.config import DistillationSettings
from src.distillation.generate import ensure_targets
from src.distillation.trainer import DistillationCollator, DistillationTrainer
from src.evaluation.metrics import DecisionMetrics
from src.model.loader import ModelLoader
from src.model.system_one import PreTrainedSystemOneModel
from src.training.config import LoraSettings, TrainConfig


def lora_target_modules(model: nn.Module) -> list[str]:
    """Returns the linear layers of the language model (LoRA targets).

    The vision tower, the projector and `lm_head` are left out, so they stay
    frozen.

    Raises:
        ValueError: If the model has no `language_model` linear layers.
    """
    names: list[str] = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ".language_model." in name
    ]
    if not names:
        raise ValueError("no language_model linear layers found")
    return names


def build_trainer(
    model: PreTrainedSystemOneModel,
    processor: Any,
    train_records: Any,
    eval_records: Any | None,
    lora: LoraSettings,
    training_args: TrainingArguments,
    distillation: DistillationSettings | None = None,
) -> Trainer:
    """Adds LoRA to the model and builds the `Trainer`.

    Args:
        model: The System One model (LoRA layers are added in place).
        processor: Processor matching the backbone.
        train_records: Labelled training records.
        eval_records: Labelled evaluation records, or None.
        lora: LoRA settings.
        training_args: `Trainer` arguments. A copy is used, with
            `remove_unused_columns` False, since the collator needs the raw
            items. TensorBoard logs
            go to `output_dir` (`TENSORBOARD_LOGGING_DIR`).
        distillation: Distillation settings (the records then carry the
            cached teacher targets). None keeps ordinary supervised training.

    Returns:
        The trainer.
    """
    lora_config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        target_modules=lora_target_modules(model),
    )
    max_options: int = model.config.max_options
    train_dataset = DecisionEngineDataset(train_records, max_options=max_options)
    eval_dataset = (
        DecisionEngineDataset(eval_records, shuffle_options=False, max_options=max_options)
        if eval_records is not None
        else None
    )
    print(f"train questions: {len(train_dataset)}, skipped: {train_dataset.skipped}")
    training_args = replace(training_args, remove_unused_columns=False)
    # tensorboard logs in output_dir, not in runs/<date>_<host>: the run is
    # named after output_dir
    os.environ["TENSORBOARD_LOGGING_DIR"] = training_args.output_dir
    # the lora weights are created before `Trainer` sets the seed
    set_seed(training_args.seed)
    prompt = PromptBuilder(processor, max_options)
    trainer_type = DistillationTrainer if distillation is not None else Trainer
    extra = {"distillation": distillation} if distillation is not None else {}
    return trainer_type(
        model=get_peft_model(model, lora_config),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=(DistillationCollator(prompt) if distillation is not None
                       else DecisionEngineCollator(prompt)),
        compute_metrics=DecisionMetrics(),
        **extra,
    )


def main(argv: list[str] | None = None) -> None:
    """Trains a LoRA and saves the merged model to `<output_dir>/final`.

    If the config has a `calibration` section, the merged model is calibrated
    before it is saved. To fit the temperature, `calibration.split` of the
    training records is carved out before training (seeded with
    `seed`) and used only for the fit.

    If the config has a `distillation` section, the teacher soft labels are
    built first when missing (the teacher is released before the student
    loads) and the loss mixes them with the human targets. Validation and
    calibration use only the human targets.

    Usage: `python -m src.training.train <config.yml> [--section.key value]`.

    Args:
        argv: Command-line arguments without the program name (default:
            `sys.argv[1:]`).
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit(__doc__)
    config = TrainConfig.from_yaml(argv[0], argv[1:])
    soft_labels = (
        None if config.distillation is None
        else ensure_targets(config.distillation, config.data)
    )
    model, processor = ModelLoader.load(config.backbone)
    loader = HubRecordLoader(config.data.dataset)
    train_records = loader.load(config.data.train)
    if soft_labels is not None:
        train_records = soft_labels.attach(
            train_records, config.data.dataset, "train", config.data.train,
            model.config.max_options,
        )
    calibration_records = None
    if config.calibration is not None and config.calibration.temperature == "fit":
        train_records, calibration_records = TemperatureCalibrator.split(
            train_records, config.calibration.split, config.seed
        )
    eval_records = (
        loader.load(config.data.validation) if config.data.validation else None
    )
    trainer = build_trainer(
        model,
        processor,
        train_records,
        eval_records,
        config.lora,
        config.training,
        config.distillation,
    )
    trainer.train(resume_from_checkpoint=config.checkpoint)
    merged: PreTrainedSystemOneModel = trainer.model.merge_and_unload()
    if config.calibration is not None:
        calibrator = TemperatureCalibrator(merged, processor, config.training)
        temperature: float = calibrator.calibrate(
            config.calibration, calibration_records
        )
        print(f"temperature: {temperature:.4f}")
    final: Path = Path(config.training.output_dir) / "final"
    merged.save_pretrained(final)
    processor.save_pretrained(final)


if __name__ == "__main__":
    main()
