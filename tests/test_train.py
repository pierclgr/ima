import math

import pytest
import torch
from torch import nn
from transformers import TrainingArguments

from src.model.system_one import PreTrainedSystemOneModel
from src.training.config import LoraSettings
from src.training.train import build_trainer, lora_target_modules

RECORDS: list[dict] = [
    {
        "state": "I was charged twice.",
        "questions": {
            "billing": {"type": "noul", "instructions": "Billing?", "label": True},
            "tone": {
                "type": "choice",
                "instructions": "Tone?",
                "criteria": {"calm": None, "angry": None, "sad": None},
                "label": "angry",
            },
        },
    }
] * 4


def training_args(tmp_path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-2,
        report_to=[],
        save_strategy="no",
        use_cpu=True,
    )


def test_lora_targets_only_language_model(model: PreTrainedSystemOneModel) -> None:
    targets = lora_target_modules(model)
    assert targets
    assert all(".language_model." in name for name in targets)
    assert not any("vision" in name or "projector" in name for name in targets)
    assert "backbone.lm_head" not in targets


def test_no_language_model_raises() -> None:
    with pytest.raises(ValueError):
        lora_target_modules(nn.Sequential(nn.Linear(2, 2)))


def test_trainer_trains_only_lora(model, processor, tmp_path) -> None:
    frozen = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "vision_tower" in name or "lm_head" in name
    }
    trainer = build_trainer(
        model, processor, RECORDS, RECORDS, LoraSettings(), training_args(tmp_path)
    )
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    trainable = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    merged = trainer.model.merge_and_unload()
    for name, before in frozen.items():
        assert torch.equal(dict(merged.named_parameters())[name], before)


def test_trainer_scales_loss_itself(model, processor, tmp_path) -> None:
    trainer = build_trainer(
        model, processor, RECORDS, None, LoraSettings(), training_args(tmp_path)
    )
    assert trainer.model_accepts_loss_kwargs is False


def test_trainer_reports_eval_loss(model, processor, tmp_path) -> None:
    trainer = build_trainer(
        model, processor, RECORDS, RECORDS, LoraSettings(), training_args(tmp_path)
    )
    metrics = trainer.evaluate()
    for name in ("eval_loss", "eval_accuracy", "eval_brier", "eval_ece"):
        assert math.isfinite(metrics[name])


def test_merged_model_saves_and_loads(model, processor, tmp_path) -> None:
    trainer = build_trainer(
        model, processor, RECORDS, None, LoraSettings(), training_args(tmp_path)
    )
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(tmp_path / "final")
    loaded = PreTrainedSystemOneModel.from_pretrained(tmp_path / "final")
    assert isinstance(loaded, PreTrainedSystemOneModel)


def test_trainer_with_gradient_checkpointing(model, processor, tmp_path) -> None:
    args = training_args(tmp_path)
    args.gradient_checkpointing = True
    trainer = build_trainer(model, processor, RECORDS, None, LoraSettings(), args)
    result = trainer.train()
    assert math.isfinite(result.training_loss)
    assert trainer.model.is_gradient_checkpointing
    lora = [p for n, p in trainer.model.named_parameters() if "lora_B" in n]
    # lora_B starts at zero, so a change means gradients reached the adapter
    assert any(p.abs().sum() > 0 for p in lora)


def test_lora_init_follows_seed(config, processor, tmp_path) -> None:
    weights = []
    for _ in range(2):
        # each build gets a new model, with the rng moved in between
        torch.rand(10)
        trainer = build_trainer(
            PreTrainedSystemOneModel(config), processor, RECORDS, None,
            LoraSettings(), training_args(tmp_path),
        )
        weights.append(
            {n: p.detach().clone() for n, p in trainer.model.named_parameters()
             if "lora_A" in n}
        )
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][n], weights[1][n]) for n in weights[0])


def test_tensorboard_logs_in_output_dir(
    model, processor, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # restores the variable that build_trainer sets
    monkeypatch.setenv("TENSORBOARD_LOGGING_DIR", "")
    args = training_args(tmp_path)
    args.report_to = ["tensorboard"]
    trainer = build_trainer(model, processor, RECORDS, None, LoraSettings(), args)
    trainer.train()
    assert list(tmp_path.glob("events.out.tfevents.*"))
    assert not (tmp_path / "runs").exists()
