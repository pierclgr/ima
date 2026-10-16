import json
from pathlib import Path

import datasets
import numpy as np
import pytest
import torch
import yaml
from transformers import TrainingArguments

from conftest import TinyBackbone, TinyProcessor
from src.distillation.cache import TeacherTargetCache
from src.distillation import generate
from src.modal import cache as cache_task
from src.model.system_one import PreTrainedSystemOneModel
from src.training import train
from src.training.train import build_trainer
from src.training.config import LoraSettings


@pytest.mark.parametrize("precache", [False, True])
def test_generate_train_calibrate_merge_and_reload(
    tmp_path: Path, records: datasets.Dataset,
    tiny_model: PreTrainedSystemOneModel, monkeypatch: pytest.MonkeyPatch,
    precache: bool,
) -> None:
    original = tmp_path / "original"
    tiny_model.save_pretrained(original)
    teacher_model = tmp_path / "teacher_model"
    tiny_model.save_pretrained(teacher_model)
    processor = TinyProcessor()
    monkeypatch.setattr(
        generate.HubRecordLoader, "load", lambda self, split: records,
    )
    cache = tmp_path / "cache"
    expected = tiny_model(
        input_ids=torch.tensor([[0]]), num_options=3,
    ).probabilities[0].detach().tolist()
    calls = []

    def load_student(
        name: str,
    ) -> tuple[PreTrainedSystemOneModel, TinyProcessor]:
        calls.append(name)
        return tiny_model, processor

    monkeypatch.setattr(train.ModelLoader, "load", load_student)
    output = tmp_path / "student"
    student = tmp_path / "student.yml"
    distillation = {
        "teacher": str(teacher_model), "soft_labels": str(cache),
        "alpha": 0.25, "temperature": 1,
        "generating": {"use_cpu": True, "report_to": "none",
                       "per_device_eval_batch_size": 2},
    }
    student.write_text(yaml.safe_dump({
        "backbone": str(original), "seed": 42,
        "data": {"dataset": "dataset", "train": "train",
                 "validation": "validation"},
        "distillation": distillation,
        "lora": {"r": 2, "alpha": 4},
        "training": {"output_dir": str(output), "use_cpu": True,
                     "report_to": "none", "max_steps": 2,
                     "per_device_train_batch_size": 2,
                     "per_device_eval_batch_size": 2,
                     "save_strategy": "steps", "save_steps": 1,
                     "eval_strategy": "steps", "eval_steps": 1,
                     "disable_tqdm": True},
        "calibration": {"temperature": "fit", "split": 0.5},
    }))

    if precache:
        labels = tmp_path / "labels.yml"
        labels.write_text(yaml.safe_dump({
            "data": {"dataset": "dataset", "train": "train",
                     "validation": "validation"},
            "distillation": distillation,
        }))
        cache_task.main([str(labels)])
        assert calls == [str(teacher_model)]
        cache_task.main([str(labels)])
        assert calls == [str(teacher_model)]
        calls.clear()
    else:
        # an existing empty cache directory must also build automatically
        cache.mkdir()
    train.main([str(student)])
    assert calls == ([str(original)] if precache
                     else [str(teacher_model), str(original)])
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["temperature"] == 1.7
    assert manifest["splits"]["validation"]["questions"] == 12
    enriched = TeacherTargetCache(cache).attach(
        records, "dataset", "train", "train", 26,
    )
    assert np.isclose(sum(enriched[0]["teacher_targets"]["choice"]), 1)
    assert enriched[0]["teacher_targets"]["choice"] == pytest.approx(expected)
    assert (output / "final" / "processor.json").is_file()
    assert (output / "checkpoint-2" / "adapter_model.safetensors").is_file()
    assert not hasattr(tiny_model, "peft_config") or not tiny_model.peft_config
    monkeypatch.setattr(
        "src.model.system_one.AutoModelForMultimodalLM.from_config",
        lambda config: TinyBackbone(),
    )
    reloaded = PreTrainedSystemOneModel.from_pretrained(output / "final")
    torch.testing.assert_close(
        reloaded(input_ids=torch.tensor([[0]]), num_options=3).logits,
        tiny_model(input_ids=torch.tensor([[0]]), num_options=3).logits,
    )
    assert reloaded.config.temperature > 0

    # resume from an adapter checkpoint over the same original student weights
    resumed = PreTrainedSystemOneModel.from_pretrained(original)
    monkeypatch.setattr(
        train.ModelLoader, "load", lambda name: (resumed, processor),
    )
    train.main([str(student), "--checkpoint", str(output / "checkpoint-2"),
                "--training.max_steps", "3"])
    state_file = output / "checkpoint-3" / "trainer_state.json"
    state = json.loads(state_file.read_text())
    assert state["global_step"] == 3


def test_existing_supervised_trainer_stays_human_only(
    tmp_path: Path, records: datasets.Dataset,
    tiny_model: PreTrainedSystemOneModel,
) -> None:
    trainer = build_trainer(tiny_model, TinyProcessor(), records, records,
                            LoraSettings(r=2, alpha=4), TrainingArguments(
                                output_dir=str(tmp_path), use_cpu=True,
                                report_to="none", max_steps=1,
                                per_device_train_batch_size=2,
                            ))
    assert "teacher_targets" not in next(iter(trainer.get_train_dataloader()))
    assert not tiny_model.backbone.lm_head.weight.requires_grad
    assert all(
        "lora_" in name for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    )
    trainer.train()


def test_raw_backbones_as_teacher_and_student(
    tmp_path: Path, records: datasets.Dataset,
    tiny_model: PreTrainedSystemOneModel, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # raw backbone configs have another `model_type`; the loader wraps them
    # as System One models with temperature 1
    teacher = tmp_path / "teacher_backbone"
    student = tmp_path / "student_backbone"
    for backbone in (teacher, student):
        backbone.mkdir()
        (backbone / "config.json").write_text('{"model_type": "gpt2"}')
    tiny_model.config.temperature = 1.0
    monkeypatch.setattr(
        generate.HubRecordLoader, "load", lambda self, split: records,
    )
    monkeypatch.setattr(
        train.ModelLoader, "load", lambda name: (tiny_model, TinyProcessor()),
    )
    output = tmp_path / "student"
    config = tmp_path / "student.yml"
    config.write_text(yaml.safe_dump({
        "backbone": str(student), "seed": 42,
        "distillation": {
            "teacher": str(teacher), "soft_labels": str(tmp_path / "cache"),
            "generating": {"use_cpu": True, "report_to": "none",
                           "per_device_eval_batch_size": 2},
        },
        "data": {"dataset": "dataset", "train": "train"},
        "lora": {"r": 2, "alpha": 4},
        "training": {"output_dir": str(output), "use_cpu": True,
                     "report_to": "none", "max_steps": 1,
                     "per_device_train_batch_size": 2,
                     "save_strategy": "no", "disable_tqdm": True},
    }))
    train.main([str(config)])
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest["temperature"] == 1.0
    assert (output / "final" / "processor.json").is_file()


def test_adapter_weights_are_loaded_and_trainable(
    tmp_path: Path, records: datasets.Dataset,
    tiny_model: PreTrainedSystemOneModel,
) -> None:
    fresh = PreTrainedSystemOneModel(tiny_model.config, backbone=TinyBackbone())
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
                             report_to="none", max_steps=1,
                             per_device_train_batch_size=2)
    source = build_trainer(tiny_model, TinyProcessor(), records, records,
                           LoraSettings(r=2, alpha=4), args)
    with torch.no_grad():
        for name, weight in source.model.named_parameters():
            if "lora_B" in name:
                weight.fill_(0.5)
    source.model.save_pretrained(tmp_path / "adapter")
    trainer = build_trainer(fresh, TinyProcessor(), records, records,
                            LoraSettings(r=99, alpha=1), args,
                            adapter=str(tmp_path / "adapter"))
    lora = {name: weight for name, weight in trainer.model.named_parameters()
            if "lora_" in name}
    assert trainer.model.peft_config["default"].r == 2
    assert lora and all(weight.requires_grad for weight in lora.values())
    assert all(torch.all(weight == 0.5) for name, weight in lora.items()
               if "lora_B" in name)
