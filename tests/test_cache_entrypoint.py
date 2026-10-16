from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import pytest

from conftest import TinyProcessor
from src.distillation.config import DistillationConfig
from src.distillation.generate import ensure_targets
from src.modal import cache
from src.model.system_one import PreTrainedSystemOneModel


@pytest.mark.parametrize("local", [False, True])
def test_regular_cache_keeps_download_flow(
    tmp_path: Path, local: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone = tmp_path / "model"
    if local:
        backbone.mkdir()
    name = str(backbone) if local else "org/model"
    config = tmp_path / "train.yml"
    config.write_text(
        f"backbone: {name}\ndata: {{dataset: data, train: train}}\n"
        f"training: {{output_dir: {tmp_path / 'run'}, report_to: none}}\n",
    )
    calls = []
    monkeypatch.setattr(cache, "snapshot_download",
                        lambda repo_id: calls.append(("model", repo_id)))
    monkeypatch.setattr(cache.HubRecordLoader, "load",
                        lambda self, split: calls.append(("data", split)))
    cache.main([str(config)])
    assert calls == ([("data", "train")] if local
                     else [("model", name), ("data", "train")])


def test_failed_generation_is_retryable_and_releases_teacher(
    tmp_path: Path, records: datasets.Dataset,
    tiny_model: PreTrainedSystemOneModel, monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher = tmp_path / "teacher"
    tiny_model.save_pretrained(teacher)
    targets = tmp_path / "targets"
    file = tmp_path / "student.yml"
    file.write_text(
        "data: {dataset: data, train: train}\n"
        f"distillation: {{teacher: {teacher}, soft_labels: {targets}, "
        "generating: {use_cpu: true, report_to: none}}\n",
    )
    config = DistillationConfig.from_yaml(file)
    monkeypatch.setattr("src.distillation.generate.HubRecordLoader.load",
                        lambda self, split: records)
    monkeypatch.setattr("src.distillation.generate.ModelLoader.load",
                        lambda name: (tiny_model, TinyProcessor()))
    monkeypatch.setattr("src.distillation.generate.DecisionEvaluator.predict",
                        lambda self, data: SimpleNamespace(
                            predictions=(None, np.zeros((12, 3))),
                            label_ids=np.zeros((12, 3)),
                        ))
    released = []
    monkeypatch.setattr("src.distillation.generate.gc.collect",
                        lambda: released.append(True))
    with pytest.raises(ValueError, match="probability"):
        ensure_targets(config.distillation, config.data)
    assert released == [True]
    assert not targets.exists()
    assert not list(tmp_path.glob(".targets-*"))

    monkeypatch.setattr("src.distillation.generate.DecisionEvaluator.predict",
                        lambda self, data: SimpleNamespace(
                            predictions=(None, np.tile(
                                [[0.6, 0.3, 0.1], [0.8, 0.2, -100],
                                 [0.4, 0.6, 0]], (4, 1),
                            )), label_ids=np.zeros((12, 3)),
                        ))
    ensure_targets(config.distillation, config.data)
    assert (targets / "manifest.json").exists()
    monkeypatch.setattr("src.distillation.generate.ModelLoader.load",
                        lambda name: pytest.fail("cached teacher was loaded"))
    ensure_targets(config.distillation, config.data)
    config.distillation.teacher = "other_teacher"
    with pytest.raises(ValueError, match="teacher"):
        ensure_targets(config.distillation, config.data)
