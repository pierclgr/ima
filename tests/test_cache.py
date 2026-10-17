from pathlib import Path

import pytest
import transformers.training_args

from src.modal import cache

YAML: str = """
backbone: Qwen/Qwen3.5-0.8B
data:
  dataset: org/kev
  train: train
  validation: validation
training:
  output_dir: runs/test
"""


def test_caches_backbone_and_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yml"
    config.write_text(YAML)
    calls: list[tuple] = []
    monkeypatch.setattr(
        cache, "snapshot_download", lambda repo_id: calls.append(("model", repo_id))
    )
    monkeypatch.setattr(
        cache.HubRecordLoader,
        "load",
        lambda self, split: calls.append(("dataset", self.repo_id, split)),
    )
    cache.main([str(config), "--backbone", "other/model"])
    assert calls == [("model", "other/model"), ("dataset", "org/kev", "train")]


def test_usage_without_arguments() -> None:
    with pytest.raises(SystemExit):
        cache.main([])


def test_bf16_config_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yml"
    config.write_text(YAML + "  bf16: true\n")
    # simulate a machine without bf16 GPU support
    monkeypatch.setattr(
        transformers.training_args, "is_torch_bf16_gpu_available", lambda: False
    )
    monkeypatch.setattr(cache, "snapshot_download", lambda repo_id: None)
    monkeypatch.setattr(cache.HubRecordLoader, "load", lambda self, split: None)
    cache.main([str(config)])
