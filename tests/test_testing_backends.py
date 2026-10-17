"""HF/native configuration, shared metrics, timing, and CLI progress."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml
from transformers import TrainingArguments

from src.data.dataset import DecisionEngineDataset
from src.data.hub import HubRecordLoader
from src.evaluation.llamacpp import LlamaCppEvaluator
from src.model.output import DecisionEngineOutput
from src.testing import test as script
from src.testing.config import LlamaCppTestArguments, TestConfig
from test_hub import fake_repo


class FakeRuntime:
    """Deterministic logits and times independent of Python elapsed time."""

    def __init__(self, *args) -> None:
        self.device = "CPU (test)"
        self.temperatures = {"choice": 2.0, "noul": 2.0, "score": 2.0}
        self.calls = 0
        self.closed = False

    def predict(self, item: dict) -> dict:
        self.calls += 1
        return {"logits": list(range(len(item["target"]))),
                "seconds": self.calls * 0.01}

    def close(self) -> None:
        self.closed = True


def native_config(tmp_path: Path) -> Path:
    path = tmp_path / "test.yml"
    path.write_text(yaml.safe_dump({
        "backend": "llamacpp", "model": "export",
        "data": {"dataset": "datasets", "split": "test"},
        "testing": {"output_dir": str(tmp_path / "results")},
    }))
    return path


def test_native_config_does_not_build_hf_arguments(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(TrainingArguments, "__post_init__", Mock(
        side_effect=AssertionError("HF arguments"),
    ))
    config = TestConfig.from_yaml(native_config(tmp_path))
    assert config.backend == "llamacpp"
    assert isinstance(config.testing, LlamaCppTestArguments)
    assert config.testing.per_device_eval_batch_size == 1
    assert config.llamacpp.ctx_size == 8192


@pytest.mark.parametrize("overrides", [
    ["--backend", "bad"], ["--testing.per_device_eval_batch_size", "2"],
    ["--testing.use_cpu", "true"], ["--llamacpp.ctx_size", "0"],
    ["--llamacpp.threads", "0"], ["--system_two.thinking", "off"],
    ["--temperature", "nan"], ["--temperature", "inf"],
])
def test_invalid_native_config(tmp_path: Path, overrides: list[str]) -> None:
    with pytest.raises(ValueError):
        TestConfig.from_yaml(native_config(tmp_path), overrides)


def test_hf_backend_remains_default(tmp_path: Path) -> None:
    path = native_config(tmp_path)
    values = yaml.safe_load(path.read_text())
    values.pop("backend")
    values["testing"].update(use_cpu=True, report_to="none")
    path.write_text(yaml.safe_dump(values))
    config = TestConfig.from_yaml(path)
    assert config.backend == "hf" and config.llamacpp is None
    assert isinstance(config.testing, TrainingArguments)
    with pytest.raises(ValueError):
        TestConfig.from_yaml(path, ["--llamacpp.ctx_size", "4096"])


def test_local_and_hub_parquet_paths_match(tmp_path: Path, monkeypatch) -> None:
    repo = fake_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    download = Mock(return_value=str(repo))
    monkeypatch.setattr("src.data.hub.snapshot_download", download)
    local = HubRecordLoader(str(repo)).load("train[:3]")
    download.assert_not_called()
    remote = HubRecordLoader("org/kev").load("train[:3]")
    assert len(local) == len(remote) == 3
    for first, second in zip(local, remote):
        assert first["state"] == second["state"]
        assert first["questions"] == second["questions"]
        assert [image.tobytes() for image in first["files"]] == [
            image.tobytes() for image in second["files"]
        ]


@pytest.mark.parametrize("temperature", [None, 1.0])
def test_native_metrics_keep_soft_targets_and_native_times(
    records, temperature, capsys,
) -> None:
    runtime = FakeRuntime()
    evaluator = LlamaCppEvaluator(runtime, temperature)
    result = evaluator.predict(records)
    dataset = DecisionEngineDataset(records, shuffle_options=False)
    item = dataset[0]
    count = len(item["target"])
    expected = DecisionEngineOutput.from_logits(
        torch.arange(count).float().unsqueeze(0),
        temperature or 2.0, torch.tensor([item["target"]]),
    )
    assert result.label_ids[0, :count] == pytest.approx(item["target"])
    assert result.predictions[1][0, :count] == pytest.approx(
        expected.probabilities[0].numpy(),
    )
    assert evaluator.times == pytest.approx(np.arange(1, 13) * 0.01)
    assert result.metrics["test_evaluated_questions"] == 12
    assert result.metrics["test_skipped_questions"] == 0
    assert all(np.isfinite(value) for value in result.metrics.values())
    output = capsys.readouterr()
    assert not output.out
    assert "Preparing evaluation questions" in output.err
    assert "Running inference" in output.err and "12/12" in output.err


def test_native_skips_only_questions_over_option_limit() -> None:
    record = {"state": "s", "questions": {
        "big": {"type": "choice", "instructions": "q",
                "criteria": {str(i): None for i in range(27)}, "label": "0"},
        "score": {"type": "score", "instructions": "q",
                  "criteria": [str(i) for i in range(11)], "label": 10},
    }}
    result = LlamaCppEvaluator(FakeRuntime()).predict([record])
    assert result.metrics["test_evaluated_questions"] == 1
    assert result.metrics["test_skipped_questions"] == 1
    assert result.metrics["test_accuracy"] == 1
    with pytest.raises(ValueError, match="no eligible"):
        LlamaCppEvaluator(FakeRuntime()).predict([])


def test_native_cli_reports_only_model_time_and_closes(
    records, tmp_path: Path, monkeypatch, capsys,
) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(script, "LlamaCppRuntime", lambda *a: runtime)
    monkeypatch.setattr(script.HubRecordLoader, "load", lambda *a: records)
    monkeypatch.setattr(script.ModelLoader, "load", Mock(
        side_effect=AssertionError("HF model"),
    ))
    script.main([str(native_config(tmp_path))])
    captured = capsys.readouterr()
    metrics = json.loads(captured.out)
    assert runtime.closed and runtime.calls == 12
    assert metrics["test_seconds_per_question"] == pytest.approx(0.115)
    assert metrics["test_timed_questions"] == 2
    assert metrics["test_evaluated_questions"] == 12
    assert metrics["test_backend"] == "llamacpp"
    assert metrics["test_device"] == "CPU (test)"
    assert "test_gpu" not in metrics and "test_runtime" not in metrics
    assert "Loading dataset" in captured.err and "Loaded 4 records" in captured.err
    assert "Model ready on CPU (test)" in captured.err
    assert json.loads((tmp_path / "results/test_metrics.json").read_text()) == metrics


@pytest.mark.parametrize("failure", [False, True])
def test_native_cli_closes_after_short_run_or_failure(
    records, tmp_path: Path, monkeypatch, failure: bool,
) -> None:
    runtime = FakeRuntime()
    monkeypatch.setattr(script, "LlamaCppRuntime", lambda *a: runtime)
    monkeypatch.setattr(script.HubRecordLoader, "load", lambda *a: records.select([0]))
    if failure:
        runtime.predict = Mock(side_effect=RuntimeError("native failure"))
    expected = RuntimeError if failure else ValueError
    with pytest.raises(expected, match="native failure" if failure else "warm-up"):
        script.main([str(native_config(tmp_path))])
    assert runtime.closed
    assert not (tmp_path / "results/test_metrics.json").exists()


def test_model_timer_synchronizes_mps(monkeypatch) -> None:
    sync = Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", sync)
    script.ModelTimer.now()
    sync.assert_called_once()
