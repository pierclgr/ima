import json
from pathlib import Path
from itertools import count

import pytest

from src.constants import SYSTEM_TWO_INSTRUCTION
from src.testing import test as test_script
from test_evaluation import RECORDS

YAML: str = """
model: {model}
data:
  dataset: org/kev
  split: test
testing:
  output_dir: {output_dir}
  per_device_eval_batch_size: 2
  use_cpu: true
  report_to: none
"""


def run(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    overrides: list[str], model_type: str = "system_one",
) -> list[str]:
    """Runs the script on a model dir of `model_type`; returns the loaders used."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    config = tmp_path / "test.yml"
    config.write_text(YAML.format(model=model_dir, output_dir=tmp_path / "out"))
    monkeypatch.setattr(
        "src.model.loader.AutoProcessor.from_pretrained", lambda *a, **k: processor,
    )
    ticks = count()
    monkeypatch.setattr(test_script.ModelTimer, "now", staticmethod(lambda: next(ticks)))
    loaders: list[str] = []
    for loader in ("from_pretrained", "from_backbone"):
        monkeypatch.setattr(
            test_script.PreTrainedSystemOneModel, loader,
            lambda *a, loader=loader, **k: loaders.append(loader) or model,
        )
    monkeypatch.setattr(test_script.HubRecordLoader, "load", lambda self, s: RECORDS * 4)
    test_script.main([str(config), *overrides])
    return loaders


def test_reports_seconds_per_question(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(model, processor, tmp_path, monkeypatch, [])
    metrics = json.loads((tmp_path / "out" / "test_metrics.json").read_text())
    assert metrics["test_seconds_per_question"] == pytest.approx(0.5)
    assert metrics["test_timed_questions"] == 4
    assert metrics["test_evaluated_questions"] == 24
    assert "test_runtime" not in metrics


@pytest.mark.parametrize(
    ("overrides", "expected"), [([], 2.0), (["--temperature", "1.5"], 1.5)]
)
def test_temperature_override(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    overrides: list[str], expected: float,
) -> None:
    # stands for the fitted temperature saved in config.json
    model.config.temperature = 2.0
    run(model, processor, tmp_path, monkeypatch, overrides)
    assert model.config.temperature == expected


@pytest.mark.parametrize(
    ("model_type", "loader"),
    [("system_one", "from_pretrained"), ("qwen3_5", "from_backbone")],
)
def test_loads_trained_model_or_backbone(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    model_type: str, loader: str,
) -> None:
    assert run(model, processor, tmp_path, monkeypatch, [], model_type) == [loader]


def test_system_two_mode(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(
        model, processor, tmp_path, monkeypatch,
        ["--system_two.max_new_tokens", "2"],
    )
    metrics = json.loads((tmp_path / "out" / "test_metrics.json").read_text())
    assert 0.0 <= metrics["test_errors"] <= 1.0
    # System Two asks for one word and keeps thinking off
    text = processor.messages[0][0]["content"][-1]["text"]
    assert text.endswith(SYSTEM_TWO_INSTRUCTION)
    assert processor.kwargs["enable_thinking"] is False
    assert "reasoning_effort" not in processor.kwargs


def test_system_one_mode_has_no_errors(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(model, processor, tmp_path, monkeypatch, [])
    metrics = json.loads((tmp_path / "out" / "test_metrics.json").read_text())
    assert "test_errors" not in metrics


@pytest.mark.parametrize(
    ("template", "expected"),
    [("{{ enable_thinking }}", {"enable_thinking": True}),
     ("{{ enable_thinking }} {{ reasoning_effort }}",
      {"enable_thinking": True, "reasoning_effort": "low"})],
)
def test_thinking_kwargs_follow_the_template(
    model, processor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    template: str, expected: dict,
) -> None:
    # a variable the template does not use would replace the processor
    # kwargs (padding) in transformers, so reasoning_effort is sent only when
    # the template has it
    processor.chat_template = template
    run(
        model, processor, tmp_path, monkeypatch,
        ["--system_two.thinking", "low", "--system_two.max_new_tokens", "2"],
    )
    assert {k: processor.kwargs[k] for k in processor.kwargs
            if k in ("enable_thinking", "reasoning_effort")} == expected
