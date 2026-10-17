"""Export metadata, lazy unwrapping, native prompts, and atomic output."""

import json
import string
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from jinja2 import Environment

from conftest import FakeTokenizer
from src.common.prompt import PromptBuilder
from src.common.types import Question
from src.llamacpp import export
from src.llamacpp.prompt import build_template


class ExportProcessor:
    """Minimal processor preserving the source chat boundaries."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.image_processor = SimpleNamespace(size={
            "shortest_edge": 65536, "longest_edge": 1048576,
        })

    def apply_chat_template(self, messages: list, **kwargs) -> str:
        assert kwargs == {
            "tokenize": False, "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "<user>" + messages[0]["content"][0]["text"] + "</user><assistant>"


@pytest.fixture
def export_source(tmp_path: Path, monkeypatch):
    model = tmp_path / "source"
    model.mkdir()
    config = {
        "model_type": "system_one", "temperature": 1.75, "max_options": 26,
        "option_token_ids": [ord(c) for c in string.ascii_uppercase],
        "backbone_config": {"model_type": "qwen3_5",
                            "architectures": ["Qwen3_5ForConditionalGeneration"],
                            "vision_config": {}},
    }
    (model / "config.json").write_text(json.dumps(config))
    (model / "model.safetensors").write_bytes(b"source weights")
    processor = ExportProcessor()
    monkeypatch.setattr(
        export.AutoProcessor, "from_pretrained", Mock(return_value=processor),
    )
    instances = []

    class Writer:
        def __init__(self) -> None:
            self.values = {}

        def add_decision_type(self, value: str) -> None:
            self.values["decision_type"] = value

        def add_decision_temperature(self, kind: str, value: float) -> None:
            self.values[f"temperature_{kind}"] = value

        def add_chat_template(self, value: list) -> None:
            self.values["templates"] = value

        def add_vision_min_pixels(self, value: int) -> None:
            self.values["min_pixels"] = value

        def add_vision_max_pixels(self, value: int) -> None:
            self.values["max_pixels"] = value

    class Converter:
        model_arch = "qwen35"

        def __init__(self, source: Path, precision, target: Path, **kwargs) -> None:
            self.target = target
            self.gguf_writer = Writer()
            self.hparams = kwargs["hparams"]
            instances.append(self)

        @classmethod
        def filter_tensors(cls, item):
            return item

        def prepare_metadata(self, vocab_only: bool) -> None:
            pass

        def write(self) -> None:
            assert self.no_mtp
            self.prepare_metadata(False)
            self.target.write_text(json.dumps(self.gguf_writer.values))

    classes = Mock(return_value=Converter)
    monkeypatch.setattr(export, "load_tools", lambda _: (
        SimpleNamespace(get_model_class=classes),
        SimpleNamespace(LlamaFileType=SimpleNamespace(MOSTLY_F16=1)),
    ))
    checkout = tmp_path / "llama.cpp"
    quantizer = checkout / "build/bin/llama-quantize"
    quantizer.parent.mkdir(parents=True)
    quantizer.touch()
    return model, checkout, instances, classes


@pytest.mark.parametrize("precision", ["F16", "Q8_0", "Q4_K_M"])
def test_export_keeps_metadata_projector_and_source(
    export_source, tmp_path: Path, monkeypatch, precision: str,
) -> None:
    model, checkout, instances, classes = export_source
    before = {p.name: p.read_bytes() for p in model.iterdir()}
    calls = []

    def quantize(command: list, **kwargs) -> None:
        calls.append(command)
        assert kwargs == {"check": True}
        Path(command[2]).write_bytes(Path(command[1]).read_bytes())

    monkeypatch.setattr(export.subprocess, "run", quantize)
    output = tmp_path / "export"
    target = export.export_model(model, output, checkout, precision)
    assert target.name == f"model-{precision}.gguf"
    assert sorted(p.name for p in output.iterdir()) == sorted([
        target.name, "mmproj-F16.gguf",
    ])
    metadata = json.loads(target.read_text())
    assert metadata["decision_type"] == "openjev"
    assert all(
        metadata[f"temperature_{kind}"] == 1.75
        for kind in ("choice", "noul", "score")
    )
    assert metadata["templates"][0]["name"] == "systemone"
    assert json.loads((output / "mmproj-F16.gguf").read_text()) == {
        "min_pixels": 65536, "max_pixels": 1048576,
    }
    lazy = lambda: "tensor"
    assert instances[0].filter_tensors(("backbone.weight", lazy)) == ("weight", lazy)
    with pytest.raises(ValueError, match="non-backbone"):
        instances[0].filter_tensors(("wrapper.weight", lazy))
    assert {p.name: p.read_bytes() for p in model.iterdir()} == before
    assert [call.kwargs["mmproj"] for call in classes.call_args_list] == [False, True]
    assert len(calls) == (0 if precision == "F16" else 1)


def test_failed_quantization_leaves_no_export(
    export_source, tmp_path: Path, monkeypatch,
) -> None:
    model, checkout, _, _ = export_source
    monkeypatch.setattr(export.subprocess, "run", Mock(
        side_effect=subprocess.CalledProcessError(1, "quantize"),
    ))
    output = tmp_path / "export"
    with pytest.raises(subprocess.CalledProcessError):
        export.export_model(model, output, checkout, "Q8_0")
    assert not output.exists()
    assert not list(tmp_path.glob(".gguf-*"))


@pytest.mark.parametrize("change", [
    {"temperature": 0}, {"temperature": float("nan")}, {"max_options": 27},
    {"model_type": "raw"}, {"option_token_ids": [1] * 26},
])
def test_invalid_source_is_rejected(
    export_source, tmp_path: Path, change: dict,
) -> None:
    model, checkout, _, _ = export_source
    path = model / "config.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(ValueError):
        export.export_model(model, tmp_path / "export", checkout)


def test_adapters_and_existing_outputs_are_rejected(
    export_source, tmp_path: Path,
) -> None:
    model, checkout, _, _ = export_source
    with pytest.raises(FileExistsError):
        export.export_model(model, model, checkout)
    (model / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="merged"):
        export.export_model(model, tmp_path / "export", checkout)


@pytest.mark.parametrize("kind,options,question", [
    ("choice", [{"key": "red", "description": "first"},
                {"key": "blue", "description": None}],
     Question("q", ["red", "blue"], ["first", None])),
    ("noul", [{"key": "true", "description": None},
              {"key": "false", "description": None}],
     Question("q", ["Yes", "No"])),
    ("score", [{"key": "0", "description": "low"}, {"key": "1", "description": "high"}],
     Question("q", ["low", "high"])),
])
def test_native_template_matches_hf_text(
    kind: str, options: list, question: Question,
) -> None:
    processor = ExportProcessor()
    template = Environment().from_string(build_template(processor, 26))
    native = template.render(state="state", instructions="q", type=kind,
                             options=options, images=["<image>"])
    messages = PromptBuilder(processor, 26).build_messages("state", question, [])
    text = messages[0]["content"][0]["text"]
    assert native == "<user><image>" + text + "</user><assistant>"


def test_chat_template_must_preserve_text() -> None:
    processor = ExportProcessor()
    processor.apply_chat_template = lambda *a, **k: "lost user text"
    with pytest.raises(ValueError, match="preserve"):
        build_template(processor, 26)


def test_missing_or_incompatible_converter(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="conversion package"):
        export.load_tools(tmp_path)
    (tmp_path / "conversion").mkdir()
    (tmp_path / "conversion/__init__.py").touch()
    monkeypatch.setattr(
        export.importlib, "import_module",
        lambda _: SimpleNamespace(GGUFWriter=object),
    )
    monkeypatch.setattr(export.sys, "path", list(export.sys.path))
    with pytest.raises(ValueError, match="decision support"):
        export.load_tools(tmp_path)
