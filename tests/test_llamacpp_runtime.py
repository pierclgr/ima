"""Worker builds, JSON transport, image encoding, and process cleanup."""

import base64
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from src.common.types import Question
from src.llamacpp import runtime as native
from src.testing.config import LlamaCppSettings


@pytest.fixture
def worker(tmp_path: Path, monkeypatch, capsys):
    model = tmp_path / "model-Q8_0.gguf"
    model.touch()
    (tmp_path / "mmproj-F16.gguf").touch()
    process = SimpleNamespace(
        stdin=io.StringIO(), stdout=io.StringIO(json.dumps({
            "device": "CPU", "temperatures": {"choice": 2, "noul": 2, "score": 2},
        }) + '\n' + json.dumps({"logits": [1, 0], "seconds": 0.125}) + '\n'),
        wait=Mock(), kill=Mock(),
    )
    build = Mock(return_value=tmp_path / "worker")
    start = Mock(return_value=process)
    monkeypatch.setattr(native, "build_worker", build)
    monkeypatch.setattr(native.subprocess, "Popen", start)
    runtime = native.LlamaCppRuntime(str(tmp_path), LlamaCppSettings(
        llama_cpp_dir=str(tmp_path / "llama.cpp"), gpu_layers=0, threads=2,
    ))
    captured = capsys.readouterr()
    assert not captured.out
    assert "Preparing llama.cpp worker" in captured.err
    assert "Loading GGUF model" in captured.err
    yield runtime, process, start
    if not runtime.log.closed:
        runtime.close()


@pytest.mark.parametrize("kind,options,expected", [
    ("choice", ["red", "blue"], ["red", "blue"]),
    ("noul", ["Yes", "No"], ["true", "false"]),
    ("score", ["low", "high"], ["0", "1"]),
])
def test_worker_protocol(
    worker, kind: str, options: list, expected: list, capsys,
) -> None:
    runtime, process, start = worker
    result = runtime.predict({"state": "s", "type": kind,
                              "question": Question("q", options),
                              "images": [Image.new("RGB", (4, 4))]})
    request = json.loads(process.stdin.getvalue())
    assert [option["key"] for option in request["options"]] == expected
    if kind == "score":
        assert [option["description"] for option in request["options"]] == options
    assert base64.b64decode(request["files"][0]).startswith(b'\x89PNG\r\n\x1a\n')
    assert result == {"logits": [1, 0], "seconds": 0.125}
    assert runtime.device == "CPU" and runtime.temperatures["score"] == 2
    assert start.call_args.args[0][-3:] == ["8192", "0", "2"]
    captured = capsys.readouterr()
    assert not captured.out


@pytest.mark.parametrize("response", ['{"error":"context overflow"}\n', 'bad\n', ''])
def test_worker_errors_are_not_skipped(worker, response: str) -> None:
    runtime, process, _ = worker
    process.stdout = io.StringIO(response)
    runtime.log.write("native diagnostic")
    with pytest.raises(RuntimeError, match="native diagnostic"):
        runtime.predict({"state": "s", "type": "noul",
                         "question": Question("q", ["Yes", "No"]), "images": []})


def test_close_kills_unresponsive_worker(worker) -> None:
    runtime, process, _ = worker
    process.wait.side_effect = [subprocess.TimeoutExpired("worker", 10), 0]
    runtime.close()
    process.kill.assert_called_once()
    assert process.stdin.closed and process.stdout.closed and runtime.log.closed


def test_build_worker_is_cached_locally(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = Mock()
    monkeypatch.setattr(native.subprocess, "run", run)
    binary = native.build_worker(tmp_path / "llama.cpp")
    assert binary.is_relative_to(tmp_path / ".cache/llamacpp")
    assert binary.name == "ima-llamacpp-worker"
    assert run.call_count == 2
    assert run.call_args_list[0].args[0][0] == "cmake"
    assert run.call_args_list[1].args[0][:2] == ["cmake", "--build"]
    assert native.build_worker(tmp_path / "other") != binary


@pytest.mark.parametrize("error,match", [
    (FileNotFoundError(), "CMake"),
    (subprocess.CalledProcessError(1, "cmake", "out", "build detail"), "build detail"),
])
def test_build_failures_include_diagnostics(
    tmp_path: Path, monkeypatch, error, match: str,
) -> None:
    monkeypatch.setattr(native.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match=match):
        native.build_worker(tmp_path)


def test_invalid_export_fails_before_build(tmp_path: Path, monkeypatch) -> None:
    build = Mock()
    monkeypatch.setattr(native, "build_worker", build)
    with pytest.raises(ValueError, match="exactly one"):
        native.LlamaCppRuntime(str(tmp_path), LlamaCppSettings())
    build.assert_not_called()


def test_startup_failure_releases_process(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "model-F16.gguf").touch()
    process = SimpleNamespace(
        stdin=io.StringIO(), stdout=io.StringIO('{"error":"load failed"}\n'),
        wait=Mock(), kill=Mock(),
    )
    monkeypatch.setattr(native, "build_worker", lambda _: tmp_path / "worker")
    monkeypatch.setattr(native.subprocess, "Popen", Mock(return_value=process))
    with pytest.raises(RuntimeError, match="load failed"):
        native.LlamaCppRuntime(str(tmp_path), LlamaCppSettings())
    process.wait.assert_called_once()
    assert process.stdin.closed and process.stdout.closed


def test_broken_pipe_includes_worker_diagnostics(worker) -> None:
    runtime, process, _ = worker
    process.stdin = Mock()
    process.stdin.write.side_effect = BrokenPipeError()
    runtime.log.write("worker exited unexpectedly")
    with pytest.raises(RuntimeError, match="worker exited unexpectedly"):
        runtime.predict({"state": "s", "type": "noul",
                         "question": Question("q", ["Yes", "No"]),
                         "images": []})
