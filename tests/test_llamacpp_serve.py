"""The server launcher delegates inference to the native executable."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.llamacpp import serve


@pytest.mark.parametrize("projector", [False, True])
def test_native_server_arguments(tmp_path: Path, monkeypatch, projector: bool) -> None:
    export = tmp_path / "export"
    export.mkdir()
    model = export / "model-Q8_0.gguf"
    model.touch()
    if projector:
        (export / "mmproj-F16.gguf").touch()
    checkout = tmp_path / "llama.cpp"
    binary = checkout / "build/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.touch()
    execute = Mock()
    monkeypatch.setattr(serve.os, "execv", execute)
    serve.main([str(export), "--llama-cpp-dir", str(checkout),
                "--host", "0.0.0.0", "--port", "9000", "--ctx-size", "4096",
                "--gpu-layers", "0", "--api-key", "local-test"])
    args = [str(binary), "--model", str(model), "--host", "0.0.0.0",
            "--port", "9000", "--ctx-size", "4096", "--n-gpu-layers", "0"]
    if projector:
        args += ["--mmproj", str(export / "mmproj-F16.gguf")]
    args += ["--api-key", "local-test"]
    execute.assert_called_once_with(str(binary), args)


def test_missing_or_ambiguous_exports_and_binary(tmp_path: Path) -> None:
    argv = [str(tmp_path), "--llama-cpp-dir", str(tmp_path)]
    with pytest.raises(ValueError, match="exactly one"):
        serve.main(argv)
    (tmp_path / "model-F16.gguf").touch()
    with pytest.raises(ValueError, match="build llama.cpp"):
        serve.main(argv)
    (tmp_path / "model-Q8_0.gguf").touch()
    with pytest.raises(ValueError, match="exactly one"):
        serve.main(argv)
