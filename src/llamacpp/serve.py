"""Starts llama.cpp's native server for an exported System One model.

Usage:
    python -m src.llamacpp.serve EXPORT --llama-cpp-dir CHECKOUT
        [--host 127.0.0.1] [--port 8080] [--ctx-size 8192]
        [--gpu-layers auto] [--api-key KEY]

This module uses only the Python standard library. Model inference and the
HTTP API are provided by the native llama-server process.
"""

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Starts llama-server in this process with native signal handling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--gpu-layers", default="auto")
    parser.add_argument("--api-key")
    args = parser.parse_args(argv)
    directory: Path = args.export.resolve()
    models: list[Path] = list(directory.glob("model-*.gguf"))
    if len(models) != 1:
        raise ValueError("export must contain exactly one model-*.gguf")
    binary: Path = (
        args.llama_cpp_dir.resolve() / "build" / "bin" / "llama-server"
    )
    if not binary.is_file():
        raise ValueError("build llama.cpp's llama-server before serving")
    command: list[str] = [
        str(binary), "--model", str(models[0]),
        "--host", args.host, "--port", str(args.port),
        "--ctx-size", str(args.ctx_size), "--n-gpu-layers", args.gpu_layers,
    ]
    projector: Path = directory / "mmproj-F16.gguf"
    if projector.is_file():
        command.extend(["--mmproj", str(projector)])
    if args.api_key:
        command.extend(["--api-key", args.api_key])
    os.execv(str(binary), command)


if __name__ == "__main__":
    main()
