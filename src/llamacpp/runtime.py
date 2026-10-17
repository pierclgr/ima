"""Persistent native inference worker for offline System One evaluation."""

import base64
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.common.types import Question
from src.testing.config import LlamaCppSettings


def build_worker(checkout: Path) -> Path:
    """Builds the small adapter against an already built llama.cpp checkout.

    Build output is cached per source/checkout location. CMake checks source,
    header, and library changes on every run, outside inference timing.
    """
    source = Path(__file__).resolve().parent / "native"
    identity = hashlib.sha256(f"{source}:{checkout}".encode()).hexdigest()[:16]
    build = Path(".cache/llamacpp").resolve() / identity
    commands = [
        ["cmake", "-S", str(source), "-B", str(build),
         f"-DLLAMA_CPP_DIR={checkout}", "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build), "--parallel", "2"],
    ]
    for command in commands:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise RuntimeError("install CMake and a C++17 compiler") from error
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "native worker build failed; build llama.cpp's shared "
                f"libraries first:\n{error.stdout}\n{error.stderr}"
            ) from error
    return build / "ima-llamacpp-worker"


class LlamaCppRuntime:
    """Loads one GGUF export and exchanges untimed JSON with a C++ worker.

    Prints worker setup and model loading status to stderr.

    Args:
        model: Export directory containing one `model-*.gguf`.
        settings: Native context and device settings.

    Attributes:
        temperatures: Saved calibration temperatures by question type.
        device: Native device names, independent of PyTorch's CUDA state.
    """

    def __init__(self, model: str, settings: LlamaCppSettings) -> None:
        directory = Path(model).expanduser().resolve()
        models = list(directory.glob("model-*.gguf"))
        if len(models) != 1:
            raise ValueError("model must contain exactly one model-*.gguf")
        checkout = Path(settings.llama_cpp_dir).expanduser().resolve()
        print("Preparing llama.cpp worker...", file=sys.stderr, flush=True)
        binary = build_worker(checkout)
        projector = directory / "mmproj-F16.gguf"
        print(f"Loading GGUF model {models[0]}...", file=sys.stderr, flush=True)
        self.log = tempfile.TemporaryFile(mode="w+t")
        self.process = subprocess.Popen(
            [str(binary), str(models[0]),
             str(projector) if projector.is_file() else "",
             str(settings.ctx_size), str(settings.gpu_layers),
             str(settings.threads or 0)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1,
        )
        try:
            metadata = self._read()
            self.temperatures: dict[str, float] = metadata["temperatures"]
            self.device: str = metadata["device"]
        except BaseException:
            self.close()
            raise

    def _read(self) -> dict[str, Any]:
        """Reads a result, raising on native errors or an unexpected exit."""
        line = self.process.stdout.readline()
        try:
            response = json.loads(line)
        except json.JSONDecodeError as error:
            self.log.seek(0)
            raise RuntimeError(
                f"native worker returned invalid output: {line!r}\n"
                + self.log.read()[-8000:]
            ) from error
        if "error" in response:
            self.log.seek(0)
            raise RuntimeError(
                f"native inference failed: {response['error']}\n"
                + self.log.read()[-8000:]
            )
        return response

    def predict(self, item: dict[str, Any]) -> dict[str, Any]:
        """Returns raw option logits and native model seconds for one item.

        PNG encoding and transport happen before the native inference clock.
        The worker also prepares images and tokens before starting its clock.
        """
        question: Question = item["question"]
        kind: str = item["type"]
        descriptions = question.descriptions or [None] * len(question.options)
        options = []
        for i, (key, description) in enumerate(zip(
            question.options, descriptions
        )):
            if kind == "score":
                key, description = str(i), key
            elif kind == "noul":
                key = "true" if i == 0 else "false"
            options.append({"key": key, "description": description})
        files: list[str] = []
        for image in item["images"]:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            files.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        request = {"state": item["state"], "instructions": question.text,
                   "type": kind, "options": options, "files": files}
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.log.seek(0)
            raise RuntimeError(
                "native worker exited:\n" + self.log.read()[-8000:]
            ) from error
        return self._read()

    def close(self) -> None:
        """Closes the worker and releases its model, also after failure."""
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()
        self.log.close()
