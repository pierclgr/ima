"""Optional CPU checks against actual F16/Q8/Q4 exports and native API."""

import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.common.question_types import question_type
from src.data.dataset import DecisionEngineDataset
from src.evaluation.llamacpp import LlamaCppEvaluator
from src.llamacpp.runtime import LlamaCppRuntime
from src.model.output import DecisionEngineOutput
from src.testing.config import LlamaCppSettings


@pytest.fixture(scope="module", params=["f16", "q8", "q4"])
def native_export(request):
    checkout = os.environ.get("LLAMA_CPP_DIR")
    exports = os.environ.get("IMA_TEST_EXPORTS")
    if not checkout or not exports:
        pytest.skip("set LLAMA_CPP_DIR and IMA_TEST_EXPORTS for native checks")
    model = Path(exports).expanduser() / f"qwen3_5_0_8b_{request.param}"
    assert model.is_dir(), f"missing export: {model}"
    return Path(checkout).expanduser(), model


@pytest.fixture(scope="module")
def native_runtime(native_export):
    checkout, model = native_export
    runtime = LlamaCppRuntime(str(model), LlamaCppSettings(
        llama_cpp_dir=str(checkout), gpu_layers=0, threads=2,
    ))
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture(scope="module")
def native_api(native_export, tmp_path_factory):
    checkout, model = native_export
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path_factory.mktemp("native-server") / "server.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "src.llamacpp.serve", str(model),
            "--llama-cpp-dir", str(checkout), "--port", str(port),
            "--gpu-layers", "0",
        ], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(log_path.read_text()[-8000:])
                try:
                    with urllib.request.urlopen(url + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            else:
                pytest.fail("native server startup timed out")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize("image", [False, True])
def test_native_worker_matches_api_and_resets_context(
    native_runtime, native_api: str, image: bool,
) -> None:
    questions = {
        "choice": {"type": "choice", "instructions": "Which color?",
                   "criteria": {"red": None, "blue": "a color"}, "label": "red"},
        "noul": {"type": "noul", "instructions": "Is this red?", "label": True},
        "score": {"type": "score", "instructions": "How bright?",
                  "criteria": ["dark", "medium", "bright"], "label": 1},
    }
    files = [Image.new("RGB", (56, 56), "red")] if image else []
    record = {"state": "A red square.", "questions": questions, "files": files}
    dataset = DecisionEngineDataset([record], shuffle_options=False)
    request = {"state": record["state"], "questions": questions, "files": []}
    if image:
        buffer = io.BytesIO()
        files[0].save(buffer, format="PNG")
        request["files"] = [
            "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        ]
    with urllib.request.urlopen(urllib.request.Request(
        native_api + "/v1/systemone", data=json.dumps(request).encode(),
        headers={"Content-Type": "application/json"},
    ), timeout=60) as response:
        api = json.load(response)
    tokens = 0
    first = None
    for i, kind in enumerate(questions):
        item = dataset[i]
        prediction = native_runtime.predict(item)
        if first is None:
            first = prediction["logits"]
        tokens += prediction["input_tokens"]
        assert np.isfinite(prediction["seconds"]) and prediction["seconds"] > 0
        output = DecisionEngineOutput.from_logits(
            torch.tensor([prediction["logits"]]), native_runtime.temperatures[kind],
        )
        expected = question_type(kind).format_answer(
            item["question"], output.probabilities[0].tolist(),
        )
        answer = api["answers"][kind]
        key = "noul" if kind == "noul" else "probabilities"
        assert answer[key] == pytest.approx(expected[key], abs=0.003)
    assert api["usage"]["input_tokens"] == tokens
    repeated = native_runtime.predict(dataset[0])["logits"]
    assert repeated == pytest.approx(first, abs=1e-4)


def test_native_offline_score_eleven_and_metrics(native_runtime) -> None:
    record = {"state": "A rating.", "questions": {"q": {
        "type": "score", "instructions": "Rate it.",
        "criteria": [str(i) for i in range(11)], "label": 5,
    }}}
    result = LlamaCppEvaluator(native_runtime).predict([record])
    assert result.metrics["test_evaluated_questions"] == 1
    assert result.metrics["test_skipped_questions"] == 0
    assert result.label_ids[0, 5] == 1
    assert np.all(np.isfinite(list(result.metrics.values())))


def test_native_context_overflow_fails(native_export) -> None:
    checkout, model = native_export
    runtime = LlamaCppRuntime(str(model), LlamaCppSettings(
        llama_cpp_dir=str(checkout), ctx_size=128, gpu_layers=0, threads=2,
    ))
    try:
        dataset = DecisionEngineDataset([{
            "state": "long text " * 1000, "questions": {"q": {
                "type": "noul", "instructions": "Yes?", "label": True,
            }},
        }])
        with pytest.raises(RuntimeError, match="ctx_size"):
            runtime.predict(dataset[0])
    finally:
        runtime.close()
