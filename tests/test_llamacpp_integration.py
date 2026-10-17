"""Optional real converter checks, using a local merged source checkpoint."""

import hashlib
import os
from pathlib import Path

import pytest

from src.llamacpp.export import export_model, load_tools


def digest(path: Path) -> str:
    """Hashes a local source file without keeping it in memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@pytest.mark.parametrize("precision", ["F16", "Q8_0", "Q4_K_M"])
def test_real_export_preserves_source(tmp_path: Path, precision: str) -> None:
    checkout = os.environ.get("LLAMA_CPP_DIR")
    source = os.environ.get("IMA_TEST_MODEL")
    if not checkout or not source:
        pytest.skip("set LLAMA_CPP_DIR and IMA_TEST_MODEL for real GGUF exports")
    model = Path(source).expanduser()
    before = {
        path.name: digest(path)
        for path in model.iterdir() if path.is_file()
    }
    target = export_model(model, tmp_path / precision, Path(checkout), precision)
    assert target.is_file() and target.stat().st_size > 0
    _, gguf = load_tools(Path(checkout))
    reader = gguf.GGUFReader(str(target))
    assert "tokenizer.chat_template.systemone" in reader.fields
    assert any(key.endswith("decision.type") for key in reader.fields)
    assert (target.parent / "mmproj-F16.gguf").is_file()
    for name, before_digest in before.items():
        assert digest(model / name) == before_digest
