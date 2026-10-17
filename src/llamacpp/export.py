"""Exports a merged System One checkpoint for llama.cpp's native decision API.

Usage:
    python -m src.llamacpp.export MODEL OUTPUT --llama-cpp-dir CHECKOUT
        [--quantization F16|Q8_0|Q4_K_M]
"""

import argparse
import importlib
import json
import logging
import math
import string
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from transformers import AutoProcessor

from src.llamacpp.prompt import build_template


def load_tools(checkout: Path) -> tuple[Any, Any]:
    """Loads the official converter and its bundled GGUF writer.

    Args:
        checkout: A llama.cpp checkout with native decision support.

    Returns:
        The conversion and gguf modules.

    Raises:
        ValueError: If the checkout lacks conversion or decision support.
    """
    if not (checkout / "conversion" / "__init__.py").is_file():
        raise ValueError(
            "llama-cpp-dir must contain llama.cpp's conversion package"
        )
    sys.path[:0] = [str(checkout), str(checkout / "gguf-py")]
    conversion: Any = importlib.import_module("conversion")
    gguf: Any = importlib.import_module("gguf")
    if not hasattr(gguf.GGUFWriter, "add_decision_type"):
        raise ValueError("llama.cpp checkout lacks native decision support")
    return conversion, gguf


def export_model(
    model: Path, output: Path, llama_cpp_dir: Path, quantization: str = "F16",
) -> Path:
    """Exports unwrapped weights, decision metadata, and the vision projector.

    The official converter reads weights lazily. Only the outer `backbone.`
    prefix and wrapper config are removed; model tensors are converted by
    llama.cpp. A completed export directory is published only on success.

    Args:
        model: Merged System One directory, including its saved processor.
        output: New directory for the GGUF files.
        llama_cpp_dir: llama.cpp source checkout; quantization also needs
            `build/bin/llama-quantize`.
        quantization: F16, Q8_0, or Q4_K_M; projector stays F16.

    Returns:
        Path to the exported language model GGUF.

    Raises:
        FileExistsError: If output already exists.
        ValueError: If the source is not merged or its settings or backbone
            are unsupported.
    """
    model, output, llama_cpp_dir = (
        Path(model).resolve(), Path(output).resolve(),
        Path(llama_cpp_dir).resolve(),
    )
    if output.exists():
        raise FileExistsError(f"export directory already exists: {output}")
    config_path: Path = model / "config.json"
    if (model / "adapter_config.json").exists() or not config_path.is_file():
        raise ValueError(
            "export needs a merged System One checkpoint and processor"
        )
    config: dict[str, Any] = json.loads(config_path.read_text())
    if config.get("model_type") != "system_one":
        raise ValueError("export needs a merged System One checkpoint")
    backbone: dict[str, Any] = config["backbone_config"]
    temperature: float = float(config.get("temperature", 1.0))
    max_options: int = int(config.get("max_options", 26))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    if not 2 <= max_options <= 26:
        raise ValueError("max_options must be in [2, 26]")
    if quantization not in ("F16", "Q8_0", "Q4_K_M"):
        raise ValueError("quantization must be F16, Q8_0, or Q4_K_M")
    quantize: Path = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if quantization != "F16" and not quantize.is_file():
        raise ValueError("build llama.cpp's llama-quantize before quantizing")

    conversion, gguf = load_tools(llama_cpp_dir)
    architecture: str = backbone["architectures"][0]
    converters: list[tuple[bool, Any]] = []
    for mmproj in (False, True) if "vision_config" in backbone else (False,):
        try:
            converters.append((mmproj, conversion.get_model_class(
                architecture, mmproj=mmproj,
            )))
        except NotImplementedError as error:
            raise ValueError(
                f"llama.cpp does not support {architecture}, mmproj={mmproj}"
            ) from error
    processor: Any = AutoProcessor.from_pretrained(
        model, local_files_only=True,
    )
    letter_ids: list[int] = []
    for letter in string.ascii_uppercase + string.ascii_lowercase:
        ids: list[int] = processor.tokenizer.encode(
            letter, add_special_tokens=False,
        )
        if len(ids) != 1:
            raise ValueError(f"decision label {letter!r} must be one token")
        letter_ids.append(ids[0])
    if config.get("option_token_ids") != letter_ids[:max_options]:
        raise ValueError("processor and model use different option token ids")
    template: str = build_template(processor, max_options)

    output.parent.mkdir(parents=True, exist_ok=True)
    filename: str = f"model-{quantization}.gguf"
    with TemporaryDirectory(prefix=".gguf-", dir=output.parent) as temporary:
        artifacts: Path = Path(temporary) / "export"
        artifacts.mkdir()
        for mmproj, converter in converters:
            # extend conversion only; the neural network stays in llama.cpp
            class DecisionConverter(converter):
                model_arch = converter.model_arch
                # decisions never use a multi-token prediction draft head
                no_mtp = True

                @classmethod
                def filter_tensors(
                    cls, item: tuple[str, Callable[[], Any]],
                ) -> tuple[str, Callable[[], Any]] | None:
                    name, tensor = item
                    if not name.startswith("backbone."):
                        raise ValueError(
                            f"unexpected non-backbone tensor: {name}"
                        )
                    return super().filter_tensors((
                        name[len("backbone."):], tensor,
                    ))

                def prepare_metadata(self, vocab_only: bool) -> None:
                    super().prepare_metadata(vocab_only)
                    if mmproj and backbone.get("model_type") in (
                        "qwen3_5", "qwen3_5_moe",
                    ):
                        # Qwen's size bounds are pixel areas, not side lengths
                        size: dict[str, int] = processor.image_processor.size
                        self.gguf_writer.add_vision_min_pixels(
                            size["shortest_edge"],
                        )
                        self.gguf_writer.add_vision_max_pixels(
                            size["longest_edge"],
                        )
                    elif not mmproj:
                        self.gguf_writer.add_decision_type("openjev")
                        for kind in ("choice", "noul", "score"):
                            self.gguf_writer.add_decision_temperature(
                                kind, temperature,
                            )
                        self.gguf_writer.add_chat_template([
                            {"name": "systemone", "template": template},
                        ])

            target: Path = artifacts / (
                "mmproj-F16.gguf" if mmproj else "model-F16.gguf"
            )
            instance: Any = DecisionConverter(
                model, gguf.LlamaFileType.MOSTLY_F16, target,
                hparams=backbone, model_name=model.parent.name,
            )
            instance.write()
            del instance
        if quantization != "F16":
            intermediate: Path = artifacts / "model-F16.gguf"
            subprocess.run(
                [str(quantize), str(intermediate),
                 str(artifacts / filename), quantization],
                check=True,
            )
            intermediate.unlink()
        artifacts.replace(output)
    return output / filename


def main(argv: list[str] | None = None) -> None:
    """Exports a model using command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument(
        "--quantization", choices=("F16", "Q8_0", "Q4_K_M"), default="F16",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    print(export_model(
        args.model, args.output, args.llama_cpp_dir, args.quantization,
    ))


if __name__ == "__main__":
    main()
