import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import TrainingArguments

from src.common.config_parser import ConfigParser


# seed of the random questions of `TestDataSettings.sample`
SAMPLE_SEED: int = 42


@dataclass
class TestDataSettings:
    """Test data on the HF Hub or in a local Parquet root.

    Attributes:
        dataset: HF dataset id or directory with split subdirectories.
        split: Test split (slices like `test[:500]` work).
        sample: Number of random questions of the split to test (seed
            `SAMPLE_SEED`, the same ones in every run), or None for all.
    """

    # not a pytest test class
    __test__ = False

    dataset: str
    split: str
    sample: int | None = None


# System Two thinking: off, or the reasoning effort (Qwen3.8 template levels)
THINKING: tuple[str, ...] = ("off", "low", "medium", "xhigh")


@dataclass
class SystemTwoSettings:
    """System Two mode: the model generates its answer (see `SystemTwoModel`).

    Attributes:
        thinking: `off`, or a reasoning effort `low`, `medium`, `xhigh`
            (thinking on; Qwen3.5 has no levels, any of them is just on).
            YAML reads a bare `off` as false, which is accepted as `off`.
        max_new_tokens: Cap on generated tokens, thinking included. Each
            question stops at `</answer>` or the model's end token; the cap
            only stops outputs that never do. Default: 256 with thinking off
            (an answer block is about 10 tokens), 32768 with thinking (Qwen's
            suggested output length for thinking).

    Raises:
        ValueError: If `thinking` is not one of the levels.
    """

    thinking: str = "off"
    max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.thinking is False:
            self.thinking = "off"
        if self.thinking not in THINKING:
            raise ValueError(f"thinking must be one of {THINKING}")
        if self.max_new_tokens is None:
            self.max_new_tokens = 256 if self.thinking == "off" else 32768
        self.max_new_tokens = int(self.max_new_tokens)


@dataclass
class LlamaCppSettings:
    """Native runtime settings, independent of HF device checks.

    Attributes:
        llama_cpp_dir: Checkout with built shared libraries in `build/bin`.
        ctx_size: Maximum context positions per question.
        gpu_layers: Offloaded layers (-1 for all, 0 for CPU).
        threads: CPU threads, or None for the native default.
    """

    llama_cpp_dir: str = "~/llama.cpp"
    ctx_size: int = 8192
    gpu_layers: int = -1
    threads: int | None = None

    def __post_init__(self) -> None:
        if self.ctx_size <= 0:
            raise ValueError("ctx_size must be > 0")
        if self.threads is not None and self.threads <= 0:
            raise ValueError("threads must be > 0")


@dataclass
class LlamaCppTestArguments:
    """Output location and the fixed single-question native batch size."""

    output_dir: str
    per_device_eval_batch_size: int = 1

    def __post_init__(self) -> None:
        if self.per_device_eval_batch_size != 1:
            raise ValueError("llamacpp requires per_device_eval_batch_size: 1")


@dataclass
class TestConfig:
    """Test run configuration, loaded from a YAML file.

    The YAML has `model`, optional `temperature` and `backend`, and the
    sections `data` and `testing`. HF accepts `TrainingArguments`; GGUF
    accepts `LlamaCppTestArguments` and optional `llamacpp` settings.
    `testing.output_dir` receives `test_metrics.json`.

    Attributes:
        model: HF model id/checkpoint or a GGUF export directory.
        data: Test data.
        testing: HF Trainer arguments or single-question GGUF arguments.
        temperature: Temperature instead of the model's saved one (e.g. 1.0
            for uncalibrated scores), or None to use the saved one.
        system_two: System Two settings (optional `system_two` section, even
            empty), or None for System One.
        backend: `hf` (default) or `llamacpp`.
        llamacpp: Native settings, required only for the GGUF backend.

    Raises:
        ValueError: If the temperature is not > 0.
    """

    # not a pytest test class
    __test__ = False

    model: str
    data: TestDataSettings
    testing: TrainingArguments | LlamaCppTestArguments
    temperature: float | None = None
    system_two: SystemTwoSettings | None = None
    backend: str = "hf"
    llamacpp: LlamaCppSettings | None = None

    def __post_init__(self) -> None:
        if self.temperature is not None and (
            not math.isfinite(self.temperature) or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and > 0")
        if self.backend not in ("hf", "llamacpp"):
            raise ValueError("backend must be hf or llamacpp")
        if self.backend == "llamacpp" and self.system_two is not None:
            raise ValueError("llamacpp testing does not support System Two")
        if self.backend == "hf" and self.llamacpp is not None:
            raise ValueError("llamacpp settings require backend: llamacpp")

    @classmethod
    def from_yaml(
        cls, path: str | Path, overrides: list[str] | None = None
    ) -> "TestConfig":
        """Loads the configuration from a YAML file.

        Args:
            path: The YAML file.
            overrides: CLI overrides as `--key value` pairs, with dotted keys
                for sections, e.g. `["--data.split", "test_ood"]`.

        Returns:
            The configuration.

        Raises:
            ValueError: If an override is malformed, or a section or key is
                unknown or missing.
        """
        values: dict[str, Any] = ConfigParser.load(
            path,
            overrides,
            keys=["model", "temperature", "backend"],
            sections=["data", "testing", "system_two", "llamacpp"],
        )
        backend = str(values.pop("backend", "hf"))
        if backend not in ("hf", "llamacpp"):
            raise ValueError("backend must be hf or llamacpp")
        model: str = str(values.pop("model"))
        temperature: Any = values.pop("temperature", None)
        system_two: SystemTwoSettings | None = ConfigParser.settings(
            values, "system_two", SystemTwoSettings
        )
        data = ConfigParser.section(values, "data", TestDataSettings)
        testing = ConfigParser.section(
            values, "testing",
            (LlamaCppTestArguments if backend == "llamacpp"
             else TrainingArguments),
        )
        llamacpp = (
            ConfigParser.section(values, "llamacpp", LlamaCppSettings)
            if backend == "llamacpp" else None
        )
        ConfigParser.check_empty(values)
        return cls(
            model=model,
            data=data,
            testing=testing,
            temperature=None if temperature is None else float(temperature),
            system_two=system_two,
            backend=backend,
            llamacpp=llamacpp,
        )
