from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from transformers import HfArgumentParser


class ConfigParser:
    """Loads YAML run configs with CLI overrides into typed dataclasses.

    Shared by the training, calibration and test configs.
    """

    @staticmethod
    def load(
        path: str | Path,
        overrides: list[str] | None,
        keys: Iterable[str],
        sections: Iterable[str],
    ) -> dict[str, Any]:
        """Reads a YAML file and applies CLI overrides.

        The `modal` and `runpod` sections are dropped: they belong to the
        cloud launchers.

        Args:
            path: The YAML file.
            overrides: `--key value` pairs: `--<key>` for a top-level key or
                `--<section>.<key>` for a section key (more dots for nested
                keys, e.g. `--training.distillation.alpha`).
            keys: Allowed top-level keys for overrides.
            sections: Allowed sections for overrides.

        Returns:
            The raw values.

        Raises:
            ValueError: If an override is malformed or unknown.
        """
        values: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        # read only by the cloud launchers (`src/modal/app.py`, `src/runpod/app.py`)
        values.pop("modal", None)
        values.pop("runpod", None)
        overrides = overrides or []
        if len(overrides) % 2:
            raise ValueError("overrides must be --key value pairs")
        for flag, value in zip(overrides[::2], overrides[1::2]):
            if not flag.startswith("--"):
                raise ValueError(f"override {flag!r} must start with --")
            section, _, key = flag[2:].partition(".")
            if not key and section in keys:
                values[section] = value
            elif key and section in sections:
                values[section] = values.get(section) or {}
                node = values[section]
                *parents, last = key.split(".")
                for parent in parents:
                    node[parent] = node.get(parent) or {}
                    node = node[parent]
                node[last] = value
            else:
                raise ValueError(f"unknown override {flag!r}")
        return values

    @staticmethod
    def parse(dataclass_type: type, values: dict[str, Any]) -> Any:
        """Builds a dataclass with HF's argument parser, which converts types.

        Raises:
            ValueError: If a key is unknown or a required one is missing.
        """
        argv: list[str] = []
        for key, value in values.items():
            # null keeps the field default
            if value is None:
                continue
            argv.append(f"--{key}")
            if isinstance(value, list):
                argv.extend(str(item) for item in value)
            else:
                argv.append(str(value))
        parser = HfArgumentParser(dataclass_type)
        try:
            parsed, remaining = parser.parse_args_into_dataclasses(
                args=argv, return_remaining_strings=True
            )
        except SystemExit as error:
            raise ValueError(f"invalid {dataclass_type.__name__}: {values}") from error
        if remaining:
            raise ValueError(f"unknown {dataclass_type.__name__} keys: {remaining}")
        return parsed

    @staticmethod
    def section(values: dict[str, Any], name: str, dataclass_type: type) -> Any:
        """Pops a section and builds it with `parse` (missing: defaults)."""
        return ConfigParser.parse(dataclass_type, values.pop(name, None) or {})

    @staticmethod
    def settings(values: dict[str, Any], name: str, settings_type: type) -> Any:
        """Pops a section and builds a dataclass that converts its own values.

        For types `parse` cannot read (e.g. `float | str`).

        Returns:
            The settings, or None if the section is missing.

        Raises:
            ValueError: If a key is unknown.
        """
        if name not in values:
            return None
        try:
            return settings_type(**(values.pop(name) or {}))
        except TypeError as error:
            raise ValueError(f"unknown {name} keys") from error

    @staticmethod
    def check_empty(values: dict[str, Any]) -> None:
        """Checks that all keys were read.

        Raises:
            ValueError: If keys are left.
        """
        if values:
            raise ValueError(f"unknown keys: {sorted(values)}")
