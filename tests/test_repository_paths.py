"""Keep regression source in Git and generated benchmark results in runs."""

import subprocess
from pathlib import Path

import yaml


def test_test_source_is_not_gitignored() -> None:
    paths = sorted(str(path) for path in Path("tests").glob("*.py"))
    paths.append("tests/test_future_regression.py")
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        capture_output=True, text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr


def test_benchmark_results_are_gitignored() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "runs/llamacpp/f16/test/test_metrics.json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_shipped_test_configs_write_under_runs() -> None:
    paths = sorted(Path("configs/test").glob("*.yml"))
    assert paths
    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert Path(config["testing"]["output_dir"]).parts[0] == "runs", path
