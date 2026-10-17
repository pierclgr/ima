from pathlib import Path

import pytest

import src.modal.app as modal_app


def test_app_defines_the_tasks() -> None:
    assert modal_app.app.name == "decision-engine"
    assert sorted(modal_app.TASKS) == ["cache", "calibrate", "test", "train"]


def test_app_has_no_web_endpoint() -> None:
    # a web function has a fixed URL, which blocks parallel runs of the app
    assert not hasattr(modal_app, "tensorboard")


@pytest.mark.parametrize(
    ("text", "flag", "expected"),
    [
        ("modal:\n  gpu: H100\n", "", "H100"),
        ("modal:\n  gpu: H100\n", "A100-80GB", "A100-80GB"),
        ("backbone: x\n", "", "H200"),
        ("modal:\n", "", "H200"),
    ],
)
def test_gpu_from_config(tmp_path: Path, text: str, flag: str, expected: str) -> None:
    config = tmp_path / "config.yml"
    config.write_text(text)
    assert modal_app.config_gpu(str(config), flag) == expected
