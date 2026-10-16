from pathlib import Path
from types import SimpleNamespace

import pytest

from src.runpod.app import MODULES, pod_request, start_command


def test_runpod_has_no_distill_task_and_train_serves_tensorboard() -> None:
    assert "distill" not in MODULES
    command = start_command("train", ["config.yml"], "abc123")
    assert "python -m src.training.train config.yml" in command
    assert "tensorboard --logdir" in command


@pytest.mark.parametrize("distillation", [False, True])
def test_runpod_cache_selects_cpu_or_gpu(
    tmp_path: Path, distillation: bool,
) -> None:
    config = tmp_path / "config.yml"
    config.write_text(
        "distillation: {teacher: t, soft_labels: s}\n" if distillation
        else "backbone: raw\n",
    )
    request = pod_request(
        "cache", [str(config)], "abc123", "GPU", "DC", "volume",
    )
    assert ("gpu" in request) == distillation
    assert ("cpu" in request) != distillation
    assert "ports" not in request
    assert "python -m src.modal.cache" in request["cmd"][0]


@pytest.mark.parametrize("distillation", [False, True])
def test_modal_cache_selects_cpu_or_gpu(
    tmp_path: Path, distillation: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.modal import app

    config = tmp_path / "config.yml"
    config.write_text(
        "distillation: {teacher: t, soft_labels: s}\n" if distillation
        else "backbone: raw\n",
    )
    assert "distill" not in app.TASKS
    assert not hasattr(app, "cache_distillation")
    calls = []

    class Function:
        """Captures launch options without contacting Modal."""

        def __init__(self, name: str) -> None:
            self.name = name

        def with_options(self, **options: str | int) -> "Function":
            calls.append((self.name, options))
            return self

        def spawn(self, argv: list[str]) -> SimpleNamespace:
            calls.append((self.name, argv))
            return SimpleNamespace(get=lambda: None)

    monkeypatch.setitem(app.TASKS, "cache", Function("cache"))
    app.main("cache", str(config), gpu="H200")
    if distillation:
        assert calls == [("cache", {"gpu": "H200", "memory": 65536}),
                         ("cache", [str(config)])]
    else:
        assert calls == [("cache", [str(config)])]
