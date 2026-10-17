import src.modal.tensorboard_app as tensorboard_app


def test_app_is_separate() -> None:
    assert tensorboard_app.app.name == "decision-engine-tensorboard"


class FakeVolume:
    """Counts `reload` calls."""

    def __init__(self) -> None:
        self.reloads: int = 0

    def reload(self) -> None:
        self.reloads += 1


def test_app_defines_tensorboard() -> None:
    assert tensorboard_app.tensorboard is not None


def test_runs_reload_middleware_reloads_on_page_load(monkeypatch) -> None:
    volume: FakeVolume = FakeVolume()
    monkeypatch.setitem(tensorboard_app.VOLUMES, "runs", volume)
    middleware = tensorboard_app.RunsReloadMiddleware(lambda environ, start: [b"ok"])
    assert middleware({"PATH_INFO": "/"}, None) == [b"ok"]
    assert middleware({"PATH_INFO": "/data/runs"}, None) == [b"ok"]
    assert volume.reloads == 1
