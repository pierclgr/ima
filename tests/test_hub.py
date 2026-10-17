import io
import json
from pathlib import Path

import datasets
import pytest
from PIL import Image

from src.data.dataset import DecisionEngineDataset
from src.data.hub import HubRecordLoader

QUESTIONS: dict = {
    "q": {"type": "noul", "instructions": "Red?", "label": True, "src": "x"}
}
README: str = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: train/*.parquet
  - split: validation
    path: validation/*.parquet
---
"""


def raw_dataset(n: int = 2) -> datasets.Dataset:
    """Rows in the kev-vision layout: JSON strings and an image struct.

    Even rows have an image and a text state, odd rows no image and a JSON
    state.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    image = {"bytes": buffer.getvalue(), "path": None}
    return datasets.Dataset.from_dict(
        {
            "id": [str(i) for i in range(n)],
            "state": [
                json.dumps("A photo." if i % 2 == 0 else {"game": "pong"})
                for i in range(n)
            ],
            "questions": [json.dumps(QUESTIONS)] * n,
            "image": [image if i % 2 == 0 else None for i in range(n)],
        }
    )


def fake_repo(root: Path) -> Path:
    """A dataset repo snapshot laid out like kev-vision."""
    repo = root / "repo"
    for split, n in (("train", 6), ("validation", 4)):
        (repo / split).mkdir(parents=True)
        raw_dataset(n).to_parquet(repo / split / "part.parquet")
    (repo / "README.md").write_text(README)
    return repo


def test_convert_to_records() -> None:
    records = HubRecordLoader.convert(raw_dataset())
    first, second = records[0], records[1]
    assert set(first) == {"state", "questions", "files"}
    assert first["state"] == "A photo."
    assert second["state"] == {"game": "pong"}
    assert first["questions"] == QUESTIONS
    assert isinstance(first["files"][0], Image.Image)
    assert second["files"] == []


def test_records_feed_dataset() -> None:
    dataset = DecisionEngineDataset(HubRecordLoader.convert(raw_dataset()))
    assert len(dataset) == 2
    assert len(dataset[0]["images"]) == 1
    assert dataset[0]["target"] == [1.0, 0.0]


def test_load_downloads_to_datasets_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = fake_repo(tmp_path)
    calls: dict = {}

    def snapshot_download(repo_id: str, repo_type: str, cache_dir: Path) -> str:
        calls.update(repo_id=repo_id, repo_type=repo_type, cache_dir=cache_dir)
        return str(repo)

    monkeypatch.setattr("src.data.hub.snapshot_download", snapshot_download)
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    records = HubRecordLoader("org/kev").load("validation[:3]")
    assert calls == {
        "repo_id": "org/kev",
        "repo_type": "dataset",
        "cache_dir": tmp_path / "cache" / "hub",
    }
    assert len(records) == 3
    assert records[0]["questions"] == QUESTIONS
    assert len(HubRecordLoader("org/kev").load("train")) == 6
