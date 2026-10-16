import json
from pathlib import Path
from typing import Any

import datasets
from huggingface_hub import snapshot_download


class HubRecordLoader:
    """Loads labelled records from a HF Hub dataset in the kev-vision layout.

    Rows hold `state` and `questions` as JSON strings and an optional `image`
    column (`{bytes, path}` or None). They are converted on access to records
    for `DecisionEngineDataset`: `state`, `questions` and `media`. An optional
    JSON `teacher_targets` column is also passed through for distillation.

    Args:
        repo_id: HF dataset id, e.g. `Jacqkues/kev-vision-decisions-full`.
    """

    def __init__(self, repo_id: str) -> None:
        self.repo_id: str = repo_id

    def load(self, split: str) -> datasets.Dataset:
        """Loads a split (slices like `validation[:64]` work) as records.

        The dataset repo is downloaded once into `<HF_DATASETS_CACHE>/hub`
        (not the HF Hub model cache), then loaded from that local copy.
        """
        cache_dir = Path(datasets.config.HF_DATASETS_CACHE) / "hub"
        local_dir: str = snapshot_download(
            repo_id=self.repo_id, repo_type="dataset", cache_dir=cache_dir
        )
        return self.convert(datasets.load_dataset(local_dir, split=split))

    @staticmethod
    def convert(dataset: datasets.Dataset) -> datasets.Dataset:
        """Returns the dataset with its rows converted to records on access."""
        dataset = dataset.cast_column("image", datasets.Image())
        return dataset.with_transform(HubRecordLoader._to_records)

    @staticmethod
    def _to_records(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        records = {
            "state": [json.loads(state) for state in batch["state"]],
            "questions": [json.loads(questions) for questions in batch["questions"]],
            "media": [
                [{"type": "image", "data": image}] if image is not None else []
                for image in batch["image"]
            ],
        }
        if "teacher_targets" in batch:
            records["teacher_targets"] = [
                json.loads(targets) for targets in batch["teacher_targets"]
            ]
        return records
