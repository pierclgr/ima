import pytest
import torch
from PIL import Image

from src.data.dataset import DecisionEngineDataset

CHOICE: dict = {
    "type": "choice",
    "instructions": "Tone?",
    "criteria": {"calm": None, "frustrated": "annoyed", "angry": None},
    "label": "frustrated",
}
NOUL: dict = {"type": "noul", "instructions": "Billing?", "label": False}
SCORE: dict = {
    "type": "score",
    "instructions": "Urgency?",
    "criteria": ["low", "mid", "high"],
    "label": 2,
}
RECORD: dict = {
    "state": {"text": "I was charged twice."},
    "questions": {"tone": CHOICE, "billing": NOUL, "urgency": SCORE},
}


def by_type(dataset: DecisionEngineDataset, kind: str) -> dict:
    return next(
        dataset[i] for i in range(len(dataset)) if dataset[i]["type"] == kind
    )


def test_one_item_per_question() -> None:
    assert len(DecisionEngineDataset([RECORD, RECORD])) == 6


def test_item_content() -> None:
    item = DecisionEngineDataset([RECORD], shuffle_options=False)[0]
    assert item["state"] == '{\n  "text": "I was charged twice."\n}'
    assert item["images"] == []
    assert item["question"].text == "Tone?"
    assert item["question"].options == ["calm", "frustrated", "angry"]
    assert item["question"].descriptions == [None, "annoyed", None]


def test_one_hot_targets() -> None:
    dataset = DecisionEngineDataset([RECORD], shuffle_options=False)
    assert by_type(dataset, "choice")["target"] == [0.0, 1.0, 0.0]
    assert by_type(dataset, "noul")["target"] == [0.0, 1.0]
    assert by_type(dataset, "score")["target"] == [0.0, 0.0, 1.0]


def test_noul_true_is_first_option() -> None:
    record = {"state": "s", "questions": {"q": {**NOUL, "label": True}}}
    assert DecisionEngineDataset([record])[0]["target"] == [1.0, 0.0]


def test_score_target_counts_are_normalized() -> None:
    score = {**SCORE, "target": {"0": 2, "1": 3, "2": 5}}
    record = {"state": "s", "questions": {"q": score}}
    assert DecisionEngineDataset([record])[0]["target"] == pytest.approx([0.2, 0.3, 0.5])


def test_choice_target_is_keyed_by_option() -> None:
    choice = {**CHOICE, "target": {"angry": 1.0, "calm": 1.0, "frustrated": 2.0}}
    record = {"state": "s", "questions": {"q": choice}}
    item = DecisionEngineDataset([record], shuffle_options=False)[0]
    assert item["target"] == pytest.approx([0.25, 0.5, 0.25])


def test_missing_target_keys_get_zero() -> None:
    score = {**SCORE, "target": {"2": 4}}
    record = {"state": "s", "questions": {"q": score}}
    assert DecisionEngineDataset([record])[0]["target"] == [0.0, 0.0, 1.0]


def test_too_many_options_are_skipped() -> None:
    criteria = {f"o{i}": None for i in range(30)}
    big = {"type": "choice", "instructions": "i", "criteria": criteria, "label": "o1"}
    record = {"state": "s", "questions": {"big": big, "ok": NOUL}}
    dataset = DecisionEngineDataset([record])
    assert len(dataset) == 1
    assert dataset.skipped == 1
    assert dataset[0]["type"] == "noul"


def test_files_are_loaded() -> None:
    image = Image.new("RGB", (4, 4))
    record = {**RECORD, "files": [image]}
    assert DecisionEngineDataset([record])[0]["images"] == [image]


def test_choice_shuffle_keeps_target_on_label() -> None:
    torch.manual_seed(0)
    dataset = DecisionEngineDataset([RECORD])
    orders = set()
    for _ in range(30):
        item = dataset[0]
        options, target = item["question"].options, item["target"]
        assert options[target.index(1.0)] == "frustrated"
        descriptions = dict(zip(options, item["question"].descriptions))
        assert descriptions["frustrated"] == "annoyed"
        orders.add(tuple(options))
    assert len(orders) > 1


def test_noul_and_score_are_not_shuffled() -> None:
    torch.manual_seed(0)
    dataset = DecisionEngineDataset([RECORD])
    for _ in range(10):
        assert by_type(dataset, "noul")["question"].options == ["Yes", "No"]
        assert by_type(dataset, "score")["question"].options == [
            "low", "mid", "high"
        ]


def test_unknown_choice_label_raises() -> None:
    record = {"state": "s", "questions": {"q": {**CHOICE, "label": "happy"}}}
    with pytest.raises(ValueError):
        DecisionEngineDataset([record])[0]


@pytest.mark.parametrize(
    "spec",
    [
        {**SCORE, "target": {"5": 1}},
        {**CHOICE, "target": {"happy": 1}},
        {**SCORE, "target": {"0": 0}},
        {**SCORE, "label": 3},
    ],
)
def test_bad_target_raises(spec: dict) -> None:
    record = {"state": "s", "questions": {"q": spec}}
    with pytest.raises(ValueError):
        DecisionEngineDataset([record])[0]
