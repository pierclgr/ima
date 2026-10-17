"""Native image inputs and the distinct API/offline score limits."""

import base64
import copy
import io

import pytest
from PIL import Image

from src.common.question_types import ScoreType
from src.common.request import RequestParser
from src.data.dataset import DecisionEngineDataset
from src.pipeline.system_one import SystemOnePipeline


@pytest.fixture
def image_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_files_alias_and_chat_images_preserve_request(image_url: str) -> None:
    request = {
        "state": {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Look"},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}]},
        "files": [image_url], "images": [image_url],
    }
    original = copy.deepcopy(request)
    text, images = RequestParser.parse_inputs(request)
    assert request == original
    assert len(images) == 3
    assert all(image.size == (4, 4) for image in images)
    assert "Look" in text and "image_url" not in text


@pytest.mark.parametrize("key", ["files", "images"])
def test_native_image_aliases(key: str, image_url: str) -> None:
    text, images = RequestParser.parse_inputs({"state": "s", key: [image_url]})
    assert text == "s" and len(images) == 1
    assert images[0].getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("extra", [
    {"media": []}, {"videos": ["video"]}, {"files": "not an array"},
    {"files": ["https://example.com/image.png"]}, {"images": [42]},
])
def test_unsupported_inputs_raise(extra: dict) -> None:
    with pytest.raises(ValueError):
        RequestParser.parse_inputs({"state": "s", **extra})


def test_eight_image_limit_and_null_state(image_url: str) -> None:
    assert len(RequestParser.parse_inputs({
        "state": "s", "files": [image_url] * 8,
    })[1]) == 8
    with pytest.raises(ValueError, match="at most 8"):
        RequestParser.parse_inputs({"state": "s", "files": [image_url] * 9})
    with pytest.raises(ValueError, match="null"):
        RequestParser.parse_inputs({"state": None})


@pytest.mark.parametrize("levels", [0, 1])
def test_score_requires_two_levels(levels: int) -> None:
    with pytest.raises(ValueError):
        ScoreType().build("q", [str(i) for i in range(levels)])


@pytest.mark.parametrize("levels", [11, 26])
def test_offline_score_levels_are_preserved(levels: int, model, processor) -> None:
    spec = {"type": "score", "instructions": "q",
            "criteria": [str(i) for i in range(levels)], "label": levels - 1}
    record = {"state": "s", "questions": {"q": spec}}
    dataset = DecisionEngineDataset([record])
    assert len(dataset) == 1 and dataset.skipped == 0
    assert len(dataset[0]["target"]) == levels
    assert dataset[0]["target"][-1] == 1
    pipeline = SystemOnePipeline(model=model, processor=processor, device="cpu")
    with pytest.raises(ValueError, match="at most 10"):
        next(pipeline.preprocess(record))


def test_score_confidence_matches_native_spread() -> None:
    kind = ScoreType()
    question = kind.build("q", ["low", "medium", "high"])
    confidence = kind.format_answer(question, [0.0, 0.56, 0.44])["confidence"]
    assert confidence == pytest.approx(0.34)
    assert kind.format_answer(question, [0.0, 1.0, 0.0])["confidence"] == 1
    assert kind.format_answer(question, [1 / 3] * 3)["confidence"] == pytest.approx(0)
