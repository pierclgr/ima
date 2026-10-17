import json

import pytest
import torch
from PIL import Image

from src.model.output import DecisionEngineOutput
from src.pipeline.system_one import SystemOnePipeline

INPUT_IDS: torch.Tensor = torch.tensor([[1, 2, 3]])

REQUEST: dict = {
    "state": "I was charged twice.",
    "questions": {
        "billing": {"type": "noul", "instructions": "About billing?"},
        "tone": {
            "type": "choice",
            "instructions": "Customer tone?",
            "criteria": {"calm": None, "frustrated": "annoyed", "angry": None},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["can wait", "this week", "today"],
        },
    },
}


@pytest.fixture
def pipe(model, processor) -> SystemOnePipeline:
    return SystemOnePipeline(model=model, processor=processor, device="cpu")


def fix_probabilities(model, monkeypatch, probabilities: list[float]) -> None:
    """Makes the model return fixed probabilities."""
    fixed = torch.tensor([probabilities])
    monkeypatch.setattr(
        model, "forward", lambda **kwargs: DecisionEngineOutput(probabilities=fixed)
    )


def test_response_format(pipe) -> None:
    out = pipe(REQUEST)
    assert set(out) == {"model", "answers", "usage"}
    assert set(out["answers"]) == {"billing", "tone", "urgency"}
    assert out["usage"] == {"input_tokens": 9, "output_tokens": 0}
    assert out["answers"]["billing"]["type"] == "noul"
    assert set(out["answers"]["tone"]) == {
        "type", "choice", "confidence", "probabilities"
    }
    assert set(out["answers"]["urgency"]) == {
        "type", "score", "confidence", "legend", "probabilities"
    }


def test_noul_is_probability_of_first_option(pipe, model) -> None:
    expected = model(input_ids=INPUT_IDS, num_options=2).probabilities[0, 0]
    out = pipe(REQUEST)["answers"]["billing"]["noul"]
    assert out == pytest.approx(expected.item(), abs=1e-5)


def test_choice_answer(pipe, model, monkeypatch) -> None:
    fix_probabilities(model, monkeypatch, [0.47, 0.28, 0.25])
    request = {"state": "s", "questions": {"tone": REQUEST["questions"]["tone"]}}
    answer = pipe(request)["answers"]["tone"]
    assert answer["choice"] == "calm"
    assert answer["probabilities"] == pytest.approx(
        {"calm": 0.47, "frustrated": 0.28, "angry": 0.25}
    )
    assert answer["confidence"] == pytest.approx(0.205, abs=1e-3)


def test_score_answer(pipe, model, monkeypatch) -> None:
    fix_probabilities(model, monkeypatch, [0.0, 0.56, 0.44])
    request = {"state": "s", "questions": {"u": REQUEST["questions"]["urgency"]}}
    answer = pipe(request)["answers"]["u"]
    assert answer["score"] == pytest.approx(1.44)
    assert answer["confidence"] == pytest.approx(0.34, abs=1e-2)
    assert answer["legend"] == {"0": "can wait", "1": "this week", "2": "today"}
    assert answer["probabilities"] == pytest.approx({"0": 0.0, "1": 0.56, "2": 0.44})


def test_score_uniform_has_zero_confidence(pipe, model, monkeypatch) -> None:
    fix_probabilities(model, monkeypatch, [1 / 3, 1 / 3, 1 / 3])
    request = {"state": "s", "questions": {"u": REQUEST["questions"]["urgency"]}}
    assert pipe(request)["answers"]["u"]["confidence"] == pytest.approx(0.0)


def test_list_of_requests(pipe) -> None:
    outs = pipe([REQUEST, REQUEST])
    assert len(outs) == 2
    assert set(outs[0]["answers"]) == {"billing", "tone", "urgency"}


def test_dict_state_is_rendered_as_json(pipe, processor) -> None:
    state = {"task": "pay", "screen": "<image:1>"}
    pipe({"state": state, "questions": {"q": REQUEST["questions"]["billing"]}})
    text = processor.messages[0][0]["content"][-1]["text"]
    assert json.dumps(state, indent=2) in text


def test_files_images_come_first(pipe, processor) -> None:
    files = [Image.new("RGB", (4, 4))]
    pipe({"state": "s", "files": files,
          "questions": {"q": REQUEST["questions"]["billing"]}})
    parts = processor.messages[0][0]["content"]
    assert [p["type"] for p in parts] == ["image", "text"]


def test_non_image_media_raises(pipe) -> None:
    media = [{"type": "video", "data": "x"}]
    with pytest.raises(ValueError):
        pipe({"state": "s", "media": media,
              "questions": {"q": REQUEST["questions"]["billing"]}})


def test_unknown_question_type_raises(pipe) -> None:
    request = {"state": "s", "questions": {"q": {"type": "x", "instructions": "i"}}}
    with pytest.raises(ValueError):
        pipe(request)


def test_too_many_options_raises(pipe) -> None:
    criteria = {str(i): None for i in range(27)}
    request = {"state": "s", "questions": {
        "q": {"type": "choice", "instructions": "i", "criteria": criteria}}}
    with pytest.raises(ValueError):
        pipe(request)


def test_mismatched_letter_ids_raise(model, processor) -> None:
    model.config.option_token_ids = list(range(26))
    with pytest.raises(ValueError):
        SystemOnePipeline(model=model, processor=processor, device="cpu")


def test_thinking_is_off(pipe, processor) -> None:
    pipe(REQUEST)
    assert processor.kwargs["enable_thinking"] is False
