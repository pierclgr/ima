import torch
from PIL import Image

from src.common.prompt import PromptBuilder
from src.common.types import Question
from src.data.collator import DecisionEngineCollator


def items() -> list[dict]:
    image = Image.new("RGB", (4, 4))
    return [
        {
            "type": "choice",
            "state": "s1",
            "images": [image],
            "question": Question("q1", ["a", "b", "c"]),
            "target": [0.0, 1.0, 0.0],
        },
        {
            "type": "noul",
            "state": "s2",
            "images": [],
            "question": Question("q2", ["Yes", "No"]),
            "target": [1.0, 0.0],
        },
    ]


def collator(processor) -> DecisionEngineCollator:
    return DecisionEngineCollator(PromptBuilder(processor, 26))


def test_sets_left_padding(processor) -> None:
    collator(processor)
    assert processor.tokenizer.padding_side == "left"


def test_batch_keys_and_shapes(processor) -> None:
    batch = collator(processor)(items())
    assert {"input_ids", "attention_mask", "labels", "option_mask",
            "num_options"} <= set(batch)
    assert batch["input_ids"].shape[0] == 2
    assert batch["num_options"] == 3
    assert batch["labels"].shape == (2, 3)
    assert batch["option_mask"].dtype == torch.bool


def test_labels_and_mask_are_padded(processor) -> None:
    batch = collator(processor)(items())
    assert batch["labels"].tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    assert batch["option_mask"].tolist() == [[True, True, True],
                                             [True, True, False]]


def test_messages_built_with_prompt(processor) -> None:
    collator(processor)(items())
    first, second = processor.messages
    assert [p["type"] for p in first[0]["content"]] == ["image", "text"]
    assert "A. Yes" in second[0]["content"][-1]["text"]


def test_batch_feeds_model(processor, model) -> None:
    model.train()
    batch = collator(processor)(items())
    out = model(**batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert out.probabilities[1, 2] == 0.0


def test_thinking_is_off_by_default(processor) -> None:
    collator(processor)(items())
    assert processor.kwargs["enable_thinking"] is False


def test_template_kwargs(processor) -> None:
    kwargs = {"enable_thinking": True, "reasoning_effort": "low"}
    DecisionEngineCollator(
        PromptBuilder(processor, 26, template_kwargs=kwargs)
    )(items())
    assert processor.kwargs["enable_thinking"] is True
    assert processor.kwargs["reasoning_effort"] == "low"
