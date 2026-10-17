from types import SimpleNamespace

import pytest
from PIL import Image

from src.common.prompt import PromptBuilder
from src.constants import SYSTEM_ONE_INSTRUCTION, SYSTEM_TWO_INSTRUCTION
from src.common.types import Question
from src.model.system_two import ANSWER

from conftest import FakeProcessor


class MultiTokenTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1, 2]


def test_letter_ids() -> None:
    builder = PromptBuilder(FakeProcessor(), max_options=26)
    assert builder.letter_ids[:3] == [ord("A"), ord("B"), ord("C")]
    assert len(builder.letter_ids) == 26


def test_multi_token_letter_raises() -> None:
    with pytest.raises(ValueError):
        PromptBuilder(SimpleNamespace(tokenizer=MultiTokenTokenizer()), 26)


def test_build_messages_text_only() -> None:
    builder = PromptBuilder(FakeProcessor(), max_options=26)
    question = Question(text="Is it safe?", options=["yes", "no"])
    messages = builder.build_messages("some state", question, [])
    content = messages[0]["content"]
    assert messages[0]["role"] == "user"
    assert [part["type"] for part in content] == ["text"]
    text = content[0]["text"]
    assert "some state" in text
    assert "Is it safe?" in text
    assert "A. yes" in text and "B. no" in text


def test_build_messages_with_descriptions() -> None:
    builder = PromptBuilder(FakeProcessor(), max_options=26)
    question = Question(
        text="q", options=["x", "y"], descriptions=["first one", None]
    )
    text = builder.build_messages("s", question, [])[0]["content"][0]["text"]
    assert "A. x: first one" in text
    assert "B. y\n" in text


def test_build_messages_images_first() -> None:
    builder = PromptBuilder(FakeProcessor(), max_options=26)
    question = Question(text="q", options=["x", "y"])
    image = Image.new("RGB", (4, 4))
    content = builder.build_messages("s", question, [image, image])[0]["content"]
    assert [part["type"] for part in content] == ["image", "image", "text"]
    assert content[0]["image"] is image


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [(None, SYSTEM_ONE_INSTRUCTION), (SYSTEM_TWO_INSTRUCTION, SYSTEM_TWO_INSTRUCTION)],
)
def test_answer_instruction(instruction: str | None, expected: str) -> None:
    kwargs = {} if instruction is None else {"instruction": instruction}
    builder = PromptBuilder(FakeProcessor(), max_options=26, **kwargs)
    question = Question(text="q", options=["x", "y"])
    text = builder.build_messages("s", question, [])[0]["content"][0]["text"]
    assert text.endswith(expected)


def test_system_two_asks_for_the_answer_block() -> None:
    assert "<answer>\n[letter of the correct option]\n</answer>" in SYSTEM_TWO_INSTRUCTION
    # with an example that the answer regex parses
    assert ANSWER.search(SYSTEM_TWO_INSTRUCTION).group(1) == "A"
