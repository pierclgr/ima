import string
from typing import Any

from PIL import Image
from transformers import BatchFeature

from src.constants import SYSTEM_ONE_INSTRUCTION
from src.common.request import RequestParser
from src.common.types import Question


class PromptBuilder:
    """Builds and encodes chat messages, maps option letters to token ids.

    The single place where prompts are turned into model inputs, shared by
    inference and training, so both get the same inputs.

    Args:
        processor: Processor matching the backbone. Its tokenizer is set to
            left padding, so the answer slot is the last position.
        max_options: Number of letters to resolve (A, B, C, ...).
        instruction: Final line of the prompt (`SYSTEM_ONE_INSTRUCTION` or
            `SYSTEM_TWO_INSTRUCTION`).
        template_kwargs: Chat template variables. Default: thinking off
            (System One reads the answer letter right after the prompt; some
            templates, e.g. Qwen3.8, think by default).

    Raises:
        ValueError: If a letter is not a single token.
    """

    def __init__(
        self,
        processor: Any,
        max_options: int,
        instruction: str = SYSTEM_ONE_INSTRUCTION,
        template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        processor.tokenizer.padding_side = "left"
        self.processor: Any = processor
        self.instruction: str = instruction
        self.template_kwargs: dict[str, Any] = (
            {"enable_thinking": False} if template_kwargs is None else template_kwargs
        )
        self.letters: list[str] = list(string.ascii_uppercase[:max_options])
        self.letter_ids: list[int] = []
        for letter in self.letters:
            ids: list[int] = processor.tokenizer.encode(
                letter, add_special_tokens=False
            )
            if len(ids) != 1:
                raise ValueError(f"letter {letter!r} is not a single token")
            self.letter_ids.append(ids[0])

    def build_messages(
        self, state: str, question: Question, images: list[Image.Image]
    ) -> list[dict[str, Any]]:
        """Builds the chat messages: images first, then state and question."""
        descriptions: list[str | None] = question.descriptions or [None] * len(
            question.options
        )
        options: str = "\n".join(
            f"{letter}. {RequestParser.render_text(option)}"
            + (f": {RequestParser.render_text(description)}" if description else "")
            for letter, option, description in zip(
                self.letters, question.options, descriptions
            )
        )
        text: str = (
            f"{state}\n\nQuestion: {question.text}\n\nOptions:\n{options}\n\n"
            f"{self.instruction}"
        )
        content: list[dict[str, Any]] = [
            {"type": "image", "image": image} for image in images
        ]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def encode(self, conversations: list[list[dict[str, Any]]]) -> BatchFeature:
        """Encodes a batch of conversations (from `build_messages`).

        Returns:
            The processor outputs (`input_ids`, `attention_mask`, image
            tensors), left padded.
        """
        return self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
            **self.template_kwargs,
        )
