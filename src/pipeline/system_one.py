from collections.abc import Iterator
from os import PathLike
from typing import Any

from transformers.pipelines.base import ChunkPipeline

from src.common.prompt import PromptBuilder
from src.common.question_types import question_type
from src.common.request import RequestParser
from src.common.types import Question
from src.model.loader import ModelLoader
from src.model.system_one import PreTrainedSystemOneModel


class SystemOnePipeline(ChunkPipeline):
    """System One pipeline with the Jev `/v1/systemone` request and response.

    Each question of a request is one chunk: one forward pass, no generation.
    See `docs/MODEL_SPEC.md` for the request and response format.

    Args:
        model: The System One model.
        processor: Processor matching the backbone.
        **kwargs: Passed to `Pipeline` (e.g. `device`).

    Raises:
        ValueError: If the processor's letter token ids differ from the ones
            saved in the model config.
    """

    def __init__(self, model: PreTrainedSystemOneModel, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        n: int = model.config.max_options
        self.prompt: PromptBuilder = PromptBuilder(self.processor, n)
        if self.prompt.letter_ids != (model.config.option_token_ids or [])[:n]:
            raise ValueError("processor and model use different letter token ids")

    @classmethod
    def from_pretrained(cls, name: str, **kwargs: Any) -> "SystemOnePipeline":
        """Loads a pipeline saved with `save_pretrained`, or a raw backbone.

        Args:
            name: HF id or local path of a trained model or of a multimodal
                backbone (see `ModelLoader.load`).
            **kwargs: Passed to `ModelLoader.load` (e.g. `dtype`,
                `device_map`).

        Returns:
            The pipeline.
        """
        model, processor = ModelLoader.load(name, **kwargs)
        return cls(model=model, processor=processor)

    def save_pretrained(self, save_directory: str | PathLike, **kwargs: Any) -> None:
        """Saves the model and the processor to `save_directory`."""
        super().save_pretrained(save_directory, **kwargs)
        self.processor.save_pretrained(save_directory)

    def _sanitize_parameters(self, **kwargs: Any) -> tuple[dict, dict, dict]:
        return {}, {}, {}

    def preprocess(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Yields the model inputs of each question of the request."""
        state_text, images = RequestParser.parse_inputs(request)
        questions: dict[str, dict[str, Any]] = request["questions"]
        for i, (question_id, spec) in enumerate(questions.items()):
            question = RequestParser.build_question(spec)
            if spec["type"] == "score" and len(question.options) > 10:
                raise ValueError("API score criteria allow at most 10 levels")
            inputs = self.prompt.encode(
                [self.prompt.build_messages(state_text, question, images)]
            )
            yield {
                "is_last": i == len(questions) - 1,
                "id": question_id,
                "type": spec["type"],
                "question": question,
                "model_inputs": inputs,
            }

    def _forward(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        inputs = model_inputs["model_inputs"]
        question: Question = model_inputs["question"]
        out = self.model(**inputs, num_options=len(question.options))
        meta = {k: v for k, v in model_inputs.items() if k != "model_inputs"}
        return {
            **meta,
            "probabilities": out.probabilities[0],
            "input_tokens": inputs["input_ids"].shape[-1],
        }

    def postprocess(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Builds the `/v1/systemone` response from the chunks."""
        answers = {
            chunk["id"]: question_type(chunk["type"]).format_answer(
                chunk["question"], chunk["probabilities"].tolist()
            )
            for chunk in chunks
        }
        return {
            "model": self.model.config.backbone_config.name_or_path,
            "answers": answers,
            "usage": {
                "input_tokens": sum(chunk["input_tokens"] for chunk in chunks),
                "output_tokens": 0,
            },
        }
