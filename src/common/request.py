import json
from typing import Any

from PIL import Image
from transformers.image_utils import load_image

from src.common.question_types import question_type
from src.common.types import Question


class RequestParser:
    """Parses `/v1/systemone` requests, shared by inference and training.

    The pipeline and the dataset both use it, so training and inference build
    the same prompt.
    """

    @staticmethod
    def render_text(value: Any) -> str:
        """Returns a state or instructions as text.

        A string is returned as is, anything else as indented JSON.
        """
        if isinstance(value, str):
            return value
        return json.dumps(value, indent=2, ensure_ascii=False)

    @staticmethod
    def parse_inputs(request: dict[str, Any]) -> tuple[str, list[Image.Image]]:
        """Reads native llama.cpp state and images without changing the request.

        Args:
            request: State, optional `files` and `images` arrays of image
                data URLs. Dataset records can hold PIL images directly.
                Image parts in a chat state are also accepted.

        Returns:
            State text and images, in files, images, then chat order.

        Raises:
            ValueError: If state is null, media use the old format, or an
                input is not an image data URL. At most eight images.
        """
        if "media" in request:
            raise ValueError("media is removed; use files or images")
        state: Any = request["state"]
        if state is None:
            raise ValueError("state must not be null")
        if request.get("videos"):
            raise ValueError("videos are not supported")
        files: list[Any] = []
        for key in ("files", "images"):
            values: Any = request.get(key)
            if values is not None:
                if not isinstance(values, list):
                    raise ValueError(f"{key} must be an array")
                files.extend(values)
        wrapped: bool = isinstance(state, dict) and "messages" in state
        messages: Any = state["messages"] if wrapped else state
        if isinstance(messages, list):
            cleaned: list[Any] = []
            for message in messages:
                if not isinstance(message, dict) or not isinstance(
                    message.get("content"), list
                ):
                    cleaned.append(message)
                    continue
                content: list[Any] = []
                for part in message["content"]:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "image_url"
                    ):
                        url: Any = part["image_url"]
                        files.append(
                            url["url"] if isinstance(url, dict) else url
                        )
                    else:
                        content.append(part)
                cleaned.append({**message, "content": content})
            state = {**state, "messages": cleaned} if wrapped else cleaned
        if len(files) > 8:
            raise ValueError("at most 8 images are supported")
        images: list[Image.Image] = []
        for value in files:
            if not isinstance(value, Image.Image) and not (
                isinstance(value, str) and value.startswith("data:image/")
            ):
                raise ValueError("files must be image data URLs")
            images.append(load_image(value))
        return RequestParser.render_text(state), images

    @staticmethod
    def build_question(spec: dict[str, Any]) -> Question:
        """Turns a typed question spec into a lettered `Question`.

        The options depend on the question type (see `question_types`).

        Raises:
            ValueError: If the question type is unknown.
        """
        return question_type(spec["type"]).build(
            RequestParser.render_text(spec["instructions"]), spec.get("criteria")
        )
