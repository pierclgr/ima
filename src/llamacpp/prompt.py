import json
import string
from typing import Any

from src.constants import SYSTEM_ONE_INSTRUCTION


def build_template(processor: Any, max_options: int) -> str:
    """Builds the native decision template with the source chat boundaries.

    llama.cpp supplies state, instructions, type, options, and image markers.
    Its multimodal processor adds the image boundary tokens around markers.
    Text formatting follows `PromptBuilder`, with thinking disabled.

    Args:
        processor: Processor saved with the trained model.
        max_options: The source model's option limit.

    Returns:
        A Jinja template for the GGUF's named `systemone` chat template.

    Raises:
        ValueError: If the chat template does not preserve the user text.
    """
    marker: str = "__IMA_DECISION_TEXT__"
    chat: str = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": marker}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if not isinstance(chat, str) or chat.count(marker) != 1:
        raise ValueError("source chat template must preserve the user text")
    prefix, suffix = chat.split(marker)
    # the named decision template runs without HF's template variables
    return (
        "{% if options | length < 2 or options | length > " + str(max_options)
        + " %}{{ raise_exception('unsupported option count') }}{% endif %}"
        "{% set letters = "
        + json.dumps(string.ascii_uppercase[:max_options]) + " %}"
        "{{ " + json.dumps(prefix) + " }}"
        "{% for image in images %}{{ image }}{% endfor %}"
        "{{ state if state is string else state | tojson(indent=2) }}"
        "{{ '\\n\\nQuestion: ' }}"
        "{{ instructions if instructions is string "
        "else instructions | tojson(indent=2) }}"
        "{{ '\\n\\nOptions:\\n' }}"
        "{% for o in options %}"
        "{% if not loop.first %}{{ '\\n' }}{% endif %}"
        "{{ letters[loop.index0] }}. "
        "{% if type == 'noul' %}{{ 'Yes' if o.key == 'true' else 'No' }}"
        "{% elif type == 'score' %}"
        "{{ o.description if o.description is string "
        "else o.description | tojson(indent=2) }}"
        "{% else %}{{ o.key }}{% endif %}"
        "{% if type != 'score' and o.description %}: "
        "{{ o.description if o.description is string "
        "else o.description | tojson(indent=2) }}"
        "{% endif %}{% endfor %}"
        "{{ " + json.dumps("\n\n" + SYSTEM_ONE_INSTRUCTION + suffix) + " }}"
    )
