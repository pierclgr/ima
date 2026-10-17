import json
import string
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import datasets
import pytest
import torch
from torch import nn
from transformers import (
    BatchFeature, CLIPVisionConfig, GPT2Config, LlamaConfig, LlavaConfig,
)

from src.data.hub import HubRecordLoader
from src.model.config import DecisionEngineConfig
from src.model.system_one import PreTrainedSystemOneModel


class TinyBackbone(nn.Module):
    """Small backbone exercising the real option-logit and LoRA paths."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.proj = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 26, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        hidden = torch.nn.functional.one_hot(input_ids[:, -1] % 8, 8).float()
        hidden = self.model.language_model.proj(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden).unsqueeze(1))


class TinyProcessor:
    """Processor that needs no tokenizer download."""

    def __init__(self) -> None:
        self.tokenizer = self
        self.padding_side = "right"

    def encode(self, letter: str, **kwargs: Any) -> list[int]:
        return [ord(letter) - ord("A")]

    def apply_chat_template(
        self, conversations: list[list[dict[str, Any]]], **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.arange(len(conversations)).unsqueeze(1)}

    def save_pretrained(self, path: Path) -> None:
        (path / "processor.json").write_text("{}")


@pytest.fixture
def records() -> datasets.Dataset:
    """Hub-style records with multiple questions and human soft targets."""
    questions = {
        "choice": {
            "type": "choice", "instructions": "Which?",
            "criteria": {"red": None, "blue": None, "green": None},
            "label": "red", "target": {"red": 3, "blue": 1},
        },
        "binary": {"type": "noul", "instructions": "Yes?", "label": True},
        "score": {
            "type": "score", "instructions": "Level?",
            "criteria": ["low", "high"], "label": 1,
        },
    }
    raw = datasets.Dataset.from_dict({
        "state": [json.dumps({"id": i}) for i in range(4)],
        "questions": [json.dumps(questions)] * 4,
        "image": [None] * 4,
    })
    return HubRecordLoader.convert(raw)


@pytest.fixture
def tiny_model() -> PreTrainedSystemOneModel:
    config = DecisionEngineConfig(
        backbone_config=GPT2Config(n_embd=8, n_layer=1, n_head=1),
        max_options=26, option_token_ids=list(range(26)), temperature=1.7,
    )
    return PreTrainedSystemOneModel(config, backbone=TinyBackbone())


LETTER_IDS: list[int] = [ord(c) for c in string.ascii_uppercase]


class FakeTokenizer:
    """Maps a single character to its code point."""

    padding_side: str = "right"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int], **kwargs) -> str:
        return "".join(chr(i) for i in ids)


class FakeProcessor:
    """Records chat messages and template kwargs, returns fixed input ids."""

    def __init__(self) -> None:
        self.tokenizer: FakeTokenizer = FakeTokenizer()
        self.messages: list[dict] | None = None
        self.kwargs: dict = {}
        self.chat_template: str = ""

    def apply_chat_template(self, messages: list, **kwargs) -> BatchFeature:
        self.messages = messages
        self.kwargs = kwargs
        # a list of conversations is a batch
        batch_size: int = len(messages) if isinstance(messages[0], list) else 1
        input_ids = torch.tensor([[1, 2, 3]] * batch_size)
        return BatchFeature(
            {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        )


@pytest.fixture
def backbone_config() -> LlavaConfig:
    text = LlamaConfig(
        vocab_size=300,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    vision = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
        projection_dim=16,
    )
    return LlavaConfig(
        text_config=text, vision_config=vision, image_token_index=299,
        image_seq_length=4,
    )


@pytest.fixture
def config(backbone_config: LlavaConfig) -> DecisionEngineConfig:
    return DecisionEngineConfig(
        backbone_config=backbone_config, option_token_ids=LETTER_IDS
    )


@pytest.fixture
def model(config: DecisionEngineConfig) -> PreTrainedSystemOneModel:
    torch.manual_seed(0)
    return PreTrainedSystemOneModel(config).eval()


@pytest.fixture
def processor() -> FakeProcessor:
    return FakeProcessor()
