import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import datasets
import pytest
import torch
from torch import nn
from transformers import GPT2Config

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
