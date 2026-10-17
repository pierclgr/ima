import pytest

from src.common.types import Question
from src.model.config import DecisionEngineConfig


def test_config_defaults() -> None:
    config = DecisionEngineConfig()
    assert config.temperature == 1.0
    assert config.max_options == 26
    assert config.option_token_ids is None


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_config_rejects_bad_temperature(temperature: float) -> None:
    with pytest.raises(ValueError):
        DecisionEngineConfig(temperature=temperature)


def test_config_rejects_too_many_max_options() -> None:
    with pytest.raises(ValueError):
        DecisionEngineConfig(max_options=27)


def test_config_keeps_backbone_config(backbone_config) -> None:
    config = DecisionEngineConfig(backbone_config=backbone_config)
    assert config.backbone_config.model_type == "llava"


def test_question_needs_two_options() -> None:
    with pytest.raises(ValueError):
        Question(text="q", options=["only"])


def test_question_descriptions_must_match_options() -> None:
    with pytest.raises(ValueError):
        Question(text="q", options=["a", "b"], descriptions=["only one"])
