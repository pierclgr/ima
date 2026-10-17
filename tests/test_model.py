import pytest
import torch
from transformers import AutoModelForMultimodalLM

from src.model.config import DecisionEngineConfig
from src.model.system_one import PreTrainedSystemOneModel

INPUT_IDS: torch.Tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])


def test_forward_shapes_and_softmax(model: PreTrainedSystemOneModel) -> None:
    out = model(input_ids=INPUT_IDS, num_options=3)
    assert out.logits.shape == (2, 3)
    assert out.probabilities.shape == (2, 3)
    assert torch.allclose(out.probabilities.sum(-1), torch.ones(2))


def test_forward_reads_only_letter_logits(model: PreTrainedSystemOneModel) -> None:
    full = model.backbone(input_ids=INPUT_IDS).logits[:, -1]
    out = model(input_ids=INPUT_IDS, num_options=4)
    letter_ids = model.config.option_token_ids[:4]
    assert torch.allclose(out.logits, full[:, letter_ids], atol=1e-5)


def test_temperature_flattens_distribution(config: DecisionEngineConfig) -> None:
    torch.manual_seed(0)
    sharp = PreTrainedSystemOneModel(config).eval()
    config_hot = DecisionEngineConfig(
        backbone_config=config.backbone_config,
        option_token_ids=config.option_token_ids,
        temperature=100.0,
    )
    hot = PreTrainedSystemOneModel(config_hot, backbone=sharp.backbone).eval()
    p_sharp = sharp(input_ids=INPUT_IDS, num_options=3).probabilities
    p_hot = hot(input_ids=INPUT_IDS, num_options=3).probabilities
    assert p_hot.max() < p_sharp.max()


def test_no_labels_no_loss(model: PreTrainedSystemOneModel) -> None:
    assert model(input_ids=INPUT_IDS, num_options=3).loss is None


def test_loss_with_index_labels(model: PreTrainedSystemOneModel) -> None:
    labels = torch.tensor([0, 2])
    out = model(input_ids=INPUT_IDS, num_options=3, labels=labels)
    expected = torch.nn.functional.cross_entropy(out.logits.float(), labels)
    assert torch.allclose(out.loss, expected)


def test_loss_with_soft_labels(model: PreTrainedSystemOneModel) -> None:
    labels = torch.tensor([[0.7, 0.2, 0.1], [0.0, 0.5, 0.5]])
    out = model(input_ids=INPUT_IDS, num_options=3, labels=labels)
    expected = -(labels * torch.log_softmax(out.logits.float(), -1)).sum(-1).mean()
    assert torch.allclose(out.loss, expected)


def test_loss_ignores_temperature(model: PreTrainedSystemOneModel) -> None:
    labels = torch.tensor([0, 2])
    before = model(input_ids=INPUT_IDS, num_options=3, labels=labels).loss
    model.config.temperature = 5.0
    after = model(input_ids=INPUT_IDS, num_options=3, labels=labels).loss
    assert torch.allclose(before, after)


def test_loss_backpropagates(model: PreTrainedSystemOneModel) -> None:
    model.train()
    out = model(input_ids=INPUT_IDS, num_options=3, labels=torch.tensor([1, 0]))
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


MASK: torch.Tensor = torch.tensor([[True, True, True], [True, True, False]])


def test_masked_options_get_zero_probability(
    model: PreTrainedSystemOneModel,
) -> None:
    out = model(input_ids=INPUT_IDS, num_options=3, option_mask=MASK)
    assert out.probabilities[1, 2] == 0.0
    assert torch.allclose(out.probabilities.sum(-1), torch.ones(2))


def test_masked_loss_matches_unmasked_subset(
    model: PreTrainedSystemOneModel,
) -> None:
    labels = torch.tensor([[0.0, 1.0, 0.0], [0.3, 0.7, 0.0]])
    out = model(input_ids=INPUT_IDS, num_options=3, labels=labels, option_mask=MASK)
    raw = model(input_ids=INPUT_IDS, num_options=3).logits.float()
    first = torch.nn.functional.cross_entropy(raw[:1], labels[:1])
    second = torch.nn.functional.cross_entropy(raw[1:, :2], labels[1:, :2])
    assert torch.isfinite(out.loss)
    assert torch.allclose(out.loss, (first + second) / 2, atol=1e-5)


def test_too_many_options_raises(model: PreTrainedSystemOneModel) -> None:
    model.config.max_options = 3
    with pytest.raises(ValueError):
        model(input_ids=INPUT_IDS, num_options=4)


def test_missing_option_token_ids_raises(backbone_config) -> None:
    config = DecisionEngineConfig(backbone_config=backbone_config)
    model = PreTrainedSystemOneModel(config).eval()
    with pytest.raises(ValueError):
        model(input_ids=INPUT_IDS, num_options=2)


def test_from_backbone_matches_plain_backbone(
    backbone_config, config: DecisionEngineConfig, tmp_path
) -> None:
    AutoModelForMultimodalLM.from_config(backbone_config).save_pretrained(tmp_path)
    plain = AutoModelForMultimodalLM.from_pretrained(tmp_path).eval()
    wrapped = PreTrainedSystemOneModel.from_backbone(
        str(tmp_path), config.option_token_ids
    ).eval()
    assert (
        wrapped.backbone.config._attn_implementation
        == plain.config._attn_implementation
    )
    a = plain(input_ids=INPUT_IDS).logits
    b = wrapped.backbone(input_ids=INPUT_IDS).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_save_load_round_trip(model: PreTrainedSystemOneModel, tmp_path) -> None:
    model.config.temperature = 2.5
    model.save_pretrained(tmp_path)
    loaded = PreTrainedSystemOneModel.from_pretrained(tmp_path).eval()
    assert loaded.config.temperature == 2.5
    assert loaded.config.option_token_ids == model.config.option_token_ids
    a = model(input_ids=INPUT_IDS, num_options=5).logits
    b = loaded(input_ids=INPUT_IDS, num_options=5).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_default_dtype() -> None:
    expected = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    assert PreTrainedSystemOneModel.default_dtype() == expected
