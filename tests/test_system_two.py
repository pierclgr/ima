import pytest
import torch

from src.common.prompt import PromptBuilder
from src.model.system_two import AnswerStop, LetterLogitsRecorder, SystemTwoModel

from conftest import LETTER_IDS, FakeProcessor, FakeTokenizer

MIN: float = torch.finfo(torch.float32).min


def find(texts: list[str], num_options: list[int], thinking: bool = False):
    """Runs `find_answers` on outputs of one token per character. The letter
    logits of step s are s * 10 + (letter index) / 100, so the step a value
    came from can be told apart."""
    length: int = max(len(text) for text in texts)
    # 0 pads the shorter outputs (generation pads finished rows)
    tokens = torch.tensor([[ord(c) for c in text] + [0] * (length - len(text))
                           for text in texts])
    steps = torch.arange(length, dtype=torch.float32)[:, None] * 10
    letters = steps + torch.arange(26, dtype=torch.float32) / 100
    step_logits = letters.expand(len(texts), -1, -1).clone()
    mask = torch.zeros(len(texts), max(num_options), dtype=torch.bool)
    for row, n in enumerate(num_options):
        mask[row, :n] = True
    return SystemTwoModel.find_answers(
        tokens, step_logits, FakeTokenizer(), mask, thinking
    )


def test_answer_block() -> None:
    logits, errors = find(["<answer>\nB\n</answer>"], [3])
    assert errors.tolist() == [False]
    # "B" is the 10th character: step 9
    assert logits[0].tolist() == pytest.approx([90.0, 90.01, 90.02])


def test_text_around_the_block() -> None:
    text = "ok <answer>\nC\n</answer> done"
    logits, errors = find([text], [3])
    assert errors.tolist() == [False]
    step = text.index("C")
    assert logits[0].tolist() == pytest.approx([step * 10 + i / 100 for i in range(3)])


@pytest.mark.parametrize(
    "text", ["B", "The answer is B", "<answer>\nB", "<answer>\nb\n</answer>",
             "<answer>\nBC\n</answer>"],
)
def test_no_valid_block_is_an_error(text: str) -> None:
    logits, errors = find([text], [3])
    assert errors.tolist() == [True]
    # uniform: zero logits on the options
    assert logits[0].tolist() == [0.0, 0.0, 0.0]


def test_letter_beyond_options_is_an_error() -> None:
    _, errors = find(["<answer>\nE\n</answer>"], [3])
    assert errors.tolist() == [True]


def test_masked_options_get_float_min() -> None:
    logits, errors = find(["<answer>\nA\n</answer>"] * 2, [3, 2])
    assert errors.tolist() == [False, False]
    assert logits[1, 2].item() == MIN
    assert logits[0, 2].item() != MIN


def test_thinking_uses_the_block_after_thinking() -> None:
    text = "<answer>\nA\n</answer> no</think><answer>\nC\n</answer>"
    logits, errors = find([text], [3], thinking=True)
    assert errors.tolist() == [False]
    step = text.rindex("C")
    assert logits[0, 0].item() == pytest.approx(step * 10)


def test_thinking_without_end_is_an_error() -> None:
    _, errors = find(["<answer>\nA\n</answer>"], [3], thinking=True)
    assert errors.tolist() == [True]


def test_forward_on_tiny_backbone(model) -> None:
    system_two = SystemTwoModel(model, PromptBuilder(FakeProcessor(), 26), max_new_tokens=3)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    option_mask = torch.tensor([[True, True, True], [True, True, False]])
    labels = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    output = system_two(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        option_mask=option_mask,
        num_options=3,
    )
    assert output.logits.shape == output.probabilities.shape == (2, 3)
    # 3 random tokens are never a full answer block: errors, uniform
    assert output.errors.tolist() == [True, True]
    assert torch.isfinite(output.loss)
    assert output.probabilities[0].tolist() == pytest.approx([1 / 3] * 3)
    assert output.probabilities[1].tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_exposes_the_model_config(model) -> None:
    assert SystemTwoModel(model, PromptBuilder(FakeProcessor(), 26), 3).config is model.config


def test_keeps_only_letter_logits(model, monkeypatch: pytest.MonkeyPatch) -> None:
    # full-vocabulary logits of every step would fill the GPU with thinking
    calls: list[dict] = []
    generate = model.backbone.generate

    def spy(**kwargs):
        calls.append(kwargs)
        return generate(**kwargs)

    monkeypatch.setattr(model.backbone, "generate", spy)
    input_ids = torch.tensor([[1, 2, 3]])
    SystemTwoModel(model, PromptBuilder(FakeProcessor(), 26), 4)(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids)
    )
    assert not calls[0].get("output_logits") and not calls[0].get("output_scores")


def test_letter_recorder_records_each_step() -> None:
    recorder = LetterLogitsRecorder(LETTER_IDS)
    scores = torch.arange(300, dtype=torch.float32)[None].repeat(2, 1)
    for _ in range(3):
        assert recorder(torch.zeros(2, 1, dtype=torch.long), scores) is scores
    steps = recorder.stacked()
    assert steps.shape == (2, 3, 26)
    assert steps[0, 0].tolist() == [float(i) for i in LETTER_IDS]


def ids(prompt: list[int], text: str) -> torch.Tensor:
    return torch.tensor([prompt + [ord(c) for c in text]])


@pytest.mark.parametrize(
    ("text", "thinking", "stop"),
    [
        ("<answer>\nB\n</answer>", False, True),
        ("<answer>\nB\n</ans", False, False),
        ("chatter", False, False),
        # thinking: a block written while thinking does not stop generation
        ("<answer>\nB\n</answer>", True, False),
        ("hmm</think><answer>\nB\n</answer>", True, True),
    ],
)
def test_answer_stop(text: str, thinking: bool, stop: bool) -> None:
    criterion = AnswerStop(FakeTokenizer(), prompt_length=3, thinking=thinking)
    done = criterion(ids([1, 2, 3], text), torch.zeros(1, 300))
    assert done.tolist() == [stop]


def test_answer_stop_ignores_the_prompt() -> None:
    # the prompt's own example block must not stop generation
    prompt = [ord(c) for c in "<answer>\nA\n</answer>"]
    criterion = AnswerStop(FakeTokenizer(), prompt_length=len(prompt), thinking=False)
    assert criterion(ids(prompt, "x"), torch.zeros(1, 300)).tolist() == [False]


def test_generation_stops_at_the_answer(model, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    generate = model.backbone.generate

    def spy(**kwargs):
        calls.append(kwargs)
        return generate(**kwargs)

    monkeypatch.setattr(model.backbone, "generate", spy)
    input_ids = torch.tensor([[1, 2, 3]])
    SystemTwoModel(model, PromptBuilder(FakeProcessor(), 26), 4, thinking=True)(
        input_ids=input_ids, attention_mask=torch.ones_like(input_ids)
    )
    (criterion,) = calls[0]["stopping_criteria"]
    assert isinstance(criterion, AnswerStop)
    assert (criterion.prompt_length, criterion.thinking) == (3, True)
