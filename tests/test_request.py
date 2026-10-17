from src.common.request import RequestParser


def test_render_text() -> None:
    assert RequestParser.render_text("plain") == "plain"
    assert RequestParser.render_text({"a": 1}) == '{\n  "a": 1\n}'


def test_dict_instructions_are_rendered() -> None:
    spec = {
        "type": "noul",
        "instructions": {"question": "Is it red?", "focus": "Color only."},
    }
    question = RequestParser.build_question(spec)
    assert question.text == RequestParser.render_text(spec["instructions"])
