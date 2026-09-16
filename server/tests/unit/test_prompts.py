"""Template rendering tests (no model needed)."""

import string

from system_one_lite.prompts import (
    JSON_ANSWER_INSTRUCTION,
    as_text,
    chat_filled,
    confidence,
    filled,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return f"<user>{messages[0]['content']}<assistant>"


def test_layout_and_mark():
    text, marks = filled(" state ", [("Which team?", ["a desc", "b desc"])])
    assert text == ("State:\nstate\n\nQuestion 1: Which team?\nA: a desc\nB: b desc\nAnswer: A")
    assert len(marks) == 1
    assert text[marks[0]] == "A"
    assert text[: marks[0]].endswith("Answer: ")


def test_codes_past_z_continue_with_two_letters():
    codes = list(string.ascii_uppercase) + ["AA", "AB", "AC"]
    text, marks = filled("s", [("pick one", [f"opt {i}" for i in range(29)])], codes=codes)
    assert "\nZ: opt 25\nAA: opt 26\nAB: opt 27\nAC: opt 28\n" in text
    assert text[marks[0]] == "A"  # the filler stays the first code


def test_multi_question_numbering_and_marks():
    text, marks = filled("s", [("q1", ["x", "y"]), ("q2", ["p", "q", "r"])])
    assert "Question 1: q1" in text
    assert "Question 2: q2" in text
    assert marks[0] < marks[1]
    for mark in marks:
        assert text[mark] == "A"
        assert text[:mark].endswith("Answer: ")


def test_structured_instructions_match_as_text():
    instr = {"field_spec": {"path": "price"}, "main_question": "supported?"}
    text, _ = filled("s", [(instr, ["yes", "no"])])
    assert as_text(instr) in text


def test_chat_layout_uses_model_card_json_answer():
    text, marks = chat_filled(FakeTokenizer(), "s", [("pick one", ["X", "Y"])])
    assert JSON_ANSWER_INSTRUCTION in text
    assert "A: X\nB: Y" in text
    assert text[marks[0]] == "A"
    assert text[: marks[0]].endswith('<assistant>{"answer": "')
    assert text[marks[0] :] == 'A"}'


def test_choice_label_includes_key_and_description():
    from system_one_lite.prompts import label

    assert label("Payment issues", "billing") == "billing — Payment issues"
    assert label({"kind": "payment"}, "billing") == ('billing — {"kind": "payment"}')
    assert label(None, "billing") == "billing"


def test_confidence_uniform_zero_one_hot_one():
    assert abs(confidence([1 / 3, 1 / 3, 1 / 3])) < 1e-9
    assert abs(confidence([1.0, 0.0, 0.0]) - 1.0) < 1e-9
    # 3-way peak 0.6: (0.6 - 1/3) / (2/3) = 0.4
    assert abs(confidence([0.6, 0.38, 0.02]) - 0.4) < 1e-9
