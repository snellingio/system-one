from system_sdk import Choice, Score
from system_sdk.responses import NoulAnswer, SystemOneResponse, Usage


def test_structured_question_content_serializes():
    choice = Choice(
        {"field": "department"},
        {"billing": {"reason": "payment"}, "other": None},
    )
    score = Score(
        ["rate", "severity"],
        [{"level": "low"}, ["high", "urgent"]],
    )

    assert choice.to_wire()["instructions"] == {"field": "department"}
    assert choice.to_wire()["criteria"]["billing"] == {"reason": "payment"}
    assert score.to_wire()["criteria"][1] == ["high", "urgent"]


def test_structured_score_legend_parses():
    response = SystemOneResponse.from_wire({
        "model": "test",
        "answers": {
            "severity": {
                "type": "score",
                "score": 0.5,
                "legend": {"0": {"level": "low"}, "1": ["high", "urgent"]},
                "probabilities": {"0": 0.5, "1": 0.5},
                "confidence": 0.0,
            },
        },
        "usage": {"input_tokens": 1, "output_tokens": 0},
    })

    answer = response.answers["severity"]

    assert answer is response.scores["severity"]
    assert answer.type == "score"
    assert answer.legend[0] == {"level": "low"}
    assert answer.probabilities == {0: 0.5, 1: 0.5}


def test_answers_and_typed_views_share_answer_objects():
    response = SystemOneResponse.from_wire({
        "model": "test",
        "answers": {
            "billing": {"type": "noul", "noul": 0.9},
            "tone": {
                "type": "choice",
                "choice": "calm",
                "probabilities": {"calm": 0.8, "angry": 0.2},
                "confidence": 0.6,
            },
        },
        "usage": {"input_tokens": 1, "output_tokens": 0},
    })

    assert response.answers["billing"] is response.nouls["billing"]
    assert response.answers["billing"].type == "noul"
    assert response.answers["tone"] is response.choices["tone"]
    assert response.answers["tone"].type == "choice"


def test_existing_positional_response_constructor_keeps_field_order():
    billing = NoulAnswer(noul=0.9)
    response = SystemOneResponse(
        "test",
        Usage(input_tokens=1, output_tokens=0),
        {"billing": billing},
        {},
        {},
    )

    assert response.nouls == {"billing": billing}
    assert response.answers == {}
