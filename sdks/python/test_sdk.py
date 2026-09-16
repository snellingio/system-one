from system_sdk import Choice, Score
from system_sdk.responses import SystemOneResponse


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

    assert response.scores["severity"].legend["0"] == {"level": "low"}
