"""Server shape tests against docs/api.md (minus its request `model`
field), through the FastAPI TestClient."""

import pytest
from fastapi.testclient import TestClient

from system_one_lite.api import app, create_app
from system_one_lite.engine import RequestContractError
from system_one_lite.schemas import (
    MAX_QUESTIONS,
    MAX_REQUEST_BYTES,
    MAX_STATE_CHARS,
)

client = None
test_app = None


@pytest.fixture(scope="module", autouse=True)
def configure_client(engine):
    global client, test_app
    test_app = create_app(engine=engine)
    with TestClient(test_app) as active_client:
        client = active_client
        yield


STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)


def evaluate(questions):
    return client.post("/evaluate", json={"state": STATE, "questions": questions})


def test_module_import_does_not_load_engine():
    assert app.state.engine is None


def test_quickstart_mixed_three_questions():
    """The quickstart.md request: Choice + Score + Noul, one pass, full shape."""
    r = evaluate(
        {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this",
                "criteria": {
                    "billing": "Payment or subscription issues",
                    "technical": "Bugs or integration problems",
                    "sales": "Pricing or account questions",
                },
            },
            "frustration": {
                "type": "score",
                "instructions": "How frustrated the customer appears",
                "criteria": [
                    "Calm, just stating facts",
                    "Frustrated but civil",
                    "Very angry, strong language",
                ],
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "The message conveys urgency or time-sensitivity",
            },
        }
    )
    assert r.status_code == 200
    body = r.json()

    assert body["model"].startswith("mlx-community/Qwen")
    assert body["usage"]["output_tokens"] == 0
    assert body["usage"]["input_tokens"] > 0

    dept = body["answers"]["department"]
    assert dept["type"] == "choice"
    assert dept["choice"] in dept["probabilities"]
    assert abs(sum(dept["probabilities"].values()) - 1.0) < 1e-5
    n = len(dept["probabilities"])
    pmax = dept["probabilities"][dept["choice"]]
    assert abs(dept["confidence"] - (pmax - 1 / n) / (1 - 1 / n)) < 1e-6

    frustration = body["answers"]["frustration"]
    assert frustration["type"] == "score"
    assert set(frustration["legend"]) == {"0", "1", "2"}
    probs = frustration["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    expected_score = sum(int(i) * p for i, p in probs.items())
    assert abs(frustration["score"] - expected_score) < 1e-6
    n = len(probs)
    assert abs(frustration["confidence"] - (max(probs.values()) - 1 / n) / (1 - 1 / n)) < 1e-6
    assert 0.0 <= frustration["score"] <= 2.0

    urgent = body["answers"]["is_urgent"]
    assert urgent["type"] == "noul"
    assert set(urgent) == {"type", "noul"}
    assert 0.0 <= urgent["noul"] <= 1.0


def test_empty_questions_rejected():
    r = evaluate({})
    assert r.status_code == 422


def test_choice_with_no_options_rejected_by_schema():
    r = evaluate(
        {
            "empty": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {},
            },
        }
    )
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)


def test_unknown_fields_rejected_at_every_request_level():
    top = client.post(
        "/evaluate",
        json={
            "state": STATE,
            "questions": {"x": {"type": "noul", "instructions": "urgent?"}},
            "extra": True,
        },
    )
    question = evaluate(
        {
            "x": {"type": "noul", "instructions": "urgent?", "extra": True},
        }
    )
    criteria = evaluate(
        {
            "x": {
                "type": "noul",
                "instructions": "urgent?",
                "criteria": {"ture": "time-sensitive"},
            },
        }
    )
    assert top.status_code == question.status_code == criteria.status_code == 422


def test_request_shape_limits_are_enforced():
    too_many = {
        f"q{i}": {"type": "noul", "instructions": "urgent?"} for i in range(MAX_QUESTIONS + 1)
    }
    assert evaluate(too_many).status_code == 422
    oversized = client.post(
        "/evaluate",
        json={
            "state": "x" * (MAX_STATE_CHARS + 1),
            "questions": {"x": {"type": "noul", "instructions": "urgent?"}},
        },
    )
    assert oversized.status_code == 422


def test_aggregate_request_content_limit_is_enforced():
    questions = {f"q{i}": {"type": "noul", "instructions": "x" * 20_000} for i in range(51)}
    assert len(str(questions).encode("utf-8")) > MAX_REQUEST_BYTES
    r = evaluate(questions)
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)


def test_busy_engine_rejected_without_waiting():
    inference_slot = test_app.state.inference_slot
    assert inference_slot.acquire(blocking=False)
    try:
        r = evaluate(
            {
                "x": {"type": "noul", "instructions": "urgent?"},
            }
        )
    finally:
        inference_slot.release()
    assert r.status_code == 503


def test_only_request_contract_errors_become_422(monkeypatch):
    def contract_error(*args, **kwargs):
        raise RequestContractError("bad prompt")

    monkeypatch.setattr(test_app.state.engine, "evaluate", contract_error)
    r = evaluate({"x": {"type": "noul", "instructions": "urgent?"}})
    assert r.status_code == 422
    assert r.json()["detail"] == "bad prompt"


def test_internal_value_error_is_not_misreported_as_422(monkeypatch):
    def internal_error(*args, **kwargs):
        raise ValueError("internal failure")

    monkeypatch.setattr(test_app.state.engine, "evaluate", internal_error)
    with pytest.raises(ValueError, match="internal failure"):
        evaluate({"x": {"type": "noul", "instructions": "urgent?"}})


def test_unknown_question_type_rejected():
    r = evaluate({"x": {"type": "mystery", "instructions": "q"}})
    assert r.status_code == 422


def test_choice_with_two_letter_codes():
    """30 options cross the Z/AA code boundary and still answer in shape."""
    criteria = {f"opt{i}": None for i in range(30)}
    r = evaluate({"big": {"type": "choice", "instructions": "pick", "criteria": criteria}})
    assert r.status_code == 200
    probs = r.json()["answers"]["big"]["probabilities"]
    assert set(probs) == set(criteria)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert r.json()["answers"]["big"]["choice"] in criteria


def test_too_many_options_rejected():
    engine = test_app.state.engine
    criteria = {f"opt{i}": None for i in range(len(engine.codes) + 1)}
    r = evaluate({"big": {"type": "choice", "instructions": "pick", "criteria": criteria}})
    assert r.status_code == 422
    assert str(len(engine.codes)) in r.json()["detail"]


def test_score_needs_two_levels():
    r = evaluate({"s": {"type": "score", "instructions": "rate", "criteria": ["only"]}})
    assert r.status_code == 422


def test_noul_with_criteria():
    r = evaluate(
        {
            "x": {
                "type": "noul",
                "instructions": "urgent?",
                "criteria": {"true": "time-sensitive", "false": "can wait"},
            }
        }
    )
    assert r.status_code == 200
    assert "noul" in r.json()["answers"]["x"]


def test_v1_systemone_alias():
    r = client.post(
        "/v1/systemone",
        json={"state": STATE, "questions": {"x": {"type": "noul", "instructions": "urgent?"}}},
    )
    assert r.status_code == 200
    assert "noul" in r.json()["answers"]["x"]
