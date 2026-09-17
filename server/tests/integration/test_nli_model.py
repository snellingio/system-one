"""Opt-in checks against the full OpenJEV checkpoint."""

import os

import pytest
from fastapi.testclient import TestClient

from system_one_lite.api import create_app
from system_one_lite.errors import RequestContractError
from system_one_lite.nli_engine import HuggingFaceNLIScorer, NLIEngine

pytestmark = pytest.mark.skipif(
    os.environ.get("SYSTEM_ONE_RUN_NLI_MODEL") != "1",
    reason="set SYSTEM_ONE_RUN_NLI_MODEL=1 to load the OpenJEV checkpoint",
)


@pytest.fixture(scope="module")
def scorer():
    return HuggingFaceNLIScorer(
        device=os.environ.get("SYSTEM_ONE_NLI_DEVICE", "auto"),
        batch_size=8,
        max_length=512,
    )


def sample_pairs():
    state = "The parcel was due Friday and remains in transit. " * 12
    premise = f"State:\n{state}\n\nQuestion:\nWhich team should handle this request?"
    return [
        (premise, "The correct answer is: deliveries — Late or missing orders"),
        (premise, "The correct answer is: billing — Charges or refunds"),
        (premise, "The correct answer is: account — Login or profile problems"),
    ]


def test_prefix_cache_matches_full_pairs_and_processes_fewer_tokens(scorer):
    pairs = sample_pairs()
    previous_batch_size = scorer.batch_size
    scorer.batch_size = 2
    try:
        scorer.prefix_cache = False
        full, full_tokens = scorer.score_grouped_pairs(pairs, [len(pairs)])
        scorer.prefix_cache = True
        cached, cached_tokens = scorer.score_grouped_pairs(pairs, [len(pairs)])
    finally:
        scorer.batch_size = previous_batch_size

    assert cached == pytest.approx(full, abs=2e-3)
    assert cached_tokens < full_tokens


def test_pairs_over_the_configured_token_limit_are_rejected(scorer):
    previous_limit = scorer.max_length
    scorer.max_length = 8
    try:
        with pytest.raises(RequestContractError, match="limit is 8"):
            scorer.score_pairs(sample_pairs()[:1])
    finally:
        scorer.max_length = previous_limit


def test_openjev_backend_serves_existing_api_shape(scorer):
    scorer.prefix_cache = True
    engine = NLIEngine(scorer=scorer)
    app = create_app(engine=engine)
    with TestClient(app) as client:
        response = client.post(
            "/evaluate",
            json={
                "state": "The parcel was due Friday and remains in transit.",
                "questions": {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team should handle this request?",
                        "criteria": {
                            "deliveries": "Late or missing orders",
                            "billing": "Charges or refunds",
                        },
                    },
                    "needs_reply": {
                        "type": "noul",
                        "instructions": "Does the customer need a reply?",
                    },
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model"].startswith("AlexWortega/openjev:qwen3.5-4b-nli@")
    assert set(body["answers"]) == {"team", "needs_reply"}
    assert sum(body["answers"]["team"]["probabilities"].values()) == pytest.approx(1.0)
    assert 0 <= body["answers"]["needs_reply"]["noul"] <= 1
    assert body["usage"]["input_tokens"] > 0
