"""Engine behavior tests."""

import mlx.core as mx
import pytest

from system_one_lite import engine as engine_module
from system_one_lite.engine import MAX_PROMPT_TOKENS, RequestContractError, TooManyOptions

STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)
QUESTION = "Which team should handle this?"
CRITERIA = {
    "billing": "Payment or subscription issues",
    "technical": "Bugs or integration problems",
    "sales": "Pricing or account questions",
}


def choice(engine, state=STATE, question=QUESTION, criteria=CRITERIA):
    labels = [desc or key for key, desc in criteria.items()]
    results, input_tokens, ms = engine.evaluate(state, [(question, labels)])
    return dict(zip(criteria, results[0])), input_tokens, ms


def test_choice_distribution_shape(engine):
    probs, input_tokens, ms = choice(engine)
    assert set(probs) == set(CRITERIA)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert input_tokens > 0 and ms > 0


def test_deterministic(engine):
    assert choice(engine)[0] == choice(engine)[0]


def test_base_model_picks_technical(engine):
    probs, _, _ = choice(engine)
    assert max(probs, key=probs.get) == "technical"


def test_more_options_than_codes_rejected(engine):
    labels = [f"option {i}" for i in range(len(engine.codes) + 1)]
    with pytest.raises(TooManyOptions):
        engine.evaluate(STATE, [(QUESTION, labels)])


def test_two_letter_codes_past_z(engine):
    """Option 27 onward gets two-letter codes; the read stays a distribution."""
    assert engine.codes[25] == "Z" and engine.codes[26] == "AA"
    n = 30
    results, _, _ = engine.evaluate(STATE, [(QUESTION, [f"opt {i}" for i in range(n)])])
    assert len(results[0]) == n
    assert abs(sum(results[0]) - 1.0) < 1e-5
    assert all(0.0 <= p <= 1.0 for p in results[0])


@pytest.mark.parametrize("option_count", [27, 255, 570])
def test_large_option_contract_prepares_without_model_inference(engine, option_count):
    labels = [f"option {i}" for i in range(option_count)]
    jobs, _ = engine.prepare(STATE, [(QUESTION, labels)])
    _, _, mask = jobs[0]
    assert len(mask) == option_count
    assert len(set(mask)) == option_count


def test_production_prepare_uses_cached_registry_ids(engine, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("production request re-tokenized every answer code")

    monkeypatch.setattr(engine_module, "letter_ids_after", fail)
    jobs, _ = engine.prepare(STATE, [(QUESTION, list(CRITERIA))])
    assert jobs[0][2] == list(engine.code_token_ids[: len(CRITERIA)])


def test_no_questions_rejected(engine):
    with pytest.raises(ValueError):
        engine.evaluate(STATE, [])


def test_empty_options_rejected(engine):
    with pytest.raises(ValueError):
        engine.evaluate(STATE, [(QUESTION, [])])


def test_multi_question_one_pass(engine):
    questions = [
        (QUESTION, [d or k for k, d in CRITERIA.items()]),
        ("Does this convey urgency?", ["yes", "no"]),
        (
            "How frustrated is the customer?",
            ["Calm, just stating facts", "Frustrated but civil", "Very angry"],
        ),
    ]
    results, input_tokens, ms = engine.evaluate(STATE, questions)
    assert len(results) == 3
    assert len(results[0]) == 3 and len(results[1]) == 2 and len(results[2]) == 3
    for probs in results:
        assert abs(sum(probs) - 1.0) < 1e-5
    # the department question still picks technical when asked alongside others
    assert max(zip(CRITERIA, results[0]), key=lambda kv: kv[1])[0] == "technical"


def test_later_question_matches_alone(engine):
    q1 = (QUESTION, [d or k for k, d in CRITERIA.items()])
    q2 = ("Does this convey urgency?", ["yes", "no"])
    together, _, _ = engine.evaluate(STATE, [q1, q2])
    alone, _, _ = engine.evaluate(STATE, [q2])
    assert together[1] == alone[0]


def test_first_question_matches_alone(engine):
    q1 = (QUESTION, [d or k for k, d in CRITERIA.items()])
    q2 = ("Does this convey urgency?", ["yes", "no"])
    together, _, _ = engine.evaluate(STATE, [q1, q2])
    alone, _, _ = engine.evaluate(STATE, [q1])
    assert together[0] == alone[0]


def test_multi_question_tokens_count_each_full_prompt(engine):
    q1 = (QUESTION, [d or k for k, d in CRITERIA.items()])
    q2 = ("Does this convey urgency?", ["yes", "no"])
    _, t1, _ = engine.evaluate(STATE, [q1])
    _, t2, _ = engine.evaluate(STATE, [q2])
    _, tb, _ = engine.evaluate(STATE, [q1, q2])
    assert tb == t1 + t2


def test_input_tokens_exclude_answer_filler(engine):
    labels = [d or k for k, d in CRITERIA.items()]
    text, marks = engine.template(STATE, [(QUESTION, labels)])
    prompt, _, _ = engine._slot(text, marks[0], QUESTION, labels)
    _, input_tokens, _ = engine.evaluate(STATE, [(QUESTION, labels)])
    assert input_tokens == len(prompt)


def test_production_result_matches_direct_forward_pass(engine):
    labels = [d or k for k, d in CRITERIA.items()]
    text, marks = engine.template(STATE, [(QUESTION, labels)])
    prompt, slot, mask = engine._slot(text, marks[0], QUESTION, labels)
    cached, _, _ = engine.evaluate(STATE, [(QUESTION, labels)])

    logits = engine.model(mx.array([prompt]))
    expected = mx.softmax(logits[0, slot].astype(mx.float32)[mx.array(mask)])
    mx.eval(expected)
    assert cached[0] == expected.tolist()


def test_prompt_token_limit_checked_before_inference(engine):
    state = "word " * (MAX_PROMPT_TOKENS + 1)
    with pytest.raises(RequestContractError, match="token limit"):
        engine.evaluate(state, [("pick", ["yes", "no"])])
