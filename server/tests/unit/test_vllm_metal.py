import math

import pytest

from system_one_lite import vllm_metal as vllm_module
from system_one_lite.engine import DEFAULT_MODEL, RequestContractError, TooManyOptions
from system_one_lite.vllm_metal import (
    MAX_LOGPROB_TOKEN_IDS,
    VllmMetalEngine,
    VllmMetalError,
    restricted_softmax,
)


class FakeTokenizer:
    marker = '{"answer": "'

    def __init__(self, token_count=12):
        self.token_count = token_count

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def encode(self, text):
        before, marker, suffix = text.rpartition(self.marker)
        if not marker:
            return list(range(self.token_count))
        tokens = list(range(self.token_count))
        if suffix:
            tokens.append(32)
            tokens.extend(ord(character) for character in suffix[1:])
        return tokens

    def decode(self, token_ids):
        return ",".join(map(str, token_ids))


def response_for(payload, scores=None, prompt_tokens=None):
    token_ids = payload["logprob_token_ids"]
    scores = scores or [-float(index) for index in range(len(token_ids))]
    return {
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": choice_index,
                "logprobs": {
                    "top_logprobs": [
                        {
                            f"token_id:{token_id}": score
                            for token_id, score in zip(token_ids, scores)
                        }
                    ]
                },
            }
            for choice_index in range(len(payload["prompt"]))
        ],
        "usage": {
            "prompt_tokens": prompt_tokens
            if prompt_tokens is not None
            else len(payload["prompt"]) * 12
        },
    }


def fake_engine(transport, token_count=12):
    return VllmMetalEngine(
        DEFAULT_MODEL,
        transport=transport,
        tokenizer=FakeTokenizer(token_count),
    )


def test_exact_requested_token_logprobs_are_normalized():
    calls = []
    tokenizer = FakeTokenizer()

    def transport(path, payload):
        calls.append((path, payload))
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        return response_for(payload, [-2.0, -1.0, -3.0])

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    probabilities, input_tokens, elapsed_ms = engine.evaluate(
        "state", [("pick", ["one", "two", "three"])]
    )

    assert probabilities[0] == pytest.approx(restricted_softmax([-2.0, -1.0, -3.0]))
    assert input_tokens == 12
    assert elapsed_ms >= 0
    assert [path for path, _ in calls] == ["/detokenize", "/v1/completions"]
    payload = calls[-1][1]
    assert payload["model"] == DEFAULT_MODEL
    assert payload["max_tokens"] == 1
    assert payload["logprob_token_ids"] == list(engine.code_token_ids[:3])
    assert payload["prompt"] == [list(range(12))]
    assert engine.output_tokens_for([("pick", ["one", "two", "three"])]) == 1


def test_questions_are_independent_and_token_usage_is_summed():
    payloads = []
    tokenizer = FakeTokenizer(token_count=7)

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        payloads.append(payload)
        return response_for(payload, prompt_tokens=14)

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    probabilities, input_tokens, _ = engine.evaluate(
        "shared", [("first", ["yes", "no"]), ("second", ["up", "down"])]
    )

    assert len(probabilities) == 2
    assert input_tokens == 14
    assert len(payloads) == 1
    assert len(payloads[0]["prompt"]) == 2


def test_more_than_128_options_are_read_in_exact_chunks():
    tokenizer = FakeTokenizer(token_count=5)
    payloads = []

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        payloads.append(payload)
        return response_for(payload, prompt_tokens=5)

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    labels = [f"option {index}" for index in range(MAX_LOGPROB_TOKEN_IDS + 2)]
    probabilities, input_tokens, _ = engine.evaluate("state", [("pick", labels)])

    assert [len(payload["logprob_token_ids"]) for payload in payloads] == [128, 2]
    assert len(probabilities[0]) == 130
    assert sum(probabilities[0]) == pytest.approx(1.0)
    assert input_tokens == 10
    assert engine.output_tokens_for([("pick", labels)]) == 2


def test_mixed_option_counts_skip_finished_questions_on_later_chunks():
    tokenizer = FakeTokenizer(token_count=5)
    batch_sizes = []

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        batch_sizes.append(len(payload["prompt"]))
        return response_for(payload, prompt_tokens=len(payload["prompt"]) * 5)

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    questions = [
        ("small", ["yes", "no"]),
        ("large", [f"option {index}" for index in range(129)]),
    ]
    _, input_tokens, _ = engine.evaluate("state", questions)

    assert batch_sizes == [2, 1]
    assert input_tokens == 15
    assert engine.output_tokens_for(questions) == 3


def test_server_tokenizer_mismatch_is_an_error_before_inference():
    calls = []

    def transport(path, payload):
        calls.append(path)
        return {"prompt": "wrong"}

    engine = fake_engine(transport)
    with pytest.raises(VllmMetalError, match="pinned tokenizer"):
        engine.evaluate("state", [("pick", ["yes", "no"])])

    assert calls == ["/detokenize"]


def test_missing_requested_token_is_an_error():
    tokenizer = FakeTokenizer()

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        response = response_for(payload)
        response["choices"][0]["logprobs"]["top_logprobs"][0].popitem()
        return response

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    with pytest.raises(VllmMetalError, match="omitted requested"):
        engine.evaluate("state", [("pick", ["yes", "no"])])


def test_wrong_response_model_is_an_error():
    tokenizer = FakeTokenizer()

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        return {**response_for(payload), "model": "wrong-model"}

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    with pytest.raises(VllmMetalError, match="expected"):
        engine.evaluate("state", [("pick", ["yes", "no"])])


@pytest.mark.parametrize("prompt_tokens", ["12", -1, True])
def test_invalid_prompt_token_count_is_an_error(prompt_tokens):
    tokenizer = FakeTokenizer()

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        return response_for(payload, prompt_tokens=prompt_tokens)

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    with pytest.raises(VllmMetalError, match="prompt token count"):
        engine.evaluate("state", [("pick", ["yes", "no"])])


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "-1.0", True])
def test_invalid_requested_logprob_is_an_error(bad_value):
    tokenizer = FakeTokenizer()

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        return response_for(payload, scores=[bad_value, -1.0])

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    with pytest.raises(VllmMetalError, match="invalid token logprobs"):
        engine.evaluate("state", [("pick", ["yes", "no"])])


def test_non_mapping_logprobs_are_an_error():
    tokenizer = FakeTokenizer()

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        response = response_for(payload)
        response["choices"][0]["logprobs"]["top_logprobs"][0] = []
        return response

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    with pytest.raises(VllmMetalError, match="invalid token logprobs"):
        engine.evaluate("state", [("pick", ["yes", "no"])])


def test_contract_checks_happen_before_transport(monkeypatch):
    def transport(path, payload):
        raise AssertionError("transport should not run")

    engine = fake_engine(transport)
    with pytest.raises(RequestContractError, match="at least one"):
        engine.evaluate("state", [])
    with pytest.raises(RequestContractError, match="custom templates"):
        engine.evaluate("state", [("pick", ["yes"])], template=object())
    with pytest.raises(TooManyOptions):
        engine.evaluate("state", [("pick", ["x"] * (len(engine.codes) + 1))])

    monkeypatch.setattr(vllm_module, "MAX_PROMPT_TOKENS", 11)
    with pytest.raises(RequestContractError, match="prompt for"):
        engine.evaluate("state", [("pick", ["yes", "no"])])


def test_total_token_limit_is_checked_before_transport(monkeypatch):
    def transport(path, payload):
        raise AssertionError("transport should not run")

    engine = fake_engine(transport, token_count=10)
    monkeypatch.setattr(vllm_module, "MAX_TOTAL_INPUT_TOKENS", 19)

    with pytest.raises(RequestContractError, match="request exceeds"):
        engine.evaluate(
            "state",
            [("first", ["yes", "no"]), ("second", ["up", "down"])],
        )


def test_prompt_at_input_limit_keeps_one_remote_output_slot(monkeypatch):
    tokenizer = FakeTokenizer(token_count=12)

    def transport(path, payload):
        if path == "/detokenize":
            return {"prompt": tokenizer.decode(payload["tokens"])}
        return response_for(payload, prompt_tokens=12)

    engine = VllmMetalEngine(DEFAULT_MODEL, transport=transport, tokenizer=tokenizer)
    monkeypatch.setattr(vllm_module, "MAX_PROMPT_TOKENS", 12)

    probabilities, input_tokens, _ = engine.evaluate("state", [("pick", ["yes", "no"])])

    assert len(probabilities[0]) == 2
    assert input_tokens == 12


def test_restricted_softmax_applies_system_one_temperature():
    probabilities = restricted_softmax([-2.0, -1.0])
    expected_second = 1 / (1 + math.exp(-1 / 0.7))

    assert probabilities == pytest.approx([1 - expected_second, expected_second])
