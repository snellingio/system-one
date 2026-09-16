"""NLI backend tests that do not load the real checkpoint."""

import math

import pytest

from system_one_lite.errors import RequestContractError, TooManyOptions
from system_one_lite.nli_engine import (
    MAX_NLI_CANDIDATES,
    NLIEngine,
    common_token_prefix,
    load_nli_assets,
    nli_pairs,
    normalize_entailment,
    resolve_entailment_index,
    resolve_nli_revision,
    select_dtype,
)


class FakeScorer:
    model_id = "fake/openjev"

    def __init__(self, values, input_tokens=123):
        self.values = values
        self.input_tokens = input_tokens
        self.pairs = None

    def score_pairs(self, pairs):
        self.pairs = pairs
        return self.values, self.input_tokens


class FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"

    class cuda:
        @staticmethod
        def is_bf16_supported():
            return True

    @staticmethod
    def device(value):
        return type("Device", (), {"type": value.split(":", 1)[0]})()


class FakeLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.calls.append((args, kwargs))
        return cls


def test_common_token_prefix():
    assert common_token_prefix([[1, 2, 3], [1, 2, 4], [1, 2]]) == 2
    assert common_token_prefix([]) == 0
    assert common_token_prefix([[1], [2]]) == 0


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda:0", "bfloat16"), ("mps:0", "float16"), ("cpu", "bfloat16")],
)
def test_dtype_uses_qualified_device_type(device, expected):
    assert select_dtype(FakeTorch, device) == expected


def test_huggingface_assets_use_the_same_pinned_revision():
    class FakeTokenizerLoader(FakeLoader):
        calls = []

    class FakeModelLoader(FakeLoader):
        calls = []

    tokenizer, model = load_nli_assets(
        FakeTokenizerLoader,
        FakeModelLoader,
        "example/model",
        "classifier",
        "abc123",
        "float16",
    )

    assert tokenizer is FakeTokenizerLoader
    assert model is FakeModelLoader
    assert FakeTokenizerLoader.calls == [
        (("example/model",), {"subfolder": "classifier", "revision": "abc123"})
    ]
    assert FakeModelLoader.calls == [
        (
            ("example/model",),
            {"subfolder": "classifier", "revision": "abc123", "dtype": "float16"},
        )
    ]


def test_builtin_revision_is_pinned_and_custom_model_requires_one():
    assert resolve_nli_revision("AlexWortega/openjev") == (
        "8c9db06441316f3fff7a68feb7da3fea79a5eff7"
    )
    assert resolve_nli_revision("example/model", "abc123") == "abc123"
    with pytest.raises(ValueError, match="required for a custom NLI model"):
        resolve_nli_revision("example/model")


def test_entailment_index_comes_from_model_config():
    config = type("Config", (), {"label2id": {"neutral": 0, "ENTAILMENT": 2}})()
    assert resolve_entailment_index(config) == 2

    missing = type("Config", (), {"label2id": {"LABEL_0": 0}})()
    with pytest.raises(ValueError, match="define an entailment label"):
        resolve_entailment_index(missing)


def test_pairs_share_one_premise_per_question():
    pairs, sizes = nli_pairs(
        "A parcel is late.",
        [
            ("Who should handle it?", ["delivery", "billing"]),
            ("Is a reply needed?", ["yes", "no"]),
        ],
    )

    assert sizes == [2, 2]
    assert len(pairs) == 4
    assert pairs[0][0] == pairs[1][0]
    assert pairs[2][0] == pairs[3][0]
    assert pairs[0][0] != pairs[2][0]
    assert pairs[0][1] == "The correct answer is: delivery"


def test_entailment_normalization_preserves_ranking():
    probabilities = normalize_entailment([0.1, 0.6, 0.3])

    assert probabilities == pytest.approx([0.1, 0.6, 0.3])
    assert sum(probabilities) == pytest.approx(1.0)


def test_temperature_changes_distribution_not_winner():
    cool = normalize_entailment([0.2, 0.8], temperature=0.5)
    warm = normalize_entailment([0.2, 0.8], temperature=2.0)

    assert cool[1] > warm[1] > 0.5


@pytest.mark.parametrize("values", [[math.nan], [-0.1], [1.1]])
def test_invalid_entailment_scores_rejected(values):
    with pytest.raises(ValueError, match="probabilities"):
        normalize_entailment(values)


def test_engine_batches_every_candidate_in_one_scorer_call():
    scorer = FakeScorer([0.1, 0.7, 0.2, 0.8, 0.2])
    engine = NLIEngine(scorer=scorer)

    results, input_tokens, elapsed_ms = engine.evaluate(
        "A parcel is late.",
        [
            ("Who should handle it?", ["delivery", "billing", "account"]),
            ("Is a reply needed?", ["yes", "no"]),
        ],
    )

    assert engine.model_id == "fake/openjev"
    assert len(scorer.pairs) == 5
    assert results[0] == pytest.approx([0.1, 0.7, 0.2])
    assert results[1] == pytest.approx([0.8, 0.2])
    assert input_tokens == 123
    assert elapsed_ms >= 0


def test_engine_rejects_wrong_score_count():
    engine = NLIEngine(scorer=FakeScorer([0.5]))
    with pytest.raises(RuntimeError, match="1 scores for 2 candidates"):
        engine.evaluate("state", [("question", ["one", "two"])])


def test_empty_questions_and_options_rejected():
    with pytest.raises(RequestContractError, match="at least one"):
        nli_pairs("state", [])
    with pytest.raises(RequestContractError, match="no options"):
        nli_pairs("state", [("question", [])])


def test_request_candidate_limit():
    labels = [f"option {index}" for index in range(MAX_NLI_CANDIDATES + 1)]
    with pytest.raises(TooManyOptions, match=str(MAX_NLI_CANDIDATES)):
        nli_pairs("state", [("question", labels)])


def test_answer_slot_template_rejected():
    engine = NLIEngine(scorer=FakeScorer([1.0]))
    with pytest.raises(RequestContractError, match="does not accept"):
        engine.evaluate("state", [("question", ["yes"])], template=object())
