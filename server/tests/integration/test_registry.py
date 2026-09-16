"""Model-specific answer-code registry checks without loading model weights."""

from functools import partial

import pytest
from mlx_lm.utils import load_tokenizer

from system_one_lite import engine as engine_module
from system_one_lite.engine import (
    DEFAULT_MODEL,
    LARGER_MODEL,
    MODEL_REVISIONS,
    QWEN3_1_7B_MODEL,
    Engine,
    TooManyOptions,
    configured_model,
    load_code_registry,
    resolve_model_snapshot,
)
from system_one_lite.prompts import chat_filled


@pytest.fixture(scope="module")
def engine_contract():
    model_path, revision = resolve_model_snapshot(DEFAULT_MODEL, tokenizer_only=True)
    tokenizer = load_tokenizer(model_path)
    assert tokenizer.chat_template
    registry_file, codes, token_ids = load_code_registry(
        DEFAULT_MODEL, tokenizer, model_path, revision
    )
    engine = Engine.__new__(Engine)
    engine.model_id = DEFAULT_MODEL
    engine.model_path = model_path
    engine.model_revision = revision
    engine.tokenizer = tokenizer
    engine.registry_file = registry_file
    engine.codes = codes
    engine.code_token_ids = token_ids
    engine.context_window = 32_768
    engine.template = partial(chat_filled, tokenizer, codes=codes)
    return engine


@pytest.mark.parametrize("model_id", [DEFAULT_MODEL, LARGER_MODEL])
def test_shipped_model_registry_matches_tokenizer(model_id):
    model_path, revision = resolve_model_snapshot(model_id, tokenizer_only=True)
    tokenizer = load_tokenizer(model_path)
    _, codes, token_ids = load_code_registry(model_id, tokenizer, model_path, revision)
    assert len(codes) == 578
    assert len(set(token_ids)) == 578


def test_model_registry_is_complete(engine_contract):
    assert len(engine_contract.codes) == 578
    assert len(set(engine_contract.code_token_ids)) == 578


def test_model_accepts_578_options(engine_contract):
    labels = [f"option {index}" for index in range(578)]
    jobs, _ = engine_contract.prepare("state", [("pick one", labels)])
    assert len(jobs[0][2]) == 578


def test_model_rejects_579_options(engine_contract):
    labels = [f"option {index}" for index in range(579)]
    with pytest.raises(TooManyOptions):
        engine_contract.prepare("state", [("pick one", labels)])


def test_supported_models_have_pinned_registries():
    for model_id, expected_revision in MODEL_REVISIONS.items():
        model_path, revision = resolve_model_snapshot(model_id, tokenizer_only=True)
        tokenizer = load_tokenizer(model_path)
        _, codes, token_ids = load_code_registry(model_id, tokenizer, model_path, revision)

        assert revision == expected_revision
        assert codes[0] == "A"
        assert len(codes) == len(set(token_ids))


def test_small_model_uses_non_thinking_prompt():
    model_path, _ = resolve_model_snapshot(QWEN3_1_7B_MODEL, tokenizer_only=True)
    tokenizer = load_tokenizer(model_path)
    text, _ = chat_filled(tokenizer, "state", [("pick one", ["yes", "no"])])

    assert "<think>\n\n</think>" in text
    assert text.endswith('{"answer": "A"}')


def test_model_can_be_selected_by_environment(monkeypatch):
    monkeypatch.setenv("SYSTEM_ONE_MODEL", "larger")

    assert configured_model() == LARGER_MODEL
    assert configured_model(DEFAULT_MODEL) == DEFAULT_MODEL


def test_model_download_uses_exact_pin(tmp_path, monkeypatch):
    revision = MODEL_REVISIONS[QWEN3_1_7B_MODEL]
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    calls = []

    def fake_download(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return snapshot

    monkeypatch.setattr(engine_module, "snapshot_download", fake_download)

    path, resolved_revision = resolve_model_snapshot(QWEN3_1_7B_MODEL)

    assert path == snapshot
    assert resolved_revision == revision
    assert calls == [(QWEN3_1_7B_MODEL, {"revision": revision, "allow_patterns": None})]


def test_small_model_runs_masked_read():
    engine = Engine(QWEN3_1_7B_MODEL)
    probabilities, input_tokens, elapsed_ms = engine.evaluate(
        "The API returns an integration error.",
        [("Which team?", ["billing", "technical"])],
    )

    assert len(probabilities) == 1
    assert abs(sum(probabilities[0]) - 1.0) < 1e-5
    assert input_tokens > 0
    assert elapsed_ms > 0
