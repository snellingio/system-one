"""Model-specific answer-code registry checks without loading model weights."""

from functools import partial

import pytest
from mlx_lm.utils import load_tokenizer

from system_one_lite.engine import (
    DEFAULT_MODEL,
    LARGER_MODEL,
    Engine,
    TooManyOptions,
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
