"""Model-specific answer-code registry checks without loading model weights."""

from functools import partial

import pytest
from mlx_lm.utils import load_tokenizer

from system_one_lite.engine import (
    SMALL_MODEL,
    Engine,
    TooManyOptions,
    load_code_registry,
    resolve_model_snapshot,
)
from system_one_lite.prompts import filled


@pytest.fixture(scope="module")
def small_engine_contract():
    model_path, revision = resolve_model_snapshot(SMALL_MODEL, tokenizer_only=True)
    tokenizer = load_tokenizer(model_path)
    registry_file, codes, token_ids = load_code_registry(
        SMALL_MODEL, tokenizer, model_path, revision
    )
    engine = Engine.__new__(Engine)
    engine.model_id = SMALL_MODEL
    engine.model_path = model_path
    engine.model_revision = revision
    engine.tokenizer = tokenizer
    engine.registry_file = registry_file
    engine.codes = codes
    engine.code_token_ids = token_ids
    engine.context_window = 32_768
    engine.template = partial(filled, codes=codes)
    return engine


def test_small_model_registry_is_complete(small_engine_contract):
    assert len(small_engine_contract.codes) == 552
    assert len(set(small_engine_contract.code_token_ids)) == 552


def test_small_model_accepts_552_options(small_engine_contract):
    labels = [f"option {index}" for index in range(552)]
    jobs, _ = small_engine_contract.prepare("state", [("pick one", labels)])
    assert len(jobs[0][2]) == 552


def test_small_model_rejects_553_options(small_engine_contract):
    labels = [f"option {index}" for index in range(553)]
    with pytest.raises(TooManyOptions):
        small_engine_contract.prepare("state", [("pick one", labels)])
