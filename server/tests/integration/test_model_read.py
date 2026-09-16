"""Tokenizer and registry contract checks for the answer-slot read.

Every answer code in the engine's registry (A-Z, then the two-letter codes
the registry tool kept) must be a single token after the slot prefix, all at the
same cut position, so the masked read compares tokens at the same position.
"""

import json

from system_one_lite.engine import common_prefix_len, tokenizer_sha256

STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)
QUESTION = "Which team should handle this?"
LABELS = [
    "Payment or subscription issues",
    "Bugs or integration problems",
    "Pricing or account questions",
]


def test_slot_starts_the_assistant_json_value(engine):
    text, marks = engine.template(STATE, [(QUESTION, LABELS)])
    prefix = text[: marks[0]]
    tokenizer = engine.tokenizer
    filled_ids = tokenizer.encode(prefix + "A")
    cut = common_prefix_len(filled_ids, tokenizer.encode(prefix))
    assert text[: marks[0]].endswith('{"answer": "')
    assert text[marks[0] :].startswith('A"}')
    assert tokenizer.decode([filled_ids[cut]]) == "A"


def test_registry_codes_single_token_at_one_cut(engine):
    tokenizer = engine.tokenizer
    text, marks = engine.template(STATE, [(QUESTION, LABELS)])
    prefix = text[: marks[0]]
    base = tokenizer.encode(prefix)
    cuts = set()
    for code in engine.codes:
        cand = tokenizer.encode(prefix + code)
        cut = common_prefix_len(cand, base)
        assert len(cand) - cut == 1, (
            f"code {code!r} is not a single token after the slot; "
            "regenerate the model answer-code registry"
        )
        cuts.add(cut)
    assert len(cuts) == 1, f"codes land at different positions: {sorted(cuts)}"


def test_registry_is_pinned_to_loaded_model_and_tokenizer(engine):
    data = json.loads(engine.registry_file.read_text())
    assert data["model"] == engine.model_id
    assert data["model_revision"] == engine.model_revision
    assert data["tokenizer_sha256"] == tokenizer_sha256(engine.model_path)
    assert [entry["code"] for entry in data["codes"]] == list(engine.codes)
    assert [entry["token_id"] for entry in data["codes"]] == list(engine.code_token_ids)
