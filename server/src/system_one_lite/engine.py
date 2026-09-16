"""The decision engine: masked read at each answer slot.

Each question is its own full chat sequence (state + that question). Later
questions do not see earlier answers. Full prompts are evaluated without a
cache because the cached experiment must prove probability parity first.
One forward pass runs per question.
"""

import hashlib
import json
import os
import string
import time
from functools import partial
from importlib.resources import files
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load

from .prompts import chat_filled

QWEN3_1_7B_MODEL = "mlx-community/Qwen3-1.7B-4bit"
QWEN3_4B_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISIONS = {
    QWEN3_1_7B_MODEL: "3b1b1768f8f8cf8351c712464f906e86c2b8269e",
    QWEN3_4B_MODEL: "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b",
}
DEFAULT_MODEL = QWEN3_4B_MODEL
CODES_FILE = files("system_one_lite.data").joinpath("qwen3_4b_instruct_2507_4bit_answer_codes.json")
TEMPERATURE = 0.7
MAX_PROMPT_TOKENS = 32_768
MAX_TOTAL_INPUT_TOKENS = 131_072
TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
ANSWER_CODE_CANDIDATES = tuple(string.ascii_uppercase) + tuple(
    a + b for a in string.ascii_uppercase for b in string.ascii_uppercase
)
_REGISTERED_IDS = object()


class RequestContractError(ValueError):
    pass


class TooManyOptions(RequestContractError):
    pass


def configured_model(model_id=None):
    """Return an explicit model, the environment setting, or the default."""
    return model_id or os.environ.get("SYSTEM_MODEL") or DEFAULT_MODEL


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def resolve_model_snapshot(model_id, tokenizer_only=False, revision=None):
    """Return a local model snapshot and its Hub revision when available."""
    local = Path(model_id).expanduser()
    if local.exists():
        return local.resolve(), None
    revision = revision or MODEL_REVISIONS.get(model_id)
    if revision is None:
        raise ValueError(
            f"{model_id!r} has no pinned revision; pass one when generating its registry"
        )
    patterns = list(TOKENIZER_FILES) if tokenizer_only else None
    snapshot = Path(snapshot_download(model_id, revision=revision, allow_patterns=patterns))
    resolved_revision = snapshot.name if snapshot.parent.name == "snapshots" else None
    return snapshot, resolved_revision


def tokenizer_sha256(model_path):
    """Hash every tokenizer file that can change answer-code tokenization."""
    files = [model_path / name for name in TOKENIZER_FILES if (model_path / name).is_file()]
    if not files:
        raise ValueError(f"no tokenizer files found in {model_path}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def answer_code_entries(tokenizer):
    """Build the complete registry for the supported answer-code search."""
    text, marks = chat_filled(tokenizer, "code registry", [("pick one", ["yes", "no"])])
    prefix = text[: marks[0]]
    base = tokenizer.encode(prefix)
    entries, cuts = [], set()
    for code in ANSWER_CODE_CANDIDATES:
        cand = tokenizer.encode(prefix + code)
        cut = common_prefix_len(cand, base)
        if len(cand) - cut != 1:
            continue
        token_id = cand[cut]
        cuts.add(cut)
        entries.append(
            {
                "code": code,
                "token": tokenizer.decode([token_id]),
                "token_id": token_id,
            }
        )
    if len(cuts) != 1:
        raise ValueError(f"codes land at different cut positions: {sorted(cuts)}")
    return entries


def _registry_files():
    return sorted(
        (
            item
            for item in files("system_one_lite.data").iterdir()
            if item.name.endswith("_answer_codes.json")
        ),
        key=lambda item: item.name,
    )


def load_code_registry(model_id, tokenizer, model_path, model_revision):
    """Find and fully validate one model's answer-code registry."""
    matches = []
    for path in _registry_files():
        data = json.loads(path.read_text())
        if data.get("model") == model_id:
            matches.append((path, data))
    if len(matches) != 1:
        raise ValueError(
            f"expected one answer-code registry for {model_id}, found "
            f"{len(matches)}; run python -m tools.gen_answer_codes --model {model_id}"
        )

    path, data = matches[0]
    expected_hash = tokenizer_sha256(model_path)
    if data.get("model_revision") != model_revision:
        raise ValueError(
            f"{path.name} holds model revision {data.get('model_revision')}, "
            f"but the loaded revision is {model_revision}; regenerate it"
        )
    if data.get("tokenizer_sha256") != expected_hash:
        raise ValueError(f"{path.name} does not match the loaded tokenizer; regenerate it")

    expected = answer_code_entries(tokenizer)
    if data.get("codes") != expected:
        raise ValueError(f"{path.name} is incomplete or invalid for {model_id}; regenerate it")
    codes = tuple(entry["code"] for entry in expected)
    token_ids = tuple(entry["token_id"] for entry in expected)
    if not codes or codes[0] != "A":
        raise ValueError(f"{path.name} must start with the A answer code")
    if len(codes) != len(set(codes)) or len(token_ids) != len(set(token_ids)):
        raise ValueError(f"{path.name} contains duplicate codes or token ids")
    validate_code_contexts(tokenizer, codes, token_ids)
    return path, codes, token_ids


def letter_ids_after(tokenizer, prefix, codes):
    """The single token each answer code becomes when it follows `prefix`.

    All codes must land at the same cut position: if some merged with the
    prefix's last token and others did not, the masked read would compare
    tokens at different positions.
    """
    base = tokenizer.encode(prefix)
    ids, cuts = {}, set()
    for code in codes:
        cand = tokenizer.encode(prefix + code)
        cut = common_prefix_len(cand, base)
        suffix = cand[cut:]
        if len(suffix) != 1:
            raise ValueError(f"code {code!r} is {len(suffix)} tokens after the slot")
        cuts.add(cut)
        ids[code] = suffix[0]
    if len(cuts) != 1:
        raise ValueError(f"codes tokenize at different positions: {sorted(cuts)}")
    return ids


def validate_code_contexts(tokenizer, codes, token_ids):
    """Check that registry IDs stay fixed across varied production prompts."""
    cases = (
        ("", "pick one", ["yes", "no"]),
        (
            {"status": "late", "note": "café 東京"},
            {"task": "route", "urgent": True},
            ["billing — payment issue", "technical — API bug"],
        ),
        (
            "line one\nline two\n" + "long state " * 40,
            "Choose one after a long state.",
            [f"option {index}" for index in range(30)],
        ),
    )
    expected = dict(zip(codes, token_ids))
    for state, instructions, labels in cases:
        text, marks = chat_filled(tokenizer, state, [(instructions, labels)], codes=codes)
        actual = letter_ids_after(tokenizer, text[: marks[0]], codes)
        if actual != expected:
            raise ValueError("answer-code token IDs change across production prompt contexts")


class Engine:
    def __init__(self, model_id=None):
        self.model_id = configured_model(model_id)
        self.model_path, self.model_revision = resolve_model_snapshot(self.model_id)
        self.model, self.tokenizer = load(self.model_path)
        args = self.model.args
        text_config = getattr(args, "text_config", {})
        self.context_window = (
            getattr(args, "max_position_embeddings", None)
            or text_config.get("max_position_embeddings")
            or MAX_PROMPT_TOKENS
        )
        self.registry_file, self.codes, self.code_token_ids = load_code_registry(
            self.model_id, self.tokenizer, self.model_path, self.model_revision
        )
        # every question is coded from the registry, not just A-Z
        self.template = partial(chat_filled, self.tokenizer, codes=self.codes)
        # warm up: compile Metal kernels before the first real request
        self.evaluate("warm up", [("pick one", ["yes", "no"])])

    def evaluate(self, state, questions, template=None):
        """Run one independent full prompt per question.

        questions: list of (instructions, labels) in request order.
        template: builds (text, marks) from (state, questions); defaults to
        the production template over this engine's answer codes. Experiments
        pass their own builder.
        Returns (list of probability lists, input_tokens, ms).
        """
        probabilities, input_tokens, elapsed_ms, _, _ = self.evaluate_profiled(
            state, questions, template
        )
        return probabilities, input_tokens, elapsed_ms

    def prepare(self, state, questions, template=None):
        """Validate and tokenize a request without running the model."""
        production_template = template is None
        if production_template:
            template = self.template
        if not questions:
            raise RequestContractError("at least one question is required")
        for instructions, labels in questions:
            if len(labels) > len(self.codes):
                raise TooManyOptions(
                    f"{len(labels)} options for {instructions!r}; "
                    f"the engine has {len(self.codes)} single-token "
                    f"answer codes"
                )
            if not labels:
                raise RequestContractError(f"{instructions!r} has no options")

        jobs = []
        n_tokens = 0
        for instructions, labels in questions:
            text, marks = template(state, [(instructions, labels)])
            registered_ids = self.code_token_ids if production_template else None
            job = self._slot(text, marks[0], instructions, labels, registered_ids)
            if len(job[0]) > min(self.context_window, MAX_PROMPT_TOKENS):
                raise RequestContractError(
                    f"prompt for {instructions!r} exceeds the "
                    f"{min(self.context_window, MAX_PROMPT_TOKENS)} token limit"
                )
            jobs.append(job)
            n_tokens += len(job[0])
            if n_tokens > MAX_TOTAL_INPUT_TOKENS:
                raise RequestContractError(
                    f"request exceeds the {MAX_TOTAL_INPUT_TOKENS} input token limit"
                )

        return jobs, n_tokens

    def evaluate_profiled(self, state, questions, template=None):
        """Evaluate and also return prompt-building and model time."""
        t0 = time.perf_counter()
        jobs, n_tokens = self.prepare(state, questions, template)
        t1 = time.perf_counter()
        probs = []
        for full, slot, mask in jobs:
            logits = self.model(mx.array([full]))
            p = mx.softmax(logits[0, slot].astype(mx.float32)[mx.array(mask)] / TEMPERATURE)
            mx.eval(p)
            probs.append(p)
        t2 = time.perf_counter()

        return (
            [p.tolist() for p in probs],
            n_tokens,
            (t2 - t0) * 1000,
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
        )

    def _slot(self, text, mark, instructions, labels, registered_ids=_REGISTERED_IDS):
        """Token index to read and the option-code token ids."""
        if registered_ids is _REGISTERED_IDS:
            registered_ids = self.code_token_ids
        prefix = text[:mark]
        full = self.tokenizer.encode(text)
        base = self.tokenizer.encode(prefix)
        cand = self.tokenizer.encode(prefix + self.codes[0])
        cut = common_prefix_len(cand, base)
        if len(cand) - cut != 1:
            raise RequestContractError(f"slot filler for {instructions!r} is not a single token")
        if cand[: cut + 1] != full[: cut + 1]:
            raise RequestContractError(
                f"slot for {instructions!r} misaligned with the full template"
            )
        codes = self.codes[: len(labels)]
        if registered_ids is None:
            try:
                ids_by_code = letter_ids_after(self.tokenizer, prefix, codes)
            except ValueError as e:
                raise RequestContractError(str(e)) from e
            mask = [ids_by_code[code] for code in codes]
        else:
            mask = list(registered_ids[: len(labels)])
        # The final filler code is needed only to validate token alignment.
        # Logits at cut - 1 already predict it, so do not evaluate that token.
        return full[:cut], cut - 1, mask
