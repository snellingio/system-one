"""System One backend for seeded DiffusionGemma reads through MLX-VLM."""

import json
import math
import os
import random
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mlx_lm.utils import load_tokenizer

from .engine import (
    DIFFUSION_GEMMA_MODEL,
    MAX_PROMPT_TOKENS,
    MAX_TOTAL_INPUT_TOKENS,
    TEMPERATURE,
    RequestContractError,
    TooManyOptions,
    diffusion_answer_code_entries,
    packaged_code_registry,
    resolve_model_id,
    resolve_model_snapshot,
    tokenizer_sha256,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 120.0
CANVAS_PREFIX = "<|channel>thought\n<channel|>"
CANVAS_SUFFIX = "<turn|>\n"
DEFAULT_DIFFUSION_SEED = 42


class MlxVlmError(RuntimeError):
    """The MLX-VLM server could not complete a structured diffusion read."""


def restricted_softmax(logprobs, temperature=TEMPERATURE):
    """Renormalize vocabulary log probabilities over allowed answer codes."""
    scaled = [value / temperature for value in logprobs]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


def _encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def build_prompt(tokenizer, state, questions, codes, compact=False):
    """Build one encoder prompt containing every independent question."""
    if compact:
        parts = [state.strip()]
        for index, (instructions, labels) in enumerate(questions, 1):
            prefix = "" if len(questions) == 1 else f"q{index} "
            options = " ".join(f"{codes[i]} {item}" for i, item in enumerate(labels))
            parts.append(f" {prefix}{instructions} {options}")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "".join(parts)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return _encode(tokenizer, text)

    parts = [
        "Answer each question about the state. Each question lists its allowed answers.\n\n",
        "State:\n",
        state.strip(),
    ]
    for index, (instructions, labels) in enumerate(questions, 1):
        parts.append(f"\n\nQuestion q{index}: {instructions}\n")
        parts.append("\n".join(f"  {codes[i]}: {item}" for i, item in enumerate(labels)))
    parts.append('\n\nReply with one line per question, in order, formatted as "- label".')
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "".join(parts)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _encode(tokenizer, text)


def build_seed_canvas(
    tokenizer,
    question_count,
    canvas_length,
    vocab_size,
    compact=False,
):
    """Build the active answer template and randomize only its answer slots."""
    rng = random.Random(DEFAULT_DIFFUSION_SEED)
    if compact:
        if question_count > canvas_length:
            raise RequestContractError(
                f"{question_count} questions need {question_count} canvas tokens; "
                f"DiffusionGemma provides {canvas_length}"
            )
        return [rng.randrange(vocab_size) for _ in range(question_count)], list(
            range(question_count)
        )

    canvas = _encode(tokenizer, CANVAS_PREFIX)
    positions = []
    for _ in range(question_count):
        canvas.extend(_encode(tokenizer, "-"))
        positions.append(len(canvas))
        canvas.append(rng.randrange(vocab_size))
        canvas.extend(_encode(tokenizer, "\n"))
    canvas.extend(_encode(tokenizer, CANVAS_SUFFIX))
    if len(canvas) > canvas_length:
        raise RequestContractError(
            f"{question_count} questions need {len(canvas)} canvas tokens; "
            f"DiffusionGemma provides {canvas_length}"
        )
    return canvas, positions


class MlxVlmDiffusionEngine:
    """Read all question slots in one DiffusionGemma denoising forward."""

    def __init__(self, model_id=None, base_url=None, transport=None, compact=False):
        self.model_id = resolve_model_id(model_id or "diffusion")
        if self.model_id != DIFFUSION_GEMMA_MODEL:
            raise ValueError("the mlx-vlm-diffusion backend requires the diffusion model profile")
        self.model_path, self.model_revision = resolve_model_snapshot(self.model_id)
        self.tokenizer = load_tokenizer(self.model_path)
        self.registry_file, registry, self.codes, self.code_token_ids = packaged_code_registry(
            self.model_id
        )
        if registry.get("model_revision") != self.model_revision:
            raise ValueError(f"{self.registry_file.name} does not match the pinned model revision")
        if registry.get("tokenizer_sha256") != tokenizer_sha256(self.model_path):
            raise ValueError(f"{self.registry_file.name} does not match the pinned tokenizer")
        if registry.get("codes") != diffusion_answer_code_entries(self.tokenizer):
            raise ValueError(f"{self.registry_file.name} is incomplete or invalid")

        config = json.loads((self.model_path / "config.json").read_text())
        self.canvas_length = int(config["canvas_length"])
        self.vocab_size = int(config["text_config"]["vocab_size"])
        self.base_url = (
            base_url or os.environ.get("SYSTEM_ONE_MLX_VLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = float(os.environ.get("SYSTEM_ONE_MLX_VLM_TIMEOUT", DEFAULT_TIMEOUT))
        self.api_key = os.environ.get("SYSTEM_ONE_MLX_VLM_API_KEY")
        self.transport = transport or self._post
        self.compact = compact

    def output_tokens_for(self, questions):
        """A read-only canvas does not commit generated tokens."""
        return 0

    def evaluate(self, state, questions, template=None):
        """Return one answer-code distribution per slot in a shared canvas."""
        if template is not None:
            raise RequestContractError("custom templates are not supported by DiffusionGemma")
        self._validate_questions(questions)
        if self.compact and len(questions) != 1:
            raise RequestContractError(
                "the compact DiffusionGemma backend requires exactly one question"
            )
        input_ids = build_prompt(
            self.tokenizer,
            state,
            questions,
            self.codes,
            compact=self.compact,
        )
        if len(input_ids) > MAX_PROMPT_TOKENS:
            raise RequestContractError(f"prompt exceeds the {MAX_PROMPT_TOKENS} token limit")
        if len(input_ids) > MAX_TOTAL_INPUT_TOKENS:
            raise RequestContractError(
                f"request exceeds the {MAX_TOTAL_INPUT_TOKENS} input token limit"
            )
        seed_canvas, positions = build_seed_canvas(
            self.tokenizer,
            len(questions),
            self.canvas_length,
            self.vocab_size,
            compact=self.compact,
        )
        slots = [
            {
                "position": position,
                "token_ids": list(self.code_token_ids[: len(labels)]),
            }
            for position, (_, labels) in zip(positions, questions)
        ]
        payload = {
            "model": str(self.model_path),
            "input_ids": input_ids,
            "seed_canvas": seed_canvas,
            "slots": slots,
            "candidate_only": True,
        }
        started = time.perf_counter()
        response = self.transport("/v1/diffusion/reads", payload)
        elapsed_ms = (time.perf_counter() - started) * 1000
        probabilities = self._parse_response(
            response,
            slots,
            len(input_ids),
            str(self.model_path),
        )
        return probabilities, len(input_ids), elapsed_ms

    def _validate_questions(self, questions):
        if not questions:
            raise RequestContractError("at least one question is required")
        for instructions, labels in questions:
            if not labels:
                raise RequestContractError(f"{instructions!r} has no options")
            if len(labels) > len(self.codes):
                raise TooManyOptions(
                    f"{len(labels)} options for {instructions!r}; "
                    f"the engine has {len(self.codes)} single-token answer codes"
                )

    def _parse_response(
        self,
        response,
        slots,
        input_tokens,
        expected_model,
    ):
        try:
            reads = response["reads"]
            usage = response["usage"]
            reported_tokens = usage["prompt_tokens"]
            denoising_steps = usage["denoising_steps"]
            candidate_only = usage["candidate_only"]
            model = response["model"]
        except (KeyError, TypeError) as error:
            raise MlxVlmError("MLX-VLM returned an invalid diffusion read response") from error
        if model != expected_model:
            raise MlxVlmError(f"MLX-VLM served {model!r}, expected pinned model {expected_model!r}")
        if denoising_steps != 1:
            raise MlxVlmError(f"MLX-VLM performed {denoising_steps!r} denoising steps, expected 1")
        if candidate_only is not True:
            raise MlxVlmError("MLX-VLM did not confirm candidate-only scoring")
        if reported_tokens != input_tokens:
            raise MlxVlmError(
                "MLX-VLM token count differs from the pinned tokenizer: "
                f"expected {input_tokens}, got {reported_tokens}"
            )
        if len(reads) != len(slots):
            raise MlxVlmError("MLX-VLM returned incomplete diffusion read slots")

        probabilities = []
        for read, slot in zip(reads, slots):
            try:
                token_ids = read["token_ids"]
                logprobs = read["logprobs"]
                position = read["position"]
            except (KeyError, TypeError) as error:
                raise MlxVlmError("MLX-VLM returned an invalid diffusion read slot") from error
            if position != slot["position"] or token_ids != slot["token_ids"]:
                raise MlxVlmError("MLX-VLM returned a mismatched diffusion read slot")
            if len(logprobs) != len(token_ids) or not all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in logprobs
            ):
                raise MlxVlmError("MLX-VLM returned invalid token log probabilities")
            probabilities.append(restricted_softmax(logprobs))
        return probabilities

    def _post(self, path, payload):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read(4096).decode("utf-8", errors="replace")
            raise MlxVlmError(f"MLX-VLM returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise MlxVlmError(f"could not reach MLX-VLM at {self.base_url}") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise MlxVlmError("MLX-VLM returned invalid JSON") from error
