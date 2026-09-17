"""System One backend for a separate vllm-metal OpenAI server."""

import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mlx_lm.utils import load_tokenizer

from .engine import (
    MAX_PROMPT_TOKENS,
    MAX_TOTAL_INPUT_TOKENS,
    TEMPERATURE,
    RequestContractError,
    TooManyOptions,
    common_prefix_len,
    configured_model,
    load_code_registry,
    packaged_code_registry,
    resolve_model_snapshot,
)
from .prompts import chat_filled

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 120.0
MAX_LOGPROB_TOKEN_IDS = 128


class VllmMetalError(RuntimeError):
    """The vllm-metal server could not complete a structured read."""


def restricted_softmax(logprobs, temperature=TEMPERATURE):
    """Renormalize raw vocabulary log-probabilities over allowed codes."""
    scaled = [value / temperature for value in logprobs]
    peak = max(scaled)
    weights = [math.exp(value - peak) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


class VllmMetalEngine:
    """Read answer-code logits through vllm-metal's completions API.

    vllm-metal does not yet implement DiffusionGemma or seeded canvas reads.
    This backend sends System One's exact prompt token IDs in one batch. Each
    prompt stays independent. Requested answer codes are split into exact
    groups of 128 because that is vLLM's per-read limit.
    """

    def __init__(self, model_id=None, base_url=None, transport=None, tokenizer=None):
        self.model_id = configured_model(model_id)
        if tokenizer is None:
            model_path, model_revision = resolve_model_snapshot(self.model_id, tokenizer_only=True)
            self.tokenizer = load_tokenizer(model_path)
            self.registry_file, self.codes, self.code_token_ids = load_code_registry(
                self.model_id,
                self.tokenizer,
                model_path,
                model_revision,
            )
        else:
            self.tokenizer = tokenizer
            self.registry_file, _, self.codes, self.code_token_ids = packaged_code_registry(
                self.model_id
            )
        self.base_url = (
            base_url or os.environ.get("SYSTEM_ONE_VLLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = float(os.environ.get("SYSTEM_ONE_VLLM_TIMEOUT", DEFAULT_TIMEOUT))
        self.api_key = os.environ.get("SYSTEM_ONE_VLLM_API_KEY")
        self.transport = transport or self._post

    def output_tokens_for(self, questions):
        """Count the internal one-token probes needed for this request."""
        return sum(math.ceil(len(labels) / MAX_LOGPROB_TOKEN_IDS) for _, labels in questions)

    def evaluate(self, state, questions, template=None):
        """Return one exact code distribution per independent question."""
        prompts, input_tokens = self._prepare(state, questions, template)
        started = time.perf_counter()
        self._verify_server_tokenizer()
        probabilities, reported_input_tokens = self._read_batches(prompts, questions)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if reported_input_tokens != input_tokens:
            raise VllmMetalError(
                "vllm-metal token count differs from the pinned tokenizer: "
                f"expected {input_tokens}, got {reported_input_tokens}"
            )
        return probabilities, input_tokens, elapsed_ms

    def _prepare(self, state, questions, template):
        if template is not None:
            raise RequestContractError("custom templates are not supported by vllm-metal")
        if not questions:
            raise RequestContractError("at least one question is required")

        prompts = []
        input_tokens = 0
        for instructions, labels in questions:
            if not labels:
                raise RequestContractError(f"{instructions!r} has no options")
            if len(labels) > len(self.codes):
                raise TooManyOptions(
                    f"{len(labels)} options for {instructions!r}; "
                    f"the engine has {len(self.codes)} single-token answer codes"
                )
            prompt = self._prompt_token_ids(state, instructions, labels)
            if len(prompt) > MAX_PROMPT_TOKENS:
                raise RequestContractError(
                    f"prompt for {instructions!r} exceeds the {MAX_PROMPT_TOKENS} token limit"
                )
            input_tokens += len(prompt) * math.ceil(len(labels) / MAX_LOGPROB_TOKEN_IDS)
            if input_tokens > MAX_TOTAL_INPUT_TOKENS:
                raise RequestContractError(
                    f"request exceeds the {MAX_TOTAL_INPUT_TOKENS} input token limit"
                )
            prompts.append(prompt)
        return prompts, input_tokens

    def _prompt_token_ids(self, state, instructions, labels):
        text, marks = chat_filled(
            self.tokenizer,
            state,
            [(instructions, labels)],
            codes=self.codes,
        )
        prefix = text[: marks[0]]
        base = self.tokenizer.encode(prefix)
        candidate = self.tokenizer.encode(prefix + self.codes[0])
        cut = common_prefix_len(candidate, base)
        full = self.tokenizer.encode(text)
        if len(candidate) - cut != 1 or candidate[: cut + 1] != full[: cut + 1]:
            raise RequestContractError(f"slot for {instructions!r} is not a single token")
        return full[:cut]

    def _verify_server_tokenizer(self):
        token_ids = list(self.code_token_ids)
        response = self.transport(
            "/detokenize",
            {"model": self.model_id, "tokens": token_ids},
        )
        try:
            remote_text = response["prompt"]
        except (KeyError, TypeError) as error:
            raise VllmMetalError("vllm-metal returned an invalid detokenize response") from error
        if remote_text != self.tokenizer.decode(token_ids):
            raise VllmMetalError("vllm-metal does not match the pinned tokenizer")

    def _read_batches(self, prompts, questions):
        max_options = max(len(labels) for _, labels in questions)
        scores = [{} for _ in questions]
        prompt_tokens = 0
        for start in range(0, max_options, MAX_LOGPROB_TOKEN_IDS):
            end = min(start + MAX_LOGPROB_TOKEN_IDS, max_options)
            active = [
                (index, prompt)
                for index, (prompt, (_, labels)) in enumerate(zip(prompts, questions))
                if len(labels) > start
            ]
            token_ids = list(self.code_token_ids[start:end])
            payload = {
                "model": self.model_id,
                "prompt": [prompt for _, prompt in active],
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 0,
                "logprob_token_ids": token_ids,
                "return_tokens_as_token_ids": True,
            }
            response = self.transport("/v1/completions", payload)
            try:
                choices = sorted(response["choices"], key=lambda choice: choice["index"])
                response_model = response["model"]
                batch_prompt_tokens = response["usage"]["prompt_tokens"]
            except (KeyError, TypeError, ValueError) as error:
                raise VllmMetalError("vllm-metal returned an invalid logprobs response") from error
            if response_model != self.model_id:
                raise VllmMetalError(
                    f"vllm-metal served {response_model!r}, expected {self.model_id!r}"
                )
            if (
                isinstance(batch_prompt_tokens, bool)
                or not isinstance(batch_prompt_tokens, int)
                or batch_prompt_tokens < 0
            ):
                raise VllmMetalError("vllm-metal returned an invalid prompt token count")
            prompt_tokens += batch_prompt_tokens
            if [choice["index"] for choice in choices] != list(range(len(active))):
                raise VllmMetalError("vllm-metal returned incomplete batch choices")
            for choice, (question_index, _) in zip(choices, active):
                option_count = len(questions[question_index][1])
                requested_ids = list(self.code_token_ids[start : min(end, option_count)])
                scores[question_index].update(self._choice_logprobs(choice, requested_ids))

        probabilities = [
            restricted_softmax([score[token_id] for token_id in self.code_token_ids[: len(labels)]])
            for score, (_, labels) in zip(scores, questions)
        ]
        return probabilities, prompt_tokens

    def _choice_logprobs(self, choice, token_ids):
        try:
            top_logprobs = choice["logprobs"]["top_logprobs"][0]
            if not isinstance(top_logprobs, dict):
                raise TypeError
            by_token_id = {
                int(token.removeprefix("token_id:")): logprob
                for token, logprob in top_logprobs.items()
                if token.startswith("token_id:")
            }
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            raise VllmMetalError("vllm-metal returned invalid token logprobs") from error
        missing = [token_id for token_id in token_ids if token_id not in by_token_id]
        if missing:
            raise VllmMetalError(
                "vllm-metal omitted requested answer-code token IDs: "
                + ", ".join(map(str, missing))
            )
        if any(
            isinstance(by_token_id[token_id], bool)
            or not isinstance(by_token_id[token_id], (int, float))
            or not math.isfinite(by_token_id[token_id])
            for token_id in token_ids
        ):
            raise VllmMetalError("vllm-metal returned invalid token logprobs")
        return {token_id: by_token_id[token_id] for token_id in token_ids}

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
            raise VllmMetalError(f"vllm-metal returned HTTP {error.code}: {detail}") from error
        except (URLError, TimeoutError) as error:
            raise VllmMetalError(f"could not reach vllm-metal at {self.base_url}") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise VllmMetalError("vllm-metal returned invalid JSON") from error
