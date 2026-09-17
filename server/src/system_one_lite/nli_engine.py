"""Batched natural-language-inference backend for System One requests."""

import math
import os
import time

from .errors import RequestContractError, TooManyOptions

DEFAULT_NLI_MODEL = "AlexWortega/openjev"
DEFAULT_NLI_SUBFOLDER = "qwen3.5-4b-nli"
DEFAULT_NLI_REVISION = "8c9db06441316f3fff7a68feb7da3fea79a5eff7"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 4096
MAX_NLI_CANDIDATES = 4096
PREMISE_TEMPLATE = "State:\n{state}\n\nQuestion:\n{instructions}"
HYPOTHESIS_TEMPLATE = "The correct answer is: {label}"


def nli_pairs(state, questions):
    """Flatten questions into premise-hypothesis pairs and group sizes."""
    if not questions:
        raise RequestContractError("at least one question is required")

    pairs = []
    group_sizes = []
    for instructions, labels in questions:
        if not labels:
            raise RequestContractError(f"{instructions!r} has no options")
        premise = PREMISE_TEMPLATE.format(state=state.strip(), instructions=instructions.strip())
        group_sizes.append(len(labels))
        pairs.extend((premise, HYPOTHESIS_TEMPLATE.format(label=label.strip())) for label in labels)

    if len(pairs) > MAX_NLI_CANDIDATES:
        raise TooManyOptions(
            f"request has {len(pairs)} candidates; the NLI backend allows {MAX_NLI_CANDIDATES}"
        )
    return pairs, group_sizes


def normalize_entailment(values, temperature=1.0):
    """Turn independent entailment probabilities into one choice distribution."""
    if not values:
        raise RequestContractError("an NLI question has no candidate scores")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("NLI temperature must be finite and positive")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("NLI entailment probabilities must be finite and in [0, 1]")

    log_weights = [math.log(max(value, 1e-12)) / temperature for value in values]
    peak = max(log_weights)
    weights = [math.exp(value - peak) for value in log_weights]
    total = sum(weights)
    return [value / total for value in weights]


def select_device(torch, requested=None):
    """Choose an available PyTorch device unless the caller chose one."""
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(torch, device):
    device_type = torch.device(device).type
    if device_type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device_type == "mps":
        return torch.float16
    return torch.bfloat16


def common_token_prefix(rows):
    """Return the number of leading token IDs shared by every row."""
    if not rows:
        return 0
    limit = min(len(row) for row in rows)
    index = 0
    while index < limit and all(row[index] == rows[0][index] for row in rows[1:]):
        index += 1
    return index


def resolve_nli_revision(model_id, revision=None):
    """Pin the built-in model and require a pin for custom repositories."""
    if revision:
        return revision
    if model_id == DEFAULT_NLI_MODEL:
        return DEFAULT_NLI_REVISION
    raise ValueError("SYSTEM_ONE_NLI_REVISION is required for a custom NLI model")


def resolve_entailment_index(config):
    """Find the entailment class without assuming one label order."""
    for label, index in getattr(config, "label2id", {}).items():
        if str(label).strip().casefold() == "entailment":
            return int(index)
    raise ValueError("NLI model config must define an entailment label")


def load_nli_assets(auto_tokenizer, auto_model, model_id, subfolder, revision, dtype):
    """Load the tokenizer and classifier from the same pinned Hub revision."""
    tokenizer = auto_tokenizer.from_pretrained(
        model_id,
        subfolder=subfolder,
        revision=revision,
    )
    model = auto_model.from_pretrained(
        model_id,
        subfolder=subfolder,
        revision=revision,
        dtype=dtype,
    )
    return tokenizer, model


class HuggingFaceNLIScorer:
    """Load OpenJEV and return one entailment probability per text pair."""

    def __init__(
        self,
        model_id=DEFAULT_NLI_MODEL,
        subfolder=DEFAULT_NLI_SUBFOLDER,
        revision=None,
        device=None,
        batch_size=DEFAULT_BATCH_SIZE,
        max_length=DEFAULT_MAX_LENGTH,
        prefix_cache=False,
    ):
        revision = resolve_nli_revision(model_id, revision)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "the NLI backend needs the nli dependency group; run uv sync --group nli"
            ) from error

        if batch_size < 1 or max_length < 1:
            raise ValueError("NLI batch size and maximum length must be positive")
        self.torch = torch
        self.model_id = model_id
        self.subfolder = subfolder
        self.revision = revision
        self.device = select_device(torch, device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.prefix_cache = prefix_cache
        self.tokenizer, self.model = load_nli_assets(
            AutoTokenizer,
            AutoModelForSequenceClassification,
            model_id,
            subfolder,
            self.revision,
            select_dtype(torch, self.device),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        text_config = self.model.config.get_text_config()
        if text_config.pad_token_id is None:
            text_config.pad_token_id = self.tokenizer.pad_token_id
        self.entailment_index = resolve_entailment_index(self.model.config)
        self.model.to(self.device).eval()
        self.template = self.model.config.nli_template

    def score_pairs(self, pairs):
        """Return entailment probabilities and the repeated-pair token count."""
        probabilities = []
        input_tokens = 0
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                group = pairs[start : start + self.batch_size]
                texts = [
                    self.template.format(premise=premise, hypothesis=hypothesis)
                    for premise, hypothesis in group
                ]
                encoded = self.tokenizer(
                    texts,
                    truncation=False,
                    padding=True,
                    return_tensors="pt",
                )
                lengths = encoded["attention_mask"].sum(dim=1)
                longest = int(lengths.max().item())
                if longest > self.max_length:
                    raise RequestContractError(
                        f"NLI pair has {longest} tokens; limit is {self.max_length}"
                    )
                input_tokens += int(lengths.sum().item())
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits.float()
                values = torch.softmax(logits, dim=-1)[:, self.entailment_index]
                probabilities.extend(values.cpu().tolist())
        return probabilities, input_tokens

    def score_grouped_pairs(self, pairs, group_sizes):
        """Score each question with optional exact shared-prefix reuse."""
        if not self.prefix_cache:
            return self.score_pairs(pairs)

        probabilities = []
        input_tokens = 0
        offset = 0
        for size in group_sizes:
            group = pairs[offset : offset + size]
            for start in range(0, len(group), self.batch_size):
                values, tokens = self._score_cached_group(group[start : start + self.batch_size])
                probabilities.extend(values)
                input_tokens += tokens
            offset += size
        return probabilities, input_tokens

    def _score_cached_group(self, pairs):
        """Run a shared causal prefix once, then batch the option suffixes."""
        if len(pairs) < 2:
            return self.score_pairs(pairs)

        texts = [
            self.template.format(premise=premise, hypothesis=hypothesis)
            for premise, hypothesis in pairs
        ]
        rows = [self.tokenizer.encode(text, truncation=False) for text in texts]
        longest = max(len(row) for row in rows)
        if longest > self.max_length:
            raise RequestContractError(f"NLI pair has {longest} tokens; limit is {self.max_length}")
        shared = common_token_prefix(rows)
        shared = min(shared, min(len(row) - 1 for row in rows))
        if shared <= 0:
            return self.score_pairs(pairs)

        torch = self.torch
        prefix_ids = torch.tensor([rows[0][:shared]], dtype=torch.long, device=self.device)
        prefix_mask = torch.ones_like(prefix_ids)
        suffixes = [row[shared:] for row in rows]
        suffix_length = max(len(row) for row in suffixes)
        suffix_ids = torch.full(
            (len(rows), suffix_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        suffix_mask = torch.zeros_like(suffix_ids)
        for index, row in enumerate(suffixes):
            suffix_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
            suffix_mask[index, : len(row)] = 1
        full_mask = torch.cat(
            [
                torch.ones(
                    (len(rows), shared),
                    dtype=suffix_mask.dtype,
                    device=self.device,
                ),
                suffix_mask,
            ],
            dim=1,
        )

        with torch.inference_mode():
            prefix_output = self.model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                use_cache=True,
            )
            cache = prefix_output.past_key_values
            if cache is None:
                raise RuntimeError("OpenJEV did not return a prefix cache")
            branch_indices = torch.zeros(len(rows), dtype=torch.long, device=self.device)
            cache.reorder_cache(branch_indices)
            logits = self.model(
                input_ids=suffix_ids,
                attention_mask=full_mask,
                past_key_values=cache,
                use_cache=False,
            ).logits.float()
            values = torch.softmax(logits, dim=-1)[:, self.entailment_index]
        processed_tokens = shared + sum(len(row) for row in suffixes)
        return values.cpu().tolist(), processed_tokens


class NLIEngine:
    """Map batched OpenJEV scores onto the existing decision-engine contract."""

    def __init__(self, model_id=None, scorer=None, temperature=None):
        model_id = model_id or DEFAULT_NLI_MODEL
        if scorer is None:
            scorer = HuggingFaceNLIScorer(
                model_id=model_id,
                subfolder=os.environ.get("SYSTEM_ONE_NLI_SUBFOLDER", DEFAULT_NLI_SUBFOLDER),
                revision=os.environ.get("SYSTEM_ONE_NLI_REVISION"),
                device=os.environ.get("SYSTEM_ONE_NLI_DEVICE", "auto"),
                batch_size=int(os.environ.get("SYSTEM_ONE_NLI_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
                max_length=int(os.environ.get("SYSTEM_ONE_NLI_MAX_LENGTH", DEFAULT_MAX_LENGTH)),
                prefix_cache=os.environ.get("SYSTEM_ONE_NLI_PREFIX_CACHE", "0") == "1",
            )
        self.scorer = scorer
        scorer_model_id = getattr(scorer, "model_id", model_id)
        scorer_subfolder = getattr(scorer, "subfolder", None)
        scorer_revision = getattr(scorer, "revision", None)
        if scorer_subfolder and scorer_revision:
            self.model_id = f"{scorer_model_id}:{scorer_subfolder}@{scorer_revision}"
        else:
            self.model_id = scorer_model_id
        self.temperature = float(
            temperature
            if temperature is not None
            else os.environ.get("SYSTEM_ONE_NLI_TEMPERATURE", "1.0")
        )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("NLI temperature must be finite and positive")

    def evaluate(self, state, questions, template=None):
        if template is not None:
            raise RequestContractError("the NLI backend does not accept answer-slot templates")
        started = time.perf_counter()
        pairs, group_sizes = nli_pairs(state, questions)
        grouped_scorer = getattr(self.scorer, "score_grouped_pairs", None)
        if grouped_scorer is None:
            entailment, input_tokens = self.scorer.score_pairs(pairs)
        else:
            entailment, input_tokens = grouped_scorer(pairs, group_sizes)
        if len(entailment) != len(pairs):
            raise RuntimeError(
                f"NLI scorer returned {len(entailment)} scores for {len(pairs)} candidates"
            )

        results = []
        offset = 0
        for size in group_sizes:
            results.append(
                normalize_entailment(
                    entailment[offset : offset + size],
                    temperature=self.temperature,
                )
            )
            offset += size
        elapsed_ms = (time.perf_counter() - started) * 1000
        return results, input_tokens, elapsed_ms
