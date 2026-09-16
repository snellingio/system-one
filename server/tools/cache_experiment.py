"""Test shared-prefix KV reuse against the safe full-prompt path.

This is not used by the server. It exits with an error unless every masked
probability matches the uncached reference within the requested tolerance.

Run from server/:
    uv run python -m tools.cache_experiment
"""

import argparse
import copy
import json
import statistics
import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from system_one_lite.engine import DEFAULT_MODEL, TEMPERATURE, Engine
from tools.benchmark import QUESTIONS, STATE


def shared_token_prefix(sequences):
    """Return the token prefix shared by every non-empty sequence."""
    if not sequences or any(not sequence for sequence in sequences):
        raise ValueError("all sequences must contain tokens")
    limit = min(len(sequence) for sequence in sequences)
    end = 0
    while end < limit and all(sequence[end] == sequences[0][end] for sequence in sequences[1:]):
        end += 1
    return sequences[0][:end]


def repeated_cache(cache, batch_size):
    """Copy a transformer KV cache across a batch dimension."""
    result = []
    for layer in cache:
        copied = copy.copy(layer)
        if not hasattr(layer, "keys") or layer.keys is None:
            raise ValueError("the model cache is not a standard KV cache")
        copied.keys = mx.repeat(layer.keys, batch_size, axis=0)
        copied.values = mx.repeat(layer.values, batch_size, axis=0)
        result.append(copied)
    return result


def cached_batch(engine, jobs):
    """Evaluate prepared jobs with one shared prefill and one suffix batch."""
    full_sequences = [job[0] for job in jobs]
    common = shared_token_prefix(full_sequences)
    suffixes = [sequence[len(common) :] for sequence in full_sequences]
    if not common or any(not suffix for suffix in suffixes):
        raise ValueError("jobs do not have a usable shared prefix")

    pad_id = engine.tokenizer.pad_token_id or 0
    max_suffix = max(len(suffix) for suffix in suffixes)
    padded = [suffix + [pad_id] * (max_suffix - len(suffix)) for suffix in suffixes]

    t0 = time.perf_counter()
    cache = make_prompt_cache(engine.model)
    prefill_logits = engine.model(mx.array([common]), cache=cache)
    mx.eval(prefill_logits)
    t1 = time.perf_counter()

    batch_cache = repeated_cache(cache, len(jobs))
    suffix_logits = engine.model(mx.array(padded), cache=batch_cache)
    mx.eval(suffix_logits)
    t2 = time.perf_counter()

    probabilities = []
    for row, (_, slot, mask) in enumerate(jobs):
        suffix_slot = slot - len(common)
        if suffix_slot < 0 or suffix_slot >= len(suffixes[row]):
            raise ValueError("answer slot is outside the real suffix")
        logits = suffix_logits[row, suffix_slot].astype(mx.float32)[mx.array(mask)]
        probs = mx.softmax(logits / TEMPERATURE)
        mx.eval(probs)
        probabilities.append(probs.tolist())

    return probabilities, {
        "common_tokens": len(common),
        "max_suffix_tokens": max_suffix,
        "prefill_ms": (t1 - t0) * 1000,
        "suffix_batch_ms": (t2 - t1) * 1000,
        "total_ms": (t2 - t0) * 1000,
    }


def compare(engine, repeats, tolerance):
    reference, _, reference_ms, _, _ = engine.evaluate_profiled(STATE, QUESTIONS)
    jobs, serial_tokens = engine.prepare(STATE, QUESTIONS)

    samples = []
    cached = None
    metrics = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        jobs, _ = engine.prepare(STATE, QUESTIONS)
        cached, metrics = cached_batch(engine, jobs)
        samples.append((time.perf_counter() - t0) * 1000)

    differences = [
        abs(left - right)
        for expected, actual in zip(reference, cached)
        for left, right in zip(expected, actual)
    ]
    winners_match = all(
        max(range(len(expected)), key=expected.__getitem__)
        == max(range(len(actual)), key=actual.__getitem__)
        for expected, actual in zip(reference, cached)
    )
    max_difference = max(differences, default=0.0)
    return {
        "model": engine.model_id,
        "questions": len(QUESTIONS),
        "serial_input_tokens": serial_tokens,
        "shared_prefix_tokens": metrics["common_tokens"],
        "max_suffix_tokens": metrics["max_suffix_tokens"],
        "serial_ms": reference_ms,
        "cached_median_ms": statistics.median(samples),
        "max_probability_difference": max_difference,
        "winners_match": winners_match,
        "tolerance": tolerance,
        "parity": winners_match and max_difference <= tolerance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.tolerance < 0:
        parser.error("--tolerance must be zero or greater")

    result = compare(Engine(args.model), args.repeats, args.tolerance)
    print(json.dumps(result, indent=2))
    if not result["parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
