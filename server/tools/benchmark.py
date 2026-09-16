"""Benchmark warmed System One engine requests with one and three questions.

Run from server/:
    uv run python -m tools.benchmark

The three-question case includes the same first question as the one-question
case. This measures the extra cost of independent question prompts for the
same state, without model-load or Metal-compilation time.
"""

import argparse
import json
import statistics
from dataclasses import asdict, dataclass

from system_one_lite.engine import DEFAULT_MODEL, Engine

STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)
QUESTIONS = (
    (
        "Which team should handle this?",
        [
            "Payment or subscription issues",
            "Bugs or integration problems",
            "Pricing or account questions",
        ],
    ),
    ("Does this convey urgency?", ["yes", "no"]),
    (
        "How frustrated is the customer?",
        ["Calm, just stating facts", "Frustrated but civil", "Very angry"],
    ),
)


@dataclass(frozen=True)
class Result:
    questions: int
    repeats: int
    input_tokens: int
    median_ms: float
    min_ms: float
    max_ms: float
    prompt_median_ms: float
    inference_median_ms: float
    questions_per_second: float
    prompt_tokens_per_second: float


def measure(engine, question_count):
    """Run one request and verify its returned shape."""
    if question_count not in (1, 3):
        raise ValueError("question_count must be 1 or 3")
    questions = QUESTIONS[:question_count]
    if hasattr(engine, "evaluate_profiled"):
        probabilities, tokens, elapsed_ms, prompt_ms, inference_ms = engine.evaluate_profiled(
            STATE, questions
        )
    else:
        probabilities, tokens, elapsed_ms = engine.evaluate(STATE, questions)
        prompt_ms, inference_ms = 0.0, elapsed_ms
    if len(probabilities) != question_count:
        raise RuntimeError("engine returned the wrong number of answers")
    if any(abs(sum(p) - 1.0) > 1e-5 for p in probabilities):
        raise RuntimeError("engine returned an invalid probability distribution")
    return tokens, elapsed_ms, prompt_ms, inference_ms


def summarize(question_count, repeats, input_tokens, samples, prompt_samples, inference_samples):
    """Turn repeated measurements of one request shape into a result."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    median_ms = statistics.median(samples)
    seconds = median_ms / 1000
    return Result(
        questions=question_count,
        repeats=repeats,
        input_tokens=input_tokens,
        median_ms=median_ms,
        min_ms=min(samples),
        max_ms=max(samples),
        prompt_median_ms=statistics.median(prompt_samples),
        inference_median_ms=statistics.median(inference_samples),
        questions_per_second=question_count / seconds,
        prompt_tokens_per_second=input_tokens / seconds,
    )


def run_case(engine, question_count, repeats):
    """Run one warmed request shape repeatedly and return its timing summary."""
    samples = []
    prompt_samples = []
    inference_samples = []
    input_tokens = None
    for _ in range(repeats):
        tokens, elapsed_ms, prompt_ms, inference_ms = measure(engine, question_count)
        if input_tokens is None:
            input_tokens = tokens
        elif tokens != input_tokens:
            raise RuntimeError("input token count changed between identical requests")
        samples.append(elapsed_ms)
        prompt_samples.append(prompt_ms)
        inference_samples.append(inference_ms)
    return summarize(
        question_count, repeats, input_tokens, samples, prompt_samples, inference_samples
    )


def run_cases(engine, repeats):
    """Alternate one- and three-question requests to reduce timing drift."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    samples = {1: [], 3: []}
    prompt_samples = {1: [], 3: []}
    inference_samples = {1: [], 3: []}
    token_counts = {}
    for index in range(repeats):
        counts = (1, 3) if index % 2 == 0 else (3, 1)
        for count in counts:
            tokens, elapsed_ms, prompt_ms, inference_ms = measure(engine, count)
            if count in token_counts and tokens != token_counts[count]:
                raise RuntimeError("input token count changed between identical requests")
            token_counts[count] = tokens
            samples[count].append(elapsed_ms)
            prompt_samples[count].append(prompt_ms)
            inference_samples[count].append(inference_ms)
    return [
        summarize(
            count,
            repeats,
            token_counts[count],
            samples[count],
            prompt_samples[count],
            inference_samples[count],
        )
        for count in (1, 3)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be zero or greater")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    engine = Engine(args.model)
    for _ in range(args.warmup):
        for count in (1, 3):
            measure(engine, count)
    results = run_cases(engine, args.repeats)

    if args.json:
        print(
            json.dumps(
                {"model": engine.model_id, "results": [asdict(r) for r in results]}, indent=2
            )
        )
        return

    print(f"model: {engine.model_id}")
    print("questions  tokens  total ms  prompt ms  model ms  questions/s  tokens/s")
    for result in results:
        print(
            f"{result.questions:9}  {result.input_tokens:6}  "
            f"{result.median_ms:8.1f}  {result.prompt_median_ms:9.1f}  "
            f"{result.inference_median_ms:8.1f}  "
            f"{result.questions_per_second:11.2f}  "
            f"{result.prompt_tokens_per_second:8.1f}"
        )


if __name__ == "__main__":
    main()
