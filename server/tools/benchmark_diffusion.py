"""Profile warmed System One DiffusionGemma requests through MLX-VLM.

Start the MLX-VLM server first. Then run this command from ``server/``:

    uv run python -m tools.benchmark_diffusion --json
"""

import argparse
import json
import math
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from system_one_lite.mlx_vlm_diffusion import (
    DEFAULT_BASE_URL,
    MlxVlmDiffusionEngine,
    build_prompt,
    build_seed_canvas,
)

REQUEST_INDEX_START = 0
STATE = "Everything is down and we have a demo at noon."
QUESTIONS = (
    (
        "Does the customer need a reply within the hour?",
        ("yes", "no"),
    ),
    (
        "Which team should handle this request?",
        (
            "outage — Service outage or availability issue",
            "billing — Charges, refunds, or payment methods",
            "feature — Feature request or product suggestion",
        ),
    ),
    (
        "What is the customer's tone?",
        ("calm", "annoyed", "furious"),
    ),
)


@dataclass(frozen=True)
class Sample:
    client_ms: float
    round_trip_ms: float


@dataclass(frozen=True)
class Result:
    concurrency: int
    requests: int
    decisions_per_request: int
    prompt_tokens: int
    canvas_tokens: int
    wall_seconds: float
    requests_per_second: float
    decisions_per_second: float
    client_median_ms: float
    client_p95_ms: float
    round_trip_median_ms: float
    round_trip_p95_ms: float


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


class DiffusionClient:
    def __init__(self, engine, questions=QUESTIONS):
        self.engine = engine
        self.questions = tuple((instructions, list(labels)) for instructions, labels in questions)
        state = self._state()
        self.prompt_tokens = len(
            build_prompt(
                engine.tokenizer,
                state,
                self.questions,
                engine.codes,
                compact=engine.compact,
            )
        )
        canvas, _ = build_seed_canvas(
            engine.tokenizer,
            len(self.questions),
            engine.canvas_length,
            engine.vocab_size,
            compact=engine.compact,
        )
        self.canvas_tokens = len(canvas)

    @staticmethod
    def _state():
        return json.dumps({"ticket": STATE})

    def call(self, _request_index):
        started = time.perf_counter()
        probabilities, prompt_tokens, round_trip_ms = self.engine.evaluate(
            self._state(), self.questions
        )
        client_ms = (time.perf_counter() - started) * 1000
        if prompt_tokens != self.prompt_tokens:
            raise RuntimeError("prompt token count changed during the benchmark")
        if len(probabilities) != len(self.questions):
            raise RuntimeError("engine returned the wrong number of decisions")
        if any(abs(sum(row) - 1.0) > 1e-5 for row in probabilities):
            raise RuntimeError("engine returned an invalid probability distribution")
        return Sample(client_ms=client_ms, round_trip_ms=round_trip_ms)


def run_level(client, concurrency, request_count, request_index_start):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        samples = list(
            executor.map(
                client.call,
                range(request_index_start, request_index_start + request_count),
            )
        )
    wall_seconds = time.perf_counter() - started
    request_rate = request_count / wall_seconds
    client_times = [sample.client_ms for sample in samples]
    round_trip_times = [sample.round_trip_ms for sample in samples]
    return Result(
        concurrency=concurrency,
        requests=request_count,
        decisions_per_request=len(client.questions),
        prompt_tokens=client.prompt_tokens,
        canvas_tokens=client.canvas_tokens,
        wall_seconds=wall_seconds,
        requests_per_second=request_rate,
        decisions_per_second=request_rate * len(client.questions),
        client_median_ms=statistics.median(client_times),
        client_p95_ms=percentile(client_times, 0.95),
        round_trip_median_ms=statistics.median(round_trip_times),
        round_trip_p95_ms=percentile(round_trip_times, 0.95),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--questions", type=int, default=len(QUESTIONS))
    parser.add_argument("--profile", choices=("full", "fast"), default="full")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.requests < 1:
        parser.error("warmup must be non-negative and requests must be positive")
    concurrency = [int(value) for value in args.concurrency.split(",")]
    if not concurrency or any(value < 1 for value in concurrency):
        parser.error("concurrency values must be positive")
    if args.questions < 1:
        parser.error("questions must be positive")
    if args.profile == "fast" and args.questions != 1:
        parser.error("the fast profile requires exactly one question")

    engine = MlxVlmDiffusionEngine(
        base_url=args.base_url,
        compact=args.profile == "fast",
    )
    engine.timeout = args.timeout
    questions = tuple(QUESTIONS[index % len(QUESTIONS)] for index in range(args.questions))
    client = DiffusionClient(engine, questions)
    timed_requests = args.requests * len(concurrency)
    for request_index in range(args.warmup):
        client.call(REQUEST_INDEX_START + timed_requests + request_index)

    results = []
    request_index = REQUEST_INDEX_START
    for level in concurrency:
        results.append(run_level(client, level, args.requests, request_index))
        request_index += args.requests

    payload = {
        "backend": ("mlx-vlm-diffusion-fast" if args.profile == "fast" else "mlx-vlm-diffusion"),
        "model": engine.model_id,
        "model_revision": engine.model_revision,
        "base_url": engine.base_url,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "state": STATE,
        "warmup": args.warmup,
        "results": [asdict(result) for result in results],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(f"model: {engine.model_id}")
    print(
        f"profile: {args.profile}; decisions/request: {len(client.questions)}; "
        f"prompt tokens: {client.prompt_tokens}; canvas tokens: {client.canvas_tokens}"
    )
    print("clients  req/s  decisions/s  client p50  client p95  roundtrip p50  roundtrip p95")
    for result in results:
        print(
            f"{result.concurrency:7}  {result.requests_per_second:5.2f}  "
            f"{result.decisions_per_second:11.2f}  "
            f"{result.client_median_ms:10.1f}  {result.client_p95_ms:10.1f}  "
            f"{result.round_trip_median_ms:13.1f}  {result.round_trip_p95_ms:13.1f}"
        )


if __name__ == "__main__":
    main()
