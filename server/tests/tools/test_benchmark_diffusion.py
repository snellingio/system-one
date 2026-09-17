import math

from tools.benchmark_diffusion import Sample, percentile, run_level


class FakeClient:
    prompt_tokens = 100
    canvas_tokens = 15

    def call(self, nonce):
        return Sample(client_ms=float(nonce + 1), round_trip_ms=float(nonce) + 0.5)


def test_percentile_uses_nearest_rank():
    assert percentile([5, 1, 4, 2, 3], 0.95) == 5


def test_run_level_reports_throughput_and_latency():
    result = run_level(FakeClient(), concurrency=2, request_count=3, nonce_start=0)

    assert result.concurrency == 2
    assert result.requests == 3
    assert result.decisions_per_request == 3
    assert result.prompt_tokens == 100
    assert result.canvas_tokens == 15
    assert math.isclose(result.decisions_per_second, result.requests_per_second * 3)
    assert result.client_median_ms == 2.0
    assert result.client_p95_ms == 3.0
    assert result.round_trip_median_ms == 1.5
    assert result.round_trip_p95_ms == 2.5
