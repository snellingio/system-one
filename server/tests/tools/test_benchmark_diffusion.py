import json
import math

from tools.benchmark_diffusion import DiffusionClient, Sample, percentile, run_level


class FakeClient:
    prompt_tokens = 100
    canvas_tokens = 15
    questions = (("first", ["yes", "no"]),) * 3

    def call(self, nonce):
        return Sample(client_ms=float(nonce + 1), round_trip_ms=float(nonce) + 0.5)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return messages[0]["content"]


class FakeEngine:
    tokenizer = FakeTokenizer()
    codes = ("A", "B")
    canvas_length = 256
    vocab_size = 200
    compact = True


def test_percentile_uses_nearest_rank():
    assert percentile([5, 1, 4, 2, 3], 0.95) == 5


def test_state_is_fixed_between_requests():
    assert json.loads(DiffusionClient._state()) == {
        "ticket": "Everything is down and we have a demo at noon."
    }


def test_client_metadata_uses_the_selected_profile():
    client = DiffusionClient(FakeEngine(), (("first", ("yes", "no")),))

    assert client.canvas_tokens == 1


def test_run_level_reports_throughput_and_latency():
    result = run_level(FakeClient(), concurrency=2, request_count=3, request_index_start=0)

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


def test_run_level_uses_the_client_question_count():
    client = FakeClient()
    client.questions = (("first", ["yes", "no"]), ("second", ["yes", "no"]))

    result = run_level(client, concurrency=1, request_count=1, request_index_start=0)

    assert result.decisions_per_request == 2
    assert math.isclose(result.decisions_per_second, result.requests_per_second * 2)
