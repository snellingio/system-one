"""Unit tests for the benchmark summary without loading an MLX model."""

import pytest

from tools.benchmark import QUESTIONS, STATE, run_case, run_cases


class FakeEngine:
    def __init__(self):
        self.calls = []

    def evaluate(self, state, questions):
        self.calls.append((state, questions))
        probabilities = [[0.25, 0.75] for _ in questions]
        return probabilities, 100 * len(questions), 10.0 * len(questions)


def test_run_case_reports_one_question_metrics():
    result = run_case(FakeEngine(), 1, repeats=3)

    assert result.questions == 1
    assert result.repeats == 3
    assert result.input_tokens == 100
    assert result.median_ms == result.min_ms == result.max_ms == 10.0
    assert result.prompt_median_ms == 0.0
    assert result.inference_median_ms == 10.0
    assert result.questions_per_second == 100.0
    assert result.prompt_tokens_per_second == 10_000.0


def test_run_case_uses_all_three_standard_questions():
    engine = FakeEngine()
    result = run_case(engine, 3, repeats=1)

    assert result.input_tokens == 300
    assert engine.calls == [(STATE, QUESTIONS)]


def test_run_cases_alternates_request_shapes():
    engine = FakeEngine()
    results = run_cases(engine, repeats=2)

    assert [result.questions for result in results] == [1, 3]
    assert [len(questions) for _, questions in engine.calls] == [1, 3, 3, 1]


@pytest.mark.parametrize("question_count,repeats", [(2, 1), (1, 0)])
def test_run_case_rejects_unsupported_counts(question_count, repeats):
    with pytest.raises(ValueError):
        run_case(FakeEngine(), question_count, repeats)
