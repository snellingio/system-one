"""Offline tests for teacher_labels: parsing, permutations, averaging.

No network: the ask function is stubbed with deterministic letter
distributions. Run with the rest of the suite:
    uv run python -m pytest -q
"""

import math

import pytest

from tools.labeling.teacher import (
    build_prompt,
    checkpoint_records,
    completed_ids,
    incomplete_ids,
    label_question,
    label_record,
    letter_probs,
    make_ask,
    question_spec,
    rotations,
    run_manifest,
)


def top(entries):
    return [{"token": t, "logprob": lp} for t, lp in entries]


def test_letter_probs_masks_and_renormalizes():
    probs = letter_probs(
        top([("The", -1.0), (" A", -0.2), ("B", -1.7), (" C", -3.2), ("\n", -2.0)]), {"A", "B", "C"}
    )
    assert set(probs) == {"A", "B", "C"}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["A"] > probs["B"] > probs["C"]
    # renormalized softmax over the letters only, ignoring non-letters
    expected = {
        letter: math.exp(logprob) for letter, logprob in {"A": -0.2, "B": -1.7, "C": -3.2}.items()
    }
    z = sum(expected.values())
    assert abs(probs["A"] - expected["A"] / z) < 1e-9


def test_letter_probs_surface_forms_dedupe_by_max():
    probs = letter_probs(top([("A", -2.0), (" A", -0.5), ("B", -1.0)]), {"A", "B"})
    assert abs(probs["A"] - math.exp(-0.5) / (math.exp(-0.5) + math.exp(-1.0))) < 1e-9


def test_letter_probs_none_when_no_letters():
    assert letter_probs(top([("The", -0.1), ("Yes", -1.0)]), {"A", "B"}) is None


def test_letter_probs_missing_letter_is_zero():
    probs = letter_probs(top([("A", -0.2), ("B", -1.7)]), {"A", "B", "C"})
    assert probs["C"] == 0.0
    assert abs(probs["A"] + probs["B"] - 1.0) < 1e-9
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_letter_probs_none_on_malformed_payload():
    assert letter_probs({"A": -0.1}, {"A"}) is None  # dict, not list
    assert (
        letter_probs([{"token": "A"}, "junk", {"logprob": -1.0}], {"A"}) is None
    )  # missing fields


def test_letter_probs_rejects_nonfinite_logprobs():
    assert letter_probs(top([("A", float("nan"))]), {"A"}) is None
    assert letter_probs(top([("A", float("inf"))]), {"A"}) is None


def test_rotations_distinct_and_start_original():
    assert rotations(["a", "b", "c"], 4) == [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]
    assert rotations(["a", "b"], 10) == [["a", "b"], ["b", "a"]]


def test_question_spec_keys_and_labels():
    keys, labels = question_spec(
        {"type": "choice", "instructions": "q", "criteria": {"x": "first", "y": None}}
    )
    assert keys == ["x", "y"] and labels == ["x — first", "y"]
    keys, labels = question_spec({"type": "score", "instructions": "q", "criteria": ["lo", "hi"]})
    assert keys == ["0", "1"] and labels == ["lo", "hi"]
    keys, labels = question_spec({"type": "noul", "instructions": "q"})
    assert keys == ["yes", "no"] and labels == ["yes", "no"]
    # noul criteria descriptors render exactly as the server does
    keys, labels = question_spec(
        {
            "type": "noul",
            "instructions": "q",
            "criteria": {"true": "time-sensitive", "false": "can wait"},
        }
    )
    assert labels == ["yes — time-sensitive", "no — can wait"]


def test_build_prompt_is_serving_prefix():
    prompt = build_prompt("s", "pick one", ["X", "Y"])
    assert prompt.endswith("Answer: ")
    assert "Question 1: pick one" in prompt
    assert prompt.startswith("State:\ns\n")
    assert "Answer: A" not in prompt


def test_build_prompt_supports_codes_past_z():
    prompt = build_prompt("s", "pick one", [f"option {i}" for i in range(27)])
    assert "\nZ: option 25\nAA: option 26\n" in prompt


def test_teacher_uses_answer_codes_past_z():
    question = {
        "type": "choice",
        "instructions": "pick",
        "criteria": {f"option_{i}": None for i in range(27)},
    }

    def ask(prompt, codes):
        assert codes[-1] == "AA"
        return {code: 1.0 / len(codes) for code in codes}

    dist, ok, total, truncated, responses = label_question(ask, "state", question, permutations=1)
    assert set(dist) == set(question["criteria"])
    assert (ok, total) == (1, 1)
    assert truncated == []
    assert responses == []


def test_make_ask_keeps_top_20_path_for_codes_past_z():
    class Response:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "top_logprobs": [
                                        {"token": " A", "logprob": -0.5},
                                        {"token": " AA", "logprob": -1.0},
                                    ]
                                }
                            ]
                        }
                    },
                ],
                "usage": {"total_tokens": 4},
                "id": "gen-1",
                "model": "teacher-actual",
                "provider": "DeepSeek",
            }

    class Client:
        def post(self, url, json):
            assert json["max_tokens"] == 1
            assert json["top_logprobs"] == 20
            assert json["logprobs"] is True
            assert json["provider"] == {
                "order": ["DeepSeek"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            assert "n" not in json
            return Response()

    ask, stats = make_ask(Client(), "teacher", "DeepSeek")
    probs = ask("prompt", {"A", "AA", "B"})
    assert probs["A"] > probs["AA"] > probs["B"]
    assert probs["B"] == 0.0
    assert stats == {"calls": 1, "tokens": 4}
    assert ask.last_response == {"id": "gen-1", "model": "teacher-actual", "provider": "DeepSeek"}


def test_teacher_requires_a_provider_lock():
    with pytest.raises(ValueError, match="provider lock"):
        make_ask(object(), "teacher")


def test_choice_averages_over_permutations():
    """A stubbed teacher that always favors the FIRST option by 0.8/0.2:
    averaging over both orders must cancel the position bias to ~50/50."""
    question = {"type": "choice", "instructions": "q", "criteria": {"x": "X", "y": "Y"}}

    def ask(prompt, letters):
        first, second = letters
        return {first: 0.8, second: 0.2}

    dist, ok, total, truncated, responses = label_question(ask, "state", question, permutations=4)
    assert abs(dist["x"] - 0.5) < 1e-9 and abs(dist["y"] - 0.5) < 1e-9
    assert (ok, total) == (2, 2)
    assert truncated == []
    assert responses == []


def test_partial_rotations_are_reported():
    """The stub fails whenever X sits in slot A: one of two orders fails,
    so the question is labeled but flagged as partially averaged."""
    question = {"type": "choice", "instructions": "q", "criteria": {"x": "X", "y": "Y"}}

    def ask(prompt, letters):
        if "A: x — X" in prompt:
            return None
        return {letter: 0.5 for letter in letters}

    dist, ok, total, truncated, responses = label_question(ask, "state", question, permutations=2)
    assert (ok, total) == (1, 2)
    assert dist == {"x": 0.5, "y": 0.5}
    assert truncated == []
    assert responses == []


def test_score_and_noul_single_call_keys():
    def ask(prompt, letters):
        return {letter: 1.0 / len(letters) for letter in letters}

    score_q = {"type": "score", "instructions": "q", "criteria": ["lo", "mid", "hi"]}
    dist, ok, total, truncated, responses = label_question(ask, "state", score_q, permutations=4)
    assert set(dist) == {"0", "1", "2"} and (ok, total) == (1, 1)
    assert truncated == []
    assert responses == []

    noul_q = {"type": "noul", "instructions": "q"}
    dist, ok, total, truncated, responses = label_question(ask, "state", noul_q, permutations=4)
    assert set(dist) == {"yes", "no"} and (ok, total) == (1, 1)
    assert truncated == []
    assert responses == []


def test_label_record_reports_failures_and_partials():
    record = {
        "id": "r1",
        "state": "s",
        "questions": {
            "ok": {"type": "noul", "instructions": "fine"},
            "bad": {"type": "noul", "instructions": "nope"},
        },
    }

    def ask(prompt, letters):
        return None if "nope" in prompt else {letter: 0.5 for letter in letters}

    soft, failed, partial, truncated, responses, preserved = label_record(
        record, ask, permutations=4
    )
    assert failed == ["bad"] and partial == {}
    assert soft["ok"] == {"yes": 0.5, "no": 0.5}
    assert soft["bad"] is None
    assert truncated == {}
    assert responses == {} and preserved == []


def test_missing_codes_get_a_small_tail_instead_of_a_hard_zero():
    question = {"type": "choice", "instructions": "q", "criteria": {"x": "X", "y": "Y", "z": "Z"}}

    def ask(prompt, codes):
        return {"A": 0.75, "B": 0.25, "C": 0.0}

    dist, _, _, truncated, responses = label_question(
        ask, "state", question, permutations=1, tail_mass=0.03
    )
    assert dist == {"x": 0.7275, "y": 0.2425, "z": 0.03}
    assert truncated[0]["missing"] == ["z"]
    assert responses == []


def test_existing_soft_target_is_preserved_by_default():
    record = {
        "id": "r1",
        "state": "s",
        "questions": {"flag": {"type": "noul", "instructions": "q"}},
        "soft_targets": {"flag": {"yes": 0.8, "no": 0.2}},
    }

    def ask(prompt, codes):
        raise AssertionError("existing target should not be replaced")

    soft, failed, partial, truncated, responses, preserved = label_record(
        record, ask, permutations=1
    )
    assert soft == record["soft_targets"]
    assert failed == [] and partial == {} and truncated == {} and responses == {}
    assert preserved == ["flag"]


def test_completed_ids_reads_a_valid_checkpoint(tmp_path):
    checkpoint = tmp_path / "labels.jsonl.tmp"
    checkpoint.write_text('{"id": "a"}\n{"id": "b"}\n')
    assert completed_ids(checkpoint) == {"a", "b"}


def test_completed_ids_discards_a_partial_final_checkpoint_line(tmp_path):
    checkpoint = tmp_path / "labels.jsonl.tmp"
    checkpoint.write_text('{"id": "a"}\n{"id":')
    assert completed_ids(checkpoint) == {"a"}


def test_completed_ids_rejects_a_partial_middle_checkpoint_line(tmp_path):
    checkpoint = tmp_path / "labels.jsonl.tmp"
    checkpoint.write_text('{"id":\n{"id": "a"}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        completed_ids(checkpoint)


def test_checkpoint_uses_the_latest_saved_record(tmp_path):
    checkpoint = tmp_path / "labels.jsonl.tmp"
    checkpoint.write_text('{"id": "a", "version": 1}\n{"id": "a", "version": 2}\n')
    assert checkpoint_records(checkpoint)["a"]["version"] == 2


def test_incomplete_ids_retry_failed_or_partial_records():
    rows = [
        {"id": "complete", "label_quality": {}},
        {"id": "failed", "label_quality": {"failed": ["q"]}},
        {"id": "partial", "label_quality": {"partial": {"q": "1/2"}}},
    ]
    assert incomplete_ids(rows) == {"failed", "partial"}


def test_retry_keeps_successful_questions_from_a_checkpoint():
    record = {
        "id": "r1",
        "state": "s",
        "questions": {
            "saved": {"type": "noul", "instructions": "saved"},
            "retry": {"type": "noul", "instructions": "retry"},
        },
        "soft_targets": {"saved": {"yes": 0.8, "no": 0.2}, "retry": None},
        "label_quality": {"failed": ["retry"]},
    }

    def ask(prompt, codes):
        assert "Question 1: retry" in prompt
        return {"A": 0.6, "B": 0.4}

    soft, failed, _, _, _, preserved = label_record(record, ask, permutations=1)
    assert soft["saved"] == {"yes": 0.8, "no": 0.2}
    assert soft["retry"] == {"yes": 0.6, "no": 0.4}
    assert failed == [] and preserved == ["saved"]


def test_retry_relabels_a_partial_question_with_a_saved_target():
    record = {
        "id": "r1",
        "state": "s",
        "questions": {"retry": {"type": "noul", "instructions": "retry"}},
        "soft_targets": {"retry": {"yes": 0.8, "no": 0.2}},
        "label_quality": {"partial": {"retry": "1/2"}},
    }

    def ask(prompt, codes):
        assert "Question 1: retry" in prompt
        return {"A": 0.6, "B": 0.4}

    soft, failed, partial, _, _, preserved = label_record(record, ask, permutations=1)
    assert soft["retry"] == {"yes": 0.6, "no": 0.4}
    assert failed == [] and partial == {} and preserved == []


def test_run_manifest_binds_input_and_label_settings():
    records = [{"id": "r1", "state": "s"}]
    manifest = run_manifest(records, "model", "provider", 4, 0.02, False)
    assert manifest == run_manifest(records, "model", "provider", 4, 0.02, False)
    assert manifest != run_manifest(records, "model", "provider", 5, 0.02, False)
    assert manifest != run_manifest(
        [{"id": "r1", "state": "changed"}], "model", "provider", 4, 0.02, False
    )
