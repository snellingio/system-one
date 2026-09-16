"""Offline tests for make_train: variant expansion and shuffling."""

import json

import pytest

from tools.labeling.make_training_set import expand_record, question_entries, source_split

RECORD = {
    "id": "r1",
    "state": "a state",
    "questions": {
        "pick": {
            "type": "choice",
            "instructions": "pick one",
            "criteria": {"x": "X", "y": "Y", "z": None},
        },
        "rate": {"type": "score", "instructions": "rate it", "criteria": ["lo", "hi"]},
        "flag": {"type": "noul", "instructions": "is it flagged"},
        "bad": {"type": "noul", "instructions": "teacher failed this"},
    },
    "soft_targets": {
        "pick": {"x": 0.7, "y": 0.2, "z": 0.1},
        "rate": {"0": 0.3, "1": 0.7},
        "flag": {"yes": 0.9, "no": 0.1},
        "bad": None,
    },
}


def test_entries_skip_failed_and_align_targets():
    entries, dropped = question_entries(RECORD)
    assert ("bad", "no teacher labels") in dropped
    assert [d for d in dropped] == [("bad", "no teacher labels")]
    pick = entries[0]
    assert pick["labels"] == ["x — X", "y — Y", "z"]
    assert pick["targets"] == [0.7, 0.2, 0.1]
    assert pick["shuffle"] is True
    assert entries[1]["shuffle"] is False and entries[2]["shuffle"] is False


def test_skip_moot_drops_null_expected_only():
    record = {**RECORD, "expected": {"pick": "x", "rate": None, "flag": True}}
    entries, dropped = question_entries(record, skip_moot=True)
    qids = [e["qid"] for e in entries]
    assert "rate" not in qids and "pick" in qids and "flag" in qids
    assert ("rate", "moot") in dropped


def test_variant0_keeps_original_order():
    examples, _ = expand_record(RECORD, variants=3, seed=0)
    pick = examples[0]["questions"][0]
    assert pick["labels"] == ["x — X", "y — Y", "z"]
    assert pick["targets"] == [0.7, 0.2, 0.1]


def test_choice_shuffles_targets_move_with_labels():
    examples, _ = expand_record(RECORD, variants=3, seed=0)
    base = dict(zip(examples[0]["questions"][0]["labels"], examples[0]["questions"][0]["targets"]))
    original = ["x — X", "y — Y", "z"]
    shuffled = [e for e in examples[1:] if e["questions"][0]["labels"] != original]
    assert shuffled, "with 3 options and 2 shuffles, at least one must reorder"
    for e in examples[1:]:
        pick = e["questions"][0]
        # same distribution, re-keyed to the new letter order
        assert dict(zip(pick["labels"], pick["targets"])) == base
        assert abs(sum(pick["targets"]) - 1.0) < 1e-9


def test_score_and_noul_never_shuffle():
    examples, _ = expand_record(RECORD, variants=5, seed=0)
    for e in examples:
        rate, flag = e["questions"][1], e["questions"][2]
        assert rate["labels"] == ["lo", "hi"] and rate["targets"] == [0.3, 0.7]
        assert flag["targets"][0] == 0.9  # yes stays first


def test_deterministic_with_same_seed():
    a, _ = expand_record(RECORD, variants=4, seed=7)
    b, _ = expand_record(RECORD, variants=4, seed=7)
    assert a == b


def test_all_questions_failed_skips_record():
    record = json.loads(json.dumps(RECORD))
    record["soft_targets"] = {k: None for k in RECORD["questions"]}
    examples, dropped = expand_record(record, variants=2, seed=0)
    assert examples == []
    assert {qid for qid, _ in dropped} == set(RECORD["questions"])


def test_single_option_choice_does_not_hang():
    record = {
        "id": "r2",
        "state": "s",
        "questions": {"only": {"type": "choice", "instructions": "q", "criteria": {"x": "X"}}},
        "soft_targets": {"only": {"x": 1.0}},
    }
    examples, dropped = expand_record(record, variants=3, seed=0)
    assert dropped == [] and len(examples) == 3
    for e in examples:
        assert e["questions"][0]["labels"] == ["x — X"]
        assert e["questions"][0]["targets"] == [1.0]


def test_invalid_targets_stop_training_file_build():
    record = json.loads(json.dumps(RECORD))
    record["soft_targets"]["pick"] = {"x": 0.8, "y": 0.2}
    with pytest.raises(ValueError, match="target keys"):
        question_entries(record)


def test_partial_teacher_labels_are_dropped_by_default():
    record = json.loads(json.dumps(RECORD))
    record["teacher"] = {"partial": {"pick": "1/2"}}
    entries, dropped = question_entries(record)
    assert "pick" not in [entry["qid"] for entry in entries]
    assert ("pick", "partial teacher label") in dropped
    entries, _ = question_entries(record, allow_partial=True)
    assert entries[0]["partial"] == "1/2"


def test_truncated_labels_keep_provenance_and_lower_weight():
    record = json.loads(json.dumps(RECORD))
    record["source"] = "demo"
    record["label_quality"] = {"truncated": {"pick": {"tail_mass": 0.02}}}
    examples, _ = expand_record(record, variants=1, seed=0, dataset_id="demo", split="train")
    example = examples[0]
    assert example["id"] == "demo:r1"
    assert example["source_id"] == "r1" and example["split"] == "train"
    assert example["questions"][0]["weight"] == 0.5
    assert example["questions"][0]["truncated"] == {"tail_mass": 0.02}


def test_question_provenance_is_kept_per_label():
    record = json.loads(json.dumps(RECORD))
    record["label_sources"] = {"pick": "openrouter", "rate": "policy"}
    record["teacher_labels"] = {"pick": {"model": "teacher", "provider": "p"}}
    examples, _ = expand_record(record, variants=1, seed=0)
    questions = examples[0]["questions"]
    assert questions[0]["label_source"] == "openrouter"
    assert questions[0]["teacher"] == {"model": "teacher", "provider": "p"}
    assert questions[1]["label_source"] == "policy"
    assert questions[1]["teacher"] is None


def test_source_split_keeps_all_variants_in_one_split():
    split = source_split("demo", "r1", 7, 0.9, 0.05)
    examples, _ = expand_record(RECORD, variants=4, seed=0, dataset_id="demo", split=split)
    assert {example["split"] for example in examples} == {split}
