"""Expand teacher soft labels into a shuffled training set.

Reads a .soft.jsonl from the teacher label tool and writes training examples:
one per (record, variant). Variant 0 keeps the original option order; the
rest reshuffle each Choice question's options (labels, letters, and target
probabilities move together, targets staying the teacher-averaged
distribution). Score levels keep their order (an ordered scale) and Noul
keeps its fixed yes/no layout, so only Choice shuffles.

The output carries structured questions, not prompts: the training loop
builds the filled template with the same engine code serving uses, so
tokenization cannot drift between training and serving.

Records are split before variants are made. Every variant of a source row
stays in one of train, validation, or test.

Usage:
    uv run python -m tools.labeling.make_training_set \
        --in ../datasets/support_tickets.soft.jsonl \
        --out ../datasets/support_tickets.train.jsonl \
        [--variants 4] [--seed 0]
"""

import argparse
import hashlib
import json
import math
import random
import sys

from tools.labeling.teacher import as_text, question_spec

TARGET_TOLERANCE = 1e-6


def target_values(soft, keys):
    """Validate and order one target distribution."""
    if not isinstance(soft, dict) or set(soft) != set(keys):
        raise ValueError("target keys do not match the question choices")
    values = [soft[key] for key in keys]
    if (
        not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in values
        )
        or abs(sum(values) - 1.0) > TARGET_TOLERANCE
    ):
        raise ValueError("targets must be finite, non-negative, and sum to 1")
    return values


def question_entries(record, skip_moot=False, allow_partial=False, truncated_weight=0.5):
    """Per-question (qid, instructions, labels, targets), skipping failures.

    With skip_moot, questions whose `expected` gold is explicitly null are
    dropped too: null marks a question that is moot for the row, and training
    on a teacher's arbitrary answer to an inapplicable question adds noise.
    """
    entries, dropped = [], []
    expected = record.get("expected") or {}
    teacher = record.get("teacher") or {}
    quality = record.get("label_quality") or {}
    partial = quality.get("partial") or teacher.get("partial") or {}
    truncations = quality.get("truncated") or {}
    for qid, question in record["questions"].items():
        soft = (record.get("soft_targets") or {}).get(qid)
        reason = None
        if not soft:
            reason = "no teacher labels"
        elif qid in partial and not allow_partial:
            reason = "partial teacher label"
        elif skip_moot and qid in expected and expected[qid] is None:
            reason = "moot"
        if reason:
            dropped.append((qid, reason))
            continue
        keys, labels = question_spec(question)
        try:
            targets = target_values(soft, keys)
        except ValueError as error:
            raise ValueError(f"{record['id']}.{qid}: {error}") from error
        truncated = truncations.get(qid)
        entries.append(
            {
                "qid": qid,
                "instructions": as_text(question["instructions"]),
                "labels": labels,
                "targets": targets,
                "shuffle": question["type"] == "choice",
                "weight": truncated_weight if truncated else 1.0,
                "truncated": truncated,
                "partial": partial.get(qid),
                "label_source": label_source(record, qid),
                "teacher": (record.get("teacher_labels") or {}).get(qid),
            }
        )
    return entries, dropped


def expand_record(
    record,
    variants,
    seed,
    skip_moot=False,
    allow_partial=False,
    truncated_weight=0.5,
    dataset_id=None,
    split=None,
):
    """Yield one training example per variant; Choice orders reshuffled."""
    state_text = as_text(record["state"])
    entries, dropped = question_entries(record, skip_moot, allow_partial, truncated_weight)
    if not entries:
        return [], dropped

    examples = []
    for variant in range(variants):
        questions = []
        for e in entries:
            if variant == 0 or not e["shuffle"]:
                labels, targets = e["labels"], e["targets"]
            else:
                # deterministic per record+question+variant, reproducible runs
                rng = random.Random(f"{record['id']}|{e['qid']}|{variant}|{seed}")
                identity = list(range(len(e["labels"])))
                order = list(identity)
                if len(order) > 1:  # a 1-option question has nothing to shuffle
                    while order == identity:
                        rng.shuffle(order)
                labels = [e["labels"][i] for i in order]
                targets = [e["targets"][i] for i in order]
            questions.append(
                {
                    "qid": e["qid"],
                    "instructions": e["instructions"],
                    "labels": labels,
                    "targets": targets,
                    "weight": e["weight"],
                    "truncated": e["truncated"],
                    "partial": e["partial"],
                    "label_source": e["label_source"],
                    "teacher": e["teacher"],
                }
            )
        source_id = record["id"]
        resolved_dataset = dataset_id or record.get("dataset_id") or record.get("source")
        example = {
            "id": f"{resolved_dataset}:{source_id}" if resolved_dataset else source_id,
            "source_id": source_id,
            "dataset_id": resolved_dataset,
            "source": record.get("source"),
            "variant": variant,
            "state_text": state_text,
            "questions": questions,
        }
        if split is not None:
            example["split"] = split
        examples.append(example)
    return examples, dropped


def label_source(record, qid):
    source = (record.get("label_sources") or {}).get(qid)
    if source:
        return source
    if qid in (record.get("teacher_labels") or {}) or "teacher" in record:
        return "openrouter"
    if "ocean" in record:
        return "policy"
    return "provided"


def dataset_identity(record, default=None):
    value = default or record.get("dataset_id") or record.get("source")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{record.get('id', '<unknown>')}: missing dataset_id or source")
    return value


def source_split(dataset_id, source_id, seed, train_fraction, validation_fraction):
    digest = hashlib.sha256(f"{dataset_id}|{source_id}|{seed}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument(
        "--variants", type=int, default=4, help="examples per record (variant 0 = original order)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--skip-moot",
        action="store_true",
        help="drop questions whose expected gold is explicitly null (moot for that row)",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="include questions labeled from fewer than all rotations",
    )
    ap.add_argument(
        "--truncated-weight",
        type=float,
        default=0.5,
        help="training weight for labels missing top_logprobs codes",
    )
    ap.add_argument(
        "--dataset-id", default=None, help="dataset namespace for records without dataset_id"
    )
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--train-fraction", type=float, default=0.9)
    ap.add_argument("--validation-fraction", type=float, default=0.05)
    args = ap.parse_args()
    if args.variants < 1:
        sys.exit("--variants must be at least 1")
    if not 0 <= args.truncated_weight <= 1:
        sys.exit("--truncated-weight must be between 0 and 1")
    if (
        not 0 < args.train_fraction < 1
        or not 0 <= args.validation_fraction < 1
        or args.train_fraction + args.validation_fraction >= 1
    ):
        sys.exit("split fractions must leave a non-empty test fraction")

    records = [json.loads(line) for line in open(args.infile) if line.strip()]
    examples, dropped_q, skipped, seen_ids = [], 0, [], set()
    for record in records:
        try:
            dataset_id = dataset_identity(record, args.dataset_id)
            source_key = (dataset_id, record["id"])
            if source_key in seen_ids:
                raise ValueError(f"duplicate source record: {dataset_id}:{record['id']}")
            seen_ids.add(source_key)
            split = source_split(
                dataset_id,
                record["id"],
                args.split_seed,
                args.train_fraction,
                args.validation_fraction,
            )
            exs, dropped = expand_record(
                record,
                args.variants,
                args.seed,
                args.skip_moot,
                args.allow_partial,
                args.truncated_weight,
                dataset_id,
                split,
            )
        except (KeyError, ValueError) as error:
            sys.exit(str(error))
        examples.extend(exs)
        dropped_q += len(dropped)
        if dropped or not exs:
            skipped.append((record["id"], dropped))

    if records and not examples:
        sys.exit(f"nothing to train on: every question was dropped ({skipped})")

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    split_counts = {
        name: sum(1 for ex in examples if ex["split"] == name)
        for name in ("train", "validation", "test")
    }

    with open(args.outfile, "w") as out:
        for ex in examples:
            out.write(json.dumps(ex) + "\n")

    print(
        f"{len(records)} records -> {len(examples)} examples "
        f"({args.variants} variants each) -> {args.outfile}"
    )
    print(f"splits: {split_counts}")
    if dropped_q:
        print(f"dropped {dropped_q} question(s): {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
