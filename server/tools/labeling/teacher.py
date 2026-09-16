"""Teacher soft labels from OpenRouter logprobs.

Reads a datasets/*.jsonl file (id, state, questions, expected), asks a
teacher model each question, reads the option-code logprobs of the first
generated token, masks + renormalizes, and writes a new jsonl with a
"soft_targets" field per record, ready to fine-tune on.

Per the README's mitigations: Choice questions are asked once per option
order (cyclic rotations, averaged) to cancel position bias; Score and Noul
are asked once in standard order. Prompt shape is the serving prefix
(filled() up to "Answer: ", no filler letter). top_logprobs caps at 20.
Codes missing from that response share a small tail mass instead of a hard zero.

Usage (tested configuration — DeepSeek's own endpoint returns logprobs; the
default router may pick providers that do not):
    OPENROUTER_API_KEY=sk-... uv run python -m tools.labeling.teacher \
        --model deepseek/deepseek-v4.1-flash --provider DeepSeek \
        --in ../datasets/support_tickets.jsonl \
        --out ../datasets/support_tickets.soft.jsonl \
        [--permutations 4] [--limit N] [--resume]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import httpx

from system_one_lite import prompts
from system_one_lite.engine import CODES_FILE
from system_one_lite.prompts import as_text, filled, label

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ANSWER_CODES = tuple(entry["code"] for entry in json.loads(CODES_FILE.read_text())["codes"])
SYSTEM = (
    "You are a precise classifier. Reply with only the answer code of the "
    "best option. No explanation."
)
LABELING_VERSION = 2
LABEL_PROTOCOL_SHA256 = hashlib.sha256(
    Path(__file__).read_bytes() + CODES_FILE.read_bytes() + Path(prompts.__file__).read_bytes()
).hexdigest()


# --- pure helpers -----------------------------------------------------------


def letter_probs(top_logprobs, letters):
    """Mask + renormalize the teacher's first-token distribution.

    top_logprobs: list of {"token": str, "logprob": float}. Tokens may carry
    surface forms ("A" or " A"); anything that is not exactly one of the
    option letters after stripping is discarded. Returns {letter: prob}
    summing to 1, or None when no option letter is in the top-k. Letters
    missing from top-k get 0 so callers can index every option. Malformed
    provider payloads (non-list, missing fields) also return None: one bad
    response fails its question, not the run.
    """
    if not isinstance(top_logprobs, list):
        return None
    best = {}
    for entry in top_logprobs:
        if not isinstance(entry, dict):
            continue
        token, logprob = entry.get("token"), entry.get("logprob")
        if (
            not isinstance(token, str)
            or not isinstance(logprob, (int, float))
            or not math.isfinite(logprob)
        ):
            continue
        token = token.strip()
        if token in letters:
            best[token] = max(best.get(token, -math.inf), logprob)
    if not best:
        return None
    peak = max(best.values())
    weights = {k: math.exp(v - peak) for k, v in best.items()}
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0:
        return None
    return {letter: weights.get(letter, 0.0) / total for letter in letters}


def rotations(keys, n):
    """Up to n distinct cyclic rotations, starting with the original order."""
    n = min(n, len(keys))
    return [list(keys[i:]) + list(keys[:i]) for i in range(n)]


def question_spec(question):
    """(keys, labels) for a dataset question; keys identify soft-target slots."""
    if question["type"] == "choice":
        keys = list(question["criteria"])
        labels = [label(d, k) for k, d in question["criteria"].items()]
    elif question["type"] == "score":
        keys = [str(i) for i in range(len(question["criteria"]))]
        labels = [label(level) for level in question["criteria"]]
    else:  # noul; criteria descriptors render exactly as the server does
        keys = ["yes", "no"]
        c = question.get("criteria") or {}
        yes = f"yes — {c['true']}" if c.get("true") else "yes"
        no = f"no — {c['false']}" if c.get("false") else "no"
        labels = [yes, no]
    return keys, labels


def build_prompt(state_text, instructions, labels, codes=ANSWER_CODES):
    """Serving prefix for one question: filled() up to the answer slot."""
    text, marks = filled(state_text, [(instructions, labels)], codes=codes)
    return text[: marks[0]]


def label_question(ask, state_text, question, permutations, tail_mass=0.02):
    """Ask one question (permuting Choice option order).

    Returns (distribution by key or None, rotations that succeeded, total,
    truncation details, response details). Missing answer codes get a small
    shared tail instead of a false hard zero.
    """
    keys, labels = question_spec(question)
    if len(labels) > len(ANSWER_CODES):
        raise ValueError(f"{len(labels)} options exceeds the {len(ANSWER_CODES)} answer-code limit")
    codes = ANSWER_CODES[: len(labels)]
    if len(labels) > 20:
        print(
            f"  warning: {len(labels)} options exceeds the 20-token "
            f"logprob cap; this label will be truncated",
            file=sys.stderr,
        )
    orders = rotations(keys, permutations) if question["type"] == "choice" else [keys]
    label_of = dict(zip(keys, labels))

    dists, truncated, responses = [], [], []
    for order in orders:
        ordered_labels = [label_of[k] for k in order]
        probs = ask(
            build_prompt(state_text, question["instructions"], ordered_labels, codes), codes
        )
        if probs is None:
            continue
        response = getattr(ask, "last_response", None)
        if response:
            responses.append(response)
        row = {order[i]: probs.get(codes[i], 0.0) for i in range(len(order))}
        if sum(row.values()) <= 0:
            continue
        missing = [key for key, value in row.items() if value == 0.0]
        if missing:
            observed_mass = 1.0 - tail_mass
            row = {
                key: (value * observed_mass if value else tail_mass / len(missing))
                for key, value in row.items()
            }
            truncated.append({"order": order, "missing": missing})
        dists.append(row)

    if not dists:
        return None, 0, len(orders), [], responses
    dist = {k: sum(d.get(k, 0.0) for d in dists) / len(dists) for k in keys}
    return dist, len(dists), len(orders), truncated, responses


def label_record(record, ask, permutations, tail_mass=0.02, replace_existing=False):
    state_text = as_text(record["state"])
    soft, failed, partial, truncated, responses, preserved = {}, [], {}, {}, {}, []
    existing = record.get("soft_targets") or {}
    old_quality = record.get("label_quality") or {}
    retry = set(old_quality.get("failed") or []) | set(old_quality.get("partial") or {})
    for qid, question in record["questions"].items():
        if not replace_existing and existing.get(qid) is not None and qid not in retry:
            soft[qid] = existing[qid]
            preserved.append(qid)
            continue
        dist, ok, total, cuts, response = label_question(
            ask, state_text, question, permutations, tail_mass
        )
        if dist is None:
            failed.append(qid)
        elif ok < total:
            partial[qid] = f"{ok}/{total}"
        if cuts:
            truncated[qid] = {"orders": cuts, "tail_mass": tail_mass}
        if response:
            responses[qid] = response
        soft[qid] = dist
    return soft, failed, partial, truncated, responses, preserved


# --- network ----------------------------------------------------------------


def make_ask(client, model, provider=None):
    """Return (ask, stats). ask(prompt, codes) -> {code: probability} or None.

    Thinking is disabled so the first generated token is the answer code.
    A locked provider avoids routes that drop logprobs.
    Retries transient failures (network, 429, 5xx). A 4xx response or a
    response without logprobs ends that call.
    """
    if not provider:
        raise ValueError("a provider lock is required for teacher labels")
    stats = {"calls": 0, "tokens": 0}

    def ask(prompt, letters):
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 1,
            "logprobs": True,
            "top_logprobs": 20,
            "reasoning": {"enabled": False},
        }
        body["provider"] = {
            "order": [provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        for attempt in range(4):
            try:
                r = client.post(API_URL, json=body)
            except httpx.TransportError as e:
                print(f"  network error (attempt {attempt + 1}): {e}", file=sys.stderr)
                time.sleep(2 * (attempt + 1) ** 2)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                print(f"  HTTP {r.status_code} (attempt {attempt + 1})", file=sys.stderr)
                time.sleep(2 * (attempt + 1) ** 2)
                continue
            if r.status_code >= 400:
                print(f"  API error {r.status_code}: {r.text[:200]}", file=sys.stderr)
                return None
            data = r.json()
            stats["calls"] += 1
            stats["tokens"] += data.get("usage", {}).get("total_tokens", 0)
            try:
                top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            except (KeyError, IndexError, TypeError):
                # null logprobs (unsupported provider) or an unexpected shape
                print(
                    "  response has no logprobs; this model or provider does not return them",
                    file=sys.stderr,
                )
                return None
            metadata = {"id": data.get("id"), "model": data.get("model")}
            actual_provider = data.get("provider") or data.get("openrouter_metadata", {}).get(
                "provider"
            )
            if actual_provider:
                metadata["provider"] = actual_provider
            ask.last_response = metadata
            return letter_probs(top, set(letters))
        return None

    return ask, stats


def checkpoint_records(path):
    """Read the latest saved version of each record from a checkpoint."""
    rows = {}
    lines = Path(path).read_text().splitlines()
    last_line = max((number for number, line in enumerate(lines, 1) if line.strip()), default=0)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == last_line:
                break  # process stop during the final buffered write
            raise ValueError(
                f"cannot resume: invalid JSON on line {line_number} of {path}"
            ) from error
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"cannot resume: missing id on line {line_number} of {path}")
        rows[record_id] = record
    return rows


def completed_ids(path):
    """Read completed record ids from a prior temporary output file."""
    return set(checkpoint_records(path))


def incomplete_ids(rows):
    """Records with failed calls or partial rotations must be retried."""
    if isinstance(rows, dict):
        rows = rows.values()
    return {
        row["id"]
        for row in rows
        if (row.get("label_quality") or {}).get("failed")
        or (row.get("label_quality") or {}).get("partial")
    }


def run_manifest(records, model, provider, permutations, tail_mass, replace_existing):
    """Stable identity for a resumable label run."""
    input_bytes = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "version": 1,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "record_count": len(records),
        "model": model,
        "provider": provider,
        "permutations": permutations,
        "truncated_tail_mass": tail_mass,
        "replace_existing": replace_existing,
        "labeling_version": LABELING_VERSION,
        "label_protocol_sha256": LABEL_PROTOCOL_SHA256,
    }


def write_json_atomic(path, value):
    temp = f"{path}.new"
    with open(temp, "w") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def rewrite_checkpoint(path, rows):
    temp = f"{path}.new"
    with open(temp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id; must return logprobs")
    ap.add_argument(
        "--provider",
        required=True,
        help="lock routing to one provider (e.g. DeepSeek); skips providers that drop logprobs",
    )
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outfile", required=True)
    ap.add_argument(
        "--permutations", type=int, default=4, help="max option orders per Choice question"
    )
    ap.add_argument(
        "--truncated-tail-mass",
        type=float,
        default=0.02,
        help="probability reserved for answer codes absent from top_logprobs (default: 0.02)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace soft targets already present in input rows",
    )
    ap.add_argument("--resume", action="store_true", help="continue from a prior .tmp output file")
    args = ap.parse_args()
    if args.permutations < 1:
        sys.exit("--permutations must be at least 1")
    if not 0 < args.truncated_tail_mass < 1:
        sys.exit("--truncated-tail-mass must be greater than 0 and less than 1")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set in the environment")

    records = [json.loads(line) for line in open(args.infile) if line.strip()]
    if args.limit:
        records = records[: args.limit]
    ids = [record.get("id") for record in records]
    if any(not isinstance(record_id, str) or not record_id for record_id in ids):
        sys.exit("every input record must have a non-empty string id")
    if len(set(ids)) != len(ids):
        sys.exit("input records have duplicate ids; use unique dataset namespaces")

    n_calls = sum(
        (min(args.permutations, len(q["criteria"])) if q["type"] == "choice" else 1)
        for record in records
        for qid, q in record["questions"].items()
        if args.replace_existing or (record.get("soft_targets") or {}).get(qid) is None
    )
    print(f"{len(records)} records, ~{n_calls} calls to {args.model}")

    client = httpx.Client(timeout=60, headers={"Authorization": f"Bearer {key}"})
    ask, stats = make_ask(client, args.model, args.provider)

    pulled_q, failed_q, partials, truncations, preserved_q = 0, 0, {}, {}, 0
    tmpfile = args.outfile + ".tmp"  # rename only on success; no truncated output
    manifestfile = tmpfile + ".manifest.json"
    final_manifest = args.outfile + ".manifest.json"
    if os.path.exists(args.outfile) or os.path.exists(final_manifest):
        sys.exit(f"{args.outfile} or its manifest already exists; choose a new output path")
    if os.path.exists(tmpfile) and not args.resume:
        sys.exit(f"{tmpfile} already exists; use --resume or inspect and remove it")
    if args.resume and not os.path.exists(tmpfile):
        sys.exit(f"cannot resume: {tmpfile} does not exist")
    manifest = run_manifest(
        records,
        args.model,
        args.provider,
        args.permutations,
        args.truncated_tail_mass,
        args.replace_existing,
    )
    if args.resume:
        if not os.path.exists(manifestfile):
            sys.exit(f"cannot resume: {manifestfile} does not exist")
        try:
            saved_manifest = json.loads(Path(manifestfile).read_text())
        except json.JSONDecodeError as error:
            sys.exit(f"cannot resume: invalid manifest {manifestfile}: {error}")
        if saved_manifest != manifest:
            sys.exit("cannot resume: input or label settings changed")
    else:
        if os.path.exists(manifestfile):
            sys.exit(f"{manifestfile} already exists; inspect and remove it")
        write_json_atomic(manifestfile, manifest)
    rows = checkpoint_records(tmpfile) if args.resume else {}
    unknown = set(rows) - set(ids)
    if unknown:
        sys.exit(f"cannot resume: {tmpfile} has ids not in this input")
    if args.resume:
        rewrite_checkpoint(tmpfile, rows.values())
    retry = incomplete_ids(rows)
    done = set(rows) - retry
    if retry:
        print(f"retrying: {len(retry)} incomplete records")
    if done:
        print(f"resuming: {len(done)} records already complete")
    with open(tmpfile, "a" if args.resume else "w") as out:
        for record in records:
            if record["id"] in done:
                continue
            record = rows.get(record["id"], record)
            soft, failed, partial, truncated, responses, preserved = label_record(
                record, ask, args.permutations, args.truncated_tail_mass, args.replace_existing
            )
            pulled_q += len(soft) - len(failed) - len(preserved)
            failed_q += len(failed)
            preserved_q += len(preserved)
            partials.update({f"{record['id']}.{qid}": n for qid, n in partial.items()})
            truncations.update(
                {f"{record['id']}.{qid}": detail for qid, detail in truncated.items()}
            )
            old_quality = record.get("label_quality") or {}
            old_sources = record.get("label_sources") or {}
            replaced = set(record["questions"]) - set(preserved)
            old_truncated = {
                qid: detail
                for qid, detail in (old_quality.get("truncated") or {}).items()
                if qid not in replaced
            }
            old_partial = {
                qid: detail
                for qid, detail in (old_quality.get("partial") or {}).items()
                if qid not in replaced
            }
            sources = {qid: source for qid, source in old_sources.items() if qid not in replaced}
            for qid in preserved:
                if qid not in sources:
                    sources[qid] = "policy" if "ocean" in record else "provided"
            for qid in responses:
                sources[qid] = "openrouter"
            teacher_labels = {
                qid: detail
                for qid, detail in (record.get("teacher_labels") or {}).items()
                if qid not in replaced
            }
            for qid, response in responses.items():
                teacher_labels[qid] = {
                    "model": args.model,
                    "provider": args.provider,
                    "permutations": args.permutations,
                    "truncated_tail_mass": args.truncated_tail_mass,
                    "responses": response,
                }
            out.write(
                json.dumps(
                    {
                        **record,
                        "soft_targets": soft,
                        "label_sources": sources,
                        "teacher_labels": teacher_labels,
                        "label_quality": {
                            **old_quality,
                            "truncated": {**old_truncated, **truncated},
                            "partial": {**old_partial, **partial},
                            "failed": failed,
                            "preserved": preserved,
                        },
                    }
                )
                + "\n"
            )
            status = f"{len(soft) - len(failed)}/{len(soft)}"
            if failed:
                status += f" (failed: {', '.join(failed)})"
            if partial:
                status += f" (partial: {partial})"
            print(f"  {record['id']}: {status}")
            out.flush()
    if partials:
        print(
            f"warning: {len(partials)} question(s) averaged over fewer than "
            f"all rotations: {partials}",
            file=sys.stderr,
        )
    if truncations:
        print(
            f"warning: {len(truncations)} question(s) had answer codes "
            f"missing from top_logprobs; tail mass was added",
            file=sys.stderr,
        )
    if failed_q or partials:
        sys.exit(
            f"incomplete: {failed_q} failed and {len(partials)} partial labels; "
            f"resume with --resume"
        )
    completed = checkpoint_records(tmpfile)
    rewrite_checkpoint(tmpfile, [completed[record_id] for record_id in ids])
    os.replace(tmpfile, args.outfile)
    os.replace(manifestfile, final_manifest)
    print(
        f"done: {pulled_q} questions labeled, {failed_q} failed, "
        f"{preserved_q} preserved, {stats['calls']} calls, {stats['tokens']} "
        f"tokens -> {args.outfile}"
    )


if __name__ == "__main__":
    main()
