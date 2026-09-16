"""Score datasets/ question golds with the local engine.

Gold conventions match datasets/README.md: Choice is the winning key,
Noul is a boolean, Score is a level index (tolerance 0.5). Extra expected
keys (route, actions, decision) and null (moot) golds are skipped.

Run:
    uv run python -m tools.evals [--limit N]
"""

import argparse
import json
from pathlib import Path

from system_one_lite.engine import DEFAULT_MODEL, Engine
from system_one_lite.prompts import as_text
from tools.labeling.teacher import question_spec

DATASETS = Path(__file__).resolve().parents[2] / "datasets"
SCORE_TOL = 0.5
LARGE_OPTION_LABELS = (
    "Advertising campaigns",
    "Accounts payable",
    "Accounts receivable",
    "Building access",
    "Business insurance",
    "Contract review",
    "Customer renewals",
    "Data exports",
    "Employee benefits",
    "Event planning",
    "Facilities maintenance",
    "Fraud review",
    "Hardware purchases",
    "Hiring and recruiting",
    "Legal notices",
    "Office supplies",
    "Partner onboarding",
    "Payment integration bugs",
    "Payroll questions",
    "Product pricing",
    "Public relations",
    "Refund processing",
    "Sales leads",
    "Security incidents",
    "Shipping delays",
    "Tax reporting",
    "Training requests",
    "Travel booking",
    "Vendor management",
    "Website content",
)
LARGE_OPTION_STATE = (
    "Our Stripe connection has failed for three days. The API returns an "
    "integration error and we are losing sales."
)


def scoreable(record):
    """Questions that have a non-null gold in expected."""
    expected = record.get("expected") or {}
    items = []
    for qid, question in record["questions"].items():
        if qid not in expected or expected[qid] is None:
            continue
        keys, labels = question_spec(question)
        items.append(
            {
                "qid": qid,
                "kind": question["type"],
                "instructions": as_text(question["instructions"]),
                "keys": keys,
                "labels": labels,
                "gold": expected[qid],
            }
        )
    return items


def judge(item, probs):
    kind, gold = item["kind"], item["gold"]
    if kind == "choice":
        got = max(zip(item["keys"], probs), key=lambda kv: kv[1])[0]
        ok = got == gold
        return ok, f"{got} expected {gold}"
    if kind == "noul":
        got = probs[0]
        ok = (got >= 0.5) == bool(gold)
        return ok, f"{got:.2f} expected {'yes' if gold else 'no'}"
    got = sum(i * v for i, v in enumerate(probs))
    ok = abs(got - float(gold)) <= SCORE_TOL
    return ok, f"{got:.2f} expected {gold}"


def run_record(engine, record):
    items = scoreable(record)
    if not items:
        return []
    results, _, _ = engine.evaluate(
        as_text(record["state"]), [(q["instructions"], q["labels"]) for q in items]
    )
    rows = []
    for q, probs in zip(items, results):
        ok, detail = judge(q, probs)
        rows.append((record["id"], q["qid"], q["kind"], ok, detail))
    return rows


def permute_choice(engine, record):
    """Rotate one Choice question; the winner must stay the same."""
    items = scoreable(record)
    choice_i = next((i for i, q in enumerate(items) if q["kind"] == "choice"), None)
    if choice_i is None:
        return None
    specs = [(q["instructions"], q["labels"]) for q in items]
    base, _, _ = engine.evaluate(as_text(record["state"]), specs)
    q = items[choice_i]
    rotated = q["keys"][1:] + q["keys"][:1]
    labels = [q["labels"][q["keys"].index(k)] for k in rotated]
    rot_specs = [
        (qq["instructions"], labels if i == choice_i else qq["labels"])
        for i, qq in enumerate(items)
    ]
    rot, _, _ = engine.evaluate(as_text(record["state"]), rot_specs)
    base_p = dict(zip(q["keys"], base[choice_i]))
    rot_p = dict(zip(rotated, rot[choice_i]))
    stable = max(base_p, key=base_p.get) == max(rot_p, key=rot_p.get)
    tv = 0.5 * sum(abs(base_p[k] - rot_p[k]) for k in base_p)
    return stable, tv


def large_option_bias(engine):
    """Measure code-order drift on the same 30-option question three ways."""
    labels = list(LARGE_OPTION_LABELS)
    orders = (labels, labels[13:] + labels[:13], list(reversed(labels)))
    distributions = []
    for order in orders:
        probs, _, _ = engine.evaluate(
            LARGE_OPTION_STATE,
            [("Which team should handle this?", order)],
        )
        distributions.append(dict(zip(order, probs[0])))

    base = distributions[0]
    winner = max(base, key=base.get)
    winners = [max(dist, key=dist.get) for dist in distributions]
    tvs = [0.5 * sum(abs(base[key] - dist[key]) for key in base) for dist in distributions[1:]]
    return {
        "options": len(labels),
        "winner": winner,
        "winners": winners,
        "winner_stable": all(item == winner for item in winners),
        "mean_tv": sum(tvs) / len(tvs),
        "max_tv": max(tvs),
    }


def summarize(rows):
    total = len(rows)
    correct = sum(r[3] for r in rows)
    by_kind = {}
    for r in rows:
        c, t = by_kind.get(r[2], (0, 0))
        by_kind[r[2]] = (c + r[3], t + 1)
    parts = [f"{correct}/{total} overall"]
    parts.extend(f"{k} {c}/{t}" for k, (c, t) in sorted(by_kind.items()))
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--datasets", default=str(DATASETS))
    args = ap.parse_args()

    root = Path(args.datasets)
    files = sorted(root.glob("*.jsonl"))
    if not files:
        ap.error(f"no JSONL datasets found in {root}")

    engine = Engine(args.model)
    all_rows = []
    permute_n, permute_ok, tvs = 0, 0, []

    for path in files:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if args.limit is not None:
            records = records[: args.limit]
        rows = []
        for record in records:
            rows.extend(run_record(engine, record))
            if permute_n < 8:
                result = permute_choice(engine, record)
                if result is not None:
                    stable, tv = result
                    permute_n += 1
                    permute_ok += stable
                    tvs.append(tv)
        all_rows.extend(rows)
        n_ok = sum(r[3] for r in rows)
        print(f"{path.name:<32} {n_ok}/{len(rows)}")
        for rec_id, qid, kind, ok, detail in rows:
            if not ok:
                print(f"  miss {rec_id} {qid} {kind} {detail}")

    print(summarize(all_rows))
    if tvs:
        print(
            f"permutation: winner stable {permute_ok}/{permute_n}, "
            f"mean TV distance {sum(tvs) / len(tvs):.2f}"
        )
    bias = large_option_bias(engine)
    print(
        f"30-option permutation: winner stable {bias['winner_stable']}, "
        f"mean TV distance {bias['mean_tv']:.2f}, "
        f"max {bias['max_tv']:.2f}"
    )


if __name__ == "__main__":
    main()
