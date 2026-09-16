"""Generate a model's single-token answer-code registry.

Run:
    uv run python -m tools.gen_answer_codes [--model MODEL] [--output FILE]

Scans A-Z, then AA-ZZ in lexicographic order, and keeps the codes that are
a single token in the real slot context (the 'Answer: ' prefix that
templates.filled builds, so they come out space-attached like ' A'), all at
one cut position. The engine assigns option codes in file order; regenerate
this file when the model or tokenizer changes.
"""

import argparse
import json
from importlib.resources import files
from pathlib import Path

from mlx_lm.utils import load_tokenizer

from system_one_lite.engine import (
    DEFAULT_MODEL,
    answer_code_entries,
    resolve_model_snapshot,
    tokenizer_sha256,
)


def default_output(model_id):
    data_dir = files("system_one_lite.data")
    for item in data_dir.iterdir():
        if not item.name.endswith("_answer_codes.json"):
            continue
        if json.loads(item.read_text()).get("model") == model_id:
            return Path(str(item))
    name = model_id.rsplit("/", 1)[-1].lower().replace("-", "_").replace(".", "_")
    return Path(str(data_dir)) / f"{name}_answer_codes.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output")
    args = ap.parse_args()

    model_path, revision = resolve_model_snapshot(args.model, tokenizer_only=True)
    tokenizer = load_tokenizer(model_path)
    entries = answer_code_entries(tokenizer)
    output = Path(args.output) if args.output else default_output(args.model)
    if not output.is_absolute():
        output = Path.cwd() / output

    out = {
        "model": args.model,
        "model_revision": revision,
        "tokenizer_sha256": tokenizer_sha256(model_path),
        "slot_context": "codes appended after 'Answer: ' (space-attached tokens)",
        "codes": entries,
    }
    output.write_text(json.dumps(out, indent=1) + "\n")

    n1 = sum(1 for e in entries if len(e["code"]) == 1)
    n2 = len(entries) - n1
    print(f"wrote {output}: {n1} one-letter + {n2} two-letter = {len(entries)} codes")
    print(
        "first:",
        [e["code"] for e in entries[:5]],
        "| around the tier boundary:",
        [e["code"] for e in entries[24:29]],
        "| last:",
        entries[-1]["code"],
    )


if __name__ == "__main__":
    main()
