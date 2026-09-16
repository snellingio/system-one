# How it works

The engine (`server/src/system_one_lite/engine.py`) never decodes text. It builds one full prompt
for each question. The next token must be an answer code. The engine runs a
forward pass and reads the code probabilities at that position. It generates
no tokens.

This page walks through the pieces so you can read the code alongside it.

## The prompt

`filled()` in `server/src/system_one_lite/prompts.py` builds the prompt. For the quickstart
question it looks like this:

```
State:
Order DB-4471 was supposed to arrive Friday and it still shows 'in transit'. ...

Question 1: Which team does this message belong to?

A: deliveries — Late, missing, or damaged orders
B: billing — Charges, refunds, payment methods
C: account — Login, profile, and app problems

Please show your choice in the answer field with only the choice letter,
e.g., "answer": "C".
```

The engine wraps this question in Qwen's chat template with thinking disabled.
The template closes an empty thinking block when the model requires one. The
scored answer starts with `{"answer": "`. The masked read scores the one
answer-code token that follows that prefix. A complete filler such as
`{"answer": "A"}` keeps the template easy to inspect.

`as_text` leaves strings unchanged and turns objects or arrays into indented
JSON. Each question becomes a `Question k:` block. Option codes start at `A`,
continue through `Z`, and then use `AA`, `AB`, and more. The template fills
each answer slot with `A` so it is a complete string. The engine does not
evaluate that filler token.

The `State:` header improved the eval result by 2 correct answers out of 20.
Option rotation stayed just as stable. An instruction line and bare numbering
matched its accuracy but were less stable.

## The masked read

Inside `Engine.evaluate`:

1. Build one prompt per question and remember the character offset of the
   answer letter (the `marks` from `filled()`).
2. Tokenize each prompt. The answer code sits at a known token index. Startup
   checks every registry code against the live tokenizer. Requests reuse those
   token IDs and still check that the answer slot has not moved.
3. Run each question as a full independent prompt. Serving does not share a
   cache. The separate cache experiment must prove matching probabilities
   before that path can be used.
4. Take the logits at the answer position. Keep only the token IDs the
   option codes can become, divide by the model-card temperature of 0.7,
   then apply softmax.

The result is a distribution over your options — and it cannot be anything
else, because only your answer codes are in the mask. A malformed answer is
impossible by construction, not by retry.

This is why `usage.output_tokens` is always 0. `usage.input_tokens` counts
all full question prompts.

## One sequence per question

Each question gets its own prompt with the shared state. Later questions never
see earlier filler codes. Adding or moving a question cannot change another
answer.

Each extra question includes another copy of the state. Prompts run one
after another. This costs more than shared-prefix caching, but it keeps
serving probabilities identical to a full prompt.

## Limits that fall out of the design

- **578 options max.** Every option needs a one-token answer
  code. Codes start with `A` through `Z`, then use two letters. The usable list
  lives in a model-specific `server/src/system_one_lite/data/*_answer_codes.json` file.
  `server/tools/gen_answer_codes.py` creates it. The engine checks the model revision,
  tokenizer hash, full code list, and unique token IDs before warm-up.
- **2–10 Score levels.** Enforced in the server schema: a one-level score
  has nothing to distribute over, and ladders past ten blur faster than
  they inform.
- **No tokenizer surprises.** If any answer code does not land as a single
  token after the slot, the engine raises and the server returns `422`. You
  get an error, never a silently wrong read.

## Confidence

The distribution from the masked read feeds `confidence()` in
`prompts.py`: `(max_p − 1/n) / (1 − 1/n)`. The formula and its trade-offs
are covered in [Confidence](confidence.md).

## Measuring it

`server/tools/evals.py` runs the engine over gold-labeled records:

- Choice: the top option must equal the gold option.
- Noul: `p(yes) ≥ 0.5` must match the gold boolean.
- Score: the weighted-average level must land within 0.5 of the gold level.
- Stability: one Choice per record runs again with rotated options. The report
  checks the winner and the distance between distributions.

From `server/`:

```bash
uv run python -m tools.evals --limit N
```

Use it to measure changes to the template, model, or confidence formula.
Compare accuracy and order stability on labeled data before and after.

## Model contract

The default engine uses `mlx-community/Qwen3-1.7B-4bit`. The larger option is
`mlx-community/Qwen3-4B-Instruct-2507-4bit`. Both downloads use exact Hub
commit pins. Each checked-in registry binds its model ID, revision, tokenizer
hash, and 578 valid answer codes. Any other model needs an exact revision, a
new registry, and a full accuracy and stability run.

`Engine`, the demo, the benchmark, and the eval tool accept `default` or
`larger` as the model name. Download either profile before a run with
`python -m tools.download_model default` or
`python -m tools.download_model larger`.

The cache experiment tool tests shared-prefix KV reuse. It is separate from the
server and exits with an error if its probabilities differ from the safe full
prompt path.
