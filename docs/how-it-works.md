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
Answer: A
```

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
3. Run each question as a full independent prompt. This model's cached
   prefix path changes its probabilities, so serving does not use it. A
   right-padded batch also moves the numbers on this Qwen3.5 (GatedDeltaNet
   conv/SSM layers), so prompts run one after another.
4. Take the logits at the answer position. Keep only the token IDs the
   option codes can become, then apply softmax.

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

- **570 options max (default model).** Every option needs a one-token answer
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

## Swapping the model

The engine takes any MLX text model. Set it with `--model` on the demo,
benchmark, or eval tool. Create its registry first:

```bash
uv run python -m tools.gen_answer_codes --model mlx-community/Qwen2.5-1.5B-Instruct-4bit
```

The generator writes a model-specific file. The template, masked read, and
answer assembly do not depend on one model.

The cache experiment tool tests shared-prefix KV reuse. It is separate from the
server and exits with an error if its probabilities differ from the safe full
prompt path.
