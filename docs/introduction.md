# Introduction

Most LLM features follow the same shape: send text, get text back, then parse
that text and hope it fits the shape your code needs. System One Lite flips
that. You send a *state* (any content) plus typed *questions*, and you get
back typed answers with probabilities. Nothing is generated, so nothing needs
parsing. Your code can branch on the answers, sort by them, and route with
them.

This is a proof of concept. It exists to test one idea: that a plain
open-weight LLM, used only as a scorer, can replace the generate-then-parse
loop for classification and judgment calls.

## One request, three parts

Every call has the same shape:

```json
{
  "state": "Order DB-4471 was supposed to arrive Friday and it still shows 'in transit'.",
  "questions": {
    "topic": {
      "type": "choice",
      "instructions": "Which team does this message belong to?",
      "criteria": {
        "deliveries": "Late, missing, or damaged orders",
        "billing": "Charges, refunds, payment methods",
        "account": "Login, profile, and app problems"
      }
    }
  }
}
```

- `state` is the content to judge. A string, or structured JSON. See
  [State](state.md).
- `questions` maps IDs you pick to typed questions. There are three types:
  **choice** (pick one option from your list), **score** (place the state on a
  ladder of levels you define), and **noul** (how likely a
  statement is to be true). See [Primitives](primitives/index.md).

The engine is fixed, so there is no `model` field to send; the response's
`model` reports the engine that actually answered.

## Why probabilities matter

Every answer is a distribution over exactly the options you defined. That
gives you two things plain text output does not:

1. **A ranking you can act on.** If `deliveries` gets 0.62 and `billing`
   0.26, you can route to deliveries and copy billing — your call, in code.
2. **Built-in doubt.** A flat distribution means the model cannot tell the
   options apart. Choice and Score answers add a `confidence` number computed
   from that distribution, so you can escalate instead of guessing.
   See [Confidence](confidence.md).

## Small questions, combined in code

The engine answers one focused question at a time. It is built for snap
judgments — the kind a person makes in a few seconds with the right context in
front of them — not for multi-step reasoning.

When a decision depends on several factors, do not write one big question.
Ask one question per factor and combine the answers in code, where you
control the weights. Changing a weight is a one-line edit; changing your mind
inside a prompt is a rewrite.

## What is honest about this POC

- The engine is a 2B model by default (`mlx-community/Qwen3.5-2B-MLX-4bit`).
  It is small, fast, and good enough to test the pattern. It is not a
  frontier model.
- Probabilities come from one masked read of the option letters
  ([How it works](how-it-works.md)). They are not tuned against real
  outcomes yet, so treat them as ranked signal, not as calibrated odds.
- `server/tools/evals.py` measures accuracy and option-order stability against
  labeled datasets, which is the current evidence that the approach holds
  up. Calibration work (temperature scaling, Brier checks) is planned but
  not done.

## Next steps

- [Quickstart](quickstart.md) — run the server and make a call.
- [Primitives](primitives/index.md) — the three question types in depth.
- [How it works](how-it-works.md) — what happens inside the engine.
