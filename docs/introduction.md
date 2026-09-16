# Introduction

Most LLM features send text, get text back, then parse it. The code hopes the
output has the expected shape. System One Lite reverses this. You send a
*state* plus typed *questions*. It returns typed answers with probabilities.
Nothing is generated or parsed. Your code can branch, sort, and route with the
answers.

This proof of concept tests one idea. A plain open-weight LLM can score
classification choices without generating text.

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
- `questions` maps your IDs to typed questions. There are three types:
  **choice**, **score**, and **noul**. See [Primitives](primitives/index.md).

The model is fixed for each server process, so there is no `model` field to
send. Set `SYSTEM_MODEL` before startup to choose a supported model. The
response's `model` field reports the engine that answered.

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

The engine answers one focused question at a time. It handles quick judgments
with clear context. It is not built for multi-step reasoning.

When a decision depends on several factors, do not write one big question.
Ask one question per factor and combine the answers in code, where you
control the weights. Changing a weight is a one-line edit; changing your mind
inside a prompt is a rewrite.

## What is honest about this POC

- The default is a 4B model
  (`mlx-community/Qwen3-4B-Instruct-2507-4bit`). A smaller pinned 1.7B model is
  also supported. Neither is a frontier model.
- Probabilities come from one masked read of the option letters
  ([How it works](how-it-works.md)). They are not tuned against real
  outcomes yet, so treat them as ranked signal, not as calibrated odds.
- `server/tools/evals.py` measures accuracy and option-order stability against
  labeled data. The public dataset is the next release step. Calibration work
  is planned but not done.

## Next steps

- [Quickstart](quickstart.md) — run the server and make a call.
- [Primitives](primitives/index.md) — the three question types in depth.
- [How it works](how-it-works.md) — what happens inside the engine.
