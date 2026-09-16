# Primitives

A primitive is one typed question plus the typed answer that comes back.
There are three, and they cover the shapes a judgment can take:

| Type | Question it answers | Answer fields |
| --- | --- | --- |
| [Choice](choice.md) | Which option applies? | `choice`, `probabilities`, `confidence` |
| [Score](score.md) | Where on this ladder? | `score`, `legend`, `probabilities`, `confidence` |
| [Noul](noul.md) | Is this true? | `noul` |

## The shape of a question

Every question is an entry in the `questions` map. You pick the ID. The body
always has `type` and `instructions`; Choice and Score add `criteria`:

```json
"wants_refund": {
  "type": "noul",
  "instructions": "Does `message` ask for money back?",
  "criteria": {
    "true": "A refund or chargeback is requested",
    "false": "No refund is requested"
  }
}
```

- **ID** (`wants_refund` above) — your handle for the answer. It never
  reaches the model, so say the whole thing in `instructions` even when the
  ID looks self-explanatory.
- **`instructions`** — the judgment, written as a question or a statement to
  judge. String, object, or array; non-strings are rendered as JSON.
- **`criteria`** — the answer space: option names and descriptions for a
  Choice, ordered levels for a Score, optional yes/no definitions for a Noul.

## Picking a type

The type follows from the shape of answer your code needs:

- **Choice** when the answer names one member of a closed, unordered set:
  which team, which category, which language. Add an `other` option so odd
  cases have somewhere honest to land.
- **Score** when the answer is a degree and you can spell out its rungs:
  severity, frustration, experience. Two to ten levels, lowest to
  highest.
- **Noul** when the answer is a plain yes/no and the interesting part is how
  strongly it leans: does this report include repro steps, is this email a
  cold outreach.

When two types both look right, pick the one whose answer needs the least
glue in code: a Choice feeds a `match`, a Score feeds a threshold, a Noul
feeds an `if`. And keep yes/no and scaling separate: a Noul at 0.5 means
"cannot tell", not "medium" — if you want medium, that is a Score.

## Asking several questions

Batch every question about a state into one request. Each question sees
only the state and its own text, so answers cannot change each other, and
adding a question never shifts another answer. Each question is still a
full prompt (the state is copied), and the engine stacks those prompts into
one forward pass. Extra questions are cheap enough to ask speculatively:
severity still costs one short prompt even on tickets that turn out not to
be bugs, and the code simply skips what it does not need.

If a follow-up truly depends on an earlier answer — for instance the answer
decides which options the next question should offer — make a second
request. That is the exception; when in doubt, ask everything up front and
sort it out in code.

## Composing answers

Each answer is a small typed value: compare it, threshold it, weight it.
When a decision needs several factors, ask one question per factor and do
the combining in code:

```python
priority = (
    2.0 * answers["severity"].score
    + 1.0 * answers["frustration"].score
    + 0.5 * answers["repro_quality"].score
)
```

The weights are plain coefficients in your code. When the mix stops matching
your team's judgment, change one and rerun — no prompt surgery.

## Next steps

- [Choice](choice.md) — options, descriptions, and reading the distribution.
- [Score](score.md) — levels, the between-levels score, and legends.
- [Noul](noul.md) — yes/no probability and how to write the statement.
- [Confidence](../confidence.md) — the spread statistic on Choice and Score
  answers, and how to threshold it.
