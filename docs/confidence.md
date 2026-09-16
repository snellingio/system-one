# Confidence

Choice and Score answers carry `confidence`, a number from 0 to 1 that says
how decided the model is. Noul answers do not carry one — the probability
itself already tells you how far the model leans.

## The exact formula

`confidence` is computed from the answer's own probability distribution by
`confidence()` in `server/src/system_one_lite/prompts.py`:

```
confidence = (max_p - 1/n) / (1 - 1/n)
```

where `n` is the number of options (or levels) and `max_p` is the largest
probability. Two anchors:

- **Uniform** distribution (`max_p = 1/n`): confidence is exactly 0. The
  model cannot tell the options apart at all.
- **One-hot** distribution (`max_p = 1`): confidence is exactly 1. All
  probability on one option.

In between, it is the top probability's lead over uniform, rescaled to the
full 0–1 range. Two consequences worth knowing:

- A fixed peak scores higher with more options. A 0.62 top probability is
  0.43 confidence among 3 options, but 0.58 among 10.
- It is deliberately not just `max_p`. A 2-option answer at 0.62/0.38 is
  barely better than a coin flip, and the formula reflects that: confidence
  0.24, not 0.62.

Because it is a plain function of `probabilities`, you can compute your own
statistic instead — the full distribution is always in the response, and
nothing on the server side depends on this particular choice.

## Reading a low confidence

Low confidence is information, not failure. It usually means one of:

- The state genuinely sits between options ("asks for refund" when the
  customer only grumbles about money).
- Two options overlap and their descriptions do not draw a sharp line.
  Rewriting the criteria fixes this.
- The state does not contain what the question needs. The honest answer is
  doubt; no prompt trick makes missing evidence appear.

## Thresholds in code

A workable starting pattern is three bands:

- **High** — act on the answer without asking anyone.
- **Middle** — act, but flag for review or get a human sign-off.
- **Low** — do not act. Escalate to a person, ask a clarifying question, or
  fall back.

Where the lines sit should follow the cost of being wrong, not a universal
number. The same 0.5 answer is fine for sorting a queue and reckless for
moving money:

```python
if dept.confidence < 0.35:
    route_to_human(ticket)          # undecided: a person looks at it
elif dept.choice == "deliveries":
    assign(ticket, "deliveries")    # cheap to undo
elif dept.choice == "billing":
    if dept.confidence > 0.75:
        draft_refund(ticket)        # higher bar for money
    else:
        ask_agent_to_confirm(ticket)
```

These thresholds are starting points. Tune them against your own data, and
re-tune when you change models — the number means "how peaked the
distribution is", and its relationship to being right depends on the engine
and on your questions.

## Calibration, honestly

A high confidence says the model found one clear option. It does not
guarantee that option is correct. Making `confidence` track real accuracy —
calibration — needs training or post-hoc scaling against labeled outcomes,
and that work is not done in this POC. Until it is:

- Use `confidence` to rank and to route, not as a probability of being
  right.
- Prefer `probabilities` when you compare across questions; the raw
  distribution survives any threshold choice you make later.
- Measure on your own inputs. `server/tools/evals.py` shows the pattern: run gold
  labeled records, then check accuracy at each confidence cutoff to find the
  bands that actually hold up.
