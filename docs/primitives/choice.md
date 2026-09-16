# Choice

A Choice asks which option fits and answers with one pick plus a probability
for every option. Use it whenever the answer is a member of a closed, unordered
set: routing, categorizing, language detection, picking a template.

When the answer is a degree rather than a pick, use a [Score](score.md).
When it is plain yes/no, use a [Noul](noul.md).

## Writing one

```json
"topic": {
  "type": "choice",
  "instructions": "Which team does this message belong to?",
  "criteria": {
    "deliveries": "Late, missing, or damaged orders",
    "billing": "Charges, refunds, payment methods",
    "account": "Login, profile, and app problems"
  }
}
```

`criteria` maps each option name to a description of what it covers. The
descriptions are the real classifier: the model sees both the name and the
description, so the description is where you draw the lines between options.
Use `null` when the name alone is unambiguous (`"english": null`).

An option needs `null` or a description, never an empty string — an empty
line in the prompt weakens that option for no reason.

Two habits that pay off:

- **Add an escape hatch.** No fixed list covers every input, so give the
  model an `"other": "None of the above"` option. Without one, odd inputs get
  force-fit into a wrong bucket with made-up confidence.
- **Split near-twins.** If two options keep splitting probability on inputs
  where you think the answer is clear, their descriptions overlap. Rewrite
  them to say what each is *not* for, or merge them and split elsewhere.

Options are answer-coded `A`–`Z`, then `AA`, `AB`, and on — every code a
single token, so the cap is 570 options on the default model, not 26. Long
option lists work, but each option is a line in the prompt and a share of the
softmax — describe them so the distinctions are real, and prefer ten sharp
options to fifty fuzzy ones.

## What comes back

From the quickstart request against the local engine:

```json
"topic": {
  "type": "choice",
  "choice": "deliveries",
  "probabilities": {
    "deliveries": 0.62,
    "billing": 0.258,
    "account": 0.122
  },
  "confidence": 0.429
}
```

- `choice` is the top option. Ties are broken by first-listed order.
- `probabilities` covers every option, sums to 1, and preserves your
  `criteria` order.
- `confidence` is `(max_p − 1/n) / (1 − 1/n)` — see
  [Confidence](../confidence.md).

## Reading the runner-up

The interesting number is often the second one. In the example above,
`billing` at 0.258 is not noise: the customer complains twice in one month,
and billing keeps a live share of the doubt. A reasonable dispatcher does
something with that:

```python
topic = answers["topic"]

if topic.confidence < 0.35:
    route_to_human(message)
else:
    assign(message, topic.choice)
    runner_up, p = max(
        ((o, p) for o, p in topic.probabilities.items() if o != topic.choice),
        key=lambda kv: kv[1],
    )
    if p > 0.25:
        notify_team(message, runner_up)   # second opinion, no re-route
```

## Order stability

Options are listed `A:`, `B:`, `C:` … and read by letter
([How it works](../how-it-works.md)), so list order can nudge probabilities
— the eval suite rotates one question's options and requires the same
winner. The effect is small but real: when a Choice matters, keep the
criteria order fixed between calls, or measure your own stability the way
`server/tools/evals.py` does (winner match plus total-variation distance under
rotation).
