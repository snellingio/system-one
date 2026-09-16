# Choice

A Choice asks which option fits and answers with one pick plus a probability
for every option. Use it whenever the answer is a member of a closed, unordered
set: routing, categorizing, language detection, picking a template.

When the answer is a degree instead of a pick, use a [Score](score.md).
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

`criteria` maps each option name to a description. The model sees both values.
Descriptions define the lines between options. Use `null` when the name alone
is clear (`"english": null`).

An option needs `null` or a description, never an empty string — an empty
line in the prompt weakens that option for no reason.

Two habits that pay off:

- **Add an escape hatch.** No fixed list covers every input, so give the
  model an `"other": "None of the above"` option. Without one, odd inputs get
  force-fit into a wrong bucket with made-up confidence.
- **Split near-twins.** If two options keep splitting probability on inputs
  where you think the answer is clear, their descriptions overlap. Rewrite
  them to say what each is *not* for, or merge them and split elsewhere.

Options use answer codes `A`–`Z`, then `AA`, `AB`, and more. Every code is one
token. The model supports 578 options, not 26. Long lists work, but each option
adds a prompt line. Prefer ten clear options to fifty vague ones.

## What comes back

An illustrative response:

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

The second-highest value can matter. In this example, `billing` at 0.258 is
large enough to consider. A dispatcher can use it like this:

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

Options appear as `A:`, `B:`, `C:`, and more. The model answers by code. List
order can change probabilities. The eval suite rotates options and checks the
winner. Keep criteria order fixed when a Choice matters. You can also measure
stability with `server/tools/evals.py`.
