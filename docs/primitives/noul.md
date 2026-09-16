# Noul

A Noul is one statement to verify against the state. The answer is one
number between 0 and 1: the probability the statement is true.

Use it for clean yes/no judgments where the lean itself is the signal — does
this email ask us to reply off-platform, does this changelog mention a
breaking change, is this candidate's phone number present. If the answer is a
degree, that is a [Score](score.md); if it is a pick from a list, that is a
[Choice](choice.md).

## Writing one

```json
"asks_for_refund": {
  "type": "noul",
  "instructions": "Does the customer ask for money back?",
  "criteria": {
    "true": "A refund or chargeback is requested",
    "false": "No refund is requested"
  }
}
```

`instructions` works best as a statement or a question with a single,
checkable condition. `criteria` is optional and exists to sharpen the edges:
`true` says what counts as yes, `false` says what counts as no. Skip it when
the statement is already unambiguous.

Write the condition so a careful reader with only the state in front of them
could mark yes or no. Two traps:

- **Vague adjectives.** "Is the candidate experienced?" — experienced by
  whose standard? Either define it ("has held a paid role using Go for at
  least two years, per `work_history`") or turn it into a Score with levels
  you write.
- **Compound statements.** "Does the report mention the version and the
  browser?" is two questions glued together; split them and combine in code.

## What comes back

```json
"asks_for_refund": {
  "type": "noul",
  "noul": 0.622
}
```

That is the whole answer. The engine scores `yes` and `no` as a two-option
Choice and returns the probability of `yes`:

- near 1 — a strong yes;
- near 0 — a strong no;
- around 0.5 — the model cannot tell from the state it was given.

There is no `confidence` field. For a two-way split, the distance from 0.5
already says how decided the model is, so a separate statistic would repeat
it.

## Using the number

You can hard-threshold it (`noul >= 0.5` is how the eval suite marks a yes),
but the value earns its keep when your code treats it as evidence:

```python
refund_p = answers["asks_for_refund"].noul

if refund_p > 0.8 and order.total < 50:
    auto_draft_refund(order)          # strong yes, small money
elif refund_p > 0.5:
    ask_agent_to_confirm(order)       # leaning yes, worth a look
else:
    attach_note(order, "no refund requested")
```

A reading like 0.622 — leaning yes without claiming it — is a normal,
honest outcome for states that hint without stating ("second late delivery
this month, where is my package" complains about money's worth but never
asks for it back). Route those cases the way you would route a colleague
saying "probably, check it".
