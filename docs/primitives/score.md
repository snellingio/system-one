# Score

A Score places the state on a ladder of levels you write, lowest to highest.
The answer is a point on that ladder — and because it is an average, it can
settle between two levels, which is often exactly what you want.

Use it when the answer is a degree, not a bucket: severity, sentiment,
seniority, confidence of an extraction. If instead you are picking a member of
a closed unordered set, use a [Choice](choice.md).

## Writing one

`criteria` lists the levels in order, 2 to 10 of them:

```json
"frustration": {
  "type": "score",
  "instructions": "How upset does the customer sound?",
  "criteria": [
    "Neutral, just asking a question",
    "Annoyed but polite",
    "Angry, ready to walk away"
  ]
}
```

The scale runs front to back: index 0 is the low end, the last entry the
high end. Write each level so a reader could sort real examples into
them without seeing the others. Three or four levels usually beat six —
finer ladders need finer distinctions, and adjacent levels that blur
together flatten the distribution.

Keep levels one-dimensional. "Angry and asking for a refund" is two axes;
make it two questions and combine them in code.

## What comes back

```json
"frustration": {
  "type": "score",
  "score": 0.762,
  "legend": {
    "0": "Neutral, just asking a question",
    "1": "Annoyed but polite",
    "2": "Angry, ready to walk away"
  },
  "probabilities": { "0": 0.377, "1": 0.484, "2": 0.139 },
  "confidence": 0.226
}
```

- `probabilities` is the distribution over level indices (as string keys),
  summing to 1.
- `score` is the probability-weighted average of the indices: here
  `0×0.377 + 1×0.484 + 2×0.139 = 0.762`. Between "neutral" and "annoyed",
  closer to "annoyed".
- `legend` maps each index back to your description, so the answer is
  self-describing in logs.
- `confidence` uses the same formula as Choice over the level distribution.

## Reading a between-levels score

A score of 1.0 says the model piled its weight on level 1. A score of 0.5
with a flat distribution means something different: the model cannot choose,
and the average happens to land mid-way. Check `probabilities` (or
`confidence`) before treating a mid-scale score as a measured "medium":

- Peaked distribution → the score is a real position. Threshold it.
- Flat distribution → the score is doubt averaged into a number. Escalate
  or improve the levels.

## Combining Scores

Scores compose cleanly because they are numbers on comparable scales.
Normalize each to 0–1 (divide by `len(criteria) − 1`), weight, and sum:

```python
def norm(ans, n_levels):
    return ans.score / (n_levels - 1)

urgency = (
    2.0 * norm(answers["severity"], 3)
    + 1.0 * norm(answers["frustration"], 3)
    + 0.5 * norm(answers["customer_tier"], 4)
)
```

When the mix is wrong, the fix is a coefficient in your code, not a new
prompt. This is the main reason to split a compound judgment into one Score
per factor.
