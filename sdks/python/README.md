# Python SDK (local clone)

A local clone of the `system-sdk` API, pointed at the System One Lite
server (`server/src/system_one_lite/api.py`). Zero dependencies, Python 3.9 or newer.

## Use

```python
from system_sdk import Choice, Noul, Score, SystemClient

with SystemClient() as client:
    response = client.system_one(
        state={"document": "I was charged twice. Please fix this ASAP."},
        questions={
            "billing": Noul(instructions="Is this ticket about billing?"),
            "tone": Choice(
                instructions="What is the customer's tone?",
                criteria={"calm": None, "frustrated": None, "angry": None},
            ),
            "urgency": Score(
                instructions="How urgent is this ticket?",
                criteria=["can wait", "this week", "today"],
            ),
        },
    )

print(response.nouls["billing"].noul)
print(response.choices["tone"].choice)
print(response.scores["urgency"].score)
```

`AsyncSystemClient` has the same shape: `async with` and
`await client.system_one(...)`. It runs the sync call in a thread, so it
needs no async HTTP dependency.

Answers carry the full shape from `docs/api.md`. Choices have `choice`,
`probabilities`, `confidence`; Scores add `score` and `legend`; Nouls are a
single `noul` number. Read every answer from `response.answers`, or use the
`nouls`, `choices`, and `scores` views. Score legend and probability keys are
integers after parsing. Errors raise `SystemRequestError` with the HTTP status
and body.

## Configuration

- `SYSTEM_BASE_URL` — where to call. Defaults to
  `http://127.0.0.1:8010`.
- You can also pass it to the constructor:
  `SystemClient(base_url=...)`.

## Files

- `system_sdk/questions.py` — `Noul`, `Choice`, `Score`.
- `system_sdk/responses.py` — typed `answers` plus the `nouls`, `choices`,
  and `scores` views.
- `system_sdk/client.py` — sync and async clients.
- `examples/quickstart.py` — the docs example, sync and async.
