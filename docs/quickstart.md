# Quickstart

Start the server, send one request with all three question types, and read
the answers. Takes about five minutes.

## 1. Start the server

```bash
cd server
uv sync
uv run uvicorn system_one_lite.api:app --port 8010
```

The first start takes a moment. The engine loads the model and runs one warm
up call to compile the Metal kernels. Watch for the usual uvicorn line:

```
INFO:     Uvicorn running on http://127.0.0.1:8010
```

You can also check it without HTTP:

```bash
cd server
uv run python -m tools.demo
```

The demo tool sends one Choice question and prints a probability bar per option.

To compare warmed engine latency for a request with one question and a request
with three questions, run:

```bash
cd server
uv run python -m tools.benchmark --repeats 10
```

The benchmark excludes model-load and initial compilation time. It prints
prompt-building time and model time separately. Add `--json` for
machine-readable output. Use the smaller test model with:

```bash
uv run python -m tools.benchmark \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --repeats 10
```

Test shared-prefix caching without changing the server path:

```bash
uv run python -m tools.cache_experiment
```

## 2. Send a request

One request carries the state and every question you want answered about it.
Mix question types freely:

```bash
curl -s http://127.0.0.1:8010/evaluate \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "state": "Order DB-4471 was supposed to arrive Friday and it still shows 'in transit'. This is the second late delivery this month and the support chat kept disconnecting me. I want to know where my package is.",
  "questions": {
    "topic": {
      "type": "choice",
      "instructions": "Which team does this message belong to?",
      "criteria": {
        "deliveries": "Late, missing, or damaged orders",
        "billing": "Charges, refunds, payment methods",
        "account": "Login, profile, and app problems"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How upset does the customer sound?",
      "criteria": [
        "Neutral, just asking a question",
        "Annoyed but polite",
        "Angry, ready to walk away"
      ]
    },
    "asks_for_refund": {
      "type": "noul",
      "instructions": "Does the customer ask for money back?",
      "criteria": {
        "true": "A refund or chargeback is requested",
        "false": "No refund is requested"
      }
    }
  }
}
EOF
```

## 3. Read the response

This is the real response from a local run of that request (a
`Qwen3.5-2B-MLX-4bit` engine, warm):

```json
{
  "model": "mlx-community/Qwen3.5-2B-MLX-4bit",
  "answers": {
    "topic": {
      "type": "choice",
      "choice": "deliveries",
      "probabilities": {
        "deliveries": 0.444,
        "billing": 0.346,
        "account": 0.210
      },
      "confidence": 0.166
    },
    "frustration": {
      "type": "score",
      "score": 0.894,
      "legend": {
        "0": "Neutral, just asking a question",
        "1": "Annoyed but polite",
        "2": "Angry, ready to walk away"
      },
      "probabilities": { "0": 0.269, "1": 0.569, "2": 0.163 },
      "confidence": 0.353
    },
    "asks_for_refund": {
      "type": "noul",
      "noul": 0.622
    }
  },
  "usage": { "input_tokens": 275, "output_tokens": 0 }
}
```

Things worth noticing:

- `topic` picks `deliveries`, but `billing` holds 0.346 — the message never
  mentions money, yet the model keeps a real share of doubt. `confidence`
  (0.166) summarizes that spread.
- `frustration` lands at 0.894, between "annoyed" (1) and "angry" (2). The
  score is an average over the levels, so it can fall between them.
- `asks_for_refund` is 0.622: leaning yes, but the customer only complains
  about lateness, never asks for money back. The doubt is the honest answer.
- `usage.output_tokens` is always 0. The engine never generates. Each question
  uses a full independent prompt.

## 4. Use the SDK (optional)

`sdks/python/` wraps the same call with typed questions and answers:

```python
from system_sdk import Choice, Noul, Score, SystemClient

with SystemClient() as client:  # defaults to http://127.0.0.1:8010
    response = client.system_one(
        state={"order": "DB-4471", "status": "in transit", "due": "Friday"},
        questions={
            "topic": Choice(
                instructions="Which team does this message belong to?",
                criteria={
                    "deliveries": "Late, missing, or damaged orders",
                    "billing": "Charges, refunds, payment methods",
                    "account": "Login, profile, and app problems",
                },
            ),
            "frustration": Score(
                instructions="How upset does the customer sound?",
                criteria=[
                    "Neutral, just asking a question",
                    "Annoyed but polite",
                    "Angry, ready to walk away",
                ],
            ),
            "asks_for_refund": Noul(
                instructions="Does the customer ask for money back?",
            ),
        },
    )

print(response.choices["topic"].choice)      # "deliveries"
print(response.scores["frustration"].score)  # 0.894
print(response.nouls["asks_for_refund"].noul)  # 0.651
```

Set `SYSTEM_BASE_URL` to point it somewhere else. See `sdks/python/README.md`.

## Where to go next

- [API](api.md) — every field, limit, and error.
- [State](state.md) — structuring the input.
- [Confidence](confidence.md) — reading `confidence` and `probabilities`.
