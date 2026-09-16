# System One Lite

Typed decisions from a local language model, without text generation.

System One Lite turns unstructured input into answers your code can use. Send
one state plus one or more typed questions. The server returns a probability
distribution for each answer. It does not generate or parse free-form text.

This is a proof of concept for routing, scoring, gating, and other small
decisions that sit inside normal software. It runs a stock open-weight model
locally with MLX.

## Why this exists

Most LLM features follow the same loop:

1. Ask for text.
2. Parse the text into a schema.
3. Retry when the text does not fit.

System One Lite replaces that loop with a masked read. Each question defines
the only valid answers. The engine reads the model's next-token scores for
those answers, masks everything else, and applies softmax. The result is typed
by construction.

This gives you:

- no malformed output;
- no generated tokens;
- a full probability distribution, not only a winning label;
- independent questions whose answers cannot affect each other; and
- local inference with a fixed model.

## Requirements

- An Apple silicon Mac. The current server uses MLX.
- Python 3.12 or newer.
- [uv](https://docs.astral.sh/uv/).
- Internet access on the first run to download the default model.

## Quickstart

Start the server:

```bash
cd server
uv sync
uv run uvicorn system_one_lite.api:app --port 8010
```

The first start loads `mlx-community/Qwen3-1.7B-4bit` and compiles the
Metal kernels. Then send a request:

```bash
curl -s http://127.0.0.1:8010/evaluate \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "state": "My order was due Friday, but it is still in transit.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {
        "deliveries": "Late, missing, or damaged orders",
        "billing": "Charges, refunds, or payment methods",
        "account": "Login, profile, or app problems"
      }
    },
    "needs_reply": {
      "type": "noul",
      "instructions": "Does the customer need a reply?"
    }
  }
}
EOF
```

The response has one typed answer for each question:

```json
{
  "model": "mlx-community/Qwen3-1.7B-4bit",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "deliveries",
      "probabilities": {
        "deliveries": 0.71,
        "billing": 0.18,
        "account": 0.11
      },
      "confidence": 0.565
    },
    "needs_reply": {
      "type": "noul",
      "noul": 0.88
    }
  },
  "usage": {
    "input_tokens": 94,
    "output_tokens": 0
  }
}
```

The numbers above show the response shape. Exact values depend on the model
and input.

For a longer example with all three question types, see the
[quickstart](docs/quickstart.md).

## Question types

| Type | Use it for | Returned value |
| --- | --- | --- |
| [Choice](docs/primitives/choice.md) | One option from a closed set | Winner, probabilities, and confidence |
| [Score](docs/primitives/score.md) | A position on an ordered scale | Weighted score, level probabilities, and confidence |
| [Noul](docs/primitives/noul.md) | A yes or no judgment | Probability of yes |

You can send up to 64 questions in one request. Each question gets its own
full prompt with the shared state. The engine runs one model pass per question
and returns every answer under the ID you chose.

## How it works

For each question, the server:

1. Writes the state, question, and allowed answers into a prompt.
2. Gives each answer a token code such as `A`, `B`, or `C`.
3. Runs the model up to the answer slot without decoding text.
4. Keeps only the logits for valid answer codes.
5. Applies softmax and maps the probabilities back to your labels.

Choice returns the most likely label and the full distribution. Score returns
the probability-weighted level. Noul returns the probability of `yes`.

Read [How it works](docs/how-it-works.md) for token alignment, limits, and the
reason each question uses an independent prompt.

## Use an SDK

The repository includes two local SDKs:

- [Python](sdks/python/README.md): sync and async clients with no runtime
  dependencies. Python 3.9 or newer.
- [JavaScript](sdks/javascript/README.md): a typed client for Node 22.18 or
  newer.

Both clients use `http://127.0.0.1:8010` by default. Set `SYSTEM_BASE_URL` to
change it.

## Measure it

Run the labeled eval sets against the local model:

```bash
cd server
uv run python -m tools.evals --limit 20
```

The 1.7B model is the `default` profile. The 4B Instruct model is the
`larger` profile. Download either model before a run with:

```bash
uv run python -m tools.download_model default
uv run python -m tools.download_model larger
```

The demo, benchmark, and eval tools accept either profile. For example:

```bash
uv run python -m tools.evals --model larger --limit 20
```

The server uses `default` unless `SYSTEM_ONE_MODEL` selects another profile:

```bash
SYSTEM_ONE_MODEL=larger uv run uvicorn system_one_lite.api:app --port 8010
```

The eval runner reports accuracy by question type and checks whether rotating
Choice options changes the winner. The repository also includes generated
examples, public benchmark samples, and game states with exact or
policy-derived labels under [`datasets/`](datasets/README.md).

Run the local checks:

```bash
cd server
uv run ruff check src tools tests
uv run ruff format --check src tools tests
uv run pytest

cd ../sdks/javascript
npm run typecheck
```

## Limits

System One Lite is an experiment, not a production decision service.

- The default model is small. It will not match a frontier model on hard
  judgments.
- Returned probabilities are model scores. They are not calibrated odds of
  being correct.
- Synthetic eval data does not cover real production traffic.
- Each question repeats the state and runs separately, so cost grows with the
  number and length of questions.
- The server accepts one inference request at a time. It returns `503` while
  the engine is busy.

Use confidence to route uncertain cases, then set thresholds from labeled data
that matches your own traffic. See [Confidence](docs/confidence.md) for the
formula and its limits.

## Project map

| Path | Purpose |
| --- | --- |
| [`server/`](server/) | FastAPI service, MLX engine, evals, and tests |
| [`sdks/python/`](sdks/python/) | Python client |
| [`sdks/javascript/`](sdks/javascript/) | TypeScript client |
| [`datasets/`](datasets/) | Labeled examples and benchmark samples |
| [`docs/`](docs/) | Guides and API reference |

Start with the [introduction](docs/introduction.md), then read the
[API reference](docs/api.md) for the full request shape, limits, and errors.
