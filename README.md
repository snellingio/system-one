# System One Lite

## Stop asking language models to write. Start making them decide.

Language models are brilliant at producing text. Software does not want text.
It wants a route, a score, a yes or no, and an honest signal when the answer is
unclear.

So why are we still asking models to write tiny essays? We parse those essays
back into data, validate the data, retry failures, and hope nothing goes off the
rails.

**System One Lite deletes the essay.**

Send one unstructured state plus up to 64 typed questions. A local model scores
only the answers you allow and returns a probability distribution for every
question.

**Unstructured state in. Typed probabilities out. Zero generated tokens.**

No free-form response. No JSON repair loop. No invented option that your code
has never heard of.

This is an independent proof of concept for the interface behind
[System One Models](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
It is not a new foundation model. It runs a stock open-weight model locally
with MLX and asks a much more interesting question:

> How much simpler does AI software become when the model is only allowed to
> decide?

## The old stack is absurd

| | A normal LLM feature | System One Lite |
| --- | --- | --- |
| Input | Unstructured text | Unstructured text or JSON |
| Output | A generated string | A typed answer over declared options |
| Uncertainty | A guess written in prose | The full probability distribution |
| Validation | Parse, validate, retry | Constrained by construction |
| Output tokens | One token at a time | **Zero** |
| Failure mode | Malformed data or invented values | A valid answer that may still be wrong |
| Deployment | Usually a hosted model | A fixed local model on Apple silicon |

The final row matters. System One Lite does not make a small model infallible.
It makes the limit between the model and your code brutally clear. The
model can choose the wrong declared answer. It cannot create a new one.

That is the difference between asking AI to behave like an API and giving it
an interface it cannot break.

## Three primitives. A ridiculous number of decisions.

| Type | Ask it to | Get back |
| --- | --- | --- |
| [Choice](docs/primitives/choice.md) | Pick from a closed set | Winner, probabilities, and confidence |
| [Score](docs/primitives/score.md) | Judge a position on an ordered scale | Weighted score, level probabilities, and confidence |
| [Noul](docs/primitives/noul.md) | Make a yes or no judgment | Probability of yes |

Route support tickets. Rank leads. Gate a workflow. Score risk. Flag content.
Decide whether a human needs to look. Combine several small judgments into a
larger rule that stays in ordinary code.

Each question is independent. One answer cannot leak into the next. Your code,
not a hidden chain of thought, decides what happens after the probabilities
arrive.

## Watch it decide

System One Lite needs an Apple silicon Mac, Python 3.12 or newer, and
[`uv`](https://docs.astral.sh/uv/). Start the server:

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

One request comes back ready for code:

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

The numbers show the response shape. Exact values depend on the input and
model. The shape does not.

For a longer example with all three question types, open the
[quickstart](docs/quickstart.md).

## Pick the model

The 1.7B model is the `default` profile. The 4B Instruct model is the
`larger` profile. Download either model before a run:

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

Both model repositories are pinned to exact commits and have checked-in
answer-code registries. Change models only after you run the same accuracy and
option-order checks on both.

## The trick is almost offensively simple

For every question, the server:

1. Writes the state, question, and allowed answers into a prompt.
2. Assigns each answer a token code such as `A`, `B`, or `C`.
3. Runs the model up to the answer slot without decoding any text.
4. Throws away every logit except the valid answer codes.
5. Applies softmax and maps the probabilities back to your labels.

The model never gets the chance to ramble. It reaches the exact point where
an answer must appear, and System One Lite reads the scores directly.

Choice returns the winning label and the full distribution. Score returns the
probability-weighted level. Noul returns the probability of `yes`. Every extra
question gets its own full prompt, so questions cannot affect one another.

Read [How it works](docs/how-it-works.md) for token alignment, option limits,
and the reason this safe path uses one model pass per question.

## Extraordinary claims, meet a local eval

There is no benchmark confetti here. The repository has an eval runner for
JSONL files that use the documented dataset envelope:

```bash
cd server
uv run python -m tools.evals --datasets /path/to/jsonl-directory --limit 20
```

The report shows accuracy by question type. It also rotates Choice options and
checks whether changing their order changes the winner. The public dataset is
the next release step and is not in Git yet. Local files under `datasets/` are
ignored, so they cannot be published by accident.

Better yet, add examples from your own traffic. A decision system earns trust
on the states it will actually see, not on a launch graphic.

## Use it from code

The repository includes two local SDKs:

- [Python](sdks/python/README.md): sync and async clients with no runtime
  dependencies. Python 3.9 or newer.
- [JavaScript](sdks/javascript/README.md): a typed client for Node 22.18 or
  newer.

Both clients use `http://127.0.0.1:8010` by default. Set `SYSTEM_BASE_URL` to
change it.

## The part most launch posts bury

System One Lite is an experiment, not a production decision service.

- The default 1.7B model is small. It will not match a frontier model on hard
  judgments.
- The returned probabilities are model scores. They are **not calibrated odds
  of being correct**.
- Synthetic eval data does not stand in for real production traffic.
- Every question repeats the state and runs separately. Cost grows with the
  number and length of questions.
- The server handles one inference request at a time and returns `503` while
  the engine is busy.
- The current server requires Apple silicon because it uses MLX.

Use confidence to route uncertain cases. Set thresholds from labeled data that
matches your traffic. Read [Confidence](docs/confidence.md) before you let a
score trigger anything expensive, sensitive, or hard to undo.

The honest pitch is still exciting. This tiny project turns a normal local LLM
into a typed decision engine. It needs no fine-tuning, text generation, or
parser. That is enough to build a surprising amount of software.

## Run every check

```bash
cd server
uv run ruff check src tools tests
uv run ruff format --check src tools tests
uv run pytest

cd ../sdks/python
uv run --with pytest pytest -q

cd ../javascript
npm ci
npm run typecheck
npm test
```

## Project map

| Path | What is inside |
| --- | --- |
| [`server/`](server/) | FastAPI service, MLX engine, evals, and tests |
| [`sdks/python/`](sdks/python/) | Python client |
| [`sdks/javascript/`](sdks/javascript/) | TypeScript client |
| [`docs/`](docs/) | Guides and API reference |

Start with the [introduction](docs/introduction.md). Then read the
[API reference](docs/api.md) for the complete request shape, limits, and error
responses.

## License

[MIT](LICENSE). The supported MLX model repositories use Apache 2.0.
