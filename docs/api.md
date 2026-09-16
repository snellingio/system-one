# API reference

One endpoint does everything. Send a state and a map of questions; get back
a typed answer for every question, keyed by the IDs you chose.

- `POST /evaluate` — the endpoint these docs use.
- `POST /v1/systemone` — an alias with the same body and the same response.

## Request body

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `state` | `string \| object \| array` | yes | The content to judge. See [State](state.md). |
| `questions` | `map<string, Question>` | yes | At least one entry. Each key is an ID you pick; the answer comes back under it. The engine never sees your IDs. |

There is no `model` field. Each server process uses one fixed model. Set
`SYSTEM_MODEL` before startup to choose a supported model. The response's
`model` field reports the engine that actually answered.

IDs only route answers back to your code. Two requests can use the same option
names with different IDs, and renaming an ID never changes the probabilities.

## Question types

Every question has `type` and `instructions`. Choice and Score add
`criteria`. `instructions` accepts a string or structured data — objects and
arrays are rendered as JSON, so structured guidance reaches the model as
readable text.

### Choice

Chooses one option from your list.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | `"choice"` | yes | |
| `instructions` | `string \| object \| array` | yes | What to decide. |
| `criteria` | `map<string, Content \| null>` | yes | At least one option. Option name → description. `null` means the name says it all. The model sees both the name and description. Options are answer-coded `A`–`Z`, then `AA`, `AB`, …; at most 578 options. |

```json
"language": {
  "type": "choice",
  "instructions": "Which language is this changelog entry written in?",
  "criteria": {
    "english": null,
    "german": null,
    "japanese": null,
    "other": "None of the above"
  }
}
```

### Score

Places the state on a ladder of levels, from low to high.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | `"score"` | yes | |
| `instructions` | `string \| object \| array` | yes | What to rate. |
| `criteria` | `Content[]` | yes | 2 to 10 level descriptions, ordered lowest to highest. |

```json
"severity": {
  "type": "score",
  "instructions": "How badly does this bug affect users?",
  "criteria": [
    "Cosmetic or trivial",
    "Blocks one user, workaround exists",
    "Blocks many users, no workaround"
  ]
}
```

### Noul

How likely the statement is to be true.

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `type` | `"noul"` | yes | |
| `instructions` | `string \| object \| array` | yes | The statement to judge. |
| `criteria` | `object` | no | Optional `true` and `false` strings that sharpen the meanings of yes and no. |

```json
"mentions_repro": {
  "type": "noul",
  "instructions": "The report includes steps to reproduce the bug.",
  "criteria": {
    "true": "Steps or a recipe a developer could follow",
    "false": "No reproduction information given"
  }
}
```

## Response body

| Field | Type | Notes |
| --- | --- | --- |
| `model` | `string` | The selected engine's model ID. The default is `mlx-community/Qwen3-4B-Instruct-2507-4bit`. |
| `answers` | `map<string, Answer>` | One answer per question, keyed by your IDs, in request order. |
| `usage.input_tokens` | `integer` | Total tokens evaluated across the independent question prompts. |
| `usage.output_tokens` | `integer` | Always 0. Nothing is generated. |

### Choice answer

| Field | Type | Notes |
| --- | --- | --- |
| `type` | `"choice"` | Answer type discriminator. |
| `choice` | `string` | The option with the highest probability. |
| `probabilities` | `map<string, number>` | Every option → probability. The values sum to 1. |
| `confidence` | `number` | 0 to 1, from the spread of `probabilities`. See [Confidence](confidence.md). |

### Score answer

| Field | Type | Notes |
| --- | --- | --- |
| `type` | `"score"` | Answer type discriminator. |
| `score` | `number` | Average level index weighted by probability. Can fall between levels. |
| `legend` | `map<string, Content>` | Level index (as a string) → your level description. |
| `probabilities` | `map<string, number>` | Level index (as a string) → probability. The values sum to 1. |
| `confidence` | `number` | Same statistic as Choice, over the level distribution. |

### Noul answer

| Field | Type | Notes |
| --- | --- | --- |
| `type` | `"noul"` | Answer type discriminator. |
| `noul` | `number` | Probability of yes, 0 to 1. There is no separate confidence field; the distance from 0.5 carries that information. |

## Limits and errors

The server validates before it runs the engine and rejects bad requests with
`422`. Explicit contract errors use a string `detail`; FastAPI schema errors
use an array of error objects.

| Rejection | Reason |
| --- | --- |
| `questions` empty or missing | At least one question is required. |
| More than 64 questions | A request may contain at most 64 questions. |
| Unknown `type` | Must be `choice`, `score`, or `noul`. |
| Unknown request field | Misspelled and unsupported fields are rejected. |
| Choice with no options | A Choice needs at least one option. |
| More options in a Choice than the engine has answer codes | Each option needs its own single-token answer code (`A`–`Z`, then two-letter codes). The model registry holds 578. |
| Fewer than 2 or more than 10 Score levels | The schema enforces both bounds. |
| Content or ID over its character limit | State: 100,000; instructions and each criterion: 20,000; IDs and option names: 200. |
| Request content over 1,000,000 bytes | The combined validated request must stay within the aggregate limit. |
| Prompt or request over its token limit | Each prompt may use at most 32,768 tokens; one request may use at most 131,072 input tokens. |

If an answer code does not land as a single token, the server returns `422`.
You get an error, never a wrong answer.

Only one inference request runs at a time. If the engine is already in use,
the server returns `503` with `detail: "the inference engine is busy"`.
Each question uses a full independent prompt. `usage.input_tokens` counts
the state again for each question.
