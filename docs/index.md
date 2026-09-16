# System One Lite docs

System One Lite is a proof of concept for a decision model. It reads a state
and answers typed questions. It returns probabilities your code can use.
The model uses a stock open-weight LLM but never generates text.
See [Introduction](introduction.md).

## Start here

- [Introduction](introduction.md) — what the system does and why it is built
  this way.
- [Quickstart](quickstart.md) — start the server and make your first call.
- [How it works](how-it-works.md) — the prompt template, the answer slot, and
  the masked read that turns one forward pass per question into a distribution.

## Reference

- [API](api.md) — request and response shapes for `POST /evaluate`.
- [State](state.md) — what goes in the `state` field and how each format is
  rendered.
- [Confidence](confidence.md) — the exact formula behind `confidence`, and how
  to pick thresholds.
- [Primitives](primitives/index.md) — the three question types:
  [Choice](primitives/choice.md), [Score](primitives/score.md), and
  [Noul](primitives/noul.md).

## Code map

| Path | What it is |
| --- | --- |
| `server/src/system_one_lite/api.py` | FastAPI server, `POST /evaluate` and `POST /v1/systemone` |
| `server/src/system_one_lite/engine.py` | the masked-read engine on MLX |
| `server/tools/gen_answer_codes.py` | creates a pinned single-token answer-code registry for each model |
| `server/src/system_one_lite/prompts.py` | the prompt template, rendering, and the confidence function |
| `server/tools/demo.py` | one-question command line demo |
| `server/tools/evals.py` | scores the engine against gold-labeled datasets |
| `server/tools/ocean/harvest.py` | builds `datasets/ocean_playouts.jsonl` from PufferLib game policies; see [Ocean games](ocean-games.md) |
| `sdks/python/` | local Python SDK against the same server |
| `sdks/javascript/` | local JavaScript SDK |
