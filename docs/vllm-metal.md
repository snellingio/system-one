# vllm-metal backend

System One can use a separate vllm-metal server instead of loading the model
in the API process. The HTTP API and answer shapes stay the same.

Install vllm-metal in its own Homebrew environment:

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

Start vllm-metal with the same pinned model revision as the answer-code
registry:

```bash
vllm serve mlx-community/Qwen3-1.7B-4bit \
  --revision 3b1b1768f8f8cf8351c712464f906e86c2b8269e \
  --max-model-len 32769 \
  --port 8000
```

The extra model-length slot is for vllM's internal output token. System One
still limits each prompt to 32,768 input tokens.

Then start System One in another terminal:

```bash
cd server
SYSTEM_ONE_BACKEND=vllm-metal \
SYSTEM_ONE_VLLM_BASE_URL=http://127.0.0.1:8000 \
uv run uvicorn system_one_lite.api:app --port 8010
```

The backend builds each prompt with System One's pinned tokenizer. It sends
the exact prompt token IDs in completion batches. It also asks vllm-metal
for the exact token IDs of every allowed answer code, then normalizes those
scores at System One's temperature. It does not use a top-k cutoff or guess a
score for a missing answer. Question IDs are still used only to route answers
and are never sent to the model.

vLLM accepts up to 128 requested token IDs in one read. A question with more
options uses more than one exact read. Each read makes vllm-metal sample one
internal token so it can return the requested log probabilities. The System
One response reports that work in `usage.output_tokens`. The local MLX backend
still reports zero output tokens.

Before inference, the backend checks both token limits. It also asks
vllm-metal to decode the full answer-code registry. A server tokenizer
mismatch stops the request instead of reading the wrong token IDs.

## Current limit

This backend does not use the seeded parallel canvas from vLLM PR 57250. Use
the [4-bit DiffusionGemma backend](diffusion-gemma.md) for that execution
shape. MLX-VLM already implements DiffusionGemma on Apple Silicon, so the
seeded read endpoint lives there instead of adapting vllm-metal's
autoregressive scheduler.

Keep vllm-metal outside the `server` uv environment. Its release pins an exact
MLX build for its native Metal extension, while System One has its own MLX
dependency.
