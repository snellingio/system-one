# Run with 4-bit DiffusionGemma

This backend uses `mlx-community/diffusiongemma-26B-A4B-it-4bit` at the pinned
revision in `engine.py`. It sends every question in one prompt and places one
`- label` answer slot per question in DiffusionGemma's active canvas. This shape
fits System One's limit of 64 questions. One read-only model pass returns the
requested answer-code scores. The canvas uses the same fixed seed for each
request, so the same input has the same canvas.

The backend needs the `feature/diffusion-gemma-reads` branch of MLX-VLM. That
branch adds `/v1/diffusion/reads` without changing normal generation.

Start the MLX-VLM server from its checkout:

```bash
cd /path/to/mlx-vlm
uv run --with-editable . python -m mlx_vlm.server --port 8080
```

Start System One in another terminal:

```bash
cd server
SYSTEM_ONE_BACKEND=mlx-vlm-diffusion \
  uv run uvicorn system_one_lite.api:app --port 8010
```

Use the compact profile for short, latency-sensitive questions:

```bash
SYSTEM_ONE_BACKEND=mlx-vlm-diffusion-fast \
  uv run uvicorn system_one_lite.api:app --port 8010
```

The compact profile accepts exactly one question. It removes the long prompt
instructions and uses one active canvas token. Use it only when the state,
question, and answer labels make the task clear without extra guidance.
It runs six full encoder layers. Later layers build lighter attention caches.
This approximation needs an accuracy check for each target task.

The first System One startup downloads the pinned 4-bit model snapshot. The
weights are about 16.5 GB. Both processes use the same Hugging Face cache.

The read endpoint accepts prompt tokens, an active seed canvas, and exact token
IDs for each answer slot. It checks every token and position before the model
runs. It scores only the requested answer codes and skips the full vocabulary
head. System One then applies its usual softmax over each question's valid
answer codes.

This path differs from the vllm-metal backend in two ways:

- All questions share one prompt and one denoising forward.
- The answer slots can attend to the fixed answer template and to one another.

The public System One request and response formats do not change. Read-only
canvas work reports zero output tokens because no tokens are committed.
