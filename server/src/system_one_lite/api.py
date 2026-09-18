"""FastAPI application for typed local model decisions.

Run with ``uv run uvicorn system_one_lite.api:app --port 8010``.
"""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException

from .engine import Engine, RequestContractError
from .mlx_vlm_diffusion import MlxVlmDiffusionEngine, MlxVlmError
from .prompts import as_text, confidence, label
from .schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluateRequest,
    EvaluateResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    Usage,
)
from .vllm_metal import VllmMetalEngine, VllmMetalError


def configured_engine():
    """Build the process-wide engine from a named profile or exact model ID."""
    backend = os.environ.get("SYSTEM_ONE_BACKEND", "mlx")
    if backend == "mlx":
        return Engine(os.environ.get("SYSTEM_ONE_MODEL"))
    if backend == "vllm-metal":
        return VllmMetalEngine(os.environ.get("SYSTEM_ONE_MODEL"))
    if backend == "mlx-vlm-diffusion":
        return MlxVlmDiffusionEngine(os.environ.get("SYSTEM_ONE_MODEL"))
    if backend == "mlx-vlm-diffusion-fast":
        return MlxVlmDiffusionEngine(
            os.environ.get("SYSTEM_ONE_MODEL"),
            compact=True,
        )
    raise ValueError(f"unknown SYSTEM_ONE_BACKEND {backend!r}")


def labels_for(question):
    """Return option lines in answer-code order."""
    if isinstance(question, ChoiceQuestion):
        return [label(description, key) for key, description in question.criteria.items()]
    if isinstance(question, NoulQuestion):
        criteria = question.criteria
        yes = f"yes — {criteria.true}" if criteria and criteria.true else "yes"
        no = f"no — {criteria.false}" if criteria and criteria.false else "no"
        return [yes, no]
    return [label(level) for level in question.criteria]


def answer_for(question, probabilities):
    """Build one typed answer from a masked probability distribution."""
    if isinstance(question, ChoiceQuestion):
        values = dict(zip(question.criteria, probabilities))
        choice = max(values, key=values.get)
        return ChoiceAnswer(
            choice=choice,
            probabilities=values,
            confidence=confidence(list(values.values())),
        )
    if isinstance(question, NoulQuestion):
        return NoulAnswer(noul=probabilities[0])
    values = {str(index): value for index, value in enumerate(probabilities)}
    return ScoreAnswer(
        score=sum(index * value for index, value in enumerate(probabilities)),
        legend={str(index): level for index, level in enumerate(question.criteria)},
        probabilities=values,
        confidence=confidence(probabilities),
    )


def create_app(
    engine: Engine | None = None,
    engine_factory: Callable[[], Engine] = Engine,
) -> FastAPI:
    """Create an app whose model loads during startup, not module import."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            app.state.engine = engine_factory()
        yield

    app = FastAPI(title="System One Lite", lifespan=lifespan)
    app.state.engine = engine
    app.state.inference_slot = BoundedSemaphore(1)

    @app.post("/evaluate", response_model=EvaluateResponse)
    @app.post("/v1/systemone", response_model=EvaluateResponse)
    def evaluate(req: EvaluateRequest):
        items = list(req.questions.items())
        if not app.state.inference_slot.acquire(blocking=False):
            raise HTTPException(503, "the inference engine is busy")

        runtime_engine = app.state.engine
        engine_questions = [
            (as_text(question.instructions), labels_for(question)) for _, question in items
        ]
        try:
            results, input_tokens, elapsed_ms = runtime_engine.evaluate(
                as_text(req.state),
                engine_questions,
            )
        except RequestContractError as error:
            raise HTTPException(422, str(error)) from error
        except VllmMetalError as error:
            raise HTTPException(502, str(error)) from error
        except MlxVlmError as error:
            raise HTTPException(502, str(error)) from error
        finally:
            app.state.inference_slot.release()

        print(
            f"{len(items)} question(s) [{', '.join(req.questions)}] "
            f"in {elapsed_ms:.0f} ms ({input_tokens} prompt tokens)",
            flush=True,
        )
        return EvaluateResponse(
            model=runtime_engine.model_id,
            answers={
                question_id: answer_for(question, probabilities)
                for (question_id, question), probabilities in zip(items, results)
            },
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=runtime_engine.output_tokens_for(engine_questions),
            ),
        )

    return app


app = create_app(engine_factory=configured_engine)
