"""FastAPI application for typed local model decisions.

Run with ``uv run uvicorn system_one_lite.api:app --port 8010``.
"""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException

from .errors import RequestContractError
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


def build_mlx_engine(model_id=None):
    from .engine import Engine

    return Engine(model_id)


def build_nli_engine(model_id=None):
    from .nli_engine import NLIEngine

    return NLIEngine(model_id)


def configured_engine():
    """Build the selected process-wide inference backend."""
    backend = os.environ.get("SYSTEM_ONE_BACKEND", "mlx")
    if backend == "mlx":
        return build_mlx_engine(os.environ.get("SYSTEM_ONE_MODEL"))
    if backend == "nli":
        return build_nli_engine(os.environ.get("SYSTEM_ONE_NLI_MODEL"))
    raise ValueError(f"unknown SYSTEM_ONE_BACKEND: {backend!r}")


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
    engine=None,
    engine_factory: Callable[[], object] = build_mlx_engine,
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
        try:
            results, input_tokens, elapsed_ms = runtime_engine.evaluate(
                as_text(req.state),
                [(as_text(question.instructions), labels_for(question)) for _, question in items],
            )
        except RequestContractError as error:
            raise HTTPException(422, str(error)) from error
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
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        )

    return app


app = create_app(engine_factory=configured_engine)
