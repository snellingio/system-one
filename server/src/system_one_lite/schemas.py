"""Validated HTTP request and response models."""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .prompts import as_text

Content = Union[str, dict, list]
MAX_QUESTIONS = 64
MAX_STATE_CHARS = 100_000
MAX_INSTRUCTION_CHARS = 20_000
MAX_CRITERION_CHARS = 20_000
MAX_ID_CHARS = 200
MAX_REQUEST_BYTES = 1_000_000


def check_content(value, name, limit):
    if len(as_text(value)) > limit:
        raise ValueError(f"{name} exceeds the {limit} character limit")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Content
    criteria: dict[str, Union[Content, None]] = Field(min_length=1)

    @field_validator("instructions")
    @classmethod
    def instructions_size(cls, value):
        return check_content(value, "instructions", MAX_INSTRUCTION_CHARS)

    @field_validator("criteria")
    @classmethod
    def criteria_size(cls, value):
        for key, description in value.items():
            check_content(key, "option name", MAX_ID_CHARS)
            if description is not None:
                check_content(
                    description,
                    f"description for {key!r}",
                    MAX_CRITERION_CHARS,
                )
        return value


class NoulCriteria(StrictModel):
    true: Union[str, None] = None
    false: Union[str, None] = None

    @field_validator("true", "false")
    @classmethod
    def criteria_size(cls, value):
        if value is not None:
            check_content(value, "noul criterion", MAX_CRITERION_CHARS)
        return value


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria | None = None

    @field_validator("instructions")
    @classmethod
    def instructions_size(cls, value):
        return check_content(value, "instructions", MAX_INSTRUCTION_CHARS)


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Content
    criteria: list[Content] = Field(min_length=2, max_length=10)

    @field_validator("instructions")
    @classmethod
    def instructions_size(cls, value):
        return check_content(value, "instructions", MAX_INSTRUCTION_CHARS)

    @field_validator("criteria")
    @classmethod
    def criteria_size(cls, value):
        for criterion in value:
            check_content(criterion, "score criterion", MAX_CRITERION_CHARS)
        return value


Question = Annotated[
    Union[ChoiceQuestion, NoulQuestion, ScoreQuestion],
    Field(discriminator="type"),
]


class EvaluateRequest(StrictModel):
    state: Content
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @field_validator("state")
    @classmethod
    def state_size(cls, value):
        return check_content(value, "state", MAX_STATE_CHARS)

    @field_validator("questions")
    @classmethod
    def question_ids_size(cls, value):
        for question_id in value:
            check_content(question_id, "question ID", MAX_ID_CHARS)
        return value

    @model_validator(mode="after")
    def request_size(self):
        if len(self.model_dump_json().encode("utf-8")) > MAX_REQUEST_BYTES:
            raise ValueError(f"request content exceeds the {MAX_REQUEST_BYTES} byte limit")
        return self


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Content]
    probabilities: dict[str, float]
    confidence: float


Answer = Union[ChoiceAnswer, NoulAnswer, ScoreAnswer]


class EvaluateResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
