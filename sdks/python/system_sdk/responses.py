"""Typed answers with a full map and question-type views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .questions import Content


@dataclass
class NoulAnswer:
    noul: float
    type: str = field(default="noul", init=False)


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: str = field(default="choice", init=False)


@dataclass
class ScoreAnswer:
    score: float
    legend: dict[int, Content]
    probabilities: dict[int, float]
    confidence: float
    type: str = field(default="score", init=False)


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class SystemOneResponse:
    model: str
    usage: Usage
    nouls: dict[str, NoulAnswer] = field(default_factory=dict)
    choices: dict[str, ChoiceAnswer] = field(default_factory=dict)
    scores: dict[str, ScoreAnswer] = field(default_factory=dict)
    answers: dict[str, Answer] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data):
        answers, nouls, choices, scores = {}, {}, {}, {}
        for name, answer in data["answers"].items():
            kind = answer["type"]
            if kind == "noul":
                parsed = NoulAnswer(noul=answer["noul"])
                nouls[name] = parsed
            elif kind == "choice":
                parsed = ChoiceAnswer(
                    choice=answer["choice"],
                    probabilities=answer["probabilities"],
                    confidence=answer["confidence"])
                choices[name] = parsed
            elif kind == "score":
                parsed = ScoreAnswer(
                    score=answer["score"],
                    legend={int(key): value for key, value in answer["legend"].items()},
                    probabilities={
                        int(key): value for key, value in answer["probabilities"].items()
                    },
                    confidence=answer["confidence"])
                scores[name] = parsed
            else:
                raise ValueError(f"unknown answer type {kind!r} for {name!r}")
            answers[name] = parsed
        usage_data = data["usage"]
        return cls(model=data["model"],
                   usage=Usage(input_tokens=usage_data["input_tokens"],
                               output_tokens=usage_data["output_tokens"]),
                   answers=answers, nouls=nouls, choices=choices, scores=scores)
