"""Typed answers, split by question type like the system-sdk response."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from .questions import Content


@dataclass
class NoulAnswer:
    noul: float


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: Dict[str, float]
    confidence: float


@dataclass
class ScoreAnswer:
    score: float
    legend: Dict[str, Content]
    probabilities: Dict[str, float]
    confidence: float


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int


@dataclass
class SystemOneResponse:
    model: str
    usage: Usage
    nouls: Dict[str, NoulAnswer] = field(default_factory=dict)
    choices: Dict[str, ChoiceAnswer] = field(default_factory=dict)
    scores: Dict[str, ScoreAnswer] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data):
        nouls, choices, scores = {}, {}, {}
        for name, answer in data["answers"].items():
            kind = answer["type"]
            if kind == "noul":
                nouls[name] = NoulAnswer(noul=answer["noul"])
            elif kind == "choice":
                choices[name] = ChoiceAnswer(
                    choice=answer["choice"],
                    probabilities=answer["probabilities"],
                    confidence=answer["confidence"])
            elif kind == "score":
                scores[name] = ScoreAnswer(
                    score=answer["score"],
                    legend=answer["legend"],
                    probabilities=answer["probabilities"],
                    confidence=answer["confidence"])
            else:
                raise ValueError(f"unknown answer type {kind!r} for {name!r}")
        usage_data = data["usage"]
        return cls(model=data["model"],
                   usage=Usage(input_tokens=usage_data["input_tokens"],
                               output_tokens=usage_data["output_tokens"]),
                   nouls=nouls, choices=choices, scores=scores)
