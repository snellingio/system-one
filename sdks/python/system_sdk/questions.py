"""Question types matching the system-sdk API."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

Content = Union[str, Dict[str, object], List[object]]


class Noul:
    """A yes/no question. The answer is the probability that it is yes."""

    def __init__(self, instructions, criteria=None):
        # type: (Content, Optional[Dict[str, str]]) -> None
        self.instructions = instructions
        self.criteria = criteria

    def to_wire(self):
        question = {"type": "noul", "instructions": self.instructions}
        if self.criteria:
            question["criteria"] = self.criteria
        return question


class Choice:
    """Pick one option from a set. criteria maps option to description or
    None when the option needs no extra detail."""

    def __init__(self, instructions, criteria):
        # type: (Content, Dict[str, Optional[Content]]) -> None
        self.instructions = instructions
        self.criteria = criteria

    def to_wire(self):
        return {"type": "choice", "instructions": self.instructions,
                "criteria": self.criteria}


class Score:
    """Rate along ordered levels, from the low end to the high end."""

    def __init__(self, instructions, criteria):
        # type: (Content, List[Content]) -> None
        self.instructions = instructions
        self.criteria = criteria

    def to_wire(self):
        return {"type": "score", "instructions": self.instructions,
                "criteria": self.criteria}
