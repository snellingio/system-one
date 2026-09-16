"""The answer-slot prompt template (README Tier 1).

filled() builds a template with every answer slot filled with the letter
"A". The engine now sends one question per sequence so later questions
do not see earlier fillers. filled() still accepts several questions for
experiments that want one concatenated prompt.
"""

import json
import string

LETTERS = string.ascii_uppercase


def as_text(value):
    """Render state or instructions the same way the server does."""
    return value if isinstance(value, str) else json.dumps(value, indent=2)


def label(value, key=None):
    """A rendered option or level line, including an option key when given."""
    if value is None:
        return key if key is not None else "null"
    rendered = value if isinstance(value, str) else json.dumps(value)
    return f"{key} — {rendered}" if key is not None else rendered


def confidence(probs):
    """Spread statistic in [0, 1]. Uniform is 0. One-hot is 1.

    (max p - 1/n) / (1 - 1/n). The docs do not pin a formula. This uses
    the whole option count, so a peaked 3-way scores higher than the same
    peak among many options. It is not max p, so the 0.60 / 0.85 bands
    are harder to reach.
    """
    n = len(probs)
    if n <= 1:
        return 1.0
    return (max(probs) - 1.0 / n) / (1.0 - 1.0 / n)


def filled(state, questions, codes=LETTERS):
    """Build the filled template.

    questions: list of (instructions, labels) in request order, where labels
    are the rendered option lines for that question.
    codes: the answer codes to letter options with, in order. The default
    covers A-Z; the engine passes its registry, which continues with the
    two-letter codes (AA, AB, ...) that stay single tokens.
    Returns (text, marks) where marks[i] is the character offset where
    question i's answer code goes (text[:mark] ends with "Answer: ").

    The "State:" header was measured on the eval suite: +2/20 accuracy over
    no header, with unchanged option-order stability. An instruction line
    and bare numbering both scored the same accuracy but reduced stability.
    """
    parts = ["State:\n" + as_text(state).strip()]
    marks = []
    for k, (instructions, labels) in enumerate(questions, 1):
        parts.append(f"\n\nQuestion {k}: {as_text(instructions)}\n")
        parts.append("\n".join(f"{codes[i]}: {label}" for i, label in enumerate(labels)))
        parts.append("\nAnswer: ")
        marks.append(sum(len(p) for p in parts))
        parts.append("A")
    return "".join(parts), marks
