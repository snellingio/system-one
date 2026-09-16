"""Spike CLI: one Choice question through the engine.

Run:
    uv run python -m tools.demo [--model mlx-community/Qwen3-4B-Instruct-2507-4bit]
"""

import argparse

from system_one_lite.engine import DEFAULT_MODEL, Engine

# The Stripe example from the original vendor quickstart
STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)
QUESTION = "Which team should handle this?"
OPTIONS = {
    "billing": "Payment or subscription issues",
    "technical": "Bugs or integration problems",
    "sales": "Pricing or account questions",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    engine = Engine(args.model)
    labels = [desc or key for key, desc in OPTIONS.items()]
    results, input_tokens, ms = engine.evaluate(STATE, [(QUESTION, labels)])
    probs = dict(zip(OPTIONS, results[0]))

    print(f"{input_tokens} prompt tokens, {ms:.0f} ms warm prefill")
    winner = max(probs, key=probs.get)
    for option, p in probs.items():
        print(f"  {option:<11} {p:6.1%} {'#' * round(p * 40)}")
    print(f"winner: {winner}")


if __name__ == "__main__":
    main()
