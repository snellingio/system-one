"""The quickstart example from the system-sdk docs, against the local
server. Run from sdks/python with the server up:

    python3 examples/quickstart.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_sdk import AsyncSystemClient, Choice, Noul, Score, SystemClient

STATE = {"document": "I was charged twice. Please fix this ASAP."}
QUESTIONS = {
    "billing": Noul(instructions="Is this ticket about billing?"),
    "tone": Choice(
        instructions="What is the customer's tone?",
        criteria={"calm": None, "frustrated": None, "angry": None},
    ),
    "urgency": Score(
        instructions="How urgent is this ticket?",
        criteria=["can wait", "this week", "today"],
    ),
}


def sync_example():
    with SystemClient() as client:
        response = client.system_one(state=STATE, questions=QUESTIONS)

    print(response.nouls["billing"].noul)
    print(response.choices["tone"].choice)
    print(response.scores["urgency"].score)


async def async_example():
    async with AsyncSystemClient() as client:
        response = await client.system_one(state=STATE, questions=QUESTIONS)

    print(response.nouls["billing"].noul)
    print(response.choices["tone"].choice)
    print(response.scores["urgency"].score)


if __name__ == "__main__":
    sync_example()
    asyncio.run(async_example())
