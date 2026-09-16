"""Sync and async clients for the System One Lite evaluate endpoint.

Zero dependencies: sync calls use urllib, async calls run the sync call in
a thread. Both clients are context managers, matching the system-sdk API.
The base URL is read from SYSTEM_BASE_URL, defaulting to the local server.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Dict, Optional, Union

from .questions import Choice, Noul, Score
from .responses import SystemOneResponse

Question = Union[Noul, Choice, Score]

DEFAULT_BASE_URL = "http://127.0.0.1:8010"


class SystemRequestError(Exception):
    """A failed evaluate call: transport trouble or an HTTP error."""


class SystemClient:
    def __init__(self, base_url=None, timeout=30):
        # type: (Optional[str], float) -> None
        self.base_url = (
            base_url or os.environ.get("SYSTEM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def system_one(self, state, questions):
        # type: (object, Dict[str, Question]) -> SystemOneResponse
        payload = {
            "state": state,
            "questions": {name: q.to_wire() for name, q in questions.items()},
        }
        return SystemOneResponse.from_wire(self._post(payload))

    def _post(self, payload):
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(
            self.base_url + "/evaluate",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:200]
            raise SystemRequestError(f"HTTP {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise SystemRequestError(
                f"cannot reach {self.base_url}: {e.reason}") from e


class AsyncSystemClient(SystemClient):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def system_one(self, state, questions):
        # type: (object, Dict[str, Question]) -> SystemOneResponse
        return await asyncio.to_thread(super().system_one, state, questions)
