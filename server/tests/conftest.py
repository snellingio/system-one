"""Shared fixtures: one Engine per test session (model load is the slow part)."""

import pytest

from system_one_lite.engine import Engine


@pytest.fixture(scope="session")
def engine():
    return Engine()
