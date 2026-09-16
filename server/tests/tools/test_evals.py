"""Eval command checks that should run without loading a model."""

import sys

import pytest

from tools import evals


def test_empty_dataset_directory_stops_before_model_load(tmp_path, monkeypatch):
    def fail_engine(*args, **kwargs):
        raise AssertionError("the model should not load without datasets")

    monkeypatch.setattr(evals, "Engine", fail_engine)
    monkeypatch.setattr(sys, "argv", ["evals", "--datasets", str(tmp_path)])

    with pytest.raises(SystemExit) as error:
        evals.main()

    assert error.value.code == 2
