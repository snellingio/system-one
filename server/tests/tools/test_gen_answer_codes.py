"""Answer-code registry output path tests."""

from system_one_lite.engine import CODES_FILE, DEFAULT_MODEL
from tools.gen_answer_codes import default_output


def test_default_model_reuses_shipped_registry():
    assert default_output(DEFAULT_MODEL) == CODES_FILE
