"""Answer-code registry output path tests."""

from system_one_lite.engine import CODES_FILE, DEFAULT_MODEL, MODEL_REVISIONS
from tools.gen_answer_codes import default_output


def test_default_model_reuses_shipped_registry():
    assert default_output(DEFAULT_MODEL) == CODES_FILE


def test_every_supported_model_has_a_shipped_registry():
    for model_id in MODEL_REVISIONS:
        assert default_output(model_id).is_file()
