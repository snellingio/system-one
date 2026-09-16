"""Answer-code registry output path tests."""

from system_one_lite.engine import CODES_FILE, DEFAULT_MODEL, LARGER_MODEL
from tools.gen_answer_codes import default_output


def test_default_model_reuses_shipped_registry():
    assert default_output(DEFAULT_MODEL) == CODES_FILE
    assert default_output("default") == CODES_FILE


def test_larger_model_reuses_shipped_registry():
    assert default_output(LARGER_MODEL).name == "qwen3_4b_instruct_2507_4bit_answer_codes.json"
    assert default_output("larger").name == "qwen3_4b_instruct_2507_4bit_answer_codes.json"
