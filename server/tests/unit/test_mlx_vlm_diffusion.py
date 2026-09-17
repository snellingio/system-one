import math
from pathlib import Path

import pytest

from system_one_lite.engine import RequestContractError
from system_one_lite.mlx_vlm_diffusion import (
    MlxVlmDiffusionEngine,
    MlxVlmError,
    build_seed_canvas,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) % 200 for char in text]

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "CHAT:" + messages[0]["content"]


def bare_engine(transport, canvas_length=256):
    engine = MlxVlmDiffusionEngine.__new__(MlxVlmDiffusionEngine)
    engine.model_id = "diffusion"
    engine.model_path = Path("/pinned/diffusion")
    engine.tokenizer = FakeTokenizer()
    engine.codes = ("A", "B", "C")
    engine.code_token_ids = (10, 11, 12)
    engine.canvas_length = canvas_length
    engine.vocab_size = 200
    engine.transport = transport
    engine.compact = False
    return engine


def test_evaluate_reads_all_questions_from_one_seeded_canvas():
    captured = {}

    def transport(path, payload):
        captured.update(path=path, payload=payload)
        return {
            "model": "/pinned/diffusion",
            "reads": [
                {
                    "position": slot["position"],
                    "token_id": slot["token_ids"][0],
                    "token_ids": slot["token_ids"],
                    "logprobs": [-0.1 * (index + 1) for index in range(len(slot["token_ids"]))],
                }
                for slot in payload["slots"]
            ],
            "usage": {
                "prompt_tokens": len(payload["input_ids"]),
                "denoising_steps": 1,
                "candidate_only": payload["candidate_only"],
            },
        }

    engine = bare_engine(transport)
    questions = [("first", ["yes", "no"]), ("second", ["a", "b", "c"])]

    probabilities, input_tokens, elapsed_ms = engine.evaluate("state", questions)

    assert captured["path"] == "/v1/diffusion/reads"
    assert captured["payload"]["model"] == "/pinned/diffusion"
    assert len(captured["payload"]["input_ids"]) == input_tokens
    assert len(captured["payload"]["seed_canvas"]) < 256
    assert captured["payload"]["candidate_only"] is True
    assert [slot["token_ids"] for slot in captured["payload"]["slots"]] == [
        [10, 11],
        [10, 11, 12],
    ]
    assert len(probabilities) == 2
    assert all(math.isclose(sum(row), 1.0) for row in probabilities)
    assert elapsed_ms >= 0
    assert engine.output_tokens_for(questions) == 0


def test_seed_canvas_rejects_more_questions_than_fit():
    with pytest.raises(RequestContractError, match="canvas tokens"):
        build_seed_canvas(FakeTokenizer(), 2, canvas_length=5, vocab_size=200)


def test_seed_canvas_fits_the_public_64_question_limit():
    first, positions = build_seed_canvas(FakeTokenizer(), 64, canvas_length=256, vocab_size=200)
    second, second_positions = build_seed_canvas(
        FakeTokenizer(), 64, canvas_length=256, vocab_size=200
    )

    assert first == second
    assert positions == second_positions
    assert len(first) <= 256
    assert len(positions) == 64


def test_compact_read_uses_one_canvas_token_per_question():
    captured = {}

    def transport(path, payload):
        captured.update(path=path, payload=payload)
        return {
            "model": "/pinned/diffusion",
            "reads": [
                {
                    "position": 0,
                    "token_id": 11,
                    "token_ids": [10, 11],
                    "logprobs": [-2.0, -0.1],
                }
            ],
            "usage": {
                "prompt_tokens": len(payload["input_ids"]),
                "denoising_steps": 1,
                "candidate_only": payload["candidate_only"],
            },
        }

    engine = bare_engine(transport)
    engine.compact = True
    probabilities, _, _ = engine.evaluate("delivered order.", [("late?", ["yes", "no"])])

    assert captured["payload"]["seed_canvas"] == [163]
    assert captured["payload"]["slots"] == [{"position": 0, "token_ids": [10, 11]}]
    assert "encoder_layers" not in captured["payload"]
    assert probabilities[0][1] > probabilities[0][0]


def test_compact_read_rejects_more_than_one_question():
    engine = bare_engine(None)
    engine.compact = True

    with pytest.raises(RequestContractError, match="exactly one question"):
        engine.evaluate(
            "state",
            [("first", ["yes", "no"]), ("second", ["yes", "no"])],
        )


def test_response_must_match_requested_slot():
    engine = bare_engine(None)
    with pytest.raises(MlxVlmError, match="mismatched"):
        engine._parse_response(
            {
                "model": "/pinned/diffusion",
                "reads": [
                    {
                        "position": 2,
                        "token_ids": [10, 11],
                        "logprobs": [-1.0, -2.0],
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "denoising_steps": 1,
                    "candidate_only": True,
                },
            },
            [{"position": 1, "token_ids": [10, 11]}],
            3,
            "/pinned/diffusion",
        )


def test_response_must_confirm_model_and_one_step():
    engine = bare_engine(None)
    base = {
        "model": "/pinned/diffusion",
        "reads": [
            {
                "position": 1,
                "token_ids": [10, 11],
                "logprobs": [-1.0, -2.0],
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "denoising_steps": 1,
            "candidate_only": True,
        },
    }

    wrong_model = {**base, "model": "another-model"}
    with pytest.raises(MlxVlmError, match="expected pinned model"):
        engine._parse_response(
            wrong_model,
            [{"position": 1, "token_ids": [10, 11]}],
            3,
            "/pinned/diffusion",
        )

    wrong_steps = {**base, "usage": {**base["usage"], "denoising_steps": 2}}
    with pytest.raises(MlxVlmError, match="expected 1"):
        engine._parse_response(
            wrong_steps,
            [{"position": 1, "token_ids": [10, 11]}],
            3,
            "/pinned/diffusion",
        )


def test_full_read_accepts_server_without_encoder_layer_metadata():
    engine = bare_engine(None)

    probabilities = engine._parse_response(
        {
            "model": "/pinned/diffusion",
            "reads": [
                {
                    "position": 1,
                    "token_ids": [10, 11],
                    "logprobs": [-1.0, -2.0],
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "denoising_steps": 1,
                "candidate_only": True,
            },
        },
        [{"position": 1, "token_ids": [10, 11]}],
        3,
        "/pinned/diffusion",
    )

    assert len(probabilities) == 1
    assert math.isclose(sum(probabilities[0]), 1.0)
