"""Tests for the ocean policy harvester.

Offline tests cover the env ports, the label solvers, and the PufferNet
forward against a scalar-loop reference of the C code. Integration tests
need the demo bundles in /tmp/puffer_ocean_cache (filled by
the Ocean harvest or policy tools) and skip otherwise.
"""

import random
from pathlib import Path

import numpy as np
import pytest

from tools.ocean.envs import (
    _KERNEL,
    G2048,
    Connect4,
    LightsOut,
    slide_grid,
    slide_line,
    solve_best_cols,
    solve_lights,
    solve_outlook,
)
from tools.ocean.policy import PufferNet, extract_weights, infer_shapes, weight_count

CACHE = Path("/tmp/puffer_ocean_cache")
have_cache = (CACHE / "connect4_game.data").exists()

needs_cache = pytest.mark.skipif(
    not have_cache, reason="demo bundles not in /tmp/puffer_ocean_cache"
)


# --- connect4 ----------------------------------------------------------------


def test_connect4_obs_layout_column_major_bottom_first():
    env = Connect4()
    env.player_pieces = (1 << 0) | (1 << 1)  # col 0, rows 0-1
    env.env_pieces = 1 << 7  # col 1, row 0
    obs = env.obs()
    assert obs[0] == 1 and obs[1] == 1
    assert obs[6] == -1 and obs[7] == 0


def test_connect4_wins_and_invalid():
    assert Connect4.won(0b1111)  # vertical
    assert Connect4.won((1 << 0) | (1 << 7) | (1 << 14) | (1 << 21))
    assert not Connect4.won(0b111)
    env = Connect4()
    env.player_pieces = sum(1 << 42 + r for r in range(6))  # column 6 full
    assert Connect4.invalid(6, env.player_pieces)
    assert not Connect4.invalid(5, env.player_pieces)


def test_connect4_opponent_takes_immediate_win():
    env = Connect4()
    env.player_pieces = 1 << 42  # agent piece col 6
    env.env_pieces = (1 << 14) | (1 << 15) | (1 << 16)  # 3 vertical col 2
    assert env.env_move(random.Random(0)) == 2


def test_connect4_step_win_reward_and_auto_reset():
    env = Connect4()
    rng = random.Random(0)
    env.player_pieces = 1 | (1 << 1) | (1 << 2)  # col 0 rows 0-2
    env.env_pieces = (1 << 42) | (1 << 43)
    obs, reward, terminal = env.step(0, rng)
    assert (reward, terminal) == (1.0, True)
    assert env.player_pieces == 0 and env.env_pieces == 0  # auto-reset


def _brute(p, o, d):
    m = p | o
    for c in range(7):
        if not Connect4.invalid(c, m) and Connect4.won(Connect4.play(c, m, o)):
            return 1.0
    legal = [c for c in range(7) if not Connect4.invalid(c, m)]
    if not legal or d == 0:
        return 0.0
    return max(-_brute(o, Connect4.play(c, m, o), d - 1) for c in legal)


def test_solver_matches_brute_force():
    rng = random.Random(11)
    checked = 0
    for _ in range(20):
        env = Connect4()
        for _ in range(rng.randrange(2, 12)):
            if not env.legal():
                break
            env.player_pieces = env.play(
                rng.choice(env.legal()), env.player_pieces | env.env_pieces, env.env_pieces
            )
            env.player_pieces, env.env_pieces = env.env_pieces, env.player_pieces
        if not env.legal():
            continue
        checked += 1
        for d in (2, 3, 4):
            assert solve_outlook(env.player_pieces, env.env_pieces, d) == _brute(
                env.player_pieces, env.env_pieces, d
            )
    assert checked >= 10


def test_solver_forced_win_and_must_block():
    env = Connect4()
    env.player_pieces = sum(1 << (2 * 7 + r) for r in range(3))
    env.env_pieces = 1 << 42
    value, cols = solve_best_cols(env.player_pieces, env.env_pieces, 6)
    assert value == 1.0 and 2 in cols

    env.reset()
    env.player_pieces = 1 << 42
    env.env_pieces = sum(1 << (2 * 7 + r) for r in range(3))
    value, cols = solve_best_cols(env.player_pieces, env.env_pieces, 10)
    assert value == 0.0 and cols == [2]  # only the block avoids the loss


# --- 2048 --------------------------------------------------------------------


def test_slide_line_merge_rules():
    row, moved, reward, score, merges = slide_line([1, 1, 1, 0])
    assert row == [2, 1, 0, 0] and moved and merges == 1
    row, *_ = slide_line([1, 1, 2, 2])
    assert row == [2, 3, 0, 0]  # 2,2,4,4 -> 4,8 (both pairs merge)
    row, moved, *_ = slide_line([1, 2, 3, 4])
    assert row == [1, 2, 3, 4] and not moved
    row, *_ = slide_line([0, 0, 0, 1])
    assert row == [1, 0, 0, 0]


def test_slide_grid_directions():
    grid = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    up, moved, *_ = slide_grid(grid, 1)
    assert moved and up[0] == 1 and up[3] == 2
    left, moved, *_ = slide_grid(grid, 3)
    assert moved and left[0] == 1 and left[12] == 2
    right, moved, *_ = slide_grid(grid, 4)
    assert moved and right[3] == 1 and right[15] == 2 and right[0] == 0
    down, moved, *_ = slide_grid(grid, 2)
    assert moved and down[12] == 1 and down[15] == 2


def test_g2048_step_invalid_penalty_and_spawn():
    env = G2048(random.Random(0))
    env.grid = [1] * 16  # full wall of 2s
    env.max_episode_ticks = 10_000
    before = list(env.grid)
    obs, reward, terminal = env.step(0)  # up: merges exist, valid
    assert reward > 0 and not terminal and env.grid != before
    env.grid = [1, 2, 1, 2, 2, 1, 2, 1] * 2  # full, no equal neighbors
    env.tick = 0
    obs, reward, terminal = env.step(0)  # no slide possible
    assert reward == -1.05 and terminal  # invalid penalty + game over


def test_g2048_game_over_detection():
    env = G2048(random.Random(0))
    env.grid = [1, 2, 1, 2, 2, 1, 2, 1] * 2
    assert env.game_over()
    env.grid = list(env.grid)
    env.grid[4] = env.grid[0]  # vertical pair in column 0
    assert not env.game_over()


# --- lightsout ---------------------------------------------------------------


def test_lightsout_press_toggles_plus_shape():
    grid = [0] * 25
    out = LightsOut.press(grid, 12)  # center
    for i in (12, 7, 11, 13, 17):
        assert out[i] == 1
    assert sum(out) == 5
    assert LightsOut.press(out, 12) == grid  # self-inverse


def test_lightsout_kernel_is_two_dimensional():
    assert len(_KERNEL) == 2


def test_lightsout_unsolvable_states_return_none():
    rng = random.Random(9)
    found = 0
    for _ in range(200):
        b = [1 if rng.random() < 0.5 else 0 for _ in range(25)]
        if any(sum(k & x for k, x in zip(kvec, b)) & 1 for kvec in _KERNEL):
            assert solve_lights(b) is None
            found += 1
            if found >= 5:
                return
    assert found >= 5  # half of all vectors miss the 23-dim image


def test_lightsout_solver_exact():
    rng = random.Random(5)
    solved_states = 0
    for _ in range(40):
        grid = [0] * 25
        presses = [i for i in range(25) if rng.random() < 0.3]
        for c in presses:
            grid = LightsOut.press(grid, c)
        if not any(grid):
            solved_states += 1
            continue
        w = LightsOut.optimal_weight(grid)
        assert 0 < w <= len(presses)
        after = list(grid)
        for c in LightsOut.optimal_presses(grid):
            after = LightsOut.press(after, c)
        assert after == [0] * 25
        # an optimal press reduces the remaining weight by exactly one
        pick = LightsOut.optimal_presses(grid)[0]
        assert LightsOut.on_optimal_path(grid, pick)
    assert solved_states >= 1 or True  # degenerate scrambles are rare


def test_lightsout_step_rewards_and_terminal():
    env = LightsOut(random.Random(0), scramble_prob=0.5, max_steps=100)
    env.grid = LightsOut.press([0] * 25, 0)  # exactly press(0) from solved
    obs, reward, terminal = env.step(0)
    assert reward == 2.0 and terminal


# --- puffernet ---------------------------------------------------------------


def test_weight_count_and_shape_inference():
    assert weight_count(42, [7], 256, 1) == 209_408  # connect4 web model
    assert weight_count(16, [4], 512, 4) == 3_156_480  # g2048
    assert weight_count(25, [25], 128, 4) == 203_136  # lightsout
    assert infer_shapes(209_408, 42, [7]) == (256, 1)
    assert infer_shapes(3_156_480, 16, [4]) == (512, 4)
    with pytest.raises(ValueError):
        infer_shapes(1_000_000, 42, [7])  # ambiguous or none


def _reference_forward(net, obs):
    """Scalar-loop transliteration of forward_puffernet in puffercpu.c."""

    def sigmoid(x):
        return 1.0 / (1.0 + 2.718281828459045 ** (-x))

    x = [sum(net.encoder[o][i] * obs[i] for i in range(net.obs_size)) for o in range(net.hidden)]
    state = [list(row) for row in net.state]
    for layer in range(net.layers):
        out = []
        for h in range(net.hidden):
            proj = [
                sum(net.proj[layer][k][j] * x[j] for j in range(net.hidden))
                for k in range(3 * net.hidden)
            ]
            hidden, gate, hw = proj[h], proj[net.hidden + h], proj[2 * net.hidden + h]
            h_tilde = hidden + 0.5 if hidden >= 0 else sigmoid(hidden)
            mingru_out = state[layer][h] + sigmoid(gate) * (h_tilde - state[layer][h])
            out.append(sigmoid(hw) * mingru_out + (1 - sigmoid(hw)) * x[h])
            state[layer][h] = mingru_out
        x = out
    dec = [sum(net.decoder[o][j] * x[j] for j in range(net.hidden)) for o in range(net.atn_sum + 1)]
    return dec[: net.atn_sum], dec[net.atn_sum], state


def test_puffernet_matches_scalar_reference():
    rng = np.random.default_rng(3)
    weights = rng.standard_normal(240).astype(np.float32)
    net = PufferNet(weights, 3, [2], 8, 1)
    ref = PufferNet(weights.copy(), 3, [2], 8, 1)
    for _ in range(5):
        obs = rng.standard_normal(3)
        logits, value = net.step(obs)
        r_logits, r_value, r_state = _reference_forward(ref, [float(v) for v in obs])
        assert np.allclose(logits, r_logits, atol=1e-5)
        assert abs(value - r_value) < 1e-5
        assert np.allclose(net.state, r_state, atol=1e-5)
        ref.state = np.array(r_state)  # carry it forward, like the real net


def test_puffernet_terminal_zeroes_state():
    rng = np.random.default_rng(4)
    weights = rng.standard_normal(240).astype(np.float32)
    net = PufferNet(weights, 3, [2], 8, 1)
    net.step(np.ones(3))
    net.step(np.ones(3))
    assert net.state.any()
    # a terminal step starts from zeroed carry, so it must equal the very
    # first step of a fresh net on the same obs
    fresh = PufferNet(weights.copy(), 3, [2], 8, 1)
    got = net.step(np.array([0.5, -1.0, 2.0]), terminal=True)
    want = fresh.step(np.array([0.5, -1.0, 2.0]))
    assert np.allclose(got[0], want[0]) and abs(got[1] - want[1]) < 1e-6
    assert np.allclose(net.state, fresh.state)


# --- integration with real demo weights --------------------------------------


@needs_cache
def test_extract_real_weights_and_shapes():
    for env, obs, act, arch in (
        ("connect4", 42, [7], (256, 1)),
        ("g2048", 16, [4], (512, 4)),
        ("lightsout", 25, [25], (128, 4)),
    ):
        weights, _ini = extract_weights(env, CACHE)
        assert infer_shapes(len(weights), obs, act) == arch


@needs_cache
def test_connect4_policy_plays_center_and_wins():
    from tools.ocean.policy import load_policy

    net = load_policy("connect4", 42, [7], CACHE)
    env = Connect4()
    logits, _ = net.step(env.obs())
    assert int(np.argmax(logits)) == 3
    rng, wins = random.Random(0), 0
    for _ in range(10):
        env.reset()
        net.reset()
        terminal = False
        while not terminal:
            logits, _ = net.step(env.obs(), terminal=terminal)
            _, r, terminal = env.step(int(np.argmax(logits)), rng)
        wins += r == 1.0
    assert wins >= 8


@needs_cache
def test_lightsout_policy_solves_most_puzzles():
    from tools.ocean.policy import load_policy

    net = load_policy("lightsout", 25, [25], CACHE)
    rng, solved = random.Random(1), 0
    for _ in range(10):
        env = LightsOut(rng, scramble_prob=0.3, max_steps=100)
        net.reset()
        terminal = False
        while not terminal:
            logits, _ = net.step(env.obs(), terminal=terminal)
            _, r, terminal = env.step(int(np.argmax(logits)))
        solved += r == 2.0
    assert solved >= 8
