"""Harvest labeled rows from PufferLib ocean demo policies.

Runs playouts with the extracted demo policies on the exact environment
ports, then writes rows in the datasets/ envelope
(id, usecase, source, state, questions, expected) plus soft_targets and
an `ocean` provenance block that names the label source per question.
Choice soft targets are the policy's action distribution; Noul and Score
soft targets are one-hots of the gold.

Label sources per game:
- connect4: gold from a depth-limited perfect-play solver (the win/loss
  game truncated at a recorded horizon); policy softmax as soft targets.
- 2048: no cheap exact gold exists; expected is the specialist's argmax
  (best_slide), a value-head quantile level (board_quality), and exact
  merge availability (merge_available).
- lightsout: gold from the exact GF(2) shortest-solution solver.

Usage:
    uv run python -m tools.ocean.harvest \
        --out ../datasets/ocean_playouts.jsonl [--seed 0]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from tools.ocean.envs import G2048, Connect4, LightsOut, solve_best_cols
from tools.ocean.policy import DEFAULT_CACHE, load_policy, softmax

SOURCE = (
    "PufferLib 5.0 ocean game states; playouts by the puffer.ai "
    "demo policy (see ocean block for weights and label sources)"
)


def pick_spread(items):
    """First, middle, and last index: the final decision of a won
    connect4 game is where win_now turns true."""
    if len(items) <= 3:
        return list(range(len(items)))
    return sorted({0, (len(items) - 1) // 2, len(items) - 1})


# --- connect4 ----------------------------------------------------------------

OUTLOOK_LEVELS = [
    "O forces a win - O has a line that beats any X defense",
    "No forced result - correct play from both sides leaves it open",
    "X forces a win - X has a line that beats any O defense",
]


def connect4_columns(env):
    mask = env.player_pieces | env.env_pieces
    cols = {}
    for c in range(7):
        name = f"col_{c + 1}"
        desc = f"Column {c + 1}" + (" (center)" if c == 3 else "")
        if env.invalid(c, mask):
            desc += " - already full"
        cols[name] = desc
    return cols


def connect4_state_text(env):
    lines = [
        "Connect Four. You play X and it is your turn. O plays well.",
        "Drop a piece in a column; it falls to the lowest empty slot.",
        "First to line up four in a row - across, down, or diagonal - wins.",
        "",
        "     1 2 3 4 5 6 7",
    ]
    for row in range(6, 0, -1):
        cells = []
        for col in range(7):
            bit = col * 7 + row - 1
            if env.player_pieces >> bit & 1:
                cells.append("X")
            elif env.env_pieces >> bit & 1:
                cells.append("O")
            else:
                cells.append(".")
        lines.append(f"{row:>3}  " + " ".join(cells))
    last = env.last_env_column()
    if last is not None:
        lines.append(f"\nO just played column {last + 1}.")
    return "\n".join(lines)


def solver_depth_for(pieces, target):
    """Stay near-empty-board positions shallow enough to finish."""
    if pieces >= 10:
        return target
    if pieces >= 6:
        return min(8, target)
    return min(6, target)


def label_connect4(snapshot, env_state, net, target_depth, budget):
    """Build one row from a playout snapshot (dict of raw facts)."""
    player, opp = env_state["player"], env_state["env"]
    dist = softmax(snapshot["logits"], TEMPERATURE)

    deadline = time.time() + budget
    value, best_cols, depth = 0.0, None, 0
    for d in range(2, solver_depth_for(env_state["pieces"], target_depth) + 1, 2):
        v, c = solve_best_cols(player, opp, d)
        value, best_cols, depth = v, c, d
        if time.time() > deadline:
            break

    argmax = int(dist.argmax())
    gold = max(best_cols, key=lambda c: dist[c]) if best_cols else argmax
    win_now = bool(Connect4.winning_cols(player, opp))
    opp_threat = bool(Connect4.winning_cols(opp, player))
    return {
        "usecase": "connect4-perfect-move",
        "source": SOURCE,
        "state": snapshot["state"],
        "questions": {
            "best_move": {
                "type": "choice",
                "instructions": "In which column should X drop its piece?",
                "criteria": snapshot["columns"],
            },
            "win_now": {
                "type": "noul",
                "instructions": "Does X have a drop this turn that "
                "immediately completes four in a row?",
                "criteria": {
                    "true": "yes - a winning drop exists",
                    "false": "no - no drop wins this turn",
                },
            },
            "opp_threat": {
                "type": "noul",
                "instructions": "Does O have a drop on its next turn that "
                "would complete four in a row?",
                "criteria": {
                    "true": "yes - O threatens a win",
                    "false": "no - O cannot win next turn",
                },
            },
            "outlook": {
                "type": "score",
                "instructions": "With perfect play by both sides over the "
                f"next {depth} plies, what is X's outlook?",
                "criteria": OUTLOOK_LEVELS,
            },
        },
        "expected": {
            "best_move": f"col_{gold + 1}",
            "win_now": win_now,
            "opp_threat": opp_threat,
            "outlook": int(value) + 1,
        },
        "soft_targets": {
            "best_move": {f"col_{c + 1}": float(dist[c]) for c in range(7)},
            "win_now": {"yes": float(win_now), "no": float(not win_now)},
            "opp_threat": {"yes": float(opp_threat), "no": float(not opp_threat)},
            "outlook": {str(i): float(i == int(value) + 1) for i in range(3)},
        },
        "ocean": {
            "game": "connect4",
            "moves_played": env_state["pieces"] // 2,
            "policy_pick": f"col_{argmax + 1}",
            "policy_value": round(snapshot["value"], 3),
            "solver_depth": depth,
            "arch": net.arch,
            "weights": net.provenance["weights"],
            "soft_temperature": TEMPERATURE,
            "labels": {
                "best_move": f"perfect-play solver (horizon {depth}); policy-preferred tie",
                "win_now": "exact",
                "opp_threat": "exact",
                "outlook": f"solver at depth {depth}",
            },
        },
    }


def play_connect4(net, rng, n_games, target_depth, budget):
    env = Connect4()
    rows, stats, seen = [], {"win": 0, "loss": 0, "draw": 0}, set()
    for _ in range(n_games):
        env.reset()
        net.reset()
        snapshots = []
        terminal = False
        while not terminal:
            logits, value = net.step(env.obs(), terminal=terminal)
            snapshots.append(
                {
                    "logits": logits.copy(),
                    "value": value,
                    "state": connect4_state_text(env),
                    "columns": connect4_columns(env),
                    "player": env.player_pieces,
                    "env": env.env_pieces,
                    "pieces": bin(env.player_pieces | env.env_pieces).count("1"),
                }
            )
            _, r, terminal = env.step(int(np.argmax(logits)), rng)
        if r == 1.0:
            stats["win"] += 1
        elif r == -1.0:
            stats["loss"] += 1
        else:
            stats["draw"] += 1
        for i in pick_spread(snapshots):
            snap = snapshots[i]
            if (snap["player"], snap["env"]) in seen:
                continue
            seen.add((snap["player"], snap["env"]))
            rows.append(
                label_connect4(
                    snap,
                    {"player": snap["player"], "env": snap["env"], "pieces": snap["pieces"]},
                    net,
                    target_depth,
                    budget,
                )
            )
    return rows, stats


# --- 2048 --------------------------------------------------------------------

QUALITY_LEVELS = [
    "Poor - crowded, mismatched tiles, close to a dead board",
    "Below average - playable but awkward, few merges left",
    "Average - ordinary board with reasonable options",
    "Good - tidy arrangement, merge options, room to build",
    "Excellent - open board, big tiles aligned, many options",
]
SLIDES = ["up", "down", "left", "right"]


def g2048_state_text(env):
    lines = [
        "2048. Slide all tiles one direction per move; equal tiles merge",
        "when they collide; a new 2 (90% of the time) or 4 appears after",
        "every successful move.",
        "",
        "     1   2   3   4",
    ]
    for r in range(4):
        cells = [f"{1 << v:4d}" if v else "   ." for v in env.grid[r * 4 : r * 4 + 4]]
        lines.append(f"{r + 1}  " + " ".join(cells))
    lines.append(f"\nScore so far: {env.score}.")
    if env.last_direction and env.last_spawn:
        r, c, tile = env.last_spawn
        lines.append(
            f"Previous move: slid {env.last_direction}; a {tile} appeared at row {r}, column {c}."
        )
    return "\n".join(lines)


def play_g2048(net, rng, n_episodes):
    env = G2048(rng)
    net.reset()
    rows, stats = [], {"episodes": 0, "best_tile": 0, "rows_this_episode": 0}
    terminal = False
    while stats["episodes"] < n_episodes:
        logits, value = net.step(env.obs(), terminal=terminal)
        terminal = False
        dist = softmax(logits, TEMPERATURE)
        argmax = int(dist.argmax())
        stats["best_tile"] = max(stats["best_tile"], max(env.grid))
        if env.moves_made % 50 == 0 and stats["rows_this_episode"] < 3:
            merges = {SLIDES[d - 1]: env.preview(d)[1] for d in range(1, 5)}
            merge_any = any(merges.values())
            rows.append(
                {
                    "usecase": "2048-best-slide",
                    "source": SOURCE,
                    "state": g2048_state_text(env),
                    "questions": {
                        "best_slide": {
                            "type": "choice",
                            "instructions": "Which direction should the player slide the tiles?",
                            "criteria": {
                                "up": "Slide everything up",
                                "down": "Slide everything down",
                                "left": "Slide everything left",
                                "right": "Slide everything right",
                            },
                        },
                        "merge_available": {
                            "type": "noul",
                            "instructions": "Does at least one direction merge "
                            "a pair of equal tiles this turn?",
                            "criteria": {
                                "true": "yes - some slide merges tiles",
                                "false": "no - no slide merges tiles",
                            },
                        },
                        "board_quality": {
                            "type": "score",
                            "instructions": "How good is this board for a strong 2048 player?",
                            "criteria": QUALITY_LEVELS,
                        },
                    },
                    "expected": {
                        "best_slide": SLIDES[argmax],
                        "merge_available": merge_any,
                        "board_quality": None,
                    },
                    "soft_targets": {
                        "best_slide": {k: float(dist[i]) for i, k in enumerate(SLIDES)},
                        "merge_available": {"yes": float(merge_any), "no": float(not merge_any)},
                        "board_quality": None,
                    },
                    "ocean": {
                        "game": "2048",
                        "moves_made": env.moves_made,
                        "policy_value": round(float(value), 3),
                        "merges_per_direction": merges,
                        "arch": net.arch,
                        "weights": net.provenance["weights"],
                        "soft_temperature": TEMPERATURE,
                        "labels": {
                            "best_slide": "policy argmax",
                            "merge_available": "exact",
                            "board_quality": "value-head quantile of this dataset",
                        },
                    },
                    "_value": float(value),
                }
            )
            stats["rows_this_episode"] += 1
        _, r, terminal = env.step(argmax)
        if terminal:
            stats["episodes"] += 1
            stats["rows_this_episode"] = 0

    raw = [row.pop("_value") for row in rows]
    order = sorted(raw)
    for row, v in zip(rows, raw):
        lv = min(4, 5 * order.index(v) // len(order))
        row["expected"]["board_quality"] = lv
        row["soft_targets"]["board_quality"] = {str(i): float(i == lv) for i in range(5)}
    stats.pop("rows_this_episode")
    stats["best_tile"] = 1 << stats["best_tile"]
    return rows, stats


# --- lightsout ---------------------------------------------------------------

DISTANCE_LEVELS = [
    "Nearly solved - 2 or fewer presses needed",
    "Close - 3 or 4 presses needed",
    "Midway - 5 or 6 presses needed",
    "Far - 7 or 8 presses needed",
    "Very far - 9 or more presses needed",
]


def lightsout_state_text(grid):
    lines = [
        "Lights Out. Pressing a cell flips it and its up, down, left, and",
        "right neighbors. Turn every light off.",
        "",
        "     1  2  3  4  5",
    ]
    for r in range(5):
        cells = ["#" if grid[r * 5 + c] else "." for c in range(5)]
        lines.append(f"{r + 1}  " + "  ".join(cells))
    on = sum(grid)
    lines.append(
        f"\n{on} light{'s' if on != 1 else ''} on. Rows and "
        "columns count from the top left; # means on."
    )
    return "\n".join(lines)


def distance_level(w):
    return 0 if w <= 2 else 1 if w <= 4 else 2 if w <= 6 else 3 if w <= 8 else 4


def play_lightsout(net, rng, n_episodes):
    rows, stats, seen = [], {"solved": 0, "timeout": 0}, set()
    for _ in range(n_episodes):
        env = LightsOut(rng, scramble_prob=rng.choice([0.15, 0.30]), max_steps=100)
        net.reset()
        terminal, steps = False, 0
        while not terminal:
            logits, value = net.step(env.obs(), terminal=terminal)
            dist = softmax(logits, TEMPERATURE)
            argmax = int(dist.argmax())
            if steps in (0, 6):
                grid = list(env.grid)
                if tuple(grid) not in seen:
                    seen.add(tuple(grid))
                    optimal = LightsOut.optimal_presses(grid)
                    assert optimal is not None
                    w = len(optimal)
                    pick_optimal = LightsOut.on_optimal_path(grid, argmax)
                    r, c = divmod(argmax, 5)
                    gold = argmax if pick_optimal else optimal[0]
                    gr, gc = divmod(gold, 5)
                    rows.append(
                        {
                            "usecase": "lightsout-optimal-press",
                            "source": SOURCE,
                            "state": lightsout_state_text(grid),
                            "questions": {
                                "best_press": {
                                    "type": "choice",
                                    "instructions": "Which cell should the player press?",
                                    "criteria": {
                                        f"cell_{rr + 1}_{cc + 1}": f"Press row {rr + 1}, column {cc + 1}"
                                        for rr in range(5)
                                        for cc in range(5)
                                    },
                                },
                                "press_is_optimal": {
                                    "type": "noul",
                                    "instructions": f"A player proposes "
                                    f"pressing row {r + 1}, "
                                    f"column {c + 1}. Is that "
                                    "press part of a shortest "
                                    "solution?",
                                    "criteria": {
                                        "true": "yes - it is on a shortest solution",
                                        "false": "no - every shortest solution skips it",
                                    },
                                },
                                "distance": {
                                    "type": "score",
                                    "instructions": "How many presses does a "
                                    "shortest solution need from "
                                    "this position?",
                                    "criteria": DISTANCE_LEVELS,
                                },
                            },
                            "expected": {
                                "best_press": f"cell_{gr + 1}_{gc + 1}",
                                "press_is_optimal": bool(pick_optimal),
                                "distance": distance_level(w),
                            },
                            "soft_targets": {
                                "best_press": {
                                    f"cell_{rr + 1}_{cc + 1}": float(dist[rr * 5 + cc])
                                    for rr in range(5)
                                    for cc in range(5)
                                },
                                "press_is_optimal": {
                                    "yes": float(pick_optimal),
                                    "no": float(not pick_optimal),
                                },
                                "distance": {
                                    str(i): float(i == distance_level(w)) for i in range(5)
                                },
                            },
                            "ocean": {
                                "game": "lightsout",
                                "presses_made": steps,
                                "lights_on": sum(grid),
                                "optimal_weight": w,
                                "policy_pick": f"row {r + 1}, column {c + 1}",
                                "policy_value": round(float(value), 3),
                                "arch": net.arch,
                                "weights": net.provenance["weights"],
                                "soft_temperature": TEMPERATURE,
                                "labels": {
                                    "best_press": "GF(2) shortest "
                                    "solution; policy pick when it "
                                    "is on one",
                                    "press_is_optimal": "exact",
                                    "distance": "exact",
                                },
                            },
                        }
                    )
            _, r, terminal = env.step(argmax)
            steps += 1
        if r == 2.0:
            stats["solved"] += 1
        else:
            stats["timeout"] += 1
    return rows, stats


# --- driver ------------------------------------------------------------------

GAMES = {
    "connect4": dict(playouts=40, obs=42, act=[7], prefix="c4"),
    "g2048": dict(playouts=25, obs=16, act=[4], prefix="2048"),
    "lightsout": dict(playouts=35, obs=25, act=[25], prefix="lo"),
}
PLAYERS = {"connect4": play_connect4, "g2048": play_g2048, "lightsout": play_lightsout}
# RL policies entropy-collapse; a temperature knob softens the harvested
# distribution without inventing a second opinion. 1.0 keeps it raw.
TEMPERATURE = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="connect4,g2048,lightsout")
    ap.add_argument("--out", default="../datasets/ocean_playouts.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="softmax temperature on policy logits for soft_targets",
    )
    ap.add_argument("--solver-depth", type=int, default=12)
    ap.add_argument(
        "--solver-budget",
        type=float,
        default=0.8,
        help="seconds of iterative deepening per connect4 row",
    )
    args = ap.parse_args()

    global TEMPERATURE
    TEMPERATURE = args.temperature
    rng = random.Random(args.seed)
    rows = []
    for game in args.games.split(","):
        cfg = GAMES[game]
        net = load_policy(game, cfg["obs"], cfg["act"], Path(args.cache))
        print(f"{game}: {net.arch}", file=sys.stderr)
        t0 = time.time()
        if game == "connect4":
            game_rows, stats = PLAYERS[game](
                net, rng, cfg["playouts"], args.solver_depth, args.solver_budget
            )
        else:
            game_rows, stats = PLAYERS[game](net, rng, cfg["playouts"])
        for i, row in enumerate(game_rows, 1):
            row["id"] = f"ocean-{cfg['prefix']}-{i:03d}"
        rows.extend(game_rows)
        print(f"{game}: {len(game_rows)} rows, {stats} ({time.time() - t0:.0f}s)", file=sys.stderr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    tmp.replace(out)
    print(f"wrote {len(rows)} rows -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
