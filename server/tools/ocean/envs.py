"""Exact Python ports of three PufferLib 5.0 ocean environments.

Ported line by line from ocean/connect4/connect4.h, ocean/g2048/g2048.h,
and ocean/lightsout/lightsout.h so the harvested policies observe exactly
what they were trained on. Only training-time logging and raylib rendering
are dropped; the dynamics, rewards, and auto-resets match the C code.

Each env: reset() at construction, then step(action) ->
(obs, reward, terminal), with obs matching the C observation layout.
Episodes auto-reset on terminal, mirroring the headless C path.
"""

import numpy as np

# --- Connect 4 ---------------------------------------------------------------
# Bitboards: 7 bits per column (6 playable rows + sentinel), bit c*7+r with
# r=0 the bottom row. The agent always moves first and owns `player_pieces`;
# the built-in opponent (negamax depth 3 + opening book) owns `env_pieces`.


class Connect4:
    ROWS, COLUMNS = 6, 7
    ACT_SIZES = [7]
    OBS_SIZE = 42
    FULL_MASK = 4432406249472  # draw() constant from connect4.h, verbatim.
    # Quirk: this equals bits 42|35 (two bottom cells), not a full board,
    # so the shipped C draw() almost never fires. Ported as-is because the
    # demo policies were trained against this exact rule.
    BOOK = {4398050705408: 2, 4398583382016: 3}

    def __init__(self):
        self.reset()

    def reset(self):
        self.player_pieces = 0
        self.env_pieces = 0
        self.last_env_bit = 0
        self.tick = 0

    # -- bitboard helpers (connect4.h) --
    @classmethod
    def top_mask(cls, col):
        return 1 << (cls.ROWS - 1) << col * (cls.ROWS + 1)

    @classmethod
    def invalid(cls, col, mask):
        return bool(mask & cls.top_mask(col))

    @staticmethod
    def play(col, mask, other_pieces):
        """Pieces of `other_pieces`'s opponent after it drops in col."""
        mask |= mask + (1 << col * 7)
        return other_pieces ^ mask

    @classmethod
    def draw(cls, mask):
        return mask == cls.FULL_MASK

    @staticmethod
    def won(pieces):
        m = pieces & (pieces >> 7)
        if m & (m >> 14):
            return True
        m = pieces & (pieces >> 6)
        if m & (m >> 12):
            return True
        m = pieces & (pieces >> 8)
        if m & (m >> 16):
            return True
        m = pieces & (pieces >> 1)
        return bool(m & (m >> 2))

    def legal(self):
        mask = self.player_pieces | self.env_pieces
        return [c for c in range(7) if not self.invalid(c, mask)]

    def obs(self):
        out = np.zeros(42, dtype=np.float32)
        for idx, bit in enumerate(i for i in range(49) if (i + 1) % 7):
            if self.player_pieces >> bit & 1:
                out[idx] = 1.0
            elif self.env_pieces >> bit & 1:
                out[idx] = -1.0
        return out

    # -- built-in opponent (compute_env_move, ported verbatim) --
    def negamax(self, pieces, other_pieces, depth):
        mask = pieces | other_pieces
        if self.won(other_pieces):
            return 10.0**depth
        if self.won(pieces):
            return 0.0
        if depth == 0 or self.draw(mask):
            return 0.0
        value = 0.0
        for col in range(7):
            if self.invalid(col, mask):
                continue
            value -= self.negamax(other_pieces, self.play(col, mask, other_pieces), depth - 1)
        return value

    def env_move(self, rng):
        mask = self.player_pieces | self.env_pieces
        h = self.player_pieces + mask + (1 << 42)
        if h in self.BOOK:
            return self.BOOK[h]
        values = [9999.0] * 7
        for col in range(7):
            if self.invalid(col, mask):
                continue
            child = self.play(col, mask, self.player_pieces)
            if self.won(child):
                return col
            values[col] = -self.negamax(self.player_pieces, child, 3)
        best = min(values)
        return rng.choice([c for c in range(7) if values[c] == best])

    def step(self, col, rng):
        """One agent move + opponent reply. Returns (obs, reward, terminal)."""
        self.tick += 1
        mask = self.player_pieces | self.env_pieces
        if self.invalid(col, mask):
            self.reset()  # invalid move loses, as in finish_game
            return self.obs(), -1.0, True
        self.player_pieces = self.play(col, mask, self.env_pieces)
        if self.won(self.player_pieces):
            self.reset()
            return self.obs(), 1.0, True
        if self.draw(self.player_pieces | self.env_pieces):
            self.reset()
            return self.obs(), 0.0, True

        col = self.env_move(rng)
        mask = self.player_pieces | self.env_pieces
        if self.invalid(col, mask):
            self.reset()  # opponent blundered into a full column: agent wins
            return self.obs(), 1.0, True
        new_env = self.play(col, mask, self.player_pieces)
        self.last_env_bit = new_env ^ self.env_pieces
        self.env_pieces = new_env
        if self.won(self.env_pieces):
            self.reset()
            return self.obs(), -1.0, True
        if self.draw(self.env_pieces | self.player_pieces):
            self.reset()
            return self.obs(), 0.0, True
        return self.obs(), 0.0, False

    # -- labeling helpers (ours, not from the C header) --
    @classmethod
    def winning_cols(cls, mover, opp):
        """Columns where a drop by `mover` completes four in a row."""
        mask = mover | opp
        return [c for c in range(7) if not cls.invalid(c, mask) and cls.won(cls.play(c, mask, opp))]

    def last_env_column(self):
        if not self.last_env_bit:
            return None
        return (self.last_env_bit.bit_length() - 1) // 7


# Depth-limited exact solver for gold labels: the win/loss game truncated
# at `depth` plies, so 0 means "nothing forced within the horizon" (draws
# and deep results both land there). Fail-soft alpha-beta with a proper
# transposition table; center-first move ordering.

_ORDER = [3, 2, 4, 1, 5, 0, 6]
_TT = {}
_EXACT, _LOWER, _UPPER = 0, 1, 2


def solve_outlook(player, opp, depth, alpha=-1.0, beta=1.0):
    """Value for the side to move (`player`): 1 forced win, -1 forced loss,
    0 draw or nothing forced within the horizon."""
    key = (player, opp, depth)
    hit = _TT.get(key)
    if hit is not None:
        value, flag = hit
        if flag == _EXACT:
            return value
        if flag == _LOWER and value >= beta:
            return value
        if flag == _UPPER and value <= alpha:
            return value

    mask = player | opp
    cols = []
    for c in _ORDER:
        if Connect4.invalid(c, mask):
            continue
        if Connect4.won(Connect4.play(c, mask, opp)):
            _TT[key] = (1.0, _EXACT)
            return 1.0
        cols.append(c)
    if not cols or Connect4.draw(mask) or depth == 0:
        _TT[key] = (0.0, _EXACT)
        return 0.0

    value = -1.0
    a, b = alpha, beta
    for c in cols:
        child = Connect4.play(c, mask, opp)
        value = max(value, -solve_outlook(opp, child, depth - 1, -b, -a))
        if value > a:
            a = value
        if a >= b:
            break

    if value <= alpha:
        _TT[key] = (value, _UPPER)
    elif value >= beta:
        _TT[key] = (value, _LOWER)
    else:
        _TT[key] = (value, _EXACT)
    return value


def solve_best_cols(player, opp, depth):
    """(best value, columns achieving it) for the side to move."""
    mask = player | opp
    results = []
    for c in _ORDER:
        if Connect4.invalid(c, mask):
            continue
        child = Connect4.play(c, mask, opp)
        v = 1.0 if Connect4.won(child) else -solve_outlook(opp, child, depth - 1)
        results.append((v, c))
    if not results:
        return 0.0, []
    best = max(v for v, _ in results)
    return best, [c for v, c in results if v == best]


# --- 2048 --------------------------------------------------------------------
# Grid stores tile exponents (0 empty, 1 = "2", ...); obs is the raw
# exponents as floats, row-major. Actions 0-3 -> up, down, left, right.

_POW15 = [0.0, 1.0, 2.83, 5.20, 8.0, 11.18, 14.70, 18.52, 22.63, 27.0, 31.62, 36.48]


def slide_line(line):
    """line[0] is the edge tiles slide toward. Returns a new list."""
    row = list(line)
    moved = False
    write = 0
    for read in range(4):
        if row[read] != 0:
            if write != read:
                row[write], row[read] = row[read], 0
                moved = True
            write += 1
    reward = 0.0
    score_add = 0
    merges = 0
    for i in range(3):
        if row[i] != 0 and row[i] == row[i + 1]:
            row[i] += 1
            reward += 0.05 if row[i] <= 6 else 0.05 + _POW15[row[i] - 6] * 0.03
            score_add += 1 << row[i]
            merges += 1
            for j in range(i + 1, 3):
                row[j] = row[j + 1]
            row[3] = 0
            moved = True
    return row, moved, reward, score_add, merges


def slide_grid(grid, direction):
    """Pure slide of a 16-int exponent grid. Returns
    (new_grid, moved, reward, score_add, merges). direction 1-4."""
    out = list(grid)
    moved_any, reward, score_add, merges = False, 0.0, 0, 0
    for k in range(4):
        if direction == 1:
            idx = [(i, k) for i in range(4)]
        elif direction == 2:
            idx = [(3 - i, k) for i in range(4)]
        elif direction == 3:
            idx = [(k, i) for i in range(4)]
        else:
            idx = [(k, 3 - i) for i in range(4)]
        line, moved, r, s, m = slide_line([out[r * 4 + c] for r, c in idx])
        reward += r
        score_add += s
        merges += m
        if moved:
            moved_any = True
            for i, (r, c) in enumerate(idx):
                out[r * 4 + c] = line[i]
    return out, moved_any, reward, score_add, merges


class G2048:
    SIZE = 4
    ACT_SIZES = [4]
    OBS_SIZE = 16
    DIR_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

    def __init__(self, rng, lifetime_max_tile=0):
        self.rng = rng
        self.lifetime_max_tile = lifetime_max_tile
        self.reset()

    def reset(self):
        self.grid = [0] * 16
        self.score = 0
        self.tick = 0
        self.episode_reward = 0.0
        self.max_tile = 0
        self.moves_made = 0
        self.max_episode_ticks = 1000
        self.last_spawn = None
        self.last_direction = None
        for _ in range(2):
            self._spawn(self._new_tile())

    def _new_tile(self):
        return 2 if self.rng.randrange(10) == 0 else 1

    def _spawn(self, tile):
        empties = [i for i, v in enumerate(self.grid) if v == 0]
        if not empties:
            return
        pos = self.rng.choice(empties)
        self.grid[pos] = tile
        self.last_spawn = (pos // 4 + 1, pos % 4 + 1, 1 << tile)

    def obs(self):
        return np.array(self.grid, dtype=np.float32)

    def game_over(self):
        if 0 in self.grid:
            return False
        for r in range(4):
            for c in range(4):
                v = self.grid[r * 4 + c]
                if r < 3 and v == self.grid[(r + 1) * 4 + c]:
                    return False
                if c < 3 and v == self.grid[r * 4 + c + 1]:
                    return False
        return True

    def step(self, action):
        """action 0-3. Returns (obs, reward, terminal)."""
        new_grid, moved, merge_reward, score_add, _ = slide_grid(self.grid, action + 1)
        reward = 0.0
        self.tick += 1
        if moved:
            self.moves_made += 1
            self.grid = new_grid
            self.max_tile = max(self.grid)
            reward += merge_reward
            self._spawn(self._new_tile())
            self.score += score_add
            mult = max(1, self.lifetime_max_tile - 8)
            self.max_episode_ticks = max(1000 * mult, self.score // 4)
            self.last_direction = self.DIR_NAMES[action]
        else:
            reward = -0.05
        terminal = False
        if self.game_over():
            reward += -1.0
            terminal = True
        elif self.tick >= self.max_episode_ticks:
            terminal = True
        self.episode_reward += reward
        if terminal:
            self.lifetime_max_tile = max(self.lifetime_max_tile, self.max_tile)
            self.reset()
        return self.obs(), reward, terminal

    # -- labeling helpers (ours) --
    def preview(self, direction):
        """(moved, merges, new_grid) for direction 1-4, without mutating."""
        new, moved, _, _, merges = slide_grid(self.grid, direction)
        return moved, merges, new


# --- Lights Out --------------------------------------------------------------
# 5x5 grid, obs = 25 lights (1 = on) row-major; action = press cell
# 0-24, toggling the cell and its orthogonal neighbors.

_GS = 5


def _press_matrix():
    """The 25x25 GF(2) matrix A: pressing cell j adds column j of A."""
    A = []
    for i in range(25):
        row = [0] * 25
        row[i] = 1
        r, c = divmod(i, _GS)
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= rr < _GS and 0 <= cc < _GS:
                row[rr * _GS + cc] = 1
        A.append(row)
    return A


# Kernel basis of the press matrix, computed once (5x5 lights out has a
# 2-dimensional kernel), plus the row-reduced form used to solve states.
def _reduce():
    A = _press_matrix()
    n = 25
    # [A | I] so row ops track the combinations; kernel rows fall out
    M = [A[i][:] + [1 if j == i else 0 for j in range(n)] for i in range(n)]
    rank = 0
    where = [-1] * n
    for c in range(n):
        piv = next((k for k in range(rank, n) if M[k][c]), None)
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        for k in range(n):
            if k != rank and M[k][c]:
                M[k] = [x ^ y for x, y in zip(M[k], M[rank])]
        where[c] = rank
        rank += 1
    kernel = [M[k][n:] for k in range(rank, n)]
    return where, M, kernel


_WHERE, _REDUCED, _KERNEL = _reduce()


def solve_lights(grid):
    """Minimal-weight press list solving `grid`, or None if unsolvable.

    _REDUCED rows carry the [A | I] reduction, so the I half of row r says
    which original rows combine into reduced row r; the same combination
    of the target grid gives that row's reduced value, and the pivot
    variable reads off directly. A is symmetric, so solvability is
    orthogonality to the kernel.
    """
    n = 25
    b = list(grid)
    if any(sum(k & x for k, x in zip(kvec, b)) & 1 for kvec in _KERNEL):
        return None
    particular = [0] * n
    for c in range(n):
        r = _WHERE[c]
        if r < 0:
            continue
        row = _REDUCED[r]
        particular[c] = sum(row[n + i] & b[i] for i in range(n)) & 1
    best = None
    for mask in range(1 << len(_KERNEL)):
        sol = particular[:]
        for bit, kvec in enumerate(_KERNEL):
            if mask >> bit & 1:
                sol = [s ^ k for s, k in zip(sol, kvec)]
        w = sum(sol)
        if best is None or w < best[0]:
            best = (w, sol)
    return best[1]


class LightsOut:
    ACT_SIZES = [_GS * _GS]
    OBS_SIZE = _GS * _GS

    def __init__(self, rng, scramble_prob=0.15, max_steps=100):
        self.rng = rng
        self.scramble_prob = scramble_prob
        self.max_steps = max_steps
        self.reset()

    @staticmethod
    def press(grid, idx):
        row, col = divmod(idx, _GS)
        out = list(grid)
        for dr, dc in ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)):
            r, c = row + dr, col + dc
            if 0 <= r < _GS and 0 <= c < _GS:
                out[r * _GS + c] ^= 1
        return out

    def _scramble(self):
        grid = [0] * 25
        for i in range(25):
            if self.rng.random() < self.scramble_prob:
                grid = self.press(grid, i)
        return grid

    def reset(self):
        # Sampling deviation from the C env, dynamics unchanged: it can
        # start fully solved, and its scramble_prob adapts live via an EMA
        # between 0.15 and 0.5. We skip solved starts and use a fixed
        # scramble_prob per episode instead.
        self.grid = self._scramble()
        while sum(self.grid) == 0:
            self.grid = self._scramble()
        self.step_count = 0
        self.prev_action = -1
        self.last_action = -1

    def obs(self):
        return np.array(self.grid, dtype=np.float32)

    def step(self, atn):
        reward = -0.02 * (36.0 / 25.0)
        prev_on = sum(self.grid)
        if not 0 <= atn < 25:
            reward -= 0.5
        else:
            if atn == self.last_action:
                reward -= 0.03
            elif atn == self.prev_action:
                reward -= 0.02
            self.grid = self.press(self.grid, atn)
            self.prev_action = self.last_action
            self.last_action = atn
            reward += 0.005 * (prev_on - sum(self.grid))
        self.step_count += 1
        terminal = False
        if sum(self.grid) == 0:
            reward = 2.0
            terminal = True
        elif self.step_count >= self.max_steps:
            reward -= 0.5
            terminal = True
        if terminal:
            self.reset()
        return self.obs(), reward, terminal

    # -- labeling helpers (ours): exact solver over GF(2) --
    @staticmethod
    def optimal_presses(grid):
        sol = solve_lights(grid)
        return None if sol is None else [i for i, v in enumerate(sol) if v]

    @classmethod
    def optimal_weight(cls, grid):
        sol = cls.optimal_presses(grid)
        return None if sol is None else len(sol)

    @classmethod
    def on_optimal_path(cls, grid, atn):
        w = cls.optimal_weight(grid)
        return w is not None and w > 0 and cls.optimal_weight(cls.press(grid, atn)) == w - 1
