# Ocean games: decisions with real ground truth

[datasets/ocean_playouts.jsonl](../datasets/ocean_playouts.jsonl) holds
181 rows from three games: Connect Four, 2048, and Lights Out. The states
come from playouts by real trained policies. PufferLib ships no
checkpoints, but every demo on puffer.ai/ocean.html embeds its policy in
the page. `server/tools/ocean/harvest.py` pulls those weights, replays the
games, and writes rows in the standard envelope.

The point for this product: most question sets can only be graded
against a teacher's opinion. These can be graded against the game.
A Confidence score of "act autonomously" finally has something real to
be right or wrong about.

## The games

| Game | Rows | Choice question | Where the gold comes from |
| ---- | ---- | --------------- | ------------------------- |
| Connect Four | 48 | Which column should X play? | Perfect-play solver over a recorded horizon; exact win-now and threat checks |
| 2048 | 73 | Which direction should the player slide? | The specialist policy's pick; merge availability is exact |
| Lights Out | 60 | Which cell should the player press? | Exact shortest solution over GF(2) |

Every row also carries `soft_targets`, ready for the training-set tool: the
Choice targets are the trained policy's own action distribution, and the
Noul and Score targets are one-hots of the gold. Each row's `ocean`
block names the label source per question, the policy's pick, its value
estimate, and the weights URL.

## Example row

`ocean-c4-003`, with the answer the policy would give next to the gold:

```
Connect Four. You play X and it is your turn. O plays well.
Drop a piece in a column; it falls to the lowest empty slot.
First to line up four in a row - across, down, or diagonal - wins.

     1 2 3 4 5 6 7
  6  . . . O . . O
  5  . . . O . . X
  4  . . . X . X X
  3  . . . O . O O
  2  . . X X O X O
  1  . . O X X X O

O just played column 5.
```

Questions and gold (policy distribution in parentheses):

- **Choice** "In which column should X drop its piece?" — gold `col_5`.
  The solver finds a forced win and column 5 wins on the spot
  (policy: col_5 at 1.00).
- **Noul** "Does X have a drop this turn that immediately completes four
  in a row?" — gold `true`.
- **Noul** "Does O have a drop on its next turn that would complete four
  in a row?" — gold `true`. O just built its own threat; X must win now,
  not later.
- **Score** "With perfect play by both sides over the next 12 plies,
  what is X's outlook?" — gold `2`, "X forces a win".

That is the demo: an unstructured board, three question types, and
answers that are facts, not opinions.

## Baseline

The untrained base model scores 189/591 (32%) on these rows via
the eval tool: Choice 12/181, Noul 130/229, Score 47/181, with option-order
permutation flipping the winner on 7 of 8 probes. That is the "before"
number. Fine-tuning on this file (it already carries `soft_targets`) is
the obvious next step, and the game-truth golds make calibration metrics
mean something afterwards.

## Regenerate

```
cd server
uv run python -m tools.ocean.harvest --out ../datasets/ocean_playouts.jsonl
uv run pytest tests/tools/test_ocean.py
```

The first run downloads the three demo bundles (about 22 MB) into
/tmp/puffer_ocean_cache and reuses them after. Use `--temperature` to
soften the policy's near-one-hot distributions, and `--seed` to reroll
the playouts.

## Caveats

- The policies are RL-trained and their distributions are sharp. The
  choice `soft_targets` are honest but close to one-hot; blend or temper
  them for calibration work.
- The 2048 golds for `best_slide` and `board_quality` are the
  specialist's opinion (argmax and a value-head quantile), not game
  truth. `merge_available` is exact. The `ocean.labels` block says which
  is which per row.
- The Connect 4 `outlook` gold is exact only within the recorded
  horizon (`ocean.solver_depth`, 6 to 12 plies here). Deeper forces may
  exist; the horizon keeps the label honest and cheap.
- The Connect 4 port keeps a quirk of the shipped PufferLib env: its
  draw rule almost never fires, so drawn games are practically
  impossible in these playouts. The policy was trained with that rule,
  so the harvest keeps it.
- Playouts act by argmax; the live web demo samples its softmax. The
  states are greedy-policy trajectories, not the demo's exact games.
- Weights are public demo assets from puffer.ai, not an official
  checkpoint release. The env code is MIT (PufferAI/PufferLib, branch
  5.0). Fine for internal PoC use; say where they came from.
