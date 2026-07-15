# MCTS internals

All of `src/sraz/core/mcts.py`. This is the heart of the engine: a single-player
AlphaZero MCTS that uses the policy/value net for guidance and a transposition
table keyed by `game.hashable_obs`. This doc walks the algorithm as implemented,
plus the several optional mechanisms bolted on for experiments.

## Data structures

- `MCTS.nodes: dict[hashable_obs, MCTSTreeNode]` — the transposition table.
  Because it is keyed by observation, two derivation paths reaching the same
  buffer share a node.
- `MCTSTreeNode` holds, per state: `direct_reward`, `is_terminal_state`,
  `nn_policy` (masked prior, possibly noise-mixed), `nn_policy_original` (clean
  prior, for re-noising on tree reuse), `nn_value`, `action_mask`, `total_N`,
  and per-action `action_Q`, `action_N`, and `action_values` (the full list of
  backed-up values, used by non-mean backup rules).

## The main entry point: `perform_simulations`

1. Record the root's `hashable_obs`. Reset the global Q min/max trackers and the
   rollout budget.
2. If the root node doesn't exist yet, expand it once (stash → `search` →
   unstash) so its prior/mask are available.
3. **Optionally inject Dirichlet noise** at the root (self-play only): sample
   `Dir(alpha)` over all actions, mask to legal moves, renormalize, and mix
   `(1-eps)*prior + eps*noise`. Only applied when the root is fresh
   (`total_N == 0`).
4. Run `n_simulations` iterations of: stash the game, `search()`, unstash. An
   assertion checks the root observation is unchanged across the round-trip.
5. Read visit counts `action_N` into a `counts` array and convert to move
   probabilities via a temperature transform.

**Special case:** `n_simulations < 0` skips search entirely and returns the
temperature-scaled masked policy prior directly (a "network-only" baseline).

### Temperature transform + fallback

`_temperature_probs_with_fallback(counts, node)` computes
$p_a \propto N_a^{1/T}$ in log-space (max-subtracted for stability). If all
counts are zero (e.g. `n_simulations == 1` only expands the root and backs up
nothing), it falls back to the masked prior, then to a uniform distribution over
the legal mask — so callers never receive an all-zero / NaN distribution. This
is pinned by `test_n_simulations_one_does_not_return_nan`.

## One simulation: `search()`

Recursive, returning the value backed up to the caller. At the current state:

1. **Create node if unseen**, recording `direct_reward = game.reward` and
   `is_terminal`.
2. **Terminal base case:** return `0.0` (the reward for *entering* a terminal
   state was already captured as the parent edge's immediate reward; a terminal
   state's own future value is 0).
3. **Unexpanded-leaf base case:** query the net (masked), store the prior, mask,
   and value. Return the value (optionally blended with a rollout estimate —
   see below).
4. **Recursive case:** pick the action maximizing the masked UCB, `step` the
   game, capture the immediate reward, recurse for the child's value, then do a
   **Bellman backup** $\text{total} = r_\text{immediate} + V(s')$ (discount is
   implicitly 1.0), update the edge, increment `total_N`, and return `total`.

Note this backs up the **full return from this node down** (immediate reward +
future value), which is what makes the tree work for dense/intermediate rewards,
not just terminal ones. The `test_mcts_env_*` suite verifies this against
analytically-known chains, bandits, delayed traps, transpositions, and
truncation.

## Selection: `calc_masked_ucbs`

PUCT-style score per legal action:

$$U(a) = \hat{Q}(a) + c_\text{explore}\, P(a)\, \frac{\sqrt{N_\text{total} + \epsilon}}{1 + N(a)}$$

where $\hat{Q}$ is the edge's Q **normalized** into $[0,1]$ using the global
min/max Q seen during this search (so the exploration term is on a comparable
scale regardless of reward magnitude; when no Q spread exists yet it defaults to
0 or 0.5). Illegal actions are set to $-\infty$. There is a fast vectorized path
for 1-D `Discrete` action spaces and a general fallback for multi-dimensional
(`MultiDiscrete`) spaces.

## Backup rules: `update_edge`

Beyond the standard **mean** backup, the edge Q can be aggregated from the full
list of backed-up values by:

- `max` — optimistic (best value seen through this edge),
- `topk` — mean of the top-$k$ values,
- `softmax` — a temperature-$\tau$ soft-max (log-sum-exp) of the values.

Selected by the `backup_rule` constructor arg. `test_mcts_env_backup_rules.py`
covers all four.

## Optional mechanisms

- **Nonterminal rollouts** (`rollout_n > 0`): at a freshly expanded leaf,
  complete the game `rollout_n` times with uniform-random legal actions
  (game-agnostic; uses only `clone`/`get_action_mask`/`step_wrapper`), aggregate
  the returns (`mean`/`max`), and blend with the net value:
  $(1-\text{blend})\cdot v_\text{rollout} + \text{blend}\cdot v_\text{net}$. A
  shared `rollout_budget` caps total rollout steps per search; if no rollout
  reaches a terminal, it falls back to the net value. See
  `test_mcts_env_rollouts.py`.
- **Tree reuse** (`perform_simulations_reuse` + `advance_to`): keep the same
  `MCTS`/tree across the moves of an episode instead of rebuilding per move.
  Noise re-injection here restores `nn_policy_original` before mixing fresh
  noise (so reused roots with existing visits still get noise, without stacking
  noise-on-noise). See `test_mcts_env_reuse_noise.py`.
- **Audit instrumentation** (`self.debug_state`): when set to a dict, `search()`
  records counters at its four return points (terminal-reward hits, leaf-value
  returns, best/first positive terminal reward seen, max depth, etc.). `None` by
  default, in which case behavior is identical to the un-instrumented path. The
  code comment cites `docs/notes/stage4/03.md`, a note not present in this
  commit (see [00-index](00-index.md) caveat).

## Defensive details worth knowing

- `query_net_masked` makes a **defensive copy** of the prior before masking
  in-place, so a net that returns a shared/cached array can't be corrupted
  (`test_uniform_policy_is_not_mutated_in_place`).
- If a state's whole mask is zero, `query_net_masked` raises `ValueError`
  ("no valid moves") — a genuine dead end; if the prior mass on legal moves is
  zero it warns and falls back to a uniform legal prior.
