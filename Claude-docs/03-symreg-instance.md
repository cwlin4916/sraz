# The symbolic-regression instance

Everything SR-specific lives in `src/sraz/instances/symreg/`. It supplies the
two engine ABCs ([02-architecture](02-architecture.md)) — a `Game` and a
`PolicyValueNet` — plus a config that wires the run together. This doc is the
code-level companion to the formal spec in `Claude-research/notes/01-game-spec.md`.

## `game.py` — the grammar game + SR reward

The file has three logical parts.

### 1. Prefix → infix conversion

`prefix_to_infix(tokens)` is a single stack pass converting a Polish-notation
token list to a parenthesized infix string, e.g.
`+ * C3 sin * C4 x + C0 * C2 * x x` → `((C3*sin((C4*x)))+(C0+(C2*(x*x))))`.
Operators (`/ * + - **`) pop two operands; functions (`cos sin`) pop one.

### 2. Grammar compilation

`SR_GRAMMAR` is the 7-production context-free grammar with a single nonterminal
`S`:

| $j$ | production | $\lvert\text{rhs}\rvert$ |
| --- | --- | --- |
| 0 | `S -> + S S` | 3 |
| 1 | `S -> C0` | 1 |
| 2 | `S -> * C1 x` | 3 |
| 3 | `S -> * C2 * x x` | 5 |
| 4 | `S -> * S S` | 3 |
| 5 | `S -> / S S` | 3 |
| 6 | `S -> * C3 sin * C4 x` | 6 |

`compile_grammar()` walks the rules and interns every symbol into an integer
token (LHS first, then unseen RHS symbols), producing a `Grammar` dataclass:
`symdict` (symbol→token), `tokenlist` (token→symbol), `nonterms` (tokens that
have productions — just `S`), `productions` (a global list of tokenized RHSs),
`proddict` (nonterminal token → its global production indices), and the pad
token. The resulting alphabet is 11 symbols
($S, +, C_0, *, C_1, x, C_2, /, C_3, \sin, C_4$) plus pad = token 11.

### 3. The SR evaluator (terminal reward)

`fit_expression(rule, xs, exact_ys)` turns a finished prefix sentence into a
scalar reward:

1. Split into tokens; register `x` as the independent variable and every `C*`
   token as a free parameter initialized to `C_INIT = 2.5`.
2. `prefix_to_infix` → `sympy_parser.parse_expr` → `sympy.lambdify` to a callable.
3. Wrap in an `lmfit.Model` and least-squares-fit the constants to `exact_ys`.
4. Compute $R^2 = 1 - \text{SS}_\text{res}/\text{SS}_\text{tot}$ and return
   $\operatorname{clip}(R^2, -1, 1)$.

It **fails soft**: any exception, a non-successful solve, or a non-finite $R^2$
yields $-1.0$ rather than raising.

### The game class: `SymRegGame`

A `Game[np.ndarray, int]` implementing the derivation MDP.

- **Observation space:** `MultiDiscrete([nsym+1] * max_len)` — the token buffer
  (default `max_len = 15`, so 15 slots over a 12-symbol alphabet).
- **Action space:** `Discrete(max_len * nprods)` = `Discrete(105)`. An action
  decodes as `pos, prod = divmod(action, nprods)`.
- **`reset()`**: buffer = `[S, pad, …]`, `real_state_len = 1`. If
  `redraw_constants`, redraw the target `C*` and clear the fit cache.
- **`get_action_mask()`**: for each occupied slot holding a nonterminal, mark
  every production of that nonterminal whose splice keeps the length under
  `max_len`. Concretely legal iff
  $\ell(s) + \lvert\text{rhs}(p_j)\rvert - 1 < \texttt{max\_len}$, so finished
  sentences carry at most `max_len - 1 = 14` tokens.
- **`step(action)`**: decode `(pos, prod)`; if invalid or overflowing, **fail
  closed** (return reward $-1$, `terminated=True`). Otherwise splice the RHS
  over position `pos` (shifting the tail right), pad-fill the remainder, and
  update `real_state_len`. If no nonterminal remains, decode the sentence and
  return the cached fit reward; else reward 0, not terminated.
- **`_fit_cached(rule)`**: memoizes `fit_expression` per sentence string. Sound
  because constants are fixed for a given instance; cleared on redraw.

**Constants:** `C_MIN, C_MAX = 1.0, 4.0`; the three target constants
$C_0, C_1, C_2$ are drawn once from `problem_seed` (default fixed, `redraw` opt
-in). Data grid: 41 points on $[1, 3]$. Target:
$4\sin(4x) + C_0 + C_1 x + C_2 x^2$.

**Custom stash/unstash:** overrides the default `deepcopy` to snapshot only the
mutable derivation state (`state`, `real_state_len`) and the `Game` bookkeeping
fields, deliberately *not* the fit cache or the (immutable) data arrays. This is
a real speed win since MCTS stashes once per simulation.

## `network.py` — `SymRegPolicyValueNet`

An MLP over a **one-hot encoding** of the token buffer. Token IDs carry no
ordinal meaning, so each of the 15 slots is one-hot over the 12-symbol
vocabulary, giving a flat $15 \times 12 = 180$-dim input; the output is a
105-way policy plus a scalar value. Built on `PolicyValueNetModel` (default 2
hidden layers of 128).

- **`train`**: Adam (lr $10^{-3}$, weight decay $10^{-4}$), MSE on the value
  head + cross-entropy on the policy head (policy target is the MCTS visit
  distribution), 10 epochs, batch 32.
- **`predict`**: runs the model on CPU, softmaxes the policy logits, returns
  `(policy_prob[105], value_scalar)`.

## `config.py` — `SymRegConfig`

A `MetaConfig` subclass whose `build()` instantiates the game, sizes the network
from the game's grammar (`n_tokens = nsym+1`, `n_actions = max_len * nprods`),
and constructs the agent and trainer. Defaults encode the documented baseline:
25 MCTS simulations, temperature 1.0, $c_\text{explore}$ 1.0, 20 self-play games
per iteration, a 10-iteration replay window, sequential self-play (`n_procs=-1`),
10 iterations total.

## Test coverage for the instance

The suite has ~254 tests total; SR-instance behavior is pinned by, among others:

- `test_symreg_game_contract.py` — Game-ABC semantics: seeded-reset determinism,
  stash/unstash exactness and reusability, clone independence, fixed vs. redraw
  constants, fit-cache reuse.
- `test_symreg_game_unit.py`, `test_grammar_env.py` — grammar/mask/step details
  and the canonical worked episode `[(0,0),(1,6),(7,0),(8,1),(9,3)]` that derives
  the 14-token frontier expression.
- `test_sr_eval.py` — the prefix→infix→fit→$R^2$ pipeline and its fail-soft paths.
