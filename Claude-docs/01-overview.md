# Overview

*Generated from a read-through of the codebase (commit `2946193`).*

## One sentence

`sraz` is a **minimal single-player AlphaZero engine** plus one concrete game
instance: **symbolic regression framed as a grammar-derivation game**, where an
agent builds a prefix-notation expression production-by-production and is
rewarded by how well that expression's constants can be least-squares-fitted to
a hidden target curve.

## What "AlphaZero on grammar games" means here

Classic AlphaZero pairs a **policy/value neural network** with **Monte-Carlo
Tree Search (MCTS)**: MCTS uses the network to guide a look-ahead search, the
search produces better move probabilities than the raw network, and the network
is then trained to imitate the search (and to predict episode returns). Repeat.

This repo strips that idea down to the **single-player** case (no opponent, no
self-play-versus-old-network arena) and applies it to a **grammar game**: the
"moves" are grammar productions, and a full game is one derivation of a sentence
in a context-free grammar. The first (and currently only) instance is
**symbolic regression** — the derived sentence is a math expression, and the
reward is the goodness-of-fit ($R^2$) of that expression against target data.

## The two layers

The design cleanly separates a **reusable engine** from a **swappable game
instance**:

```text
                ┌──────────────────────────────────────────────┐
   ENGINE       │  Game (ABC)   PolicyValueNet (ABC)            │
 (game-agnostic)│  MCTS         Agent        Trainer            │
                │  utils: checkpoint / multiprocessing / stats  │
                └──────────────────────────────────────────────┘
                                    ▲
                                    │ implements the two ABCs
                                    │
                ┌──────────────────────────────────────────────┐
  INSTANCE      │  SymRegGame            (the grammar MDP)       │
  (symreg)      │  SymRegPolicyValueNet  (an MLP over one-hot)   │
                │  SymRegConfig          (wires it all up)       │
                └──────────────────────────────────────────────┘
```

- The **engine** (`src/sraz/core/`, `src/sraz/training/`, `src/sraz/utils/`)
  knows nothing about symbolic regression. It talks to games only through the
  `Game` abstract base class and to networks only through `PolicyValueNet`.
- The **instance** (`src/sraz/instances/symreg/`) implements those two ABCs for
  the specific problem, and a config object stitches game + net + agent +
  trainer together.

To add a *new* grammar game you implement a new `Game` subclass and (optionally)
a new net, then a config — the engine, MCTS, and training loop are untouched.
See [05-file-map-and-howto](05-file-map-and-howto.md) for the recipe.

## The symbolic-regression game in brief

- **State.** A fixed 15-slot token buffer holding the current partial expression
  (the "sentential form"), left-justified and pad-filled. Starts as
  $[S, \text{pad}, \dots]$.
- **Action.** A $(\text{position}, \text{production})$ pair — rewrite the
  nonterminal $S$ at `position` using one of 7 grammar productions. Flattened to
  a single `Discrete(105)` index for the network ($15 \times 7 = 105$).
- **Grammar.** 7 productions combining four primitives — a constant $C_0$, a
  linear term $C_1 x$, a quadratic $C_2 x^2$, and a sinusoid $C_3\sin(C_4 x)$ —
  with operators $\{+, *, /\}$.
- **Termination.** When no nonterminal $S$ remains, the buffer decodes to a
  finished prefix expression.
- **Reward** (only at termination): convert prefix → infix, parse with sympy,
  least-squares-fit the free constants with `lmfit` against the target, and
  return the $R^2$ **clipped to $[-1, 1]$**. Every intermediate step scores 0.
- **Target.** $y(x) = 4\sin(4x) + C_0 + C_1 x + C_2 x^2$, sampled at 41 points
  on $[1,3]$. The $C^*$ are drawn once from the problem seed and fixed (so a
  given structure has one deterministic score); per-episode resampling is
  opt-in via `redraw_constants`.

The problem is deliberately **nontrivial**: the token buffer caps expressions at
14 tokens, but the exact generating form needs 18 tokens — so a perfect fit is
*unreachable*, and the agent must find the best expressible surrogate (the
14-token frontier scores $R^2 \approx 0.9873$). Reward is sparse (terminal only)
and the inner constant-fit is nonconvex.

See `Claude-research/notes/01-game-spec.md` for the authoritative, math-heavy specification of the
game, a worked episode, and the first training run. This `Claude-docs/` folder
focuses instead on the **engine internals** and a **file-by-file map**.

## Status of the project

Per `Claude-research/notes/01-game-spec.md` §6, the first training run (seed 42, default
hyperparameters) **converges prematurely** to the one-action expression $C_1 x$
at $R^2 = 0.8729$ and never escapes — a documented baseline, not a success. The
MCTS code contains scaffolding (nonterminal rollouts, alternative backup rules,
tree reuse, Dirichlet noise) that reads as machinery for the follow-up
experiments aimed at beating that baseline.
