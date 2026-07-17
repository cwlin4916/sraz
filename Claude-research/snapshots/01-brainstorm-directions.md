# Brainstorm — directions after the baseline

*Snapshot, 2026-07-15. Immutable; new thinking goes in a new numbered file.*

## The problem, stated sharply

The AlphaZero run does one interesting thing and stops: it locks onto $C_1 x$
($R^2 = 0.8729$) at iteration 1 and never moves. The reachable ceiling is the
14-token frontier at $R^2 = 0.9873$, and there is a whole ladder in between that
it never climbs:

| structure | $R^2$ |
| --- | --- |
| $C_1 x$ (found) | 0.8729 |
| $C_1 x + C_2 x^2$ | 0.9316 |
| $C_0 + C_1 x + C_2 x^2$ | 0.9731 |
| $C_3 \sin(C_4 x) + C_0 + C_2 x^2$ (frontier) | 0.9873 |

## Why it sticks (mechanism)

- Reward is **sparse and terminal-only**, and $C_1 x$ is a *one-action* episode
  — the shortest path to a decent reward. The compositional gateway
  ($S \to +\, S\, S$) needs many more good actions before it pays off.
- With discount $\gamma = 1$, a state's value target $\approx$ expected terminal
  $R^2$ under the *current* policy. Early on the policy fumbles compositions
  (many end in poor structures, or the $-1$ clip floor for degenerate `/` fits),
  so the $+\, S\, S$ branch *looks* low-value. The policy then avoids it,
  starving it of the experience that would reveal its true value —
  **self-reinforcing, deceptive-reward local optimum**.
- 25 simulations at temperature 1 is a tiny budget to rediscover a 14-token
  needle by exploration.

So this is a wide / deep / sparse / deceptive search problem, which is what makes
some interventions clearly higher-leverage than others.

## Option space (tiered)

| Tier | Direction | ROI take |
| --- | --- | --- |
| 1. Diagnose | Instrument self-play: do episodes *ever* reach the 0.93+ structures? How does the net's value for the $+\, S\, S$ branch evolve? | Do first — cheap; distinguishes a *search* failure from a *learning* failure. |
| 2. Exploit built-in knobs | MCTS already has optimistic backup (`max`/`topk`/`softmax`), nonterminal rollouts, tree reuse, and sim/temp/Dirichlet — all untested on symreg (config uses plain mean-backup, 25 sims, no rollouts). | Highest ROI. Runs are ~7s. `backup_rule="max"` directly counteracts the pessimism (keeps the best value ever seen through $+\, S\, S$, not the mean). |
| 3. Algorithmic upgrade | Gumbel AlphaZero — built for low-simulation regimes with guaranteed policy improvement; 25 sims is its sweet spot. | High potential, bigger build. The serious upgrade if knobs aren't enough. |
| 4. Reshape the problem | Curriculum (drop the sinusoid first, grow), raise `max_len` to 19 so the true form is reachable, novelty/entropy bonus, anti-early-termination bias. | Powerful but changes the task definition — only if that's acceptable. |
| 5. Rigor + infra | Multi-seed with error bars (currently n=1), a sweep harness (the parallel path seeds correctly per worker after the RNG fix), fix the 5 low review findings. | Necessary for a real result; not itself a research advance. |

## Recommended first move

**Diagnose (tier 1) + a small ablation grid over the already-built knobs
(tier 2)** — nearly free and maximally informative:

- One diagnostic counter: fraction of self-play terminals that are
  multi-production (does it ever see the ladder?).
- A grid over `{backup_rule in mean,max,topk} x {rollout_n in 0,8} x
  {n_simulations in 25,100} x {temperature in 1.0,1.5}`, a few seeds each. All
  config-only; the whole grid is minutes on CPU.

Prediction: `backup_rule="max"` and/or `n_simulations=100` break $C_1 x$ first.

**Caveat (honest):** naive random rollouts might *hurt* here — uniform-random
grammar completions mostly yield garbage / failed fits, which could make
compositional branches look *even worse*. Worth testing, not a safe bet.

## Open steering questions (unresolved)

1. **Goal?** publishable result (rigor/seeds/novelty) vs learning exercise
   (climb the ladder + understand why) vs capable SR system (AlphaZero may be
   the wrong tool vs genetic programming / PySR).
2. **Is changing the task fair game?** curriculum / larger buffer / reward
   shaping, or beat 0.8729 on the exact game as specified?
