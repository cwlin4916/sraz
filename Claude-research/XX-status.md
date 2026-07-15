# Research status

*Living board — updated in place. Record every material change in the Update log.*

## Current state (2026-07-15)

- **Baseline:** default run (seed 42, 25 sims × 20 games × 10 iters) converges
  at iteration 1 to $C_1 x$ ($R^2 = 0.8729$) and never escapes. Reachable
  ceiling is the 14-token frontier at $R^2 = 0.9873$.
- **Reproducible:** after the MCTS RNG fix, `run_symreg.py --seed 42` is
  bit-reproducible run-to-run (was not before).
- **Environment:** project-local `.venv` (CPU torch); a full default run is ~7s
  on CPU. 4× A10G GPUs present but not useful here (inference is CPU-pinned in
  `predict()`, and the net is a tiny 2×128 MLP).
- **Code:** correctness fixes + test reorganization committed on branch
  `review/mcts-fixes-and-test-reorg` (not yet pushed — remote permissions).
- **Working framework:** the study is organized around the **informativeness ×
  deception** 2×2 — see [02-informativeness-and-deception](02-informativeness-and-deception.md).
  The current game is the worst corner (uninformative + deceptive), consistent
  with the observed failure.
- **First best-corner experiment done:** the `additive_quadratic` instance is
  built and wired (general grammar/target parameterization + problem registry).
  AZ **climbs to R²≈1 at 100+ sims but stalls at `C2·x²` (0.929) at the default
  25** — see [03-additive-instance-results](03-additive-instance-results.md).
  This forces a refinement: V\*-monotonicity ≠ easy; a near-optimal cheap terminal
  is a *shallow trap* whose escape cost scales with search budget.

## Open decisions (need input)

1. **Goal** — publishable result / learning exercise / capable SR system.
2. **Task changes allowed?** — curriculum, larger `max_len`, reward shaping vs
   beating 0.8729 on the exact game.

## Candidate next actions

From [01-brainstorm-directions](01-brainstorm-directions.md):

- [ ] Diagnose: log fraction of self-play terminals that are multi-production;
      track the value the net assigns the `S -> + S S` branch across iterations.
- [ ] Ablation grid over built-in knobs (`backup_rule`, `rollout_n`,
      `n_simulations`, `temperature`), a few seeds each — config-only, minutes.
- [ ] (bigger) Gumbel AlphaZero for the low-simulation regime.
- [ ] (rigor) multi-seed sweep harness + error bars.

## Update log

- 2026-07-15 — Built the `additive_quadratic` best-corner instance (general
  (grammar, target) parameterization of SymRegGame + `problems.py` registry, +10
  tests, 274 passing). First AZ result recorded in
  [03-additive-instance-results](03-additive-instance-results.md): climbs to
  R²≈1 at 100/400 sims, stalls at `C2·x²` (0.929) at the default 25 — a *shallow
  trap* (near-optimal cheap terminal). Refines axis 2 into V\*-monotone vs
  exploration-accessible gradient.
- 2026-07-15 — Added [02-informativeness-and-deception](02-informativeness-and-deception.md):
  the informativeness × deception 2×2, the core conceptual tool for the study
  (each axis maps to one of the two networks; the current game is the worst
  corner). This is the framing to build experiments around.
- 2026-07-15 — Created Claude-research; captured the initial brainstorm (01) and
  this board. Baseline reproduced (7s CPU); correctness fixes + test reorg
  landed on branch `review/mcts-fixes-and-test-reorg`. Awaiting goal + task-scope
  decisions before picking a direction.
