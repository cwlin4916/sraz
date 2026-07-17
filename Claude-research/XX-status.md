# Research status

*Living board — updated in place. Record every material change in the Update log.*

## Current state (2026-07-17)

- **Docs unified.** The former `docs/` tree is merged into `Claude-research/`;
  see [00-index](00-index.md) for the layout and the cross-instance results
  table. All figure drivers now write to `Claude-research/figures/`.
- **Baseline (`sine`, seed 42):** default run (25 sims × 20 games × 10 iters)
  converges at iteration 1 to $C_1 x$ ($R^2 = 0.8729$) and never escapes.
  Reachable ceiling is the 14-token frontier at $R^2 = 0.9873$.
- **Best-corner instance (`additive_quadratic`):** built and wired (general
  (grammar, target) parameterization + problem registry). AZ **climbs to
  $R^2 \approx 1$ at 100+ sims but stalls at $C_2x^2$ (0.929) at the default
  25** — see [snapshots/03](snapshots/03-additive-instance-results.md). Forces
  the refinement: $V^*$-monotonicity ≠ easy; a near-optimal cheap terminal is a
  *shallow trap* whose escape cost scales with search budget.
- **Target families (`lin_A–D`, `quad_A–D`):** eight designed targets with
  closed-form score laws, all exactly expressible ($R^2 = 1$ reachable). Pure
  MCTS collapses to the constant on `quad_B` at $N_{\text{sim}} = 8$ — see
  [notes/02](notes/02-target-families.md).
- **Working framework:** the study is organized around the **informativeness ×
  deception** 2×2 — see [snapshots/02](snapshots/02-informativeness-and-deception.md).
  The `sine` game is the worst corner, consistent with the observed failure.
- **Reproducible:** after the MCTS RNG fix, `run_symreg.py --seed 42` is
  bit-reproducible run-to-run (was not before).
- **Environment:** project-local `.venv` (CPU torch); a full default run is ~7s
  on CPU. 4× A10G GPUs present but not useful here (inference is CPU-pinned in
  `predict()`, and the net is a tiny 2×128 MLP).

## Known defects

1. **Pure-MCTS sweep is not seed-reproducible.** Isolated to
   `MCTS._rollout_value`; a hand-written loop over the same `Game` API with the
   same seed *is* deterministic. Fits, fit cache, hash randomisation and BLAS
   threading each ruled out by measurement. Until fixed,
   [notes/02](notes/02-target-families.md) §2's **shape** is readable and its
   **digits** are not.
2. **17 of 31 figures are orphaned** — generated and committed but cited by no
   note. Enumerated in [00-index](00-index.md).
3. **`notes/01`'s seed-42 training run (§6) is missing.** Promised to "the second
   note", never landed there; `figures/symreg_reward_curve_seed42.png` is its
   orphaned figure.
4. **Dangling code ref (predates the merge):** `src/sraz/core/mcts.py:106` and
   `tests/core/test_mcts_core.py:4` cite `docs/notes/stage4/03.md`, which has
   never existed in this repo.

## Open decisions (need input)

1. **Goal** — publishable result / learning exercise / capable SR system.
2. **Task changes allowed?** — curriculum, larger `max_len`, reward shaping vs
   beating 0.8729 on the exact game.

## Candidate next actions

- [ ] **Resolve the mean-vs-max tension.** [notes/02](notes/02-target-families.md)
      §2.3 already shows averaged rollouts decay ($0.546 \to 0.165$ on `quad_A`)
      while best-of holds near $1.0$, yet `backup_rule="max"` is still listed as
      untested in [snapshots/01](snapshots/01-brainstorm-directions.md)'s
      highest-ROI tier. Cheapest open question in the repo.
- [ ] **Write up the orphaned figures**, or delete them. 17 results with no
      claim attached is the largest single gap.
- [ ] Fix the `_rollout_value` seeding defect so §2's digits become quotable.
- [ ] Diagnose: log fraction of self-play terminals that are multi-production;
      track the value the net assigns the `S -> + S S` branch across iterations.
- [ ] Ablation grid over built-in knobs (`backup_rule`, `rollout_n`,
      `n_simulations`, `temperature`), a few seeds each — config-only, minutes.
- [ ] Cleaner positive control: an additive target with *no* near-optimal single
      term, predicting AZ climbs even at 25 sims (tests axis-2 part (b) directly).
- [ ] (bigger) Gumbel AlphaZero for the low-simulation regime.
- [ ] (rigor) multi-seed sweep harness + error bars.

## Update log

- 2026-07-17 — **Merged `docs/` into `Claude-research/`.** `docs/notes/{01,02}.md`
  → `notes/`, `docs/research/` → `lit/`, `docs/comments/` → `comments/`, figures
  unioned into `figures/` (no basename collisions), existing numbered files →
  `snapshots/`. Repointed 8 figure drivers off `docs/notes/figures`; fixed the
  pre-existing dangling links `02_training_run.md` (never existed) and
  `tests/test_grammar_env.py` (renamed to `tests/instances/symreg/test_game_grammar.py`
  by the test reorg). Rewrote [00-index](00-index.md) with a cross-instance results
  table and the provisional/reproducible split. Surfaced three gaps, logged above.
- 2026-07-15 — Built the `additive_quadratic` best-corner instance (general
  (grammar, target) parameterization of SymRegGame + `problems.py` registry, +10
  tests, 274 passing). First AZ result recorded in
  [snapshots/03](snapshots/03-additive-instance-results.md): climbs to
  R²≈1 at 100/400 sims, stalls at `C2·x²` (0.929) at the default 25 — a *shallow
  trap* (near-optimal cheap terminal). Refines axis 2 into V\*-monotone vs
  exploration-accessible gradient.
- 2026-07-15 — Added [snapshots/02](snapshots/02-informativeness-and-deception.md):
  the informativeness × deception 2×2, the core conceptual tool for the study
  (each axis maps to one of the two networks; the current game is the worst
  corner). This is the framing to build experiments around.
- 2026-07-15 — Created Claude-research; captured the initial brainstorm (01) and
  this board. Baseline reproduced (7s CPU); correctness fixes + test reorg
  landed on branch `review/mcts-fixes-and-test-reorg`. Awaiting goal + task-scope
  decisions before picking a direction.
