# Claude-scripts — index

Analysis scripts for the symbolic-regression grammar game. These are
standalone measurements of the *task* (policy-free), not training runs.

Run everything through the project venv from the repo root, e.g.
`.venv/bin/python Claude-scripts/01-state-value-gap.py --help`.

| Script | What it computes |
| --- | --- |
| [01-state-value-gap.py](01-state-value-gap.py) | For one derivation state $s$: the optimal value $V^*(s)$ (exact, by exhaustive completion enumeration + parallel lmfit, disk-cached) and the random-rollout value distribution $V_{\text{rand}}(s)$ ($K$ seeded uniform-legal-action completions), plus the gap $V^*(s) - \mathbb{E}[V_{\text{rand}}(s)]$ — the policy-free informativeness signal. Writes JSON to `Claude-experiments/8-5/`. |

## Notes that bit us once (read before trusting a number)

- **The sine target's constants come from `problem_seed`, and the documented
  baseline uses `problem_seed = 42`, not the shipped default 0**
  (`run_symreg.py` maps `--seed` straight onto `problem_seed`). Clipped-$R^2$
  depends on those constants, so the fit cache is keyed by seed
  (`cache/fit_cache_sine_seed42.json`) and results must state the seed.
- $V^*$ needs **exact enumeration** — sampling (even 1000 rollouts) undershoots
  the true max, because the optimum is a specific rare structure.
