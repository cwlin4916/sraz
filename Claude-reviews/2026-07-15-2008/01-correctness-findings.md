# Correctness findings

8 findings, all CONFIRMED by adversarial verification (0 refuted). Severity is
about blast radius in this codebase, not abstract badness. "Fixed" findings
have regression tests; "Reported" ones were left unchanged per the approved
scope (reorganize + fix the 3 mediums only).

## Medium (fixed)

### #1 — Multi-dimensional Dirichlet noise is mis-sized `[FIXED]`

- **Where:** `src/sraz/core/mcts.py` root-noise draw (both `perform_simulations`
  and `perform_simulations_reuse`).
- **Bug:** noise was `np.random.dirichlet([alpha] * len(nn_policy))`. For a
  `MultiDiscrete` action space `nn_policy` is 2-D, and `len()` returns only the
  first axis. A non-square grid raised `ValueError` in `noise * mask`; a square
  grid broadcast one identical row across the grid (statistically wrong noise).
  1-D `Discrete` games (symreg) were unaffected, but the engine otherwise fully
  supports multi-dim spaces.
- **Fix:** draw `[alpha] * nn_policy.size` and `.reshape(nn_policy.shape)`.
- **Regression:** `tests/core/test_mcts_noise_reuse.py::test_noise_multidim_square_grid_reshapes_and_is_not_row_broadcast`,
  `::test_noise_multidim_nonsquare_grid_does_not_crash`,
  `::test_noise_multidim_reuse_path_does_not_crash`.

### #2 — Leaf-eval caches extracted from an unstepped game `[FIXED]`

- **Where:** `src/sraz/core/agent.py` `play_for_experience` /
  `play_for_experience_reuse_tree`.
- **Bug:** `_extract_leaf_eval_data` was called on the outer game object, which
  `play_one_round`/MCTS never step (they clone internally). `export_caches()`
  therefore always returned the empty/baseline cache, silently nullifying the
  Trainer's cross-worker cache merge. Fully latent today — no `src/` game defines
  a `leaf_evaluator` — but wrong for any that would.
- **Fix:** added `clone_game` to the play methods; the `play_for_experience`
  wrappers now pass `clone_game=False` so the game actually stepped during the
  episode is the one whose caches are exported.
- **Residual (documented):** MCTS-simulation caches accumulate on ephemeral
  clones and are not captured; a `leaf_evaluator` needing those must be shared
  by reference across clones (a design choice beyond this fix).
- **Regression:** `tests/core/test_agent.py::test_play_for_experience_extracts_leaf_eval_from_stepped_game`,
  `::test_play_for_experience_reuse_tree_extracts_leaf_eval`.

### #3 — MCTS randomness decoupled from the seeded RNG streams `[FIXED]`

- **Where:** `src/sraz/core/mcts.py` Dirichlet noise + random rollouts.
- **Bug:** exploration randomness used the process-global legacy `np.random`,
  which the Agent's four named seed streams (incl. the one literally called
  `"mcts"`) never touched. `run_symreg.py` never seeds the global RNG, so the
  "seed 42" run's exploration was non-reproducible; and under `fork`
  multiprocessing, workers inherit the same global state and draw identical
  noise (killing self-play diversity).
- **Fix:** `MCTS` takes an optional `rng: np.random.Generator`; the Agent threads
  one seeded per-episode Generator (from `random_seed`/`interaction_seed`, i.e.
  the `"mcts"` stream) through `policy` → MCTS and the reuse path. MCTS never
  touches global `np.random`.
- **Verification:** two `run_symreg.py --seed 42 ...` invocations are now
  byte-identical (were not before). Regression:
  `tests/core/test_mcts_noise_reuse.py::test_noise_uses_injected_rng_and_ignores_global_np_random`.
- **Test migration:** tests that seeded global `np.random` to control MCTS noise
  were updated to inject `rng=np.random.default_rng(seed)` (the correct pattern);
  the previously luck-passing `test_policy_add_noise_*` agent test was made
  deterministic (was flaky 16/20 after the fix until migrated).

## Low (reported, not changed)

| # | Where | Issue | Note |
| --- | --- | --- | --- |
| 4 | `training/trainer.py` (`train_multiple`, `save_checkpoint` default) | Writes a hardcoded `Path("checkpoints")`, ignoring the configured `checkpoint_dir`. | Pinned by `test_train_multiple_checkpoint_every_writes_hardcoded_dir` as "documented current behavior" — changing it means updating that test. |
| 5 | `core/config.py` `_to_serializable_dict` | numpy integer scalars are `str()`-ified in config JSON (numpy floats / py-ints survive). | Fix: normalize via `obj.item()` / `numbers.Integral` before the `str()` fallback. Provenance-only. |
| 6 | `utils/multiprocessing.py` `push()` | Appends to `self.stashes` without clearing first; a mid-push failure leaves stale stashes that mis-map on the next push. | Only reachable if `push_multiprocessing()` can raise. Fix: reset `self.stashes = []` at the top of `push()`. |
| 7 | `scripts/run/run_symreg.py:164` | Logged `fit_cache_size` is always 0 (reads the wrong game object). | Diagnostic only; no effect on training. |
| 8 | `scripts/run/run_symreg.py:90` | `config.json`'s `plot_path` points at `training_metrics.png`, never written (actual curve is `reward_curve.png`). | Cosmetic / mislabeled config field. |

## Not bugs (verification refuted or confirmed-intended)

The review also examined and cleared: the `temperature=0` NaN path (guarded +
tested as intended), terminal-state future value `= 0.0`, the reward fail-soft to
`-1`, `predict()` forcing CPU, and the strict mask length bound (`< max_len`).
These are documented, tested design choices, not defects.
