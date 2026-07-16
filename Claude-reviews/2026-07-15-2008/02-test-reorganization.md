# Test reorganization

Goal: make it trivial to find, from a source file, the tests that exercise it and
read the behaviors they pin. The suite now mirrors `src/sraz/` one-to-one.

## New layout

```
tests/
  pytest.ini is at repo root (testpaths=tests, pythonpath=.)
  conftest.py            (unchanged: pins thread env vars pre-import)
  README.md              navigation index: component -> file -> behaviors
  helpers/mcts_envs.py   shared synthetic envs + stub nets
  core/       test_game, test_agent, test_config, test_policy_value_net,
              test_mcts_core, test_mcts_search, test_mcts_backup,
              test_mcts_noise_reuse, test_mcts_rollouts, test_mcts_invariants
  training/   test_trainer
  utils/      test_common, test_statistics, test_multiprocessing, test_checkpoint
  instances/symreg/  test_game_grammar, test_game_unit, test_game_contract,
                     test_sr_eval, test_network, test_config
  e2e/        test_smoke
```

Every directory (including `tests/` and `helpers/`) has an `__init__.py`, so
duplicate basenames like `core/test_config.py` vs `instances/symreg/test_config.py`
get distinct module paths under pytest's default import mode. `pytest.ini` adds
`pythonpath=.` so `from tests.helpers.mcts_envs import ...` resolves regardless of
invocation.

## Files split (each previously mixed multiple components)

| Old file | Split into |
| --- | --- |
| `test_utils.py` | `utils/test_common.py`, `utils/test_statistics.py`, `utils/test_multiprocessing.py` |
| `test_policy_value_net.py` | `core/test_policy_value_net.py` (ABC + Torch + MLP), `instances/symreg/test_network.py` |
| `test_config.py` | `core/test_config.py` (dataclasses + MetaConfig), `instances/symreg/test_config.py` (SymRegConfig.build) |

All other files were moved 1:1 (renamed for clarity, e.g.
`test_mcts_env_search_correctness.py` -> `core/test_mcts_search.py`). `test_trainer.py`
was kept whole — it exercises the Trainer, which legitimately drives checkpoint/mp
as collaborators.

## Counts

258 tests preserved across the move/split (verified per-file), + 6 new regression
tests = **264 total** (263 passed, 1 skipped — a macOS-only env-var test).

## Regression tests added

- Fix #1 (multi-dim noise): `core/test_mcts_noise_reuse.py` — square-grid reshape
  (not row-broadcast), non-square no-crash, reuse-path no-crash.
- Fix #3 (seeded RNG): `core/test_mcts_noise_reuse.py` — same injected seed gives
  identical noise regardless of global `np.random`; different seed differs.
- Fix #2 (leaf-eval): `core/test_agent.py` — export is non-empty after an episode
  for both the standard and reuse-tree play paths.

## Test-behavior migration (from finding #3)

Tests that seeded the process-global `np.random` to make MCTS noise reproducible
now inject a seeded Generator instead (the correct pattern), and the trainer
tests' now-dead `np.random.seed(...)` lines + misleading comments were removed —
their determinism flows through the Agent's seeded `"mcts"` stream.

## How to use it (for reviewers)

- Start at `tests/README.md` for the component -> file map.
- To review a source file, open the parallel test path and read the test names
  and docstrings; behaviors are named, and intentional "documents current
  behavior" choices are commented.
