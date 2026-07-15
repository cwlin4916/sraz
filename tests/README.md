# Test suite

Tests are organized to mirror `src/sraz/`, so the tests for any source module
live at the parallel path under `tests/`. To review what a component guarantees,
open its test file and read the test names + docstrings (every test is named for
the behavior it pins).

```
tests/
  core/          -> src/sraz/core/
  training/      -> src/sraz/training/
  utils/         -> src/sraz/utils/
  instances/symreg/ -> src/sraz/instances/symreg/
  e2e/           cross-cutting end-to-end smoke
  helpers/       shared fixtures/stubs (not collected as tests)
```

Run everything: `pytest` (config in `pytest.ini` pins `testpaths=tests` and
`pythonpath=.`). Run one component: `pytest tests/core/test_mcts_search.py`.

## Component -> test file map

### `core/` — the game-agnostic engine

| Source | Test file | n | What it verifies |
| --- | --- | --: | --- |
| `core/game.py` | `core/test_game.py` | 28 | `Game` ABC wrappers (step_count/bookkeeping), `hashable_obs` for scalars/tuples/ndarrays incl. collisions, `stash`/`unstash`/`clone` deep-snapshot semantics, `EnvGame` delegation + mask error mapping |
| `core/agent.py` | `core/test_agent.py` | 24 | `policy` flat/normalized output + temperature + injected-rng noise, `play_one_round` discounted returns / obs-copy independence / seed determinism, reuse-tree variant, `play_for_experience` seeding + leaf-eval extraction, RNG-stream construction |
| `core/policy_value_net.py` | `core/test_policy_value_net.py` | 14 | `PolicyValueNet` ABC surface, `TorchPolicyValueNet` checkpoint round-trip + push/pop device moves, `PolicyValueNetModel` forward shapes / body depth / unnormalized logits |
| `core/config.py` | `core/test_config.py` | 12 | dataclass defaults + per-instance mutable-default independence, `MetaConfig` abstractness, callable/tuple serialization, `save`/`plot` |
| `core/mcts.py` | `core/test_mcts_core.py` | 6 | node invariants (`total_N == sum action_N`), `n_simulations==1` no-NaN fallback, masked actions never selected, terminal reward backprop, temperature transform |
| `core/mcts.py` | `core/test_mcts_search.py` | 8 | exact-Q search correctness on known MDPs: bandit optimal arm, adversarial prior overcome, binary-tree first action, delayed-trap Bellman, value-guided search, temperature sharpening |
| `core/mcts.py` | `core/test_mcts_backup.py` | 11 | backup rules mean/max/topk/softmax math, UCB Q-normalization regimes, masked `-inf` in 1-D and 2-D, chain integration |
| `core/mcts.py` | `core/test_mcts_noise_reuse.py` | 11 | Dirichlet noise on fresh/visited roots, reuse re-injection (no noise-on-noise), `advance_to` warm start + node retention; **regressions:** multi-dim noise reshape (square/non-square) + RNG independence from global `np.random` |
| `core/mcts.py` | `core/test_mcts_rollouts.py` | 10 | rollout leaf eval, budget accounting/reset, mean/max modes, blend formula, multidim stepping, negative-`n_simulations` direct query + masking |
| `core/mcts.py` | `core/test_mcts_invariants.py` | 14 | MultiDiscrete (`GridBandit`) UCB fallback + masking, transposition node sharing, dead-end raise, truncation-as-terminal, `total_N` invariant across envs, game-state restored after search |

### `training/`

| Source | Test file | n | What it verifies |
| --- | --- | --: | --- |
| `training/trainer.py` | `training/test_trainer.py` | 23 | example collection (mp push/pop, sequential vs parallel), replay-window trimming, entropy/`policy_kl_gap` stats, leaf-cache snapshot/merge, `train_multiple` checkpointing, end-to-end symreg iteration, cross-build determinism via seeded streams |

### `utils/`

| Source | Test file | n | What it verifies |
| --- | --- | --: | --- |
| `utils/common.py` | `utils/test_common.py` | 4 | thread-var pinning, deterministic-CUDA env, `get_device` |
| `utils/statistics.py` | `utils/test_statistics.py` | 9 | `StatisticsManager` timestamping, non-dict rejection, JSONL append vs overwrite counter semantics |
| `utils/multiprocessing.py` | `utils/test_multiprocessing.py` | 11 | `MultiprocessingManager` push/pop state machine, `starmap` sequential/pool/fallback, `validate_multiprocessing_setup` |
| `utils/checkpoint.py` | `utils/test_checkpoint.py` | 27 | `CheckpointManager` save/load/validate/list/delete, `exclude_keys`, corrupt-pickle handling |

### `instances/symreg/`

| Source | Test file | n | What it verifies |
| --- | --- | --: | --- |
| `symreg/game.py` (grammar/MDP) | `instances/symreg/test_game_grammar.py` | 7 | grammar compilation artifacts, reset obs, mask at reset, splice step, full derivation of the frontier form, no dead ends, fail-closed invalid action |
| `symreg/game.py` (MDP unit) | `instances/symreg/test_game_unit.py` | 15 | state-copy returns, illegal/terminal fail-closed, mask length boundary per production, stash/clone mid-episode fidelity, hashable keying, small-`max_len` |
| `symreg/game.py` (engine contract) | `instances/symreg/test_game_contract.py` | 6 | seeded-reset determinism, stash/unstash exactness+reuse, clone independence, fixed vs redraw constants, fit-cache reuse |
| `symreg/game.py` (SR reward) | `instances/symreg/test_sr_eval.py` | 7 | prefix→infix, fit of true/partial/constant forms, garbage→−1 fail-soft, clip to [−1,1], target formula |
| `symreg/network.py` | `instances/symreg/test_network.py` | 9 | one-hot encoding, predict shapes/normalization, seed determinism, training-params merge, train loss finite+decreasing, checkpoint round-trip |
| `symreg/config.py` | `instances/symreg/test_config.py` | 7 | `SymRegConfig` defaults, `build()` wiring + net sizing from grammar, kwargs overrides, class serialization |

### `e2e/`

| Test file | n | What it verifies |
| --- | --: | --- |
| `e2e/test_smoke.py` | 1 | one full training iteration + a greedy episode on the real symreg stack |

## Helpers (`tests/helpers/`)

`mcts_envs.py` — synthetic environments with analytically known ground truth
(`TableGame`, `make_bandit`/`make_chain`/`make_binary_tree`/`make_delayed_trap`/
`make_transposition`/`make_truncation`/`make_dead_end`, and the MultiDiscrete
`GridBanditGame`) plus stub nets (`UniformNet`, `FixedPolicyNet`,
`ValueTableNet`). Imported by the `core/test_mcts_*` files as
`from tests.helpers.mcts_envs import ...`. Games/nets that are specific to one
test file are defined locally in that file instead.

## Notes for reviewers

- Several tests deliberately pin *documented current behavior* (e.g. the
  hardcoded checkpoint dir, the `save_jsonl` overwrite counter) rather than an
  ideal — those carry an explanatory comment.
- MCTS exploration randomness is injected via a seeded `rng` Generator (never
  the process-global `np.random`); tests that need determinism pass
  `rng=np.random.default_rng(seed)`.
- See `Claude-reviews/` for the correctness review that accompanied this layout.
