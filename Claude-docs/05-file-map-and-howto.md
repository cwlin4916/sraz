# File map & how-to

## Repository map

```text
src/sraz/
  core/
    game.py              Game ABC (+ EnvGame Gymnasium adapter); stash/clone/hashable_obs
    policy_value_net.py  PolicyValueNet ABC, TorchPolicyValueNet, PolicyValueNetModel (MLP)
    mcts.py              single-player AlphaZero MCTS (selection/expansion/backup/noise/rollouts)
    agent.py             Agent: self-play, experience collection, discounted returns
    config.py            MetaConfig + the config dataclasses; save()/plot()
  training/
    trainer.py           Trainer: collect → replay window → net.train; checkpoint hooks
  utils/
    common.py            thread pinning, deterministic CUDA, get_device
    multiprocessing.py   MultiprocessingManager (push/pop GPU state, starmap)
    checkpoint.py        CheckpointManager (pickle agent state + net weights)
    statistics.py        StatisticsManager (JSONL stat logs)
  instances/symreg/
    game.py              SymRegGame + grammar compile + prefix→infix + lmfit reward
    network.py           SymRegPolicyValueNet (one-hot MLP)
    config.py            SymRegConfig (baseline hyperparameters, build())
scripts/
  run/run_symreg.py         training driver (CLI, greedy eval, experiment dir, plots)
  plotting/plot_sr_game.py  Figure 1 generator (replays the real env)
Claude-research/notes/01-game-spec.md            formal SR game spec + first training run (seed 42)
tests/                      ~254 tests: engine (mcts_env_*, game, agent, trainer, ...) + symreg
```

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU wheel is fine
pip install -r requirements.txt
pip install -e .
```

Deps: numpy, gymnasium, matplotlib, lmfit (pulls scipy), sympy, pytest. Python
$\geq 3.10$.

## Test

```bash
pytest tests/ -q          # ~30 s; conftest.py pins thread env vars first
```

## Run the documented experiment

```bash
python scripts/run/run_symreg.py --seed 42
```

Useful flags (all optional; defaults come from `SymRegConfig`):

| Flag | Meaning | Default |
| --- | --- | --- |
| `--seed` | run seed; also used as `problem_seed` and to seed the four RNG streams | 42 |
| `--n-iterations` | outer AlphaZero iterations | 10 |
| `--n-simulations` | MCTS simulations per move | 25 |
| `--n-games` | self-play games per iteration | 20 |
| `--max-len` | token-buffer length (the full target needs 19 to be expressible) | 15 |
| `--redraw-constants` | resample target constants each episode (stochastic variant) | off |
| `--out` | output root dir | `experiments/symreg` |

Outputs land in `experiments/symreg/<timestamp>_seed<seed>_mcts<n>_iter<n>/`:
`config.json`, `train_stats.jsonl`, `iteration_log.jsonl`, `reward_curve.png`,
and `checkpoints/final/` (network weights + pickled replay buffer + RNG state).

### What the driver does (`run_symreg.py`)

- Pins NumPy threads and deterministic CUDA **before** importing numpy/torch.
- Builds the config, overrides from CLI, seeds everything, saves `config.json`.
- Loops `train_iteration()` then a **greedy eval** (`compute_greedy_eval`: one
  noise-free, near-argmax episode with `temperature_override=0.01`), logging
  greedy $R^2$, best-so-far, avg self-play $R^2$, and losses per iteration.
- Its `_save_final_checkpoint` pickles `trainer.all_training_examples` directly,
  working around the `CheckpointManager`/`agent` attribute mismatch noted in
  [02-architecture](02-architecture.md).

## Regenerate Figure 1

```bash
python scripts/plotting/plot_sr_game.py   # writes Claude-research/figures/sr_game_progression.png
```

It replays the canonical scripted episode through the *real* `SymRegGame` and
fits it through the *real* reward pipeline — nothing is hand-transcribed.

## How to add a new grammar game

The engine is game-agnostic, so a new instance is self-contained:

1. **Implement a `Game` subclass** with `step`, `reset`, `get_action_mask`, and
   `action_space`/`observation_space`. Ensure `hashable_obs` is meaningful; for
   speed, override `stash_state`/`unstash_state` to snapshot only the mutable
   fields (see `SymRegGame` for the pattern) — and round-trip *all* the `Game`
   bookkeeping fields, or MCTS rewinds will corrupt.
2. **Provide a net** — reuse `PolicyValueNetModel`/`TorchPolicyValueNet` with an
   appropriate encoding of your observation, mirroring `SymRegPolicyValueNet`.
3. **Write a `MetaConfig` subclass** whose `build()` sizes the net from the game
   and returns `(game, net, agent, trainer)`.
4. **Add a driver** modeled on `run_symreg.py`, and tests modeled on
   `test_symreg_game_contract.py` (Game-ABC semantics) plus any game-specific
   unit tests.

Nothing in `core/mcts.py`, `core/agent.py`, or `training/trainer.py` should need
to change.

## Reproducibility notes

- Determinism relies on pinning BLAS/OMP threads to 1 and seeding the agent's
  four RNG streams; `conftest.py` sets the thread env vars for tests.
- With fixed constants (default), a given expression string has one
  deterministic $R^2$, and the per-sentence fit cache is sound. Pass
  `--redraw-constants` to recover the stochastic variant (value becomes an
  expectation over resampled targets; the cache is invalidated each redraw).
