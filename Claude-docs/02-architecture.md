# Architecture — the game-agnostic engine

This doc describes the reusable AlphaZero engine that lives outside the symbolic
regression instance. Everything here is generic: it operates on any
`Game`/`PolicyValueNet` pair. The SR-specific pieces are covered in
[03-symreg-instance](03-symreg-instance.md).

## The two abstract interfaces

Everything hangs off two ABCs. A concrete problem supplies both.

### `Game` — `src/sraz/core/game.py`

An abstract single-player environment, intentionally shaped so a Farama
Gymnasium env is *almost* already an implementation. Subclasses must implement:

- `step(action) -> (obs, reward, terminated, truncated, info)`
- `reset() -> (obs, info)`
- `get_action_mask()` — a boolean/0-1 array marking legal actions.

and must define `action_space` and `observation_space` (Gymnasium spaces).

The base class then layers on the machinery MCTS depends on:

| Member | Role |
| --- | --- |
| `reset_wrapper()` / `step_wrapper()` | Call `reset`/`step` and cache the results on `self.obs`, `self.reward`, `self.terminated`, `self.truncated`, `self.info`, `self.step_count`. MCTS reads these cached fields rather than plumbing return values around. |
| `hashable_obs` (property) | A hashable key for the current observation. Keys the MCTS transposition table (`nodes`). Default `_hashable_obs_impl` handles scalars, tuples/lists, and ndarrays (via `.tobytes()`). |
| `stash_state()` / `unstash_state()` | Snapshot and restore state so MCTS can descend a simulation and then rewind. Default is `deepcopy`; subclasses override for speed (SR does). |
| `clone()` | An independent playable copy (default `deepcopy`). Used by the `Agent` and by rollouts. |

`EnvGame` is a concrete adapter: wrap any Gymnasium `Env` and it becomes a
`Game`, forwarding `step`/`reset`/`get_action_mask`/`render` to `env`.

> **Key contract:** MCTS keeps its own copy of the game and mutates it in place,
> using `stash_state`/`unstash_state` to rewind after each simulation. A correct
> `stash`/`unstash` pair (round-tripping *all* of `obs`, `reward`, `terminated`,
> `truncated`, `info`, `step_count`, plus internal state) is load-bearing. The
> SR contract tests assert this explicitly.

### `PolicyValueNet` — `src/sraz/core/policy_value_net.py`

Abstract network with the AlphaZero two-head interface:

- `predict(state) -> (policy, value)` — `policy` is a probability array over the
  flat action space; `value` is a scalar estimate of the state's return.
- `train(examples)` — fit to `(state, policy_target, value_target)` tuples.
- `save_checkpoint` / `load_checkpoint`
- `push_multiprocessing` / `pop_multiprocessing` — move tensors on/off GPU around
  a multiprocessing fan-out.

`TorchPolicyValueNet` implements the checkpoint and multiprocessing methods for
any `nn.Module`, leaving `train`/`predict` to the subclass. `PolicyValueNetModel`
is a plain shared-body MLP with separate policy and value heads — the default
backbone the SR net uses.

## How the pieces connect

```text
  Trainer.train_iteration()
    │
    ├─ _collect_training_examples()          # self-play, possibly multiprocessed
    │     for each of N games:
    │        Agent.play_for_experience(game, ...)
    │          └─ play_one_round(game)
    │               loop over moves:
    │                 move_probs = Agent.policy(state)   ─────────┐
    │                 action = sample(move_probs)                 │ builds a fresh
    │                 state.step_wrapper(action)                  │ MCTS per move
    │               └─ emits [(obs, move_probs, discounted_R)]    │
    │                                                     MCTS.perform_simulations()
    │                                                       └─ n_simulations × search()
    │                                                            └─ net.predict(obs)
    ├─ _process_training_examples()          # append to replay window, flatten
    └─ _train_network()                      # net.train(flat_examples)
```

### `Agent` — `src/sraz/core/agent.py`

The agent's single responsibility is **playing games and collecting
experience**; it deliberately holds no training/checkpointing logic. Key
methods:

- `policy(state, add_noise, temperature_override)` — clones the state, spins up a
  fresh `MCTS`, runs the simulations, and returns visit-count-derived move
  probabilities (a flat array over the action space). If an `external_policy` is
  configured it is used instead of MCTS (for baselines/pitting).
- `play_one_round(game, ...)` — plays a full episode by repeatedly calling
  `policy` and sampling an action, recording `(obs, move_probs)` at each step.
  At episode end it computes **discounted returns** backward,
  $G_t = r_t + \gamma G_{t+1}$, and zips them into
  `(obs, move_probs, discounted_return)` training tuples.
- `play_one_round_reuse_tree(game, ...)` — same, but creates **one** MCTS at the
  start and reuses its tree across all moves via
  `perform_simulations_reuse()` + `advance_to()` (warm-started search).
- `play_for_experience[_reuse_tree](...)` — the multiprocessing entry points:
  re-seed, reset the game, play one round, and return the experience plus any
  leaf-evaluator caches. These are what the `Trainer` fans out across processes.

The agent owns four named RNG streams (`mcts`, `train`, `eval`,
`external_policy`) so runs are reproducible from seeds.

### `Trainer` — `src/sraz/training/trainer.py`

Orchestrates the AlphaZero outer loop. One `train_iteration()`:

1. **Collect** `n_games_per_train` self-play games (parallel via
   `MultiprocessingManager`, or sequential when `n_procs < 0`, which also logs
   per-game progress).
2. **Process**: append this iteration's examples to `all_training_examples`,
   drop iterations older than `n_past_iterations_to_train` (a sliding **replay
   window**), and flatten to one training set.
3. **Train**: call `net.train(flat_examples)` and record statistics — including
   the **MCTS target entropy** and the **policy KL gap** (policy loss minus that
   entropy floor), a diagnostic of how far the net is from imitating the search.

`train_multiple()` loops this with optional periodic checkpointing. Note the
trainer has no accept/reject arena — this is a single-player setup, so the new
network is always kept (`RunConfig.accept_threshold` exists in the config
dataclass but the v1 SR driver does not gate on it).

## Configuration — `src/sraz/core/config.py`

A set of dataclasses (`AgentConfig`, `TrainerConfig`, `EvaluatorConfig`,
`RunConfig`, `GameConfig`, `NetConfig`) bundled by an abstract `MetaConfig`. A
concrete config subclass implements `build()`, returning the assembled
`(game, net, agent, trainer)`. `MetaConfig` also provides:

- `save(path)` — JSON-serialize the whole config (callables become
  `"<callable: module.name>"` strings) for experiment provenance.
- `plot(path)` — render a 6-panel matplotlib summary of the config.

The SR instance's `SymRegConfig` (see [03](03-symreg-instance.md)) fills these in
with concrete classes and hyperparameters.

## Utilities — `src/sraz/utils/`

| Module | Purpose |
| --- | --- |
| `common.py` | `disable_numpy_multithreading()` (pin BLAS/OMP threads to 1 for clean multiprocessing), `use_deterministic_cuda()`, `get_device()`. |
| `multiprocessing.py` | `MultiprocessingManager`: `push()`/`pop()` GPU state around a fan-out, and `starmap()` which runs a function across a process `Pool` (spawn → forkserver → fork fallback), or sequentially when `n_procs < 0`. |
| `checkpoint.py` | `CheckpointManager`: pickle agent state (`game`, `all_training_examples`, `rngs`) + save/load the network; validate and list checkpoints. |
| `statistics.py` | `StatisticsManager`: accumulate per-iteration stat dicts and append them to a JSONL file. |

> **Note on checkpointing:** `CheckpointManager` expects
> `agent.all_training_examples`, but the replay buffer actually lives on the
> `Trainer`. The SR driver sidesteps this with its own `_save_final_checkpoint`
> that pickles `trainer.all_training_examples` directly — see
> [05-file-map-and-howto](05-file-map-and-howto.md).
