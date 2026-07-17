# Claude-docs — index

Durable reference documentation for the `sraz` codebase, generated from a
read-through at commit `2946193`. Read in order; each doc is a self-contained
iteration snapshot.

| # | Doc | What it covers |
| --- | --- | --- |
| 01 | [overview](01-overview.md) | What the project is; the engine/instance split; the SR game in brief |
| 02 | [architecture](02-architecture.md) | The game-agnostic engine: `Game`, `PolicyValueNet`, `Agent`, `Trainer`, `MCTS` and how they connect |
| 03 | [symreg-instance](03-symreg-instance.md) | The symbolic-regression grammar game in code: grammar compilation, the derivation MDP, the fit-based reward |
| 04 | [mcts-internals](04-mcts-internals.md) | The MCTS algorithm as implemented: selection, expansion, backup, masking, noise, tree reuse, rollouts |
| 05 | [file-map-and-howto](05-file-map-and-howto.md) | File-by-file map, how to install/test/run, how to add a new game |

## Relationship to the repo's own docs

- `README.md` — install/run quickstart and layout.
- `Claude-research/notes/01-game-spec.md` — the **authoritative, math-heavy specification** of the SR
  game plus the first documented training run (seed 42). It is the source of
  truth for the *game*; these Claude-docs concentrate on the *engine internals*
  and a code-level map, and defer to `01.md` for the formal MDP definition.

## Caveat noted during the read-through

`src/sraz/core/mcts.py:95` references `docs/notes/stage4/03.md` ("Core MCTS
audit findings"), but only `Claude-research/notes/01-game-spec.md` exists in this commit. The
referenced note appears to be from the upstream project (`AlphaZero_PP`) and was
not ported. The `self.debug_state` instrumentation it documents *is* present in
the code (see [04-mcts-internals](04-mcts-internals.md)).
