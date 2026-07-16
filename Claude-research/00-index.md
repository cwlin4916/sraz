# Claude-research — index

Progress tracker for the research on **AlphaZero applied to the
symbolic-regression grammar game** (`sraz`). Flat numbered snapshots plus one
living status board.

| # | Doc | What it is |
| --- | --- | --- |
| 01 | [brainstorm-directions](01-brainstorm-directions.md) | 2026-07-15 brainstorm: why the baseline gets stuck, and the tiered option space for what to try next |
| 02 | [informativeness-and-deception](02-informativeness-and-deception.md) | The core conceptual tool: the informativeness × deception 2×2 for when AlphaZero should work, and how it isolates the value-net vs policy/search roles |
| 03 | [additive-instance-results](03-additive-instance-results.md) | First best-corner experiment: building the `additive_quadratic` instance + AZ results (climbs at 100+ sims, shallow-trap refinement) |
| XX | [status](XX-status.md) | Living status board — current state, open decisions, candidate next actions, update log |

## Related artifacts (elsewhere in the repo)

- `docs/notes/01.md` — authoritative game spec + the first documented run (§6 baseline).
- `Claude-docs/` — codebase architecture, MCTS internals, file map.
- `Claude-reviews/2026-07-15-2008/` — correctness review (3 fixes landed, incl. the MCTS RNG seeding that made runs reproducible).

## Conventions

Numbered `NN-topic.md` files are immutable snapshots — new thinking goes into a
new numbered file. `XX-status.md` is updated in place; its Update log records
every material change (`YYYY-MM-DD — <change>`).
