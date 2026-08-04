# idea — a DAG for research idea planning

A minimal port of the `blueprint/` system of *RamifiedRelativeLanglands* from lemmas to research ideas. Each node is one idea or concept to attempt; it is settled either **mathematically** (an argument) or **empirically** (an experiment). Everything belonging to this system lives inside this directory.

## Layout

| Path | Purpose |
|---|---|
| `README.md` | This plan and the conventions in force |
| `main_dag.md` | The index — one line per node, plus the tally |
| `nodes/<id>/` | One directory per node: `node.md` (fields and the full statement), `report.md` (the one-page write-up once attempted). A child idea's directory sits **inside its parent's** |
| `nodes/_template.md` | The node template — copy it to start a node |

## The node

`node.md` carries the fields and the statement; `report.md` is written only once the idea is attempted. Fields:

- **created:** ISO date, set once, never changed.
- **status:** `stated` → `attempting` → `verified` | `refuted` | `parked`.
- **verify:** `math` | `empirical` — what settles this node: an argument, or an experiment with a stated success criterion.
- **depends-on:** the ideas this one presupposes, as links. `ready` and `blocked` are *derived* from dependency statuses, never stored.
- **required-by:** the reverse links.

## Conventions (inherited from the parent project)

- **A node's id is its path.** Split the id at each `-`, take the successive prefixes, join with `/`: `I1-2-1` lives at `nodes/I1/I1-2/I1-2-1/`, inside the idea it refines. Every directory is named with the full id of the node it holds. Decomposing a node only ever *adds* directories beneath it.
- **One node, one directory, one page.** `report.md` is at most 900 words. A report that will not fit means the node is really several nodes — split it.
- **Every fact is stored once.** A node's status lives in its `node.md` and nowhere else; `main_dag.md` index lines are a projection of the nodes, regenerated (for now: re-derived by hand) whenever a node changes, never independently edited.
- **A report is self-contained.** It restates the node's statement verbatim, then settles it. Math nodes end in an argument; empirical nodes end in what was run, where the code and artifacts live, and the observed outcome against the stated criterion. External sources are cited by link, not from memory.
- **Verification is honest.** `verified` means the argument holds or the experiment met its criterion. A negative result is `refuted`, not deleted — the report records why. An idea set aside without a verdict is `parked`.

## Minimal plan

1. **Scaffold** *(this commit)* — `README.md`, `main_dag.md`, `nodes/_template.md`.
2. **First node by hand** — copy the template to `nodes/I1/node.md`, state one real idea with its verification target, and update the index. This tests whether the conventions fit ideas as well as they fit lemmas.
3. **Attempt it** — write `nodes/I1/report.md`, move the status, and only then judge what machinery is worth porting.
4. **Port machinery on demand, not up front** — in order of expected pain: an index generator (the parent's `tools/build_dag_site.py --check` analogue), a render/link gate, then `/decompose`- and `/prove`-style commands adapted to ideas (*decompose* → sub-ideas, *prove* → verify). Nothing is ported until the manual version has actually hurt.
