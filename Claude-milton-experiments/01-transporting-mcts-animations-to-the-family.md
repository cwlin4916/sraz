# Transporting the MCTS animation scripts to the family

What `Claude-scripts/04` and `05` (the two scripts that produce the GIFs in
`Claude-experiments/8-5`, `8-6`) actually do, why `--problem lin_A` doesn't
work on them today, and the fix.

## 1. What makes the GIFs

- **`04-mcts-tree-animation.py`** drives one first-move pure-MCTS search,
  one simulation per frame: a depth-2 derivation tree (edge width = visit
  count, colour = backed-up Q) plus a bar panel of the root's options.
  Output: `mcts-tree_<key>.gif`.
- **`05-mcts-ucb-animation.py`** — same search, same fidelity, but decomposes
  each frame's UCB score into `Qtilde` (exploit) + `u` (explore) per action,
  plus a UCB-vs-simulation history panel. Output: `mcts-ucb_<key>.gif`.
- Both reuse `01-state-value-gap.py`'s `build_game()` for the root game object
  and `02`'s tree/layout helpers; the fit-value memo they read from is built
  by `01`/`02`'s exhaustive enumeration + cache. `03` (convergence sweep) and
  `06` (c-exploration sweep) import `SymRegConfig` the same way `04`/`05` do,
  so the fix below likely applies to them too — not re-verified line by line here.

Every one of these defaults to `--problem sine`, and every existing GIF in
`Claude-experiments/` is a sine run.

## 2. Why `--problem lin_A` breaks

Each script builds **two** things that have to agree, and only one of them
knows about the family:

- **The root game** — `build_game(problem, seed)` (`01-state-value-gap.py:75`):
  ```python
  if problem == "sine":
      return SymRegGame(problem_seed=problem_seed)
  return SymRegGame(target=problem)   # named targets fix their own coeffs
  ```
  This *does* resolve `"lin_A"` (via `targets.get_target`), so it wouldn't
  crash — but with no `grammar_rules`/`max_len` passed, `SymRegGame` defaults
  to the sine grammar at `max_len=15` (`game.py:228`, `:216`), not
  `ADDITIVE_GRAMMAR` at `L=12`.
- **The net + MCTS params** — via `SymRegConfig(problem=args.problem, ...)`
  (`04:341`, `05:295`). `SymRegConfig.build()` calls `get_problem(self.problem)`
  (`config.py:94`), and `problems.PROBLEMS` only has two keys: `"sine"` and
  `"additive_quadratic"`. `get_problem("lin_A")` raises `KeyError` — this path
  hard-fails, it doesn't just use the wrong grammar.

Both have to be fixed together: the net's action count is computed from
*this* config's internal game (`config.py:104`), and that net is then handed
to the `MCTS` object driving the *other* (root) game — a size mismatch between
the two would silently corrupt the search rather than error.

## 3. The fix

Two small, local edits — no new registries, no new problem-resolution logic.

**`build_game()`** (`01-state-value-gap.py:75`): the `else` branch already
targets the family; it's just missing the grammar. (`ADDITIVE_GRAMMAR` isn't
imported yet — add it to the existing `from sraz.instances.symreg.game import
SymRegGame, fit_expression` on line 53.)
```python
return SymRegGame(target=problem, grammar_rules=ADDITIVE_GRAMMAR, max_len=12)
```

**Wherever a script builds `SymRegConfig(problem=args.problem, ...)`**: don't
forward the target name as the config's `problem`. `SymRegConfig.build()`
already has a branch for exactly this case (`config.py:96–100`: "a named
target supplies its own ys... the problem still supplies the grammar") — it's
just never fed one.
```python
if args.problem == "sine":
    cfg = SymRegConfig(problem="sine", pure_mcts=True)
else:
    cfg = SymRegConfig(problem="additive_quadratic", pure_mcts=True)
    cfg.game.kwargs["target"] = args.problem   # e.g. "lin_A"
    cfg.game.kwargs["max_len"] = 12
```

**Optional, for speed not correctness**: `install_fit_memo` reads
`Claude-experiments/8-5/cache/fit_cache_<key>.json` but never writes it —
that cache is built by running `01` first. Without it, `04`/`05` still work
(the memo wrapper falls back to a live `lmfit` solve on a cache miss and just
never persists it), just slower per novel terminal. Run `01 --problem lin_A`
once before animating if that matters.

## 4. Which members to animate first

Not all 8 — GIFs are for watching a search live, not a sweep. Reuse the
writeup's own selection (`sec:family`, "the four cells" + `quad_B`): `lin_A`,
`lin_B`, `quad_A`, `quad_D`, `quad_B`. That's the inversion/no-inversion ×
common/rare 2×2 plus the plateau control — five runs, ten GIFs.

## 5. Commands, once §3 lands

```
.venv/bin/python Claude-scripts/01-state-value-gap.py --problem lin_A
.venv/bin/python Claude-scripts/04-mcts-tree-animation.py --problem lin_A --sims 64
.venv/bin/python Claude-scripts/05-mcts-ucb-animation.py --problem lin_A --sims 40 --fps 1.5
```
repeated for `lin_B`, `quad_A`, `quad_D`, `quad_B`.
