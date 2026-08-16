#!/usr/bin/env python
"""Derivation tree annotated with BOTH V* and V_rand per node.

Same idea as scripts/plotting/plot_grammar_tree.py (a leftmost-derivation tree
whose nodes are partial expressions), but each node is labelled with two
quantities instead of one:

    V*      the OPTIMAL value reachable from the node  -- exact, = max clipped-R^2
            over every legal completion of the node's partial expression.
    V~      the RANDOM-ROLLOUT value of the node       -- mean clipped-R^2 over
            K uniform-random legal-action completions (what a mean-backup /
            random-rollout search perceives).

The gap V* - V~ is the node-local informativeness signal: a node that is deep
green in V* but red in V~ is a hidden-value trap -- reachably good, but random
play almost never finds it, so the search cannot see the promise.

Every node is reachable from the root, so its completions are a subset of the
root's; the seed-matched fit cache built by 01-state-value-gap.py already holds
every fit, making all per-node V* lookups instant.

Run from repo root:
    .venv/bin/python Claude-scripts/02-grammar-tree-vstar-vrand.py \
        --problem sine --problem-seed 42 --depth 2 --rollouts 1000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import zlib
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "Claude-experiments" / "8-5"

# reuse the single-state machinery (module name starts with a digit -> load by path)
_spec = importlib.util.spec_from_file_location(
    "state_value_gap", Path(__file__).with_name("01-state-value-gap.py"))
svg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svg)


# --------------------------------------------------------------------------
# Parallel per-node rollouts (worker side)
# --------------------------------------------------------------------------
# Deeper trees have hundreds of non-terminal nodes, each needing K rollouts, so
# the V~ pass is parallelised across nodes. Each worker builds the game once
# (initializer) and reuses ONE game per node via stash/unstash, so that node's
# internal fit cache is shared across its K rollouts -- a big speed-up, since
# random rollouts hit the same terminal sentences repeatedly. Sampling and RNG
# use are byte-identical to svg.rollout_values, so the numbers are unchanged.
_ROLL_GAME = None


def _roll_init(problem, problem_seed):
    global _ROLL_GAME
    _ROLL_GAME = svg.build_game(problem, problem_seed)


def _roll_one(task):
    tokens, k, seed = task
    g = svg._game_at(tuple(tokens), _ROLL_GAME)
    stash = g.stash_state()
    rng = np.random.default_rng(seed)
    rewards = np.empty(k, dtype=float)
    for n in range(k):
        g.unstash_state(stash)
        while not (g.terminated or g.truncated):
            valid = np.flatnonzero(g.get_action_mask())
            if len(valid) == 0:
                break
            g.step_wrapper(int(valid[rng.integers(len(valid))]))
        rewards[n] = g.reward
    return float(rewards.mean()), float(rewards.std())


# --------------------------------------------------------------------------
# Tree construction (expand the leftmost nonterminal, to a fixed depth)
# --------------------------------------------------------------------------
def build_tree(game, max_depth: int) -> list[dict]:
    g = game.grammar
    nodes: list[dict] = []

    def add(tokens, depth, parent):
        nid = len(nodes)
        term = svg.leftmost_nt(tokens, g.nonterms) == -1
        nodes.append({"id": nid, "tokens": tuple(tokens), "depth": depth,
                      "parent": parent, "children": [], "terminal": term})
        if parent is not None:
            nodes[parent]["children"].append(nid)
        if not term and depth < max_depth:
            i = svg.leftmost_nt(tokens, g.nonterms)
            for j in g.proddict[tokens[i]]:
                rhs = tuple(g.productions[j])
                if len(tokens) + len(rhs) - 1 < game.state_len:
                    add(tokens[:i] + rhs + tokens[i + 1:], depth + 1, nid)
        return nid

    add((g.symdict[g.start],), 0, None)
    return nodes


def layout(nodes: list[dict]) -> dict:
    """Tidy midpoint layout: leaves get successive x, parents sit above the mean."""
    pos = {}
    counter = [0]

    def assign(nid):
        ch = nodes[nid]["children"]
        if not ch:
            x = float(counter[0]); counter[0] += 1
        else:
            x = float(np.mean([assign(c) for c in ch]))
        pos[nid] = (x, -float(nodes[nid]["depth"]))
        return x

    assign(0)
    return pos


# --------------------------------------------------------------------------
# Per-node values
# --------------------------------------------------------------------------
def compute_values(game, nodes, cache_key, workers, k, seed, problem,
                   problem_seed):
    g = game.grammar
    # 1) enumerate each node's completions once; gather the global set to fit
    per_node_comps: dict[int, list[str]] = {}
    union: set[str] = set()
    for n in nodes:
        comps = {svg.decode(c, g.tokenlist)
                 for c in svg.enumerate_completions(n["tokens"], game)}
        per_node_comps[n["id"]] = list(comps)
        union |= comps
    print(f"[tree] {len(nodes)} nodes, {len(union)} distinct completions to price")
    fits = svg.fit_all(sorted(union), game, cache_key, workers)  # all cache hits

    # per-node V~ is deterministic given (expr, k, base seed), so cache it to
    # disk -- re-rendering the plot then costs no rollouts at all. The rollout
    # seed is derived from a STABLE hash of the expression (not the node id), so
    # the same partial expression gets the same V~ at any depth/position, and
    # the cache is reused across depth settings.
    rcache_path = svg.CACHE_DIR / f"rollout_cache_{cache_key}_k{k}_s{seed}.json"
    rcache = (json.loads(rcache_path.read_text())
              if rcache_path.exists() else {})

    def node_expr(n):
        return svg.decode(n["tokens"], g.tokenlist)

    # 2) figure out which non-terminal nodes still need rollouts, then run them
    #    in parallel across nodes (one game per worker, reused via stash/unstash)
    todo = [(n["tokens"], k, seed + zlib.crc32(node_expr(n).encode()))
            for n in nodes
            if not n["terminal"] and node_expr(n) not in rcache]
    if todo:
        t0 = __import__("time").time()
        exprs = [svg.decode(tok, g.tokenlist) for tok, _, _ in todo]
        if workers == 1:
            _roll_init(problem, problem_seed)
            results = [_roll_one(t) for t in todo]
        else:
            import multiprocessing as mp
            with mp.Pool(workers, initializer=_roll_init,
                         initargs=(problem, problem_seed)) as pool:
                results = list(pool.imap(_roll_one, todo, chunksize=1))
        for expr, (mean, std) in zip(exprs, results):
            rcache[expr] = {"mean": mean, "std": std}
        dt = __import__("time").time() - t0
        print(f"[tree] rolled {len(todo)} nodes x {k} in {dt:.1f}s "
              f"({workers} workers)")
        svg.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        rcache_path.write_text(json.dumps(rcache))
    else:
        print(f"[tree] all non-terminal V~ served from cache")

    # 3) assemble: V* = max over completions; V~ from the (now-populated) cache
    values = {}
    for n in nodes:
        nid = n["id"]
        comp_vals = np.array([fits[s] for s in per_node_comps[nid]])
        v_star = float(comp_vals.max())
        if n["terminal"]:
            v_rand, v_std = v_star, 0.0    # a terminal's only completion is itself
        else:
            rc = rcache[node_expr(n)]
            v_rand, v_std = rc["mean"], rc["std"]
        values[nid] = {"v_star": v_star, "v_rand": v_rand, "v_rand_std": v_std,
                       "gap": v_star - v_rand, "n_completions": len(comp_vals)}
    return values


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw(game, nodes, pos, values, out_path, problem, seed, k):
    g = game.grammar
    cmap = matplotlib.colormaps["RdYlGn"]
    norm = Normalize(vmin=-1.0, vmax=1.0)          # the reward's clip range

    # --- geometry ---------------------------------------------------------
    # Each level is stretched horizontally so a leaf gets ~COL_IN inches, and
    # within a level nodes are cycled through TIERS vertical offsets so that
    # same-tier neighbours are TIERS slots apart -> wide labels never overlap
    # even when packed tightly. Denser trees use more tiers and tighter columns.
    DPI = 150
    AGG_MAX = 62000                                 # Agg errors above 2**16 px
    leaves = [n for n in nodes if not n["children"]]
    n_leaves = len(leaves)
    max_depth = max(n["depth"] for n in nodes)
    if n_leaves > 400:
        COL_IN, ROW_IN, TIERS, FONT = 0.60, 3.4, 4, 5.0
    elif n_leaves > 60:
        COL_IN, ROW_IN, TIERS, FONT = 1.15, 3.3, 3, 6.0
    else:
        COL_IN, ROW_IN, TIERS, FONT = 1.60, 3.0, 2, 8.0
    STAG = 0.42
    # keep the figure under the Agg pixel limit no matter how wide the tree
    COL_IN = min(COL_IN, (AGG_MAX / DPI) / max(n_leaves, 1))

    by_depth: dict[int, list[int]] = {}
    for n in nodes:
        by_depth.setdefault(n["depth"], []).append(n["id"])
    stag = {}
    for ids in by_depth.values():
        for rank, nid in enumerate(sorted(ids, key=lambda i: pos[i][0])):
            stag[nid] = (rank % TIERS) * STAG      # 0..(TIERS-1)*STAG, downward

    row_pitch = 1.0 + STAG * TIERS

    def xy(nid):
        return pos[nid][0], -float(nodes[nid]["depth"]) * row_pitch - stag[nid]

    fig, ax = plt.subplots(
        figsize=(max(16, COL_IN * n_leaves), ROW_IN * (1 + max_depth)))

    for node in nodes:                              # edges first
        x0, y0 = xy(node["id"])
        for c in node["children"]:
            x1, y1 = xy(c)
            ax.plot([x0, x1], [y0, y1], "-", color="0.55", lw=0.6, zorder=1)

    for node in nodes:
        x, y = xy(node["id"])
        v = values[node["id"]]
        expr = svg.decode(node["tokens"], g.tokenlist)
        if node["terminal"]:
            label = f"{expr}\nR² = {v['v_star']:.3f}"
        else:
            label = (f"{expr}\n"
                     f"V*={v['v_star']:.3f}  V~={v['v_rand']:.3f}\n"
                     f"gap {v['gap']:.2f}")
        # fill encodes V~ (what random rollout perceives); read V* off the label
        ax.annotate(label, (x, y), ha="center", va="center", fontsize=FONT,
                    family="monospace", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.3", ec="black", lw=0.5,
                              alpha=0.95, fc=cmap(norm(v["v_rand"]))))

    xs = [pos[n["id"]][0] for n in nodes]
    ys = [xy(n["id"])[1] for n in nodes]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="node fill = V~ (mean random-rollout R²)",
                 shrink=0.5, pad=0.01)
    ax.set_title(
        f"Derivation tree: {problem} (seed {seed}) — node = partial expression; "
        f"V* = optimal reachable R², V~ = mean random-rollout R² (K={k})",
        fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="sine")
    ap.add_argument("--problem-seed", type=int, default=42)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--rollouts", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="base rollout RNG seed")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    game = svg.build_game(args.problem, args.problem_seed)
    cache_key = (f"{args.problem}_seed{args.problem_seed}"
                 if args.problem == "sine" else args.problem)
    print(f"=== tree: {args.problem} (problem_seed={args.problem_seed}), "
          f"depth {args.depth}, target {game.target_infix} ===")
    print(f"    constants: {game.constants}")

    nodes = build_tree(game, args.depth)
    values = compute_values(game, nodes, cache_key, args.workers,
                            args.rollouts, args.seed, args.problem,
                            args.problem_seed)
    pos = layout(nodes)

    tag = (f"{args.problem}_seed{args.problem_seed}"
           if args.problem == "sine" else args.problem)
    png = OUT_DIR / f"tree_{tag}_depth{args.depth}.png"
    draw(game, nodes, pos, values, png, args.problem, args.problem_seed,
         args.rollouts)

    # dump the tree + values as JSON and print a table
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump = [{"id": n["id"], "depth": n["depth"], "parent": n["parent"],
             "terminal": n["terminal"],
             "expr": svg.decode(n["tokens"], game.grammar.tokenlist),
             **values[n["id"]]} for n in nodes]
    (OUT_DIR / f"tree_{tag}_depth{args.depth}.json").write_text(
        json.dumps({"problem": args.problem, "problem_seed": args.problem_seed,
                    "depth": args.depth, "rollouts": args.rollouts,
                    "nodes": dump}, indent=2))

    print(f"\n{'depth':>5} {'kind':>5} {'V*':>7} {'V~':>7} {'gap':>7}  expr")
    for n in nodes:
        v = values[n["id"]]
        kind = "term" if n["terminal"] else "part"
        print(f"{n['depth']:>5} {kind:>5} {v['v_star']:>7.3f} "
              f"{v['v_rand']:>7.3f} {v['gap']:>7.3f}  "
              f"{svg.decode(n['tokens'], game.grammar.tokenlist)}")
    print(f"\n[saved] {png.relative_to(REPO)}")
    print(f"[saved] {(OUT_DIR / f'tree_{tag}_depth{args.depth}.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
