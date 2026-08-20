#!/usr/bin/env python3
"""Depth-2 derivation trees for the target family, annotated with V*, V^q and rho.

Visual analogue of `02-grammar-tree-vstar-vrand.py` (which produced
`Claude-experiments/8-5/tree_sine_seed42_depth2.png`) transported to the
controlled family: same idea -- a node is a partial expression, the fill colour
is what a mean-backup search *perceives*, the label carries what is actually
*reachable* -- with three deliberate changes, all of which make the picture
sharper here than it could be on sine.

1.  The tree is the MDP's, not the grammar's. `02` expands only the leftmost
    nonterminal, so `+ S S` shows 4 children. The real action set is every
    (position, production) pair the mask allows, so `+ S S` has 8 -- and that
    branching factor is exactly what a searcher's budget is spent against.
    `--expansion leftmost` reproduces `02`'s convention for comparison.
2.  V^q is exact, not a rollout mean. `02` had to estimate V~ with K random
    completions because the sine space is far too large to enumerate. This
    family has 4,898 reachable states, so the uniform-completion value is
    computed by backward induction -- no sampling error, no K to tune.
3.  Each node also carries rho, the exact probability that uniform completion
    from that node lands on an *exact* expression. V^q says "how good does
    random play look here"; rho says "how often does random play actually win
    here". They can point in different directions, which is the point.

The root's two candidate choices are drawn on top of the tree: the action a
mean-backup search would take (argmax V^q over root actions) and the action
that is actually optimal (argmax V*). When they differ, the target satisfies the
writeup's inversion condition and the tree is labelled a trap.

Requires the terminal-score cache built by 07-family-exact-audit.py.

Outputs, imitating `02`'s naming. Per-target artifacts go in a per-target
subfolder so that everything belonging to one family member -- trees now, and
whatever each member accumulates later -- sits together; only genuinely
family-wide artifacts stay at the top level:

    <out-dir>/<target>/tree_<target>_depth<D>.png    one tree per target
    <out-dir>/<target>/tree_<target>_depth<D>.json   nodes + values
    <out-dir>/tree_family_depth<D>.png               all targets, one grid
    <out-dir>/cache/                                 shared fit cache (from 07)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import Normalize                            # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sraz.instances.symreg.game import fit_expression              # noqa: E402
from sraz.instances.symreg.targets import family_targets, get_target  # noqa: E402

_audit = __import__("07-family-exact-audit")
PRODS = _audit.PRODS
PROD_NAMES = _audit.PROD_NAMES
START = _audit.START
legal_actions = _audit.legal_actions
apply_action = _audit.apply_action
is_terminal = _audit.is_terminal
enumerate_reachable = _audit.enumerate_reachable
backward_induction = _audit.backward_induction


# --------------------------------------------------------------------------
# Tree construction
# --------------------------------------------------------------------------
def build_tree(max_depth: int, L: int, expansion: str) -> list[dict]:
    """Expand to a fixed depth. `all` uses every legal action (the MDP tree);
    `leftmost` expands only the first nonterminal (02's grammar-tree convention).
    """
    nodes: list[dict] = []

    def add(form, depth, parent, via):
        nid = len(nodes)
        nodes.append({"id": nid, "form": form, "depth": depth, "parent": parent,
                      "children": [], "terminal": is_terminal(form), "via": via})
        if parent is not None:
            nodes[parent]["children"].append(nid)
        if not is_terminal(form) and depth < max_depth:
            acts = legal_actions(form, L)
            if expansion == "leftmost":
                first = min(i for i, t in enumerate(form) if t == "S")
                acts = [(p, j) for p, j in acts if p == first]
            for pos, prod in acts:
                add(apply_action(form, pos, prod), depth + 1, nid, (pos, prod))
        return nid

    add(START, 0, None, None)
    return nodes


def layout(nodes: list[dict]) -> dict:
    """Tidy midpoint layout: leaves get successive x, parents sit above the mean."""
    pos, counter = {}, [0]

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


def root_choice(nodes: list[dict], vals: dict) -> dict:
    """Which root action mean-backup prefers, which is optimal, and the gap."""
    kids = nodes[0]["children"]
    def nm(nid):
        f = nodes[nid]["form"]
        return PROD_NAMES[nodes[nid]["via"][1]] if nodes[nid]["via"] else "?"
    vq = {nid: vals["Vq"][nodes[nid]["form"]] for nid in kids}
    vs = {nid: vals["Vstar"][nodes[nid]["form"]] for nid in kids}
    greedy = max(kids, key=lambda n: vq[n])
    vs_max = max(vs.values())
    # V* frequently ties at the root (whenever a one-move terminal is already
    # exact, e.g. lin_D). Reporting some other tied action as "the optimal one"
    # would read as a disagreement where none exists, so when the greedy action
    # is itself optimal it IS the optimal action named.
    tie = vs[greedy] >= vs_max - 1e-9
    best = greedy if tie else max(kids, key=lambda n: vs[n])
    n_optimal = sum(1 for n in kids if vs[n] >= vs_max - 1e-9)
    return {"greedy_id": greedy, "greedy_name": nm(greedy),
            "greedy_Vq": vq[greedy], "greedy_Vstar": vs[greedy],
            "optimal_id": best, "optimal_name": nm(best),
            "optimal_Vq": vq[best], "optimal_Vstar": vs[best],
            "is_trap": vs[greedy] < vs_max - 1e-9,
            "greedy_is_optimal": bool(tie),
            "n_optimal_actions": n_optimal,
            "margin": vq[greedy] - vq[best]}


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
CMAP = matplotlib.colormaps["RdYlGn"]
NORM = Normalize(vmin=-1.0, vmax=1.0)      # the reward's clip range


def node_label(node: dict, vals: dict, scores: dict) -> str:
    form = node["form"]
    expr = " ".join(form)
    if node["terminal"]:
        return f"{expr}\nR² = {scores[form]:+.3f}"
    return (f"{expr}\n"
            f"V*={vals['Vstar'][form]:.3f}  Vq={vals['Vq'][form]:+.3f}\n"
            f"gap {vals['Vstar'][form] - vals['Vq'][form]:.3f}   "
            f"ρ={vals['rho'][form]:.3f}")


def draw_tree(ax, nodes, pos, vals, scores, choice, font=8.0,
              tiers=2, stag_amt=0.46):
    """Render one tree onto `ax`. Mirrors 02's staggered-tier geometry."""
    by_depth: dict[int, list[int]] = {}
    for n in nodes:
        by_depth.setdefault(n["depth"], []).append(n["id"])
    stag = {}
    for ids in by_depth.values():
        for rank, nid in enumerate(sorted(ids, key=lambda i: pos[i][0])):
            stag[nid] = (rank % tiers) * stag_amt
    row_pitch = 1.0 + stag_amt * tiers

    def xy(nid):
        return pos[nid][0], -float(nodes[nid]["depth"]) * row_pitch - stag[nid]

    for node in nodes:                                     # edges first
        x0, y0 = xy(node["id"])
        for c in node["children"]:
            x1, y1 = xy(c)
            ax.plot([x0, x1], [y0, y1], "-", color="0.55", lw=0.6, zorder=1)

    # the root's two candidate actions, drawn over the plain edges
    x0, y0 = xy(0)
    if choice["greedy_is_optimal"]:
        x1, y1 = xy(choice["greedy_id"])
        ax.plot([x0, x1], [y0, y1], "-", color="#1b7f3b", lw=2.6, zorder=2,
                alpha=0.9, solid_capstyle="round")
    else:
        for nid, colour, style in (
                (choice["greedy_id"], "#c1121f", (0, (4, 2))),
                (choice["optimal_id"], "#1b7f3b", "-")):
            x1, y1 = xy(nid)
            ax.plot([x0, x1], [y0, y1], ls=style, color=colour, lw=2.6,
                    zorder=2, alpha=0.9, solid_capstyle="round")

    for node in nodes:
        x, y = xy(node["id"])
        fill = CMAP(NORM(vals["Vq"][node["form"]]))
        ax.annotate(node_label(node, vals, scores), (x, y), ha="center",
                    va="center", fontsize=font, family="monospace", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.3", ec="black", lw=0.5,
                              alpha=0.95, fc=fill))

    xs = [pos[n["id"]][0] for n in nodes]
    ys = [xy(n["id"])[1] for n in nodes]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
    ax.axis("off")


def legend_handles(choice) -> list:
    tied = (f", tied with {choice['n_optimal_actions'] - 1} other"
            f"{'s' if choice['n_optimal_actions'] > 2 else ''}"
            if choice["n_optimal_actions"] > 1 else "")
    if choice["greedy_is_optimal"]:
        return [Line2D([], [], color="#1b7f3b", lw=2.6,
                       label=f"mean-backup picks {choice['greedy_name']}, which is "
                             f"optimal{tied} "
                             f"(Vq={choice['greedy_Vq']:+.3f}, "
                             f"V*={choice['greedy_Vstar']:.3f})")]
    return [
        Line2D([], [], color="#c1121f", lw=2.6, ls=(0, (4, 2)),
               label=f"mean-backup picks {choice['greedy_name']}  "
                     f"(Vq={choice['greedy_Vq']:+.3f}, "
                     f"V*={choice['greedy_Vstar']:.3f})  <- cannot reach 1"),
        Line2D([], [], color="#1b7f3b", lw=2.6,
               label=f"optimal is {choice['optimal_name']}{tied}  "
                     f"(Vq={choice['optimal_Vq']:+.3f}, "
                     f"V*={choice['optimal_Vstar']:.3f})"),
    ]


def title_for(name: str, choice: dict, depth: int, expansion: str,
              n_nodes: int) -> str:
    t = get_target(name)
    verdict = ("TRAP: the greedy choice cannot reach the optimum"
               if choice["is_trap"] else "no inversion: greedy is also optimal")
    return (f"Depth-{depth} {'MDP' if expansion == 'all' else 'leftmost-derivation'} "
            f"tree: {name}   y = {t.infix}\n"
            f"node = partial expression;  V* = optimal reachable R²;  "
            f"Vq = exact uniform-completion value;  ρ = P(uniform completion is exact)\n"
            f"{n_nodes} nodes — {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--family", default="linear")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--tau", type=float, default=1e-6)
    ap.add_argument("--expansion", choices=("all", "leftmost"), default="all",
                    help="'all' = every legal action (the MDP tree); "
                         "'leftmost' = 02's grammar-tree convention")
    ap.add_argument("--out-dir", default=str(REPO / "Claude-experiments" / "8-17"))
    ap.add_argument("--no-grid", action="store_true",
                    help="skip the combined all-targets figure")
    args = ap.parse_args()

    names = args.targets or [t.name for t in family_targets(args.family)]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    L, tau = args.max_len, args.tau

    forms, terminals = enumerate_reachable(L)
    nodes = build_tree(args.depth, L, args.expansion)
    pos = layout(nodes)
    print(f"=== depth-{args.depth} tree, expansion={args.expansion}: "
          f"{len(nodes)} nodes "
          f"({sum(1 for n in nodes if n['terminal'])} terminal) ===")
    print(f"    MDP: {len(forms)} reachable forms, {len(terminals)} terminals, "
          f"L={L}, tau={tau:g}")

    per_target = {}
    for name in names:
        t = get_target(name)
        xs, ys = t.xs(), None
        ys = t.ys(xs)
        cache_path = out_dir / "cache" / f"fit_{name}.json"
        if not cache_path.exists():
            raise SystemExit(f"missing {cache_path}\n"
                             f"run 07-family-exact-audit.py first")
        cache = json.loads(cache_path.read_text())
        miss = [f for f in terminals if " ".join(f) not in cache]
        for f in miss:
            cache[" ".join(f)] = float(fit_expression(" ".join(f), xs, ys))
        if miss:
            cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True))
        scores = {f: cache[" ".join(f)] for f in terminals}
        vals = backward_induction(forms, scores, L, tau)
        choice = root_choice(nodes, vals)
        per_target[name] = (scores, vals, choice)

        print(f"\n--- {name}: y = {t.infix}")
        print(f"    mean-backup would pick {choice['greedy_name']:>2} "
              f"(Vq={choice['greedy_Vq']:+.4f}, V*={choice['greedy_Vstar']:.4f});"
              f"  optimal is {choice['optimal_name']:>2} "
              f"(Vq={choice['optimal_Vq']:+.4f}, V*={choice['optimal_Vstar']:.4f})"
              f"  ->  {'TRAP' if choice['is_trap'] else 'no inversion'}")

        # ---- one figure per target, imitating 02's output shape -----------
        n_leaves = sum(1 for n in nodes if not n["children"])
        fig, ax = plt.subplots(figsize=(max(16.0, 1.75 * n_leaves),
                                        3.1 * (1 + args.depth)))
        draw_tree(ax, nodes, pos, vals, scores, choice)
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM); sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01,
                     label="node fill = Vq (exact uniform-completion value)")
        ax.legend(handles=legend_handles(choice), loc="lower left",
                  fontsize=8.5, framealpha=0.92)
        ax.set_title(title_for(name, choice, args.depth, args.expansion,
                               len(nodes)), fontsize=12)
        fig.tight_layout()
        tdir = out_dir / name          # per-target subfolder
        tdir.mkdir(parents=True, exist_ok=True)
        png = tdir / f"tree_{name}_depth{args.depth}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)

        dump = [{"id": n["id"], "depth": n["depth"], "parent": n["parent"],
                 "terminal": n["terminal"], "expr": " ".join(n["form"]),
                 "action": (None if n["via"] is None else
                            {"pos": n["via"][0], "prod": n["via"][1],
                             "name": PROD_NAMES[n["via"][1]]}),
                 "V_star": vals["Vstar"][n["form"]],
                 "V_q": vals["Vq"][n["form"]],
                 "rho": vals["rho"][n["form"]],
                 "gap": vals["Vstar"][n["form"]] - vals["Vq"][n["form"]],
                 "R": scores[n["form"]] if n["terminal"] else None}
                for n in nodes]
        (tdir / f"tree_{name}_depth{args.depth}.json").write_text(json.dumps(
            {"target": name, "infix": t.infix, "depth": args.depth,
             "expansion": args.expansion, "max_len": L, "tau": tau,
             "root_choice": {k: v for k, v in choice.items()},
             "nodes": dump}, indent=2))
        print(f"    [saved] {png.relative_to(REPO)}")

    # ---- combined figure: the tree is identical across targets, only the
    #      values change, so stacking them is the family comparison ---------
    if not args.no_grid and len(names) > 1:
        n_leaves = sum(1 for n in nodes if not n["children"])
        fig, axes = plt.subplots(len(names), 1,
                                 figsize=(max(16.0, 1.75 * n_leaves),
                                          3.1 * (1 + args.depth) * len(names)))
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            scores, vals, choice = per_target[name]
            draw_tree(ax, nodes, pos, vals, scores, choice)
            ax.legend(handles=legend_handles(choice), loc="lower left",
                      fontsize=8.0, framealpha=0.92)
            ax.set_title(title_for(name, choice, args.depth, args.expansion,
                                   len(nodes)), fontsize=11)
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM); sm.set_array([])
        fig.colorbar(sm, ax=axes.tolist(), shrink=0.35, pad=0.01,
                     label="node fill = Vq (exact uniform-completion value)")
        png = out_dir / f"tree_family_depth{args.depth}.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[saved] {png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
