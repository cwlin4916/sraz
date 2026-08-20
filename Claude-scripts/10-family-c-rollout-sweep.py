#!/usr/bin/env python3
"""Pure-MCTS sweep over the exploration constant x rollouts-per-leaf, per target.

Family analogue of `06-c-exploration-sweep.py` (which produced
`Claude-experiments/8-6/c-sweep_sine_seed42_rb5000.png`). Same question, same
figure language, two axes swapped:

    06:  c_exploration  x  simulations-per-move
    10:  c_exploration  x  rollout_n            (simulations held fixed)

`rollout_n` is the number of uniform random completions averaged into one leaf
value. It is the more interesting second axis here because it controls *how much
the mean backup averages away*: at rollout_n=1 a leaf value IS one terminal's
reward, so a perfect completion registers as 1.0; at rollout_n=20 a perfect
completion is diluted by 19 mediocre siblings and the tree may never record a
value above the one-move decoy. Sweeping it against c asks whether the trap of
eq. (9) is escaped by buying breadth (c) or by sharpening the leaf signal
(small rollout_n) -- and whether they trade off.

Two deviations from 06, both deliberate:

1.  The primary heatmap shows MEAN R^2, not best. On sine V*=0.9966 was rarely
    attained so `best` discriminated well; here every target is exactly
    expressible (V*=1) and `best` over 24 episodes saturates at 1.000 in most
    cells, which would wash the panel out. `best` is still drawn in the left
    panel as the dashed series.
2.  A third panel reports the exact-recovery fraction, the writeup's primary
    outcome -- the share of episodes whose greedy play lands on R^2 = 1.

`rollout_budget` is DERIVED, not guessed. The longest derivation in this MDP is
11 actions, so one move can consume at most sims * rollout_n * 11 rollout steps;
the budget is set to exactly that. It is therefore provably unable to bind in
any cell, so the sweep measures c and rollout_n rather than starvation. (The
shipped default of 500 would starve almost every cell here.)

Episode seeds are shared across cells, so cells are paired: a difference between
two cells is attributable to (c, rollout_n), not to RNG.

Outputs, per target, into <out-dir>/<target>/:
    c-rollout-sweep_<target>_s<sims>.png
    c-rollout-sweep_<target>_s<sims>.json
and one comparison figure at <out-dir>/c-rollout-sweep_family_s<sims>.png
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import Normalize                            # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import sraz.instances.symreg.game as game_mod                      # noqa: E402
from sraz.instances.symreg.game import prefix_to_infix             # noqa: E402
from sraz.instances.symreg.config import SymRegConfig              # noqa: E402
from sraz.instances.symreg.targets import family_targets, get_target  # noqa: E402

MAX_ACTIONS = 11        # longest derivation under ADDITIVE_GRAMMAR at L=12
EXACT_TOL = 1e-6


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(
        fname.replace("-", "_").removesuffix(".py"),
        Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


conv = _load("03-mcts-convergence.py")      # owns greedy_episode
sweep6 = _load("06-c-exploration-sweep.py")  # owns family() / FAM_ABBR
family = sweep6.family
FAM_ABBR = sweep6.FAM_ABBR
FAMILIES = sweep6.FAMILIES


def install_memo(cache_path: Path) -> dict:
    """Point game.fit_expression at the 8-17 terminal-score cache.

    Same trick as 03's install_fit_memo, but reading the cache 07 builds
    (keyed by the space-joined sentence, which is exactly what the game passes
    in) instead of 8-5's sine-keyed file. Misses fall through to a live solve
    and are remembered, so correctness does not depend on the cache existing.
    """
    memo: dict[str, float] = {}
    if cache_path.exists():
        memo.update(json.loads(cache_path.read_text()))
        print(f"[memo] preloaded {len(memo)} fits from {cache_path.name}")
    else:
        print(f"[memo] no cache at {cache_path.name}; fitting live (slower)")
    orig = game_mod.fit_expression

    def memoised(rule, xs, exact_ys, max_nfev=None):
        v = memo.get(rule)
        if v is None:
            v = orig(rule, xs, exact_ys, max_nfev=max_nfev)
            memo[rule] = v
        return v

    game_mod.fit_expression = memoised
    return memo


def reference_levels(name: str) -> dict:
    """V* and the one-move rungs, for the horizontal reference lines."""
    t = get_target(name)
    xs, ys = t.xs(), None
    ys = t.ys(xs)
    one_move = {"C0": "C0", "* C1 x": "* C1 x", "* C2 * x x": "* C2 * x x"}
    R = {k: float(game_mod.fit_expression(v, xs, ys))
         for k, v in one_move.items()}
    best_name = max(R, key=lambda k: R[k])
    return {"V_star": 1.0, "one_move": R,
            "decoy_name": best_name, "decoy": R[best_name]}


def run_cell(agent, c, rollout_n, sims, episodes, base_seed) -> dict:
    """One (c, rollout_n) cell: `episodes` seeded greedy episodes."""
    agent.mcts_params["c_exploration"] = float(c)
    agent.mcts_params["rollout_n"] = int(rollout_n)
    agent.mcts_params["n_simulations"] = int(sims)
    # provably sufficient, so never binding -- see module docstring
    agent.mcts_params["rollout_budget"] = int(sims * rollout_n * MAX_ACTIONS)

    t0 = time.time()
    out = [conv.greedy_episode(agent, base_seed + e) for e in range(episodes)]
    dt = time.time() - t0

    r2s = np.array([r for r, _ in out])
    infixes = [prefix_to_infix(rule.split()) if rule else None for _, rule in out]
    fams = [family(s) for s in infixes]
    counts = Counter(infixes)
    modal_infix, modal_count = counts.most_common(1)[0]
    best_i = int(np.argmax(r2s))
    return {
        "c_exploration": float(c), "rollout_n": int(rollout_n),
        "n_simulations": int(sims), "episodes": episodes,
        "rollout_budget": int(sims * rollout_n * MAX_ACTIONS),
        "mean_r2": float(r2s.mean()), "best_r2": float(r2s.max()),
        "worst_r2": float(r2s.min()), "std_r2": float(r2s.std()),
        "frac_exact": float(np.mean(r2s >= 1.0 - EXACT_TOL)),
        "n_exact": int(np.sum(r2s >= 1.0 - EXACT_TOL)),
        "modal_infix": modal_infix, "modal_count": int(modal_count),
        "modal_family": family(modal_infix),
        "best_infix": infixes[best_i], "best_prefix": out[best_i][1],
        "n_distinct": len(counts),
        "family_counts": {f: fams.count(f) for f in FAMILIES if fams.count(f)},
        "r2_per_episode": r2s.tolist(), "infix_per_episode": infixes,
        "wall_clock_s": round(dt, 2),
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _heat(ax, M, cs, rns, norm, cmap, annot, title, xlabel=True):
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(rns)):
        for j in range(len(cs)):
            top, bot = annot(i, j)
            ax.text(j, i - 0.16, top, ha="center", va="center", fontsize=8,
                    fontweight="bold")
            if bot:
                ax.text(j, i + 0.22, bot, ha="center", va="center",
                        fontsize=6.8, color="0.25")
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([f"{c:g}" for c in cs])
    ax.set_yticks(range(len(rns)))
    ax.set_yticklabels([str(r) for r in rns])
    if xlabel:
        ax.set_xlabel("c_exploration")
    ax.set_ylabel("rollout_n  (completions averaged per leaf)")
    ax.set_title(title, fontsize=10)
    return im


def draw(results, cs, rns, ref, out_path, name, sims, episodes):
    grid = {(r["c_exploration"], r["rollout_n"]): r for r in results}
    fig, (axL, axM, axR) = plt.subplots(
        1, 3, figsize=(21, 5.9), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    # ---- left: R^2 vs c, one colour per rollout_n -------------------------
    cmap = matplotlib.colormaps["viridis"]
    for i, rn in enumerate(rns):
        col = cmap(i / max(len(rns) - 1, 1) * 0.85)
        axL.plot(cs, [grid[(c, rn)]["mean_r2"] for c in cs], "-o", color=col,
                 lw=2, ms=5, label=f"rollout_n={rn} (mean)")
        axL.plot(cs, [grid[(c, rn)]["best_r2"] for c in cs], "--^", color=col,
                 lw=1.2, ms=5, alpha=0.75, label=f"rollout_n={rn} (best)")
    axL.axhline(ref["V_star"], color="tab:green", ls=":", lw=1.5,
                label=f"V* = exact = {ref['V_star']:.3f}")
    axL.axhline(ref["decoy"], color="tab:red", ls=":", lw=1.4,
                label=f"one-move decoy  {ref['decoy_name']} = {ref['decoy']:.3f}")
    axL.set_xscale("log")
    axL.set_xticks(cs)
    axL.set_xticklabels([f"{c:g}" for c in cs])
    axL.set_xlabel("c_exploration  (log scale)")
    axL.set_ylabel("greedy terminal $R^2$")
    axL.set_title("Converged fit quality vs exploration constant", fontsize=10)
    axL.grid(alpha=0.3)
    axL.legend(fontsize=6.5, loc="best", ncol=2)

    # ---- middle: mean R^2 heatmap ----------------------------------------
    Mm = np.array([[grid[(c, r)]["mean_r2"] for c in cs] for r in rns])
    lo = min(ref["decoy"] - 0.05, float(Mm.min()))
    normm = Normalize(vmin=lo, vmax=1.0)
    im1 = _heat(axM, Mm, cs, rns, normm, "RdYlGn",
                lambda i, j: (f"{grid[(cs[j], rns[i])]['mean_r2']:.3f}",
                              f"{FAM_ABBR[grid[(cs[j], rns[i])]['modal_family']]}"
                              f" {grid[(cs[j], rns[i])]['modal_count']}"
                              f"/{episodes}"),
                "mean $R^2$ (top) and modal family (bottom)\n"
                "con = constant, lin = C1*x, aff = C0+C1*x, quad = has $x^2$")
    fig.colorbar(im1, ax=axM, label="mean $R^2$", shrink=0.85, pad=0.02)

    # ---- right: exact-recovery fraction ----------------------------------
    Me = np.array([[grid[(c, r)]["frac_exact"] for c in cs] for r in rns])
    im2 = _heat(axR, Me, cs, rns, Normalize(0.0, 1.0), "RdYlGn",
                lambda i, j: (f"{grid[(cs[j], rns[i])]['frac_exact']:.2f}",
                              f"{grid[(cs[j], rns[i])]['n_exact']}/{episodes}"),
                "exact-recovery fraction\n"
                "(share of episodes whose greedy play reaches $R^2=1$)")
    fig.colorbar(im2, ax=axR, label="fraction exact", shrink=0.85, pad=0.02)

    t = get_target(name)
    fig.suptitle(
        f"Pure MCTS on {name}   y = {t.infix}   —   c_exploration × rollout_n "
        f"sweep at {sims} simulations/move, {episodes} paired episodes/cell\n"
        f"shipped PUCT selection (not the writeup's eq. 17); mean backup; "
        f"rollout_budget derived per cell = sims × rollout_n × {MAX_ACTIONS}, "
        f"provably non-binding",
        fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def draw_family(per_target, cs, rns, out_path, sims, episodes):
    """One row per target: mean R^2 and exact-recovery, on shared scales."""
    names = list(per_target)
    fig, axes = plt.subplots(len(names), 2, figsize=(15, 4.6 * len(names)))
    axes = np.atleast_2d(axes)
    for row, name in enumerate(names):
        results, ref = per_target[name]
        grid = {(r["c_exploration"], r["rollout_n"]): r for r in results}
        Mm = np.array([[grid[(c, r)]["mean_r2"] for c in cs] for r in rns])
        Me = np.array([[grid[(c, r)]["frac_exact"] for c in cs] for r in rns])
        t = get_target(name)
        i1 = _heat(axes[row, 0], Mm, cs, rns, Normalize(-1.0, 1.0), "RdYlGn",
                   lambda i, j, g=grid: (f"{g[(cs[j], rns[i])]['mean_r2']:.3f}", ""),
                   f"{name}   y = {t.infix}   —   mean $R^2$",
                   xlabel=(row == len(names) - 1))
        i2 = _heat(axes[row, 1], Me, cs, rns, Normalize(0.0, 1.0), "RdYlGn",
                   lambda i, j, g=grid: (f"{g[(cs[j], rns[i])]['frac_exact']:.2f}", ""),
                   f"{name}   —   exact-recovery fraction",
                   xlabel=(row == len(names) - 1))
        fig.colorbar(i1, ax=axes[row, 0], shrink=0.85, pad=0.02)
        fig.colorbar(i2, ax=axes[row, 1], shrink=0.85, pad=0.02)
    fig.suptitle(f"Pure MCTS, linear family: c_exploration × rollout_n at "
                 f"{sims} simulations/move ({episodes} episodes/cell)\n"
                 f"left column on a shared [-1,1] scale so targets are "
                 f"comparable; right column is the writeup's primary outcome",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--family", default="linear")
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--c", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0])
    ap.add_argument("--rollout-n", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(REPO / "Claude-experiments" / "8-17"))
    args = ap.parse_args()

    names = args.targets or [t.name for t in family_targets(args.family)]
    out_dir = Path(args.out_dir).resolve()
    cs, rns = list(args.c), list(args.rollout_n)

    print(f"=== c_exploration x rollout_n sweep, {args.sims} sims/move ===")
    print(f"targets : {', '.join(names)}")
    print(f"c       : {cs}")
    print(f"rollout_n: {rns}")
    print(f"episodes: {args.episodes} per cell (paired seeds from {args.seed})")
    print(f"rollout_budget per cell = {args.sims} x rollout_n x {MAX_ACTIONS} "
          f"(provably non-binding)\n")

    per_target = {}
    for name in names:
        install_memo(out_dir / "cache" / f"fit_{name}.json")
        ref = reference_levels(name)
        t = get_target(name)
        print(f"--- {name}: y = {t.infix}   "
              f"V*=1.000  decoy {ref['decoy_name']}={ref['decoy']:+.4f}")

        cfg = SymRegConfig(problem="additive_quadratic", pure_mcts=True)
        cfg.game.kwargs["target"] = name
        cfg.game.kwargs["max_len"] = 12
        cfg.game.kwargs["redraw_constants"] = False
        cfg.agent.random_seeds = {"mcts": args.seed, "train": args.seed + 1,
                                  "eval": args.seed + 2,
                                  "external_policy": args.seed + 3}
        _, _, agent, _ = cfg.build()

        results = []
        for rn in rns:
            row = []
            for c in cs:
                r = run_cell(agent, c, rn, args.sims, args.episodes, args.seed)
                results.append(r)
                row.append(f"{r['mean_r2']:+.3f}/{r['n_exact']:>2}")
            print(f"  rollout_n={rn:>2} | " + "  ".join(row))
        print(f"             {'':9}" + "  ".join(f"{'c='+format(c,'g'):>9}"
                                                 for c in cs))

        tdir = out_dir / name
        tdir.mkdir(parents=True, exist_ok=True)
        png = tdir / f"c-rollout-sweep_{name}_s{args.sims}.png"
        draw(results, cs, rns, ref, png, name, args.sims, args.episodes)
        (tdir / f"c-rollout-sweep_{name}_s{args.sims}.json").write_text(
            json.dumps({"target": name, "infix": t.infix,
                        "n_simulations": args.sims, "c": cs, "rollout_n": rns,
                        "episodes": args.episodes, "base_seed": args.seed,
                        "reference": ref, "max_actions": MAX_ACTIONS,
                        "selection_rule": "shipped PUCT (not writeup eq. 17)",
                        "cells": results}, indent=2))
        print(f"  [saved] {png.relative_to(REPO)}")
        per_target[name] = (results, ref)

    if len(names) > 1:
        png = out_dir / f"c-rollout-sweep_family_s{args.sims}.png"
        draw_family(per_target, cs, rns, png, args.sims, args.episodes)
        print(f"\n[saved] {png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
