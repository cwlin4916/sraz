#!/usr/bin/env python
"""What equation does pure MCTS converge to as the exploration constant varies?

Sweeps `c_exploration` (the c in the PUCT selection rule) crossed with the
simulation budget, and reports the equation greedy play converges to in each
cell.

Why c is the interesting knob here. Selection scores each action as

    UCB(a) = Qtilde(a) + c * P(a) * sqrt(N_tot + EPS) / (1 + N(a))

With a uniform prior P = 1/K and an incumbent pinned at Qtilde = 1, an *untried*
action (N = 0, Qtilde = 0) only overtakes it once

    c * sqrt(N_tot) / K > 1      i.e.    N_tot > (K / c)^2

For K = 7 legal first moves and c = 1 that is ~49 simulations, which is why the
default search commits to the first stable terminal it finds (`* C1 x`, R^2 =
0.873) and leaves most siblings untouched. Raising c lowers that threshold
quadratically, so this sweep asks the direct question: does buying breadth with
c get the search out of the linear answer, and how far up the ladder does it
climb?

Reference rungs on the sine target (seed 42), all exact from the enumeration in
01/02:

    R^2 = 0.873   `C1*x`                honest linear (what c = 1 converges to)
    R^2 = 0.958   `C0 + C1*x`           affine
    R^2 = 0.997   V*, the global optimum reachable under max_len = 15

Episode seeds are shared across cells, so cells are *paired*: a difference
between two c values is attributable to c rather than to RNG.

Fits are memoised from the seed-matched cache built by 01/02 (which lives under
the 8-5 experiment folder and is keyed by problem seed, so it is reused rather
than duplicated here). Rollout terminal fits therefore become dict lookups and
the R^2 values match the V* analysis exactly.

`--rollout-budget` is exposed because the shipped default (500 steps, shared
across every simulation of one move) is exhausted after ~6 leaf evaluations, so
later leaves fall back to the net's constant 0.0 rather than a rollout estimate.
Raising it separates "the rollout signal is deceptive" from "the rollout signal
is starved".

Run from repo root:
    .venv/bin/python Claude-scripts/06-c-exploration-sweep.py
    .venv/bin/python Claude-scripts/06-c-exploration-sweep.py --rollout-budget 5000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sraz.instances.symreg.game import prefix_to_infix
from sraz.instances.symreg.config import SymRegConfig

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "Claude-experiments" / "8-6"


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(
        fname.replace("-", "_").removesuffix(".py"), Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 03 owns the fit memo (pointed at the seed-matched cache) and the greedy episode
conv = _load("03-mcts-convergence.py")

# exact reference values from the enumeration in 01/02
V_STAR = 0.9966
R_AFFINE = 0.9575
R_LINEAR = 0.8729

FAMILIES = ["sin", "quadratic", "affine", "linear", "constant", "other"]
FAM_ABBR = {"sin": "sin", "quadratic": "quad", "affine": "aff",
            "linear": "lin", "constant": "con", "other": "oth", "none": "--"}
FAM_COLOR = {"sin": "#2f8f4e", "quadratic": "#7fbf5f", "affine": "#f2c14e",
             "linear": "#e8813a", "constant": "#c0392b", "other": "0.6",
             "none": "0.85"}


def family(infix: str | None) -> str:
    """Coarse structural label for a converged expression.

    Deliberately crude -- it reads the infix string rather than the fitted
    functional form -- so the numeric R^2 is always reported alongside it.
    """
    if not infix:
        return "none"
    if "sin" in infix:
        return "sin"
    if "x**2" in infix or "x*x" in infix:
        return "quadratic"
    if "x" not in infix:
        return "constant"
    return "affine" if "+" in infix else "linear"


def run_cell(agent, c, sims, episodes, base_seed):
    """One (c, n_simulations) cell: `episodes` seeded greedy episodes."""
    agent.mcts_params["c_exploration"] = float(c)
    agent.mcts_params["n_simulations"] = int(sims)

    t0 = time.time()
    out = [conv.greedy_episode(agent, base_seed + e) for e in range(episodes)]
    dt = time.time() - t0

    r2s = np.array([r for r, _ in out])
    infixes = [prefix_to_infix(rule.split()) if rule else None for _, rule in out]
    fams = [family(s) for s in infixes]
    counts = Counter(infixes)
    best_i = int(np.argmax(r2s))
    modal_infix, modal_count = counts.most_common(1)[0]

    return {
        "c_exploration": float(c),
        "n_simulations": int(sims),
        "episodes": episodes,
        "mean_r2": float(r2s.mean()),
        "best_r2": float(r2s.max()),
        "worst_r2": float(r2s.min()),
        "std_r2": float(r2s.std()),
        "modal_infix": modal_infix,
        "modal_count": int(modal_count),
        "modal_family": family(modal_infix),
        "best_infix": infixes[best_i],
        "best_prefix": out[best_i][1],
        "n_distinct": len(counts),
        "family_counts": {f: fams.count(f) for f in FAMILIES if fams.count(f)},
        "frac_sin": float(np.mean([f == "sin" for f in fams])),
        "frac_beats_linear": float(np.mean(r2s > R_LINEAR + 1e-6)),
        "r2_per_episode": r2s.tolist(),
        "infix_per_episode": infixes,
        "converged_counts": [(k, v) for k, v in counts.most_common()],
        "wall_clock_s": round(dt, 1),
    }


def draw(results, cs, sims_list, out_path, problem, problem_seed, budget_tag):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.6),
                                   gridspec_kw={"width_ratios": [1.05, 1.0]})
    grid = {(r["c_exploration"], r["n_simulations"]): r for r in results}

    # ---- left: R^2 vs c, one colour per simulation budget ----
    cmap = matplotlib.colormaps["viridis"]
    for i, sims in enumerate(sims_list):
        col = cmap(i / max(len(sims_list) - 1, 1) * 0.85)
        means = [grid[(c, sims)]["mean_r2"] for c in cs]
        bests = [grid[(c, sims)]["best_r2"] for c in cs]
        axL.plot(cs, means, "-o", color=col, lw=2, ms=5, label=f"{sims} sims (mean)")
        axL.plot(cs, bests, "--^", color=col, lw=1.2, ms=5, alpha=0.75,
                 label=f"{sims} sims (best)")
    axL.axhline(V_STAR, color="tab:green", ls=":", lw=1.4,
                label=f"V* global optimum = {V_STAR:.3f}")
    axL.axhline(R_AFFINE, color="tab:olive", ls=":", lw=1.2,
                label=f"affine  C0+C1*x = {R_AFFINE:.3f}")
    axL.axhline(R_LINEAR, color="tab:red", ls=":", lw=1.2,
                label=f"linear  C1*x = {R_LINEAR:.3f}")
    axL.set_xscale("log")
    axL.set_xticks(cs)
    axL.set_xticklabels([f"{c:g}" for c in cs])
    axL.set_xlabel("c_exploration  (log scale)")
    axL.set_ylabel("greedy terminal $R^2$")
    axL.set_title("Converged fit quality vs exploration constant")
    axL.grid(alpha=0.3)
    axL.legend(fontsize=7, loc="lower right", ncol=2)

    # ---- right: heatmap of best R^2, annotated with the modal family ----
    M = np.array([[grid[(c, s)]["best_r2"] for c in cs] for s in sims_list])
    norm = Normalize(vmin=min(R_LINEAR - 0.05, M.min()), vmax=V_STAR)
    im = axR.imshow(M, cmap="RdYlGn", norm=norm, aspect="auto")
    for i, s in enumerate(sims_list):
        for j, c in enumerate(cs):
            r = grid[(c, s)]
            axR.text(j, i - 0.16, f"{r['best_r2']:.3f}", ha="center", va="center",
                     fontsize=8, fontweight="bold")
            axR.text(j, i + 0.22,
                     f"{FAM_ABBR[r['modal_family']]} {r['modal_count']}/{r['episodes']}",
                     ha="center", va="center", fontsize=7, color="0.25")
    axR.set_xticks(range(len(cs)))
    axR.set_xticklabels([f"{c:g}" for c in cs])
    axR.set_yticks(range(len(sims_list)))
    axR.set_yticklabels([str(s) for s in sims_list])
    axR.set_xlabel("c_exploration")
    axR.set_ylabel("simulations per move")
    axR.set_title("best $R^2$ per cell (top) and modal family (bottom)\n"
                  "lin = C1*x, aff = C0+C1*x, quad = has $x^2$, sin = has sin")
    fig.colorbar(im, ax=axR, label="best $R^2$", shrink=0.85, pad=0.02)

    fig.suptitle(f"Pure MCTS on {problem} (seed {problem_seed}) -- "
                 f"c_exploration sweep{budget_tag}", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="sine")
    ap.add_argument("--problem-seed", type=int, default=42)
    ap.add_argument("--c", type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0])
    ap.add_argument("--sims", type=int, nargs="+", default=[25, 100, 400])
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--rollout-n", type=int, default=20)
    ap.add_argument("--rollout-budget", type=int, default=None,
                    help="override the shared per-move rollout step budget "
                         "(MCTS default is 500, which starves after ~6 leaves)")
    ap.add_argument("--seed", type=int, default=42, help="base episode seed")
    args = ap.parse_args()

    cache_key = (f"{args.problem}_seed{args.problem_seed}"
                 if args.problem == "sine" else args.problem)
    conv.install_fit_memo(cache_key)

    cfg = SymRegConfig(problem=args.problem, pure_mcts=True)
    cfg.game.kwargs["problem_seed"] = args.problem_seed
    cfg.game.kwargs["redraw_constants"] = False
    cfg.agent.mcts_params["rollout_n"] = args.rollout_n
    cfg.agent.random_seeds = {"mcts": args.seed, "train": args.seed + 1,
                              "eval": args.seed + 2,
                              "external_policy": args.seed + 3}
    game, net, agent, _ = cfg.build()
    if args.rollout_budget is not None:
        agent.mcts_params["rollout_budget"] = args.rollout_budget

    budget = agent.mcts_params.get("rollout_budget", 500)
    budget_tag = "" if args.rollout_budget is None else f", rollout_budget={budget}"

    print(f"=== c_exploration sweep: {args.problem} "
          f"(problem_seed={args.problem_seed}) ===")
    print(f"target: {game.target_infix}   constants: "
          + str({k: round(v, 3) for k, v in game.constants.items()}))
    print(f"pure MCTS: uniform prior P=1/K, random rollouts "
          f"(rollout_n={args.rollout_n}, rollout_budget={budget}, mean backup)")
    print(f"greedy = argmax visit counts, {args.episodes} paired episodes per cell")
    print(f"reference: linear={R_LINEAR:.4f}  affine={R_AFFINE:.4f}  "
          f"V*={V_STAR:.4f}\n")

    results = []
    for sims in args.sims:
        print(f"--- {sims} simulations per move " + "-" * 40)
        for c in args.c:
            r = run_cell(agent, c, sims, args.episodes, args.seed)
            results.append(r)
            fam = ", ".join(f"{FAM_ABBR[f]}x{n}"
                            for f, n in sorted(r["family_counts"].items(),
                                               key=lambda kv: -kv[1]))
            print(f"  c={c:>5g}: mean={r['mean_r2']:+.4f} best={r['best_r2']:+.4f} "
                  f"distinct={r['n_distinct']:>2} [{fam}]  ({r['wall_clock_s']}s)")
            print(f"           modal {r['modal_count']:>2}/{r['episodes']}: "
                  f"{r['modal_infix']}")
            if r["best_r2"] > R_LINEAR + 1e-6:
                print(f"           BEST  {r['best_r2']:+.4f}: {r['best_infix']}")

    tag = f"{cache_key}" + ("" if args.rollout_budget is None
                            else f"_rb{args.rollout_budget}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jpath = OUT_DIR / f"c-sweep_{tag}.json"
    jpath.write_text(json.dumps({
        "problem": args.problem, "problem_seed": args.problem_seed,
        "target_infix": game.target_infix, "target_constants": game.constants,
        "rollout_n": args.rollout_n, "rollout_budget": budget,
        "episodes": args.episodes, "c_values": args.c, "sims": args.sims,
        "reference": {"linear": R_LINEAR, "affine": R_AFFINE, "v_star": V_STAR},
        "results": results}, indent=2))
    png = OUT_DIR / f"c-sweep_{tag}.png"
    draw(results, args.c, args.sims, png, args.problem, args.problem_seed,
         budget_tag)

    # ---- summary ----
    print("\n--- summary " + "-" * 56)
    best = max(results, key=lambda r: r["best_r2"])
    print(f"  overall best R^2 = {best['best_r2']:+.4f} at c={best['c_exploration']:g}, "
          f"sims={best['n_simulations']}")
    print(f"    {best['best_infix']}")
    beat = [r for r in results if r["best_r2"] > R_LINEAR + 1e-6]
    print(f"  cells beating the linear answer: {len(beat)}/{len(results)}")
    sin = [r for r in results if r["frac_sin"] > 0]
    print(f"  cells ever reaching a sin term:  {len(sin)}/{len(results)}")
    print(f"[saved] {jpath.relative_to(REPO)}")
    print(f"[saved] {png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
