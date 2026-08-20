#!/usr/bin/env python3
"""Every family member as one point: rollout preference vs rollout recovery.

x  V^q(C) - max_i V^q(D_i), the difference between the continue node and the
   best of its siblings at depth 1, under V^q -- the expected reward of a
   uniform random completion, i.e. what a rollout-fed mean-backup search
   estimates. x < 0 means rollouts rate a one-move terminal above the only
   branch whose subtree contains an exact expression.

y  rho(C), the probability that ONE uniform random completion from C is exact.

Why rho and not P(exact within B rollouts), which is what the per-target figure
(`11-inversion-vs-recovery.py`) marks: at B = 100 that probability is
1 - 1.6e-22 for lin_A/B/C and 1 - 1.0e-39 for lin_D, i.e. 1.0 to every digit a
float carries. Plotted, all four points would sit on one horizontal line and the
y axis would carry no information. rho is the same quantity per draw, keeps the
ordering, and stays readable. P(B) is printed and annotated so the saturation is
visible rather than hidden.

Colour encodes the write-up's trap test, which the x axis alone cannot express:
x < 0 is necessary for a trap but not sufficient. If the winning sibling is
ITSELF exact (R = 1) then preferring it is correct and no trap exists -- that is
lin_D. A trap needs both x < 0 AND max_i R(D_i) < 1, the two inequalities of
eq. `eq:inversion`.

Reads the tree JSONs written by `09-family-depth2-trees.py`, so the numbers are
the numbers in the `tree_<target>_depth2.png` figures.

    .venv/bin/python Claude-scripts/12-family-inversion-recovery-scatter.py
    .venv/bin/python Claude-scripts/12-family-inversion-recovery-scatter.py \
        --targets lin_A lin_B lin_C lin_D quad_A quad_B quad_C quad_D
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sraz.instances.symreg.targets import get_target               # noqa: E402

FAMILY_DIR = {"linear": "8-17", "quadratic": "8-20"}
EXACT_TOL = 1e-6
COINCIDENT_TOL = 1e-9          # closer than this and markers would overlap

C_TRAP = "#c0392b"
C_OK = "#2e8b57"
C_SOLVED = "#2c6fad"


def collect(target: str, exp_root: Path) -> dict:
    t = get_target(target)
    tj = exp_root / FAMILY_DIR[t.family] / target / f"tree_{target}_depth2.json"
    if not tj.exists():
        raise SystemExit(f"missing {tj}\nrun 09-family-depth2-trees.py "
                         f"--family {t.family} first")
    d = json.loads(tj.read_text())
    l1 = [n for n in d["nodes"] if n["depth"] == 1]
    cont = next(n for n in l1 if not n["terminal"])
    sibs = [n for n in l1 if n["terminal"]]
    for n in sibs:                       # a terminal's V^q is just its reward
        assert abs(n["V_q"] - n["R"]) < 1e-12, n["expr"]
    best = max(sibs, key=lambda n: n["V_q"])
    sib_exact = best["R"] >= 1.0 - EXACT_TOL
    diff = cont["V_q"] - best["V_q"]
    if sib_exact:
        kind = "solved at once"
    elif diff < 0:
        kind = "trap"
    else:
        kind = "no inversion"
    return {"target": target, "infix": d["infix"], "family": t.family,
            "V_q_C": cont["V_q"], "V_star_C": cont["V_star"], "rho": cont["rho"],
            "best_sib": best["expr"], "best_sib_R": best["R"],
            "best_sib_name": best["action"]["name"],
            "difference": diff, "best_sib_exact": bool(sib_exact), "kind": kind}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+",
                    default=["lin_A", "lin_B", "lin_C", "lin_D"])
    ap.add_argument("--exp-root", default=str(REPO / "Claude-experiments"))
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    exp_root = Path(args.exp_root)
    rows = [collect(t, exp_root) for t in args.targets]
    B = args.budget
    for r in rows:
        r["fail_B"] = (1.0 - r["rho"]) ** B
        r["P_B"] = 1.0 - r["fail_B"]

    print(f"{'target':8s} {'difference':>12s} {'rho(C)':>10s} "
          f"{'P(B=%d)' % B:>16s} {'best sibling':>14s} {'R':>9s}  kind")
    for r in rows:
        print(f"{r['target']:8s} {r['difference']:+12.6f} {r['rho']:10.6f} "
              f"{'1 - ' + format(r['fail_B'], '.1e'):>16s} "
              f"{r['best_sib']:>14s} {r['best_sib_R']:+9.4f}  {r['kind']}")

    # group points that would land on top of one another
    groups: list[list[dict]] = []
    for r in rows:
        for g in groups:
            if (abs(g[0]["difference"] - r["difference"]) < COINCIDENT_TOL
                    and abs(g[0]["rho"] - r["rho"]) < COINCIDENT_TOL):
                g.append(r); break
        else:
            groups.append([r])
    for g in groups:
        if len(g) > 1:
            print(f"\ncoincident (drawn as one marker): "
                  f"{', '.join(x['target'] for x in g)}   "
                  f"max |dx| = {max(abs(a['difference']-b['difference']) for a in g for b in g):.1e}")

    # ---------------- draw -------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.6, 7.4))
    xs = [r["difference"] for r in rows]
    lo, hi = min(xs), max(xs)
    pad = max(0.12, 0.22 * (hi - lo))
    xlim = (lo - pad, hi + pad)

    ax.axvspan(xlim[0], 0, color=C_TRAP, alpha=0.045, zorder=0)
    ax.axvline(0, color="black", lw=1.3, ls="--", zorder=1)
    ax.text(xlim[0] + 0.012, 0.035,
            "x < 0:  rollouts rate a one-move terminal\nABOVE the continue node",
            fontsize=9, color=C_TRAP, va="bottom", ha="left")
    ax.text(xlim[1] - 0.012, 0.035,
            "x > 0:  rollouts already\nprefer the continue node",
            fontsize=9, color=C_OK, va="bottom", ha="right")

    style = {"trap": (C_TRAP, "X", "trap: rollouts prefer a sibling AND it is not exact"),
             "solved at once": (C_SOLVED, "s", "solved at once: winning sibling IS exact, so preferring it is correct"),
             "no inversion": (C_OK, "o", "no inversion: rollouts already prefer the continue node")}
    seen_lab = set()
    for g in groups:
        r = g[0]
        col, mk, lab = style[r["kind"]]
        ax.scatter([r["difference"]], [r["rho"]], s=340, c=col, marker=mk,
                   edgecolors="black", linewidths=1.3, zorder=6,
                   label=lab if lab not in seen_lab else None)
        seen_lab.add(lab)
        if len(g) > 1:      # a second, larger ring so both members are visible
            ax.scatter([r["difference"]], [r["rho"]], s=1000, facecolors="none",
                       edgecolors=col, linewidths=2.0, zorder=5)
        name = " = ".join(x["target"] for x in g)
        note = "  (coincident)" if len(g) > 1 else ""
        ax.annotate(
            f"$\\bf{{{name.replace('_', chr(92)+'_')}}}${note}\n"
            f"y = {r['infix']}\n"
            f"diff = {r['difference']:+.4f}\n"
            f"$\\rho(C)$ = {r['rho']:.4f}\n"
            f"best sib {r['best_sib']} = {r['best_sib_R']:+.4f}\n"
            f"$P({B})$ = 1 - {r['fail_B']:.0e}",
            xy=(r["difference"], r["rho"]),
            xytext=(r["difference"] + 0.030 * (1 if r["difference"] < (lo+hi)/2 else -1),
                    r["rho"] + (0.115 if r["rho"] < 0.5 else -0.135)),
            ha="left" if r["difference"] < (lo+hi)/2 else "right",
            fontsize=8.6, family="monospace",
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=col, alpha=0.93),
            arrowprops=dict(arrowstyle="-", color=col, lw=1.1))

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"difference   $V^q(C)\;-\;\max_i V^q(D_i)$"
                  "\n(negative = random rollouts rate the continue node below its best sibling)",
                  fontsize=11)
    ax.set_ylabel(r"recovery rate   $\rho(C)$" "\n"
                  r"P(one uniform random completion from $C$ is exact)",
                  fontsize=11)
    ax.set_title(
        "Linear family: what rollouts prefer at the root, against how often "
        "rollouts find the answer\n"
        f"both axes exact, by backward induction — no sampling. "
        f"At B = {B} rollouts every member recovers with probability 1 to "
        f"machine precision,\nso the per-draw rate is plotted; "
        r"$\rho$ takes only two values here because it depends on the target "
        "only through its support (Lemma 3.2).",
        fontsize=11)
    ax.grid(alpha=0.28)
    ax.legend(fontsize=9, loc="upper center", framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_dir = (Path(args.out_dir).resolve() if args.out_dir
               else exp_root / FAMILY_DIR[rows[0]["family"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_name or "inversion-recovery_scatter"
    png = out_dir / f"{stem}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    (out_dir / f"{stem}.json").write_text(json.dumps(
        {"budget": B, "x": "V_q(C) - max_i V_q(D_i)", "y": "rho(C)",
         "points": rows}, indent=2))
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    main()
