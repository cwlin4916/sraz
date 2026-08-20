#!/usr/bin/env python3
"""Why the search avoids the one branch that contains the answer.

Two exact quantities about the continue node C = `+ S S`, drawn together
because the point is the contrast between them:

LEFT  the difference between C's value and the best of its siblings at the
      same level of the MDP, under V^q -- the expected reward of a uniform
      random completion, which is exactly what a rollout-fed mean-backup search
      estimates. The root's children are C and the three immediately-terminal
      decoys D1 = `C0`, D2 = `* C1 x`, D3 = `* C2 * x x`, so "same level" means
      depth 1, four nodes. A terminal's V^q is just its reward R.

RIGHT the probability that B uniform random completions from C hit an exact
      expression at least once, 1 - (1 - rho(C))^B, with B = 100 marked. rho(C)
      is exact, from the same backward induction, so this curve is closed-form
      rather than sampled.

Input is the tree JSON that `09-family-depth2-trees.py` writes, so the numbers
on the plot are the numbers in `tree_<target>_depth2.png` -- no recomputation,
no chance of drift.

    .venv/bin/python Claude-scripts/11-inversion-vs-recovery.py --target lin_A
    .venv/bin/python Claude-scripts/11-inversion-vs-recovery.py --target quad_D
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

# which experiment folder holds which family's trees
FAMILY_DIR = {"linear": "8-17", "quadratic": "8-20"}
BUDGET_MARK = 100

C_CONT = "#4a7fb5"      # the continue node C = `+ S S`
C_CURVE = "#2e8b57"     # the recovery curve
C_BAD = "#c0392b"       # the sibling that beats C on V^q


def load_level1(tree_json: Path) -> tuple[dict, list[dict]]:
    d = json.loads(tree_json.read_text())
    nodes = d["nodes"]
    level1 = [n for n in nodes if n["depth"] == 1]
    cont = [n for n in level1 if not n["terminal"]]
    if len(cont) != 1:
        raise SystemExit(f"expected exactly one non-terminal at depth 1, "
                         f"got {[n['expr'] for n in cont]}")
    return d, level1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="lin_A")
    ap.add_argument("--exp-root", default=str(REPO / "Claude-experiments"))
    ap.add_argument("--budget", type=int, default=BUDGET_MARK)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    t = get_target(args.target)
    exp_dir = Path(args.exp_root) / FAMILY_DIR[t.family] / args.target
    tree_json = exp_dir / f"tree_{args.target}_depth2.json"
    if not tree_json.exists():
        raise SystemExit(f"missing {tree_json}\nrun 09-family-depth2-trees.py "
                         f"--family {t.family} first")
    d, level1 = load_level1(tree_json)

    cont = next(n for n in level1 if not n["terminal"])
    sibs = [n for n in level1 if n["terminal"]]
    # A terminal's V^q equals its V* equals its R, so the sibling maximum is
    # the same under either value function; assert rather than assume.
    for n in sibs:
        assert abs(n["V_q"] - n["V_star"]) < 1e-12, n["expr"]
    best_sib = max(sibs, key=lambda n: n["V_star"])
    sib_max = best_sib["V_star"]

    diff_vq = cont["V_q"] - sib_max
    diff_vstar = cont["V_star"] - sib_max
    rho = cont["rho"]
    B = args.budget
    fail_B = (1.0 - rho) ** B          # keep the tail; 1 - p_B underflows to 0
    p_B = 1.0 - fail_B

    print(f"=== {args.target}:  y = {d['infix']} ===")
    print(f"continue node      : {cont['expr']}")
    print(f"  V^q  = {cont['V_q']:.10f}")
    print(f"  V*   = {cont['V_star']:.10f}")
    print(f"  rho  = {rho:.10f}")
    print(f"best sibling       : {best_sib['expr']}  "
          f"V^q = V* = R = {sib_max:.10f}")
    print(f"difference vs best sibling")
    print(f"  under V^q  : {diff_vq:+.10f}   <- what the search estimates")
    print(f"  under V*   : {diff_vstar:+.10f}   <- the truth")
    print(f"  inversion  : {'YES' if diff_vq < 0 < diff_vstar else 'no'}")
    print(f"P(exact within {B} random rollouts from {cont['expr']}):")
    print(f"  P       = 1 - {fail_B:.6e}   (indistinguishable from 1 in float64)")
    print(f"  failure = (1 - rho)^{B} = {fail_B:.6e}")
    for q in (0.5, 0.95, 0.99):
        k = int(np.ceil(np.log(1 - q) / np.log(1 - rho)))
        print(f"  rollouts for {q:.0%}: {k}")

    # ---------------- draw -------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.6, 6.0),
                                   gridspec_kw={"width_ratios": [1.18, 1.0]})

    # ---- left: level-1 V^q and the difference ----------------------------
    order = [cont] + sorted(sibs, key=lambda n: -n["V_q"])
    x = np.arange(len(order))
    cols = [C_CONT] + [C_BAD if n is best_sib else "0.72" for n in order[1:]]
    axL.bar(x, [n["V_q"] for n in order], 0.56, color=cols,
            edgecolor="black", linewidth=0.7)
    for xi, n in zip(x, order):
        v = n["V_q"]
        axL.text(xi, v + (0.022 if v >= 0 else -0.05), f"{v:.4f}",
                 ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=10, fontweight="bold")
    axL.axhline(sib_max, color=C_BAD, ls="--", lw=1.6,
                label=f"best sibling  {best_sib['expr']}  =  {sib_max:.4f}")
    axL.axhline(0, color="black", lw=0.8)

    # the difference, as an annotated span on the continue node
    axL.annotate("", xy=(0, cont["V_q"]), xytext=(0, sib_max),
                 arrowprops=dict(arrowstyle="<->", color=C_CONT, lw=2.6,
                                 shrinkA=0, shrinkB=0))
    axL.text(0.10, (cont["V_q"] + sib_max) / 2,
             f"difference\n{diff_vq:+.4f}", fontsize=13, fontweight="bold",
             color=C_CONT, va="center")

    axL.set_xticks(x)
    axL.set_xticklabels([f"{n['expr']}\n{n['action']['name']}" for n in order],
                        family="monospace", fontsize=10)
    axL.set_ylabel(r"$V^q$   (expected $R^2$ of a uniform random completion)")
    lo = min(n["V_q"] for n in order)
    axL.set_ylim(min(-0.2, lo - 0.15), max(1.0, sib_max + 0.28))
    axL.set_title(
        f"$V^q$ at depth 1: {cont['expr']} against its three siblings\n"
        f"the only non-terminal child, and the only one whose subtree "
        f"contains an exact expression",
        fontsize=10.5)
    axL.legend(fontsize=9.2, loc="upper right", framealpha=0.95)
    axL.grid(axis="y", alpha=0.25)
    axL.spines[["top", "right"]].set_visible(False)

    # ---- right: recovery probability from the continue node ---------------
    Bs = np.arange(1, 1001)
    P = 1.0 - (1.0 - rho) ** Bs
    axR.semilogx(Bs, P, "-", color=C_CURVE, lw=2.4,
                 label=r"$1-(1-\rho(C))^{B}$")
    axR.axvline(B, color="black", ls=":", lw=1.4)
    axR.plot([B], [p_B], "o", color=C_BAD, ms=9, zorder=5)
    axR.annotate(f"$B = {B}$\n$P = 1 - {fail_B:.1e}$",
                 xy=(B, p_B), xytext=(B * 0.10, 0.62),
                 arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.6),
                 fontsize=11, fontweight="bold", color=C_BAD)
    k95 = int(np.ceil(np.log(0.05) / np.log(1 - rho)))
    axR.axvline(k95, color="0.45", ls="--", lw=1.1)
    axR.text(k95 * 1.12, 0.16, f"$K_{{95}} = {k95}$", fontsize=9.4, color="0.30")
    axR.set_xlabel("B  =  uniform random completions drawn from "
                   f"{cont['expr']}   (log scale)")
    axR.set_ylabel("P(at least one exact expression)")
    axR.set_ylim(0, 1.045)
    axR.set_title(
        f"random rollout from {cont['expr']} finds the exact expression\n"
        rf"$\rho(C) = {rho:.6f}$ exactly, so this is closed form",
        fontsize=10.5)
    axR.grid(alpha=0.3, which="both")
    axR.legend(fontsize=9.5, loc="lower right")
    axR.spines[["top", "right"]].set_visible(False)

    rel = "below" if diff_vq < 0 else "above"
    fig.suptitle(
        f"{args.target}   y = {d['infix']}   —   the branch that rollouts rate "
        f"{rel} its best sibling is the branch that contains the answer\n"
        f"left: $V^q$({cont['expr']}) sits {abs(diff_vq):.4f} {rel} "
        f"{best_sib['expr']}   |   right: {B} random rollouts from "
        f"{cont['expr']} fail with probability only "
        f"$(1-\\rho)^{{{B}}} = {fail_B:.1e}$",
        fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out_dir = Path(args.out_dir).resolve() if args.out_dir else exp_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"inversion-recovery_{args.target}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    (out_dir / f"inversion-recovery_{args.target}.json").write_text(json.dumps({
        "target": args.target, "infix": d["infix"],
        "continue_node": {"expr": cont["expr"], "V_q": cont["V_q"],
                          "V_star": cont["V_star"], "rho": rho},
        "siblings": [{"expr": n["expr"], "name": n["action"]["name"],
                      "R": n["R"], "V_q": n["V_q"], "V_star": n["V_star"]}
                     for n in sibs],
        "best_sibling": {"expr": best_sib["expr"], "value": sib_max},
        "difference_under_V_q": diff_vq,
        "difference_under_V_star": diff_vstar,
        "inversion": bool(diff_vq < 0 < diff_vstar),
        "budget": B, "p_exact_within_budget": p_B,
        "p_failure_within_budget": fail_B,
        "rollouts_for": {str(q): int(np.ceil(np.log(1-q)/np.log(1-rho)))
                         for q in (0.5, 0.95, 0.99)},
        "source_tree_json": str(tree_json.relative_to(REPO)),
    }, indent=2))
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    main()
