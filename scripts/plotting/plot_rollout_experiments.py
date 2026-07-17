"""Figures for the rollout experiments of note 02 section 8.

Two questions, one figure each.

  mcts_deceptive_trap.png    quad_B: does budget alone escape the 0.9972
                             linear cheat? Plotted as 1 - R^2 on a log y-axis,
                             because the gap the target was built to test
                             (0.9972 vs 1.0) is one pixel on a linear axis --
                             the limitation note 02 section 6 already called out.

  mcts_rollout_sparsity.png  quad_A: greedy R^2 vs budget for pure UCT against
                             rollouts under both aggregations.

Both figures carry three series, and the third is the point: `rollout_n=5`
alone is not a condition, because `rollout_mode` decides whether rollouts help
or hurt. Drawing only mean-aggregated rollouts would have reported that
rollouts make the search worse, which is true and misleading.

Series are categorical (identity, not magnitude), so they take a fixed hue
order, validated for CVD separation rather than eyeballed. Budget is the
ordered magnitude and rides the x-axis, log-scaled: the arms differ by three
decades, so a linear x compresses every rollout point into the left edge.

Usage:
    python scripts/plotting/plot_rollout_experiments.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed hue order, assigned by series identity and never cycled.
# Validated (light surface, categorical): worst adjacent CVD dE 26.5 protan,
# 20.0 tritan; normal-vision floor 31.9; all >= 3:1 against the surface.
BLUE, ORANGE, PURPLE = "#3b7dd8", "#e8710a", "#9c27b0"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

SERIES = [
    ("rollout0",      "pure UCT — no rollouts",           BLUE,   "o", "-"),
    ("rollout5_mean", "5 rollouts, averaged",             ORANGE, "s", "--"),
    ("rollout5_max",  "5 rollouts, best-of",              PURPLE, "^", "-"),
]

# 1 - R^2 = 0 is off a log axis; floor it one decade below the smallest
# resolvable gap so an exact solve is visibly *at the floor*, not missing.
EPS = 1e-4


def _style(ax):
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.7)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_deceptive_trap(data, out: Path):
    d = data["quad_B"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))

    for key, label, color, marker, ls in SERIES:
        s = d[key]
        x = np.array(s["n_sim"], float)
        # Median, not mean: across seeds the rollout arms are bimodal (solve, or
        # fall back to a cheat), so a mean would report a value no run returns.
        y = np.array(s.get("r2_median", s["r2_mean"]), float)
        gap = np.maximum(1.0 - y, EPS)
        ax.plot(x, gap, ls, color=color, marker=marker, ms=6, lw=2,
                label=label, mec="white", mew=0.8, zorder=3)

    tr = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    ax.axhline(1.0 - 0.9972, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.985, 1.0 - 0.9972, "the 5-token linear cheat  $R^2 = 0.9972$ ",
            transform=tr, va="bottom", ha="right", fontsize=8.5, color=MUTED)
    ax.axhline(EPS, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.985, EPS, "exact  $R^2 = 1$  (floored at $\\epsilon = 10^{-4}$) ",
            transform=tr, va="bottom", ha="right", fontsize=8.5, color=MUTED)

    ax.set_yscale("log")
    ax.set_ylim(2.5e-5, 3.0)
    ax.set_xlim(3.5, 3e5)
    ax.set_xlabel("MCTS simulations per action  ($n_{\\mathrm{sim}}$, log scale)",
                  fontsize=10, color=INK)
    ax.set_ylabel("$1 - R^2_{\\mathrm{greedy}}$   (log; lower is better)",
                  fontsize=10, color=INK)
    ax.set_title("Monotonic-exterior target: budget alone never springs the trap",
                 fontsize=12.5, color=INK, pad=26, loc="left")
    ax.text(0.0, 1.02,
            "pure UCT holds the 0.9972 cheat from 50 to 102,400 simulations; "
            "max-aggregated rollouts solve it at 5",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


def plot_rollout_sparsity(data, out: Path):
    d = data["quad_A"]
    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    XLIM = (3.5, 6e4)

    handles = []
    for key, label, color, marker, ls in SERIES:
        s = d[key]
        x = np.array(s["n_sim"], float)
        y = np.array(s.get("r2_median", s["r2_mean"]), float)
        ln, = ax.plot(x, y, ls, color=color, marker=marker, ms=6, lw=2,
                      label=label, mec="white", mew=0.8, zorder=3)
        handles.append(ln)

    # Reference lines are labelled at the right edge, past every series' last
    # point (12,800), which is the only region of this panel that stays clear.
    tr = ax.get_yaxis_transform()
    ax.axhline(1.0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.985, 1.005, "exact: $C_0 + C_1x + C_2x^2$ ", transform=tr,
            va="bottom", ha="right", fontsize=8.5, color=MUTED)
    ax.axhline(0.4721, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.985, 0.4785, "linear plateau  $0.4721$ ", transform=tr,
            va="bottom", ha="right", fontsize=8.5, color=MUTED)

    ax.set_ylim(-0.14, 1.16)
    ax.set_xlim(*XLIM)
    ax.set_xlabel("MCTS simulations per action  ($n_{\\mathrm{sim}}$, log scale)",
                  fontsize=10, color=INK)
    ax.set_ylabel("median greedy $R^2$", fontsize=10, color=INK)
    ax.set_title("A   Interior-extremum target: rollouts help only if aggregated optimistically",
                 fontsize=12, color=INK, pad=26, loc="left")
    ax.text(0.0, 1.02,
            "median over 15 rollout seeds; pure UCT is deterministic (1 run)",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    _style(ax)

    # Panel B. The median in A is a step function of this panel crossing 50%,
    # so its zigzag is bimodality, not noise in the estimate.
    for key, label, color, marker, ls in SERIES:
        s = d[key]
        x = np.array(s["n_sim"], float)
        f = (np.array(s["frac_solved"], float) if "frac_solved" in s
             else (np.array(s["r2_mean"], float) > 0.999).astype(float))
        axb.plot(x, 100 * f, ls, color=color, marker=marker, ms=6, lw=2,
                 mec="white", mew=0.8, zorder=3)

    axb.axhline(50, color=MUTED, lw=1, ls=":", zorder=1)
    axb.text(0.985, 51.5, "50% — where A's median flips ", transform=axb.get_yaxis_transform(),
             va="bottom", ha="right", fontsize=8.5, color=MUTED)
    axb.set_ylim(-7, 112)
    axb.set_xlim(*XLIM)
    axb.set_xlabel("MCTS simulations per action  ($n_{\\mathrm{sim}}$, log scale)",
                   fontsize=10, color=INK)
    axb.set_ylabel("% of seeds reaching $R^2 = 1$", fontsize=10, color=INK)
    axb.set_title("B   solve rate", fontsize=12, color=INK, pad=26, loc="left")
    axb.text(0.0, 1.02, "pure UCT: 0% until 12,800, then 100%",
             transform=axb.transAxes, fontsize=9, color=MUTED, va="bottom")
    _style(axb)

    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                     fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path,
                   default=Path("Claude-research/figures/rollout_experiments.json"))
    p.add_argument("--out-dir", type=Path, default=Path("Claude-research/figures"))
    a = p.parse_args()

    data = json.loads(a.data.read_text())
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_deceptive_trap(data, a.out_dir / "mcts_deceptive_trap.png")
    plot_rollout_sparsity(data, a.out_dir / "mcts_rollout_sparsity.png")


if __name__ == "__main__":
    main()
