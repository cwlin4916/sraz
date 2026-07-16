"""Figure for note 02 section 2: pure MCTS, R^2 against simulation budget.

Two panels on top, one per target, both drawing the same measurement: greedy
R^2 against the MCTS simulation budget, five episodes per budget. Two panels
underneath drawing the root-edge Q that *decides* each curve, so the figure
explains itself rather than asking the prose to.

The row-B panels are the point. A curve of R^2 against budget says only what
happened; the root table says why, because the emitted expression is settled by
which root edge wins the visit-count argmax. Reading the two rows together is
what separates "search found the answer" from "the tie-break did".

Row B is drawn as a lollipop, not a bar. The load-bearing value there is
Q = 0 exactly -- the constant's edge, and the largest Q on quad_B -- and a bar
of length zero is invisible. A dot sitting on the zero line is not.

R^2 is a single series per panel, so it takes one hue and no legend box: the
panel title names it. The root dots are signed, so they take a diverging
encoding (one hue each side of a neutral zero), never a categorical rainbow --
the sign of Q is the whole content of row B.

Usage:
    python scripts/plotting/plot_pure_mcts_sweep.py
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuses the note's existing validated hues rather than introducing a second
# palette. Light surface: BLUE/ORANGE adjacent dE 26.5 protan, 20.0 tritan;
# normal-vision floor 31.9; both >= 3:1 against the surface.
BLUE, ORANGE = "#3b7dd8", "#e8710a"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

# Row B is signed data: Q >= 0 is "worth entering", Q < 0 is "worth avoiding".
POS, NEG, ZERO = BLUE, ORANGE, "#6b6b6b"
BAND = "#eef3fb"     # background band marking the selected edge

ROOT_BUDGET = 9      # the first collapsed budget; the root table that explains it

PANEL_TITLE = {
    "lin_D": "A1   lin_D:  $y = 2x$   (exact structure is 1 production away)",
    "quad_B": "A2   quad_B:  $y = 6 - 5x + 0.5x^2$   (deceptive)",
}

# Reachable scores worth drawing as references (docs/notes/02.md section 1.3).
# quad_B's 5-token linear cheat scores 0.9972 -- 0.003 from the exact 1.0, and
# so indistinguishable on a linear axis. It is deliberately NOT drawn as a
# separate line: two labels on one pixel would imply a resolution this axis does
# not have. The prose carries it instead.
REFERENCES = {
    "lin_D": [(1.0, "exact: the 3-token '* C1 x'")],
    "quad_B": [(1.0, "exact: the 11-token quadratic"),
               (0.0, "the lone constant 'C0'")],
}


def _style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def cell_for(data, target, n_sims):
    for c in data["cells"]:
        if c["target"] == target and c["n_simulations"] == n_sims:
            return c
    raise KeyError(f"no cell for {target} at {n_sims} simulations")


def draw_r2(ax, data, target):
    cells = sorted((c for c in data["cells"] if c["target"] == target),
                   key=lambda c: c["n_simulations"])
    xs = [c["n_simulations"] for c in cells]
    means = [c["r2_mean"] for c in cells]

    for value, label in REFERENCES[target]:
        ax.axhline(value, color=MUTED, lw=0.8, ls=":", alpha=0.75, zorder=1)
        ax.text(xs[0] - 1.3, value + 0.022, label, color=MUTED, fontsize=7.2,
                va="bottom", ha="left")

    # The five episodes, drawn individually. They differ only by their rollout
    # draws, so a spread of zero is a result -- the scatter must be visible for
    # the reader to see it collapse onto the mean rather than be told it did.
    for c in cells:
        r2 = [ep["r2"] for ep in c["episodes"]]
        ax.plot([c["n_simulations"]] * len(r2), r2, "o", ms=9, mfc="none",
                mec=BLUE, mew=1.1, alpha=0.5, zorder=3)

    ax.plot(xs, means, "-o", color=BLUE, lw=2.0, ms=7, zorder=4,
            markeredgecolor="white", markeredgewidth=1.2)

    ax.set_title(PANEL_TITLE[target], fontsize=10, color=INK, loc="left", pad=8)
    ax.set_xlabel("MCTS simulations per move", fontsize=9, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 1.8, xs[-1] + 1.4)
    ax.set_ylim(-0.16, 1.16)
    _style(ax)


def draw_root(ax, data, target):
    rows = [r for r in cell_for(data, target, ROOT_BUDGET)["root"]
            if r["Q"] is not None]
    rows = sorted(rows, key=lambda r: r["Q"])

    qs = [r["Q"] for r in rows]
    ns = [r["N"] for r in rows]
    ypos = np.arange(len(rows))
    win = int(np.argmax(ns))     # the visit-count argmax IS the move selector

    ax.axhspan(win - 0.45, win + 0.45, color=BAND, zorder=0)
    ax.axvline(0.0, color=MUTED, lw=1.0, zorder=2)

    for y, q in zip(ypos, qs):
        colour = ZERO if q == 0 else (POS if q > 0 else NEG)
        ax.plot([0, q], [y, y], "-", color=colour, lw=2.0, zorder=3)
        ax.plot([q], [y], "o", ms=9, color=colour, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)

    for y, q, n in zip(ypos, qs, ns):
        tag = f"Q={q:+.2f}   N={n}" + ("   ← selected" if y == win else "")
        ax.text(1.42, y, tag, color=INK if y == win else MUTED,
                fontsize=7.6, va="center", ha="right",
                fontweight="bold" if y == win else "normal")

    ax.set_yticks(ypos)
    ax.set_yticklabels([r["production"] for r in rows], fontsize=7.5,
                       fontfamily="monospace")
    ax.set_xlim(-1.18, 1.45)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("backed-up $Q$ at the root edge", fontsize=9, color=MUTED)
    ax.set_title(f"{'B1' if target == 'lin_D' else 'B2'}   root edges after "
                 f"{ROOT_BUDGET} simulations", fontsize=10, color=INK,
                 loc="left", pad=8)
    ax.grid(True, axis="x", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    p = argparse.ArgumentParser(description="Figure: pure MCTS R^2 vs budget")
    p.add_argument("--data", default="docs/notes/figures/pure_mcts_sweep.json")
    p.add_argument("--out", default="docs/notes/figures/pure_mcts_sweep.png")
    args = p.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    targets = data["hyperparameters"]["targets"]
    hp = data["hyperparameters"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

    for col, target in enumerate(targets):
        draw_r2(axes[0][col], data, target)
        draw_root(axes[1][col], data, target)
    axes[0][0].set_ylabel("greedy $R^2$", fontsize=9.5, color=INK)

    # The one thing the curve cannot carry: quad_B's pre-collapse point is a
    # tie-break, not a search success.
    q5 = cell_for(data, "quad_B", 5)
    if abs(q5["r2_mean"]) > 0.5:
        axes[0][1].annotate(
            "every root edge tied at $N{=}1$;\nargmax takes the lowest index,\n"
            "which happens to continue",
            xy=(5.05, q5["r2_mean"] - 0.02), xytext=(6.9, 0.58), fontsize=7.4,
            color=INK, ha="left",
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                            connectionstyle="arc3,rad=0.2"))
        axes[0][1].annotate("visits concentrate on 'C0'",
                            xy=(9.1, 0.03), xytext=(10.2, 0.20), fontsize=7.4,
                            color=INK, ha="left",
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))

    fig.suptitle(
        "Pure MCTS with mean random-rollout leaves: more search, worse expression",
        fontsize=13, color=INK, x=0.5, y=0.985)
    fig.text(0.5, 0.947,
             f"no network (uniform prior, value 0) · leaf = mean of {hp['rollout_n']} random rollouts · "
             f"{hp['n_episodes']} episodes/budget · backup={hp['backup_rule']} · "
             f"c={hp['c_exploration']} · $\\tau_{{select}}$={hp['temperature_select']} · no Dirichlet noise",
             fontsize=8, color=MUTED, ha="center")

    fig.tight_layout(rect=[0, 0, 1, 0.932])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
