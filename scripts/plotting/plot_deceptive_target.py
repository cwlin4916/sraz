"""The deceptive target of note 02 section 6.1, drawn against its linear decoy.

    mcts_deceptive_target.png   quad_B: y = 6 - 5x + x^2/2 on [-1,1], plotted
                                against the best-fitting 5-token line, plus the
                                residual that separates them.

Why the figure exists. Note 02 section 6.1 asserts that a line explains 99.7%
of this target and calls the remaining 0.3% a trap. Both numbers are variance
ratios, and a variance ratio is not a picture: the reader is asked to accept
that 0.9972 means "visually identical" without being shown it. Panel A shows
it. Panel B shows what is left over, which is the entire quantity UCB must
detect to fund an 11-token derivation over a 5-token one.

The residual is drawn on its own axis rather than as an inset, because its
scale is the point. Sharing panel A's axis would render it flat -- which is
the very compression the figure was drawn to undo.

R^2 is taken from the game's own reward pipeline (`fit_expression`), not
recomputed here, so the annotated number is the one the section 3.3 ladder
reports rather than an independent calculation that happens to agree. The
drawn curve comes from a closed-form least-squares fit, and the script
asserts the two agree before plotting: the sentence is linear in its
constants, so Levenberg-Marquardt and the normal equations reach the same
optimum, and a divergence would mean the oracle stalled rather than solved.

Usage:
    python scripts/plotting/plot_deceptive_target.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sraz.instances.symreg.game import fit_expression
from sraz.instances.symreg.targets import get_target

# Hue order shared with plot_rollout_experiments.py: the target and its decoy
# are categorical identities, not magnitudes.
BLUE, ORANGE, PURPLE = "#3b7dd8", "#e8710a", "#9c27b0"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

# The 5-token linear decoy and the 11-token exact structure, in the grammar's
# prefix notation, exactly as tests/test_symreg_targets.py scores them.
DECOY = "+ * C1 x C0"
EXACT = "+ C0 + * C1 x * C2 * x x"


def _style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_deceptive_target(out: Path):
    t = get_target("quad_B")
    xs, ys = t.xs(), t.ys(t.xs())

    # The authoritative score, straight from the reward pipeline.
    r2_decoy = fit_expression(DECOY, xs, ys, max_nfev=50)
    r2_exact = fit_expression(EXACT, xs, ys, max_nfev=50)

    # The drawn line: closed-form least squares, checked against the pipeline.
    c1, c0 = np.polyfit(xs, ys, 1)
    line = c1 * xs + c0
    ss_res = np.sum((ys - line) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    assert abs((1 - ss_res / ss_tot) - r2_decoy) < 1e-6, (
        f"closed-form fit {1 - ss_res / ss_tot:.6f} disagrees with the "
        f"pipeline's {r2_decoy:.6f}: the inner optimizer did not solve")

    resid = ys - line

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(8.0, 7.2), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.30})

    # -- Panel A: the target and the decoy, on the same axis --
    ax_a.plot(xs, ys, "-", color=BLUE, lw=2.4, zorder=3,
              label=f"target  $y = 6 - 5x + \\frac{{1}}{{2}}x^2$   ($R^2 = {r2_exact:.4f}$ at 11 tokens)")
    ax_a.plot(xs, line, "--", color=ORANGE, lw=2.0, zorder=4,
              label=f"decoy  $C_1x + C_0$   ($R^2 = {r2_decoy:.4f}$ at 5 tokens)")
    ax_a.plot(xs, ys, "o", color=BLUE, ms=3.5, zorder=5, alpha=0.55)
    _style(ax_a)
    ax_a.set_ylabel("$y$", color=INK)
    ax_a.set_title(
        "A   A line captures the target's whole slope; what it misses is a slight bow",
        color=INK, fontsize=11, loc="left", pad=10)
    ax_a.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="upper right")

    # -- Panel B: what is left over --
    ax_b.axhline(0.0, color=MUTED, lw=0.9, ls=":", zorder=2)
    ax_b.fill_between(xs, 0.0, resid, color=PURPLE, alpha=0.18, zorder=3)
    ax_b.plot(xs, resid, "-", color=PURPLE, lw=2.2, zorder=4,
              label="residual  target $-$ decoy")
    _style(ax_b)
    ax_b.set_xlabel("$x$", color=INK)
    ax_b.set_ylabel("residual", color=INK)
    ax_b.set_title(
        f"B   The whole signal: {1 - r2_decoy:.4f} of the variance, a parabola of "
        f"amplitude {resid.max() - resid.min():.2f}",
        color=INK, fontsize=11, loc="left", pad=10)
    ax_b.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="upper center")

    fig.suptitle(
        "Monotonic-exterior target: the trap a search has to see past",
        color=INK, fontsize=13, x=0.10, ha="left", y=0.975)
    fig.text(0.10, 0.025,
             f"41 points on $x \\in [-1,1]$; $R^2$ from the game's reward pipeline "
             f"under a 50-evaluation cap",
             color=MUTED, fontsize=8.5, ha="left")

    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.10)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"  decoy  {DECOY!r}  R^2 = {r2_decoy:.6f}   gap = {1 - r2_decoy:.6f}")
    print(f"  exact  {EXACT!r}  R^2 = {r2_exact:.6f}")
    print(f"  residual range [{resid.min():.4f}, {resid.max():.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("docs/notes/figures"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_deceptive_target(a.out_dir / "mcts_deceptive_target.png")


if __name__ == "__main__":
    main()
