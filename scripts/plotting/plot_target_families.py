"""Figures for the two designed target families of note 02, sections 3.1 and 3.2.

One figure per family, each answering the question its section poses.

  symreg_linear_family.png     lin_A..lin_D. A line has no vertex and no
                               discriminant, so section 3.1 partitions it by the
                               only invariants it has: where the root sits
                               relative to the domain, and how c0 and c1 are
                               scaled against one another.

  symreg_quadratic_family.png  quad_A..quad_D, partitioned by the vertex
                               x_v = -c1/2c2 and the discriminant D = c1^2-4c0c2.

Both figures share a layout, because both families pose the same two questions.

  Row A   the four members drawn RAW, one small multiple each, on its own
          y-axis. Raw and per-panel is the only faithful option here: the
          coefficient spread is 1e6 and it lives in the *offset*, so no shared
          axis holds four members (lin_C would flatten the other three onto one
          pixel) and no monotone y-transform repairs an offset spread while
          preserving linear shape. Rescaling y -> a*y is free -- the reward is
          exactly invariant to it -- but shifting y -> y + b is not, and it is
          precisely the offset that separates these members. Note the corollary:
          z-scoring, which shifts, collapses all four linear members onto ONE
          curve (checked: max pairwise spread 4.7e-10). It would erase the
          family it purports to draw.

  Row B   the reachable score as a closed form in the family's invariant, with
          the four members plotted on it. This is what makes the section 3.3
          ladder attributable to a geometric property rather than a table of
          eight numbers.

The closed forms are exact, not fits, and each is pinned against the ladder by
tests/test_symreg_targets.py. On the symmetric 41-point grid the basis
{1, x, x^2 - <x^2>} is orthogonal, which is what makes them closed forms at all:

  linear     R^2 of the 3-token `* C1 x` is clip(1 - rho^2 / <x^2>), where
             rho = |c0/c1| is the distance from the root to the origin. The
             other four ladder rows are CONSTANT down the whole family, so this
             single curve carries everything in it that varies.

  quadratic  R^2 of the 5-token `+ * C1 x C0` is lambda = x_v^2/(x_v^2+kappa),
             kappa = sum((x^2-<x^2>)^2) / (4 sum(x^2)), and R^2 of the 7-token
             `+ * C2 * x x C0` is exactly 1 - lambda. The two rows sum to 1 on
             every member: one number per target, set by the vertex alone.

Colour is assigned by member identity in declaration order and never cycled.
The house hues of plot_rollout_experiments.py are reused unchanged for
continuity; the fourth (crimson) was chosen by running the palette validator,
not by eye. Worst all-pairs CVD separation sits in the 6-8 floor band -- an
inherited property of the existing BLUE/PURPLE pair, not introduced here --
which is legal only with secondary encoding, so every member also carries its
own marker shape and a direct label.

Usage:
    python scripts/plotting/plot_target_families.py
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from sraz.instances.symreg.targets import LINEAR_TARGETS, QUADRATIC_TARGETS  # noqa: E402

# Fixed hue order, assigned by member identity and never cycled. BLUE/ORANGE/
# PURPLE are the house hues of plot_rollout_experiments.py; CRIMSON is the
# fourth slot, validated (light surface, categorical, --pairs all): worst pair
# PURPLE-BLUE dE 7.5 deutan / 9.4 tritan, normal-vision floor 18.0, all >= 3:1
# against the surface. The 6-8 band obliges secondary encoding -- hence the
# per-member marker shapes and direct labels below.
BLUE, ORANGE, PURPLE, CRIMSON = "#3b7dd8", "#e8710a", "#9c27b0", "#d81b60"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"
BAND = "#eef2f7"   # "the invariant lies inside the data domain"

LIN_SLOTS = [
    ("lin_A", "interior root",     BLUE,    "o"),
    ("lin_B", "exterior root",     ORANGE,  "s"),
    ("lin_C", "ill-conditioned",   PURPLE,  "^"),
    ("lin_D", "pure proportional", CRIMSON, "D"),
]
QUAD_SLOTS = [
    ("quad_A", "interior extremum",      BLUE,    "o"),
    ("quad_B", "monotonic exterior",     ORANGE,  "s"),
    ("quad_C", "ill-conditioned scales", PURPLE,  "^"),
    ("quad_D", "root crossing",          CRIMSON, "D"),
]

XS = np.linspace(-1.0, 1.0, 41)
M2 = float(np.mean(XS ** 2))                                        # 0.35 exactly
KAPPA = float(np.sum((XS ** 2 - M2) ** 2) / (4 * np.sum(XS ** 2)))  # 0.069875


def _poly(coeffs) -> str:
    """Render ascending-power coeffs as TeX, dropping 0 terms and unit 1s."""
    out = ""
    for power, c in enumerate(coeffs):
        if c == 0:
            continue
        mag = abs(c)
        if power == 0:
            body = f"{mag:g}"
        else:
            head = "" if mag == 1 else f"{mag:g}"
            body = head + ("x" if power == 1 else "x^2")
        if not out:
            out = f"-{body}" if c < 0 else body
        else:
            out += f" {'-' if c < 0 else '+'} {body}"
    return out or "0"


def _style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.7)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _small_multiple(ax, target, color, marker, title, subtitle):
    """One member, raw, on its own y-axis, with its 41 sample points."""
    y = target.ys(XS)
    ax.plot(XS, y, "-", color=color, lw=2, zorder=3)
    ax.plot(XS[::4], y[::4], marker, color=color, ms=5, mec="white", mew=0.8,
            ls="none", zorder=4)
    ax.set_xlim(-1.06, 1.06)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("$x$", fontsize=9.5, color=INK, labelpad=1)
    ax.set_ylabel("$y$", fontsize=9.5, color=INK, labelpad=2)
    ax.set_title(title, fontsize=9.5, color=INK, pad=26, loc="left")
    ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
            color=MUTED, va="bottom", linespacing=1.45)
    _style(ax)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.tick_params(axis="y", labelsize=7.6)
    return y


def _mark_roots(ax, target, color):
    """Ring every root that lands inside the domain, on the y = 0 line."""
    roots = target.roots_in_domain()
    if not roots:
        return
    ax.axhline(0.0, color=MUTED, lw=0.9, ls=":", zorder=1)
    for r in roots:
        ax.plot([r], [0.0], "o", color="white", mec=color, mew=2, ms=8, zorder=5)


# ---------------------------------------------------------------------------
# Figure: the linear family
# ---------------------------------------------------------------------------

def plot_linear_family(out: Path):
    fig = plt.figure(figsize=(13.0, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.75,
                          wspace=0.34)

    for k, (name, short, color, marker) in enumerate(LIN_SLOTS):
        t = LINEAR_TARGETS[name]
        root = t.real_roots[0]
        shown = "-10^{6}" if abs(root) > 1e3 else f"{root:g}"
        where = "inside" if t.roots_in_domain() else "outside"
        ax = fig.add_subplot(gs[0, k])
        _small_multiple(
            ax, t, color, marker,
            f"A{k + 1}   {name} · {short}",
            f"$y = {_poly(t.coeffs)}$\nroot at $x = {shown}$ — {where} the domain",
        )
        _mark_roots(ax, t, color)

    # lin_C's whole variation is 2e-6 on an offset of 1e3; without explicit
    # ticks matplotlib renders three identical "1000.000" labels.
    fig.axes[2].set_yticks([999.999, 1000.0, 1000.001])
    fig.axes[2].set_ylim(999.9988, 1000.0012)

    # ---- Row B: the closed form ------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    rho = np.concatenate([np.linspace(0, 0.1, 400), np.logspace(-1, 6.6, 800)])
    ax.plot(rho, np.clip(1 - rho ** 2 / M2, -1, 1), "-", color=INK, lw=1.8,
            zorder=3)

    ax.axvspan(-0.02, 1, color=BAND, zorder=0)
    ax.axvline(1.0, color=MUTED, lw=1, ls=":", zorder=1)
    xtr = ax.get_xaxis_transform()   # x in data, y in axes fraction
    ax.text(1.35, 0.30, "root leaves the domain", transform=xtr, rotation=90,
            va="bottom", ha="left", fontsize=8, color=MUTED)

    ax.axhline(0.0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.axhline(-1.0, color=MUTED, lw=1, ls=":", zorder=1)
    ytr = ax.get_yaxis_transform()   # x in axes fraction, y in data
    ax.text(0.012, 0.035, " no better than $\\bar y$", transform=ytr,
            va="bottom", ha="left", fontsize=8.5, color=MUTED)
    ax.text(0.012, -0.965, " clip floor", transform=ytr,
            va="bottom", ha="left", fontsize=8.5, color=MUTED)

    handles = []
    for name, short, color, marker in LIN_SLOTS:
        t = LINEAR_TARGETS[name]
        c0, c1 = t.coeffs
        r = abs(c0 / c1)
        h, = ax.plot([r], [float(np.clip(1 - r ** 2 / M2, -1, 1))], marker,
                     color=color, ms=11, mec="white", mew=1.2, ls="none",
                     zorder=5, label=f"{name} — {short}")
        handles.append(h)

    # Direct labels: the secondary encoding the 6-8 CVD band obliges.
    ax.text(0.006, 0.80, "lin_D\n$\\rho=0$", fontsize=8.5, color=INK,
            ha="left", va="top")
    ax.text(0.33, 0.93, "lin_A  $\\rho=0.25$", fontsize=8.5, color=INK)
    ax.text(3.0, -0.86, "lin_B  $\\rho=5$", fontsize=8.5, color=INK, ha="center")
    ax.text(3e5, -0.86, "lin_C  $\\rho=10^{6}$", fontsize=8.5, color=INK,
            ha="center")

    ax.annotate("", xy=(5, -1.0), xytext=(1e6, -1.0),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1,
                                shrinkA=7, shrinkB=7))
    ax.text(7e3, -0.955, "$10^{5}\\times$ apart in $\\rho$, identical after clipping",
            ha="center", fontsize=8.5, color=MUTED)

    ax.set_xscale("symlog", linthresh=0.1, linscale=1.0)
    ax.set_xlim(-0.02, 3e6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("$\\rho = |c_0/c_1|$ — distance from the root to the origin"
                  "   (symlog: linear below $0.1$, log above)",
                  fontsize=10, color=INK)
    ax.set_ylabel("$R^2$ of the 3-token `* C1 x`", fontsize=10, color=INK)
    ax.set_title("B   The root position alone sets every score that varies in this family",
                 fontsize=12, color=INK, pad=26, loc="left")
    ax.text(0.0, 1.02,
            "curve is the closed form $R^2 = \\mathrm{clip}(1 - \\rho^2/\\langle x^2\\rangle)$ "
            "with $\\langle x^2\\rangle = 0.35$ exactly — not a fit; the other four ladder "
            "rows of §3.3 are constant down the whole family",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    _style(ax)

    fig.suptitle("A   The linear family  $y = c_0 + c_1x$  on  $x \\in [-1,1]$, drawn raw — "
                 "each member on its own $y$-axis",
                 fontsize=12, color=INK, x=0.058, y=0.986, ha="left")

    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                     fontsize=9, bbox_to_anchor=(0.5, -0.035))
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure: the quadratic family
# ---------------------------------------------------------------------------

def plot_quadratic_family(out: Path):
    fig = plt.figure(figsize=(13.0, 7.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.75,
                          wspace=0.34)

    for k, (name, short, color, marker) in enumerate(QUAD_SLOTS):
        t = QUADRATIC_TARGETS[name]
        xv, disc = t.vertex, t.discriminant
        shown = "\\approx 0" if abs(xv) < 1e-3 else f"= {xv:g}"
        where = "inside" if t.vertex_in_domain() else "outside"
        n_roots = len(t.roots_in_domain())
        roots = (f"{n_roots} root{'s' if n_roots != 1 else ''} inside"
                 if n_roots else "no roots inside")
        ax = fig.add_subplot(gs[0, k])
        _small_multiple(
            ax, t, color, marker,
            f"A{k + 1}   {name} · {short}",
            f"$y = {_poly(t.coeffs)}$\n"
            f"$x_v {shown}$ — {where} · $\\Delta = {disc:g}$ · {roots}",
        )
        _mark_roots(ax, t, color)
        if t.vertex_in_domain():
            yv = float(t.ys(np.array([xv]))[0])
            ax.plot([xv], [yv], "v", color="white", mec=color, mew=2, ms=9,
                    zorder=5)

    # ---- Row B: the closed form ------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    grid = np.concatenate([np.linspace(0, 0.05, 300),
                           np.logspace(np.log10(0.05), np.log10(30), 700)])
    lam = grid ** 2 / (grid ** 2 + KAPPA)
    ax.plot(grid, lam, "-", color=INK, lw=1.8, zorder=3)
    ax.plot(grid, 1 - lam, "--", color=MUTED, lw=1.8, zorder=3)

    ax.text(0.004, 0.055, "$\\lambda$  =  `+ * C1 x C0`  (5 tokens)",
            fontsize=9, color=INK)
    ax.text(0.004, 0.905, "$1-\\lambda$  =  `+ * C2 * x x C0`  (7 tokens)",
            fontsize=9, color=MUTED)

    ax.axvspan(-0.01, 1.0, color=BAND, zorder=0)
    xtr = ax.get_xaxis_transform()
    ax.axvline(1.0, color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(0.87, 0.30, "vertex leaves the domain ", transform=xtr, rotation=90,
            va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.axvline(np.sqrt(KAPPA), color=MUTED, lw=1, ls=":", zorder=1)
    ax.text(np.sqrt(KAPPA) * 0.93, 0.04,
            "$\\sqrt{\\kappa}=0.264$", transform=xtr,
            rotation=90, va="bottom", ha="right", fontsize=8, color=MUTED)

    handles = []
    for name, short, color, marker in QUAD_SLOTS:
        t = QUADRATIC_TARGETS[name]
        c0, c1, c2 = t.coeffs
        axv = abs(t.vertex)
        lam_k = axv ** 2 / (axv ** 2 + KAPPA)
        ax.plot([axv, axv], [lam_k, 1 - lam_k], "-", color=color, lw=1,
                alpha=0.4, zorder=2)
        h, = ax.plot([axv], [lam_k], marker, color=color, ms=11, mec="white",
                     mew=1.2, ls="none", zorder=5, label=f"{name} — {short}")
        ax.plot([axv], [1 - lam_k], marker, color=color, ms=11, mec="white",
                mew=1.2, ls="none", alpha=0.4, zorder=5)
        handles.append(h)

    ax.text(0.0045, 0.50, "quad_C\n$x_v \\approx 0$\n$\\lambda = 0.000$",
            fontsize=8.5, color=INK, ha="left", va="center")
    ax.annotate("quad_D\n$\\lambda=0.125$", xy=(0.1, 0.125), xytext=(0.028, 0.34),
                fontsize=8.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=9))
    ax.annotate("quad_A\n$\\lambda=0.472$\n$1-\\lambda=0.528$",
                xy=(0.25, 0.472), xytext=(0.62, 0.14),
                fontsize=8.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=9))
    ax.annotate("quad_B  $\\lambda=0.9972$\nthe $0.9972$ linear cheat",
                xy=(5, 0.9972), xytext=(1.25, 0.68), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=9))

    ax.set_xscale("symlog", linthresh=0.05, linscale=1.0)
    ax.set_xlim(-0.01, 30)
    ax.set_ylim(-0.07, 1.16)
    ax.set_xlabel("$|x_v| = |c_1 / 2c_2|$ — distance from the vertex to the origin"
                  "   (symlog: linear below $0.05$, log above)",
                  fontsize=10, color=INK)
    ax.set_ylabel("$R^2$ of the two-term structures", fontsize=10, color=INK)
    ax.set_title("B   The vertex alone splits the variance, and the two two-term rows sum to $1$",
                 fontsize=12, color=INK, pad=26, loc="left")
    ax.text(0.0, 1.02,
            "curves are the closed form $\\lambda = x_v^2/(x_v^2+\\kappa)$ with "
            "$\\kappa = 0.069875$ exactly — not fits; solid marker $=\\lambda$, "
            "faded marker $=1-\\lambda$, and each member's pair sums to $1$",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    _style(ax)

    fig.suptitle("A   The quadratic family  $y = c_0 + c_1x + c_2x^2$  on  $x \\in [-1,1]$, "
                 "drawn raw — each member on its own $y$-axis",
                 fontsize=12, color=INK, x=0.058, y=0.986, ha="left")

    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
                     fontsize=9, bbox_to_anchor=(0.5, -0.035))
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("docs/notes/figures"))
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_linear_family(a.out_dir / "symreg_linear_family.png")
    plot_quadratic_family(a.out_dir / "symreg_quadratic_family.png")


if __name__ == "__main__":
    main()
