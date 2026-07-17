"""Figures for the two designed target families of note 02, sections 1.1 and 1.2.

Four figures, two per family: a family's members and its scores answer
different questions and do not fit one canvas.

  symreg_linear_family.png     FIGURE 1a. lin_A..lin_D drawn raw. A line has no
                               vertex and no discriminant, so section 1.1
                               partitions it by the only invariants it has:
                               where the root sits relative to the domain, and
                               how c0 and c1 are scaled against one another.

  symreg_linear_scores.png     FIGURE 1b. The reachable score of EVERY ladder
                               structure, as a closed form in rho = |c0/c1|.
                               One curve per candidate expression, the four
                               members marked on each: section 1.3's whole
                               linear block, drawn rather than tabulated.

  symreg_quadratic_family.png  FIGURE 2a. quad_A..quad_D drawn raw, partitioned
                               by the vertex x_v = -c1/2c2 and the discriminant
                               D = c1^2-4c0c2 -- the invariants a parabola has
                               and a line does not.

  symreg_quadratic_scores.png  FIGURE 2b. Every ladder structure again, but in
                               TWO panels rather than one axis: this family has
                               two reward-visible degrees of freedom, and no
                               single axis holds all seven rows honestly. Panel
                               A is the five rows the vertex settles; panel B is
                               the two it cannot, priced against the offset.

Figures 1a and 2a draw the members RAW, one small multiple each, on its own
y-axis. Raw and per-panel is the only faithful option here: the coefficient
spread is 1e6 and it lives in the *offset*, so no shared axis holds four members
(lin_C would flatten the other three onto one pixel) and no monotone
y-transform repairs an offset spread while preserving linear shape. Rescaling
y -> a*y is free -- the reward is exactly invariant to it -- but shifting
y -> y + b is not, and it is precisely the offset that separates these members.
Note the corollary: z-scoring, which shifts, collapses all four linear members
onto ONE curve (checked: max pairwise spread 4.7e-10). It would erase the
family it purports to draw.

WHY FIGURE 1b IS SEVEN CURVES AND NOT ONE
-----------------------------------------
The old panel B drew a single structure, `* C1 x`, on the grounds that the rest
of the linear ladder column is constant. Five of the seven are; TWO vary, and
the reason is one line of algebra worth drawing rather than asserting.

Write the target as y = c0 + c1 x. On the symmetric grid <x> = <x^3> = 0, so
y - ybar = c1 x and SS_tot = c1^2 sum(x^2). Now split the ladder by whether a
structure carries a free constant:

  WITH an intercept   the free C0 absorbs c0 exactly, whatever c0 is. The fit,
                      and hence R^2, cannot depend on c0 at all -- so the row is
                      FLAT in rho. Four rows: C0 (0), C1x+C0 (1), C2x^2+C0 (0),
                      C0+C1x+C2x^2 (1).

  WITHOUT one         the structure must pay for c0 out of its residual, so rho
                      enters. Two rows, and both are exact:

                        `* C1 x`      a*=1 (x _|_ 1), residual = c0
                                      R^2 = clip(1 - rho^2/<x^2>)
                        `* C2 * x x`  x is odd, x^2 even, so x _|_ x^2: the x
                                      component is unreachable AND the constant
                                      is paid for too
                                      R^2 = clip(-kappa2 * rho^2),
                                      kappa2 = (1 - <x^2>^2/<x^4>)/<x^2>

That is the family's single degree of freedom made visible: rho moves exactly
the two intercept-free rows and nothing else. Each closed form is pinned
against the reward pipeline by tests/test_symreg_targets.py.

The sinusoid is the one row with no closed form. Every other structure enters
its constants linearly, so its fit is linear least squares in disguise; the
sinusoid's frequency C4 is not, the 50-evaluation cap stalls on it, and it
returns the clip floor. It is drawn MEASURED and marked as such -- a property of
the oracle, not of the structure.

COLOUR
------
Two channels, two meanings, never mixed. Colour+shape is assigned by MEMBER
identity in declaration order and never cycled; the house hues of
plot_rollout_experiments.py are reused unchanged for continuity, and the fourth
(crimson) was chosen by running the palette validator, not by eye. Worst
all-pairs CVD separation sits in the 6-8 floor band -- an inherited property of
the existing BLUE/PURPLE pair, not introduced here -- which is legal only with
secondary encoding, so every member also carries its own marker shape and a
direct label. STRUCTURE identity, in figure 1b, therefore cannot use colour: it
is carried by ink value and dash pattern instead, which leaves the member hues
unambiguous where 24 markers land on seven curves. Figure 2b inherits that
grammar unchanged, including each structure's ink and dash: `* C1 x` stays INK
solid and `* C2 * x x` STEEL solid across both figures, so a row can be tracked
between them. The one thing figure 2b re-reads is which rows get the heavy solid
weight -- the rows that MOVE, which this family swaps.

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
from matplotlib.lines import Line2D

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
STEEL = "#6f7780"  # second ink value, for the structure channel of figure 1b
TEAL = "#00796b"   # reserved for the one MEASURED row (the sinusoid)

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
M4 = float(np.mean(XS ** 4))                                        # 0.220325 exactly
VAR_X2 = M4 - M2 ** 2                                               # 0.097825 exactly
KAPPA = float(np.sum((XS ** 2 - M2) ** 2) / (4 * np.sum(XS ** 2)))  # 0.069875
KAPPA2 = (1.0 - M2 ** 2 / M4) / M2                                  # 1.2685805...
KAPPA2_M2 = KAPPA2 * M2                                             # 0.444003...

RHO_ZERO = float(np.sqrt(M2))            # 0.5916  `* C1 x` crosses R^2 = 0
RHO_FLOOR = float(np.sqrt(2 * M2))       # 0.8367  `* C1 x` reaches the clip floor
RHO_FLOOR_Q = float(1 / np.sqrt(KAPPA2))  # 0.8879  `* C2 * x x` reaches it


# --- The linear ladder: R^2 of every candidate structure, as a curve in rho ---
#
# `flat` is None for the two intercept-free rows (they vary) and the constant
# value otherwise. `dodge` separates rows that coincide EXACTLY -- (C0,
# C2x^2+C0) at 0 and (C1x+C0, C0+C1x+C2x^2) at 1 -- and is applied to flat rows
# only, never to a varying curve, because those carry the cited crossings. The
# dodge is disclosed on the figure.

def r2_linear_term(rho):
    """`* C1 x`: pays for c0 out of its residual."""
    return np.clip(1.0 - rho ** 2 / M2, -1.0, 1.0)


def r2_quadratic_term(rho):
    """`* C2 * x x`: pays for c0 AND cannot reach the odd component."""
    return np.clip(-KAPPA2 * rho ** 2, -1.0, 1.0)


LADDER = [
    # key,    TeX,                     tokens, flat, dodge, ls,           colour, lw
    ("c0",    r"$C_0$",                     1,  0.0, -0.020, (0, (6, 3)),      STEEL, 1.5),
    ("c1x",   r"$C_1x$",                    3, None,  0.0,   "solid",          INK,   2.3),
    ("c2x2",  r"$C_2x^2$",                  5, None,  0.0,   "solid",          STEEL, 2.3),
    ("c1x_0", r"$C_1x + C_0$",              5,  1.0, -0.020, (0, (6, 3)),      STEEL, 1.5),
    ("c2x2_0", r"$C_2x^2 + C_0$",           7,  0.0,  0.020, (0, (1.5, 2.5)),  STEEL, 1.5),
    ("full",  r"$C_0 + C_1x + C_2x^2$",    11,  1.0,  0.020, (0, (1.5, 2.5)),  STEEL, 1.5),
    ("sin",   r"$C_3\sin(C_4x)$",           6, -1.0,  0.0,   (0, (4, 2, 1, 2)), TEAL,  1.8),
]
CURVE = {row[0]: row for row in LADDER}


def ladder_r2(key, rho):
    """R^2 of one ladder structure at rho, before dodging. Exact except `sin`."""
    if key == "c1x":
        return r2_linear_term(rho)
    if key == "c2x2":
        return r2_quadratic_term(rho)
    return np.full_like(np.asarray(rho, dtype=float), CURVE[key][3])


# --- The quadratic ladder: the same seam as figure 1b, read the other way ----
#
# The linear family has ONE reward-visible degree of freedom, so figure 1b puts
# all seven rows on one axis. This family has TWO, and no axis holds all seven.
# Which rows move, and in what, is settled by the same fact figure 1b turns on
# -- whether the structure carries a free constant -- with the membership of the
# two sets exactly swapped.
#
# Write y = c0 + c1x + c2x^2. On the symmetric grid x _|_ 1 and x _|_ (x^2 -
# <x^2>), so with ybar = c0 + c2<x^2> the variance splits cleanly:
#
#   s^2 = <(y - ybar)^2> = c1^2<x^2> + c2^2 V,        V = <x^4> - <x^2>^2
#
# Call the first share LAMBDA. The member's scale divides out of it, leaving a
# closed form in the vertex x_v = -c1/2c2 alone:
#
#   lambda = c1^2<x^2>/s^2 = x_v^2/(x_v^2 + kappa),   kappa = V/(4<x^2>)
#
#   WITH an intercept   the free C0 absorbs c0 exactly, so the fit never sees
#                       the offset and the vertex settles the row: C1x+C0 scores
#                       lambda, C2x^2+C0 scores 1-lambda, C0 scores 0, the
#                       11-token form scores 1. Four closed forms in x_v, and
#                       the two that move sum to 1. This is PANEL A.
#
#   WITHOUT one         the row pays for the offset out of its residual -- and
#                       the offset is a second coordinate, invisible to x_v. Two
#                       rows, both exact against the row's OWN normalized offset
#                       (normalized by s, which is what makes them scale-free):
#
#                         `* C1 x`      span {x}: reaches no constant at all, so
#                                       it pays the mean, whole
#                                       R^2 = clip(lambda - omega^2),
#                                       omega = ybar/s
#                         `* C2 * x x`  span {x^2}: x^2 is NOT orthogonal to 1,
#                                       so it absorbs part of the intercept --
#                                       but only part, and the rest is charged at
#                                       a fixed rate
#                                       R^2 = clip((1-lambda) - kappa2<x^2> omega0^2),
#                                       omega0 = c0/s,  kappa2<x^2> = 0.444
#
# Read the two laws with their panel-A counterparts (lambda and 1-lambda) and
# each says one thing: A ROW WITHOUT AN INTERCEPT SCORES WHAT ITS COUNTERPART
# SCORES, MINUS A QUADRATIC IN THE NORMALIZED OFFSET. That difference is PANEL
# B, and it is one curve per structure -- the second coordinate enters through
# the offset and nowhere else.
#
# Both laws are UNIVERSAL. Put lambda = 1 (c2 = 0: a line) and they collapse to
# figure 1b's two falling curves exactly -- omega^2 = rho^2/<x^2> and
# kappa2<x^2>*omega0^2 = kappa2*rho^2. Figure 1b IS the lambda = 1 slice of
# panel B, which is why panel B carries the linear members too. Pinned against
# the reward pipeline, both families, by tests/test_symreg_targets.py.

def lam_of(xv):
    """Linear share of the variance: a closed form in the vertex alone."""
    return xv ** 2 / (xv ** 2 + KAPPA)


def price_linear_term(omega):
    """R^2 that `* C1 x` gives up by having no intercept: it pays the mean."""
    return omega ** 2


def price_quadratic_term(omega0):
    """R^2 that `* C2 * x x` gives up: x^2 absorbs part of c0, at a fixed rate."""
    return KAPPA2_M2 * omega0 ** 2


def reward_coords(target):
    """(lambda, omega, omega0) -- a member's reward-visible coordinates.

    lambda is the linear share of the variance; omega = ybar/s and omega0 = c0/s
    are the mean and the intercept in units of the member's own spread on the
    grid. Defined for lines and parabolas alike -- a line is lambda = 1.
    """
    c0, c1, c2 = (list(target.coeffs) + [0.0, 0.0, 0.0])[:3]
    y = target.ys(XS)
    s = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    lam = c1 ** 2 * M2 / (c1 ** 2 * M2 + c2 ** 2 * VAR_X2)
    return lam, float(y.mean()) / s, c0 / s


# The five rows panel A can draw: everything the vertex alone determines. Dodge
# and dash follow figure 1b's convention exactly -- flat rows are dodged and
# dashed, the rows that MOVE are solid, heavy and never dodged -- even though
# this family swaps which structures belong to which set. Here the two moving
# rows only TOUCH the flat ones asymptotically (lambda -> 0 as x_v -> 0, -> 1 as
# x_v -> inf), so the dodge buys separation at the ends alone.
QUAD_LADDER = [
    # key,     TeX,                       tokens, flat, dodge,  ls,               colour, lw
    ("c0",     r"$C_0$",                       1,  0.0, -0.020, (0, (6, 3)),      STEEL, 1.5),
    ("c1x_0",  r"$C_1x + C_0$",                5, None,  0.0,   "solid",          INK,   2.3),
    ("c2x2_0", r"$C_2x^2 + C_0$",              7, None,  0.0,   "solid",          STEEL, 2.3),
    ("full",   r"$C_0 + C_1x + C_2x^2$",      11,  1.0,  0.020, (0, (1.5, 2.5)),  STEEL, 1.5),
    ("sin",    r"$C_3\sin(C_4x)$",             6, -1.0,  0.0,   (0, (4, 2, 1, 2)), TEAL, 1.8),
]


def quad_ladder_r2(key, xv):
    """R^2 of one x_v-determined row at vertex xv, before dodging."""
    if key == "c1x_0":
        return lam_of(np.asarray(xv, dtype=float))
    if key == "c2x2_0":
        return 1.0 - lam_of(np.asarray(xv, dtype=float))
    flat = {row[0]: row[3] for row in QUAD_LADDER}[key]
    return np.full_like(np.asarray(xv, dtype=float), flat)


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


def _member_legend(fig, handles, y=-0.02, ncol=4):
    leg = fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
                     fontsize=9, bbox_to_anchor=(0.5, y))
    for txt in leg.get_texts():
        txt.set_color(INK)
    return leg


# ---------------------------------------------------------------------------
# Figure 1a: the linear family, drawn raw
# ---------------------------------------------------------------------------

def plot_linear_family(out: Path):
    fig = plt.figure(figsize=(13.0, 3.75))
    gs = fig.add_gridspec(1, 4, wspace=0.34)

    handles = []
    for k, (name, short, color, marker) in enumerate(LIN_SLOTS):
        t = LINEAR_TARGETS[name]
        root = t.real_roots[0]
        shown = "-10^{6}" if abs(root) > 1e3 else f"{root:g}"
        where = "inside" if t.roots_in_domain() else "outside"
        ax = fig.add_subplot(gs[0, k])
        _small_multiple(
            ax, t, color, marker,
            f"{k + 1}   {name} · {short}",
            f"$y = {_poly(t.coeffs)}$\nroot at $x = {shown}$ — {where} the domain",
        )
        _mark_roots(ax, t, color)
        handles.append(Line2D([], [], color=color, marker=marker, ls="-", lw=2,
                              ms=7, mec="white", mew=0.8,
                              label=f"{name} — {short}"))

    # lin_C's whole variation is 2e-6 on an offset of 1e3; without explicit
    # ticks matplotlib renders three identical "1000.000" labels.
    fig.axes[2].set_yticks([999.999, 1000.0, 1000.001])
    fig.axes[2].set_ylim(999.9988, 1000.0012)

    fig.suptitle("The linear family  $y = c_0 + c_1x$  on  $x \\in [-1,1]$, drawn raw — "
                 "each member on its own $y$-axis",
                 fontsize=12, color=INK, x=0.058, y=1.10, ha="left")

    _member_legend(fig, handles, y=-0.13)
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 1b: the reachable score of every structure, as a curve in rho
# ---------------------------------------------------------------------------

def plot_linear_scores(out: Path):
    # linthresh = 1 puts the symlog break exactly at "the root leaves the
    # domain", so the linear stretch IS the shaded band and the only region
    # where anything moves; linscale = 3.5 buys that region ~35% of the width
    # instead of the 24% a 0.1 break left it with.
    fig = plt.figure(figsize=(13.4, 7.7))
    ax = fig.add_subplot(1, 1, 1)

    rho = np.concatenate([np.linspace(0, 1.0, 900), np.logspace(0, 6.5, 900)])
    LABEL_X = 3.2e3   # flat rows are labelled here: clear of lin_B (5) and lin_C (1e6)

    # --- domain band and reference lines --------------------------------
    ax.axvspan(-0.06, 1, color=BAND, zorder=0)
    ax.axvline(1.0, color=MUTED, lw=1, ls=":", zorder=1)
    xtr = ax.get_xaxis_transform()
    ax.text(1.25, 0.32, "root leaves the domain", transform=xtr, rotation=90,
            va="bottom", ha="left", fontsize=8, color=MUTED)
    for yline in (1.0, 0.0, -1.0):
        ax.axhline(yline, color=GRID, lw=0.9, zorder=1)

    # --- the seven curves -----------------------------------------------
    for key, tex, ntok, flat, dodge, ls, color, lw in LADDER:
        ax.plot(rho, ladder_r2(key, rho) + dodge, ls=ls, color=color, lw=lw,
                zorder=3, solid_capstyle="round")

    # --- the four members, on every curve -------------------------------
    # 6 curves x 4 members = 24 markers: section 1.3's linear block, drawn.
    # The sinusoid row carries no markers -- it is flat at the floor for every
    # member, and 4 more markers there would only crowd the pile-up the clip
    # already causes.
    member_handles = []
    for name, short, color, marker in LIN_SLOTS:
        t = LINEAR_TARGETS[name]
        c0, c1 = t.coeffs
        r = abs(c0 / c1)
        for key, tex, ntok, flat, dodge, ls, ccol, lw in LADDER:
            if key == "sin":
                continue
            yv = float(ladder_r2(key, np.array([r]))[0]) + dodge
            ax.plot([r], [yv], marker, color=color, ms=8.5, mec="white",
                    mew=1.1, ls="none", zorder=5)
        member_handles.append(
            Line2D([], [], color=color, marker=marker, ls="none", ms=8.5,
                   mec="white", mew=1.1, label=f"{name} — {short}"))

    # --- direct labels on the two rows that actually move ----------------
    ax.annotate(r"$C_1x$   (3 tok)" "\n" r"$R^2 = \mathrm{clip}(1 - \rho^2/\langle x^2\rangle)$",
                xy=(0.42, r2_linear_term(np.array([0.42]))[0]), xytext=(0.70, 0.72),
                fontsize=9.5, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=5))
    ax.annotate(r"$C_2x^2$   (5 tok)" "\n" r"$R^2 = \mathrm{clip}(-\kappa_2\,\rho^2)$",
                xy=(0.60, r2_quadratic_term(np.array([0.60]))[0]), xytext=(0.30, -0.74),
                fontsize=9.5, color=STEEL, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=5))

    # --- the crossings: the family's entire varying content ---------------
    ax.plot([RHO_ZERO], [0.0], "|", color=INK, ms=11, mew=1.6, zorder=6)
    ax.text(RHO_ZERO + 0.02, 0.07,
            r"$\rho=\sqrt{\langle x^2\rangle}=0.592$" "\n" r"$C_1x$ crosses $0$",
            fontsize=8.2, color=INK, ha="left", va="bottom")
    ax.plot([RHO_FLOOR, RHO_FLOOR_Q], [-1.0, -1.0], "|", color=INK, ms=11,
            mew=1.6, zorder=6)
    ax.text(0.83, -1.06,
            r"both intercept-free rows floor inside the domain:"
            "\n" r"$\rho=0.837$ ($C_1x$),  $0.888$ ($C_2x^2$)",
            fontsize=8.2, color=MUTED, ha="right", va="top")

    # --- direct labels on the five flat rows ------------------------------
    for x, y, va, tex, col in (
        (LABEL_X,  1.02, "bottom", r"$C_0 + C_1x + C_2x^2$   (11 tok)", STEEL),
        (LABEL_X,  0.98, "top",    r"$C_1x + C_0$   (5 tok)",           STEEL),
        (LABEL_X,  0.02, "bottom", r"$C_2x^2 + C_0$   (7 tok)",         STEEL),
        (LABEL_X, -0.02, "top",    r"$C_0$   (1 tok)",                  STEEL),
        (LABEL_X, -0.98, "bottom",
         r"$C_3\sin(C_4x)$   (6 tok) — measured, not a closed form",    TEAL),
    ):
        ax.text(x, y, tex, fontsize=9, color=col, ha="center", va=va)

    ax.text(LABEL_X, 0.50,
            "Every structure with a free $C_0$ is flat in $\\rho$: the constant\n"
            "absorbs $c_0$ exactly, whatever it is. Hence the two exact rows\n"
            "coincide at $1$ — on a line, $C_2$ is dead weight — and $C_2x^2+C_0$\n"
            "sits on $C_0$ at $0$, because $x$ is odd and $x^2$ even.",
            fontsize=8.8, color=MUTED, ha="center", va="center", linespacing=1.6)

    ax.annotate("", xy=(5, -1.05), xytext=(1e6, -1.05),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1,
                                shrinkA=7, shrinkB=7))
    ax.text(7e3, -1.10, "lin_B and lin_C: $10^{5}\\times$ apart in $\\rho$, "
                        "identical after clipping",
            ha="center", va="top", fontsize=8.2, color=MUTED)

    # --- axes -------------------------------------------------------------
    ax.set_xscale("symlog", linthresh=1.0, linscale=3.5)
    ax.set_xlim(-0.06, 3e6)
    ax.set_ylim(-1.34, 1.30)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1", "$10^1$", "$10^2$",
                        "$10^3$", "$10^4$", "$10^5$", "$10^6$"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("$\\rho = |c_0/c_1|$ — distance from the root to the origin"
                  "   (symlog: linear on the shaded $\\rho \\leq 1$, log above)",
                  fontsize=10.5, color=INK, labelpad=8)
    ax.set_ylabel("$R^2$ the structure would score on $y = c_0 + c_1x$",
                  fontsize=10.5, color=INK)
    ax.set_title("Every reachable score in the linear family, as a closed form in the root",
                 fontsize=12.5, color=INK, pad=76, loc="left")
    ax.text(0.0, 1.018,
            "Curves are exact arithmetic, not fits — except $C_3\\sin(C_4x)$, the one structure carrying a constant it is nonlinear in, drawn as measured under the\n"
            "50-eval cap. $\\langle x^2\\rangle = 0.35$ and $\\kappa_2 = 1.269$ are exact on the 41-point grid. Flat rows are dodged by $\\pm0.02$ where they coincide exactly\n"
            "(their true values are $0$ and $1$); the two moving curves are undodged. Each marker is one cell of §1.3's linear block — lin_B and lin_C show five, not\n"
            "six, because at the floor their two intercept-free rows land on the same point.",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom",
            linespacing=1.6)
    _style(ax)

    _member_legend(fig, member_handles, y=-0.045)
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2a: the quadratic family, drawn raw
# ---------------------------------------------------------------------------

def plot_quadratic_family(out: Path):
    fig = plt.figure(figsize=(13.0, 3.9))
    gs = fig.add_gridspec(1, 4, wspace=0.34)

    handles = []
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
            f"{k + 1}   {name} · {short}",
            f"$y = {_poly(t.coeffs)}$\n"
            f"$x_v {shown}$ — {where} · $\\Delta = {disc:g}$ · {roots}",
        )
        _mark_roots(ax, t, color)
        if t.vertex_in_domain():
            yv = float(t.ys(np.array([xv]))[0])
            ax.plot([xv], [yv], "v", color="white", mec=color, mew=2, ms=9,
                    zorder=5)
        handles.append(Line2D([], [], color=color, marker=marker, ls="-", lw=2,
                              ms=7, mec="white", mew=0.8,
                              label=f"{name} — {short}"))

    fig.suptitle("The quadratic family  $y = c_0 + c_1x + c_2x^2$  on  $x \\in [-1,1]$, "
                 "drawn raw — each member on its own $y$-axis",
                 fontsize=12, color=INK, x=0.058, y=1.10, ha="left")

    _member_legend(fig, handles, y=-0.13)
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2b: every reachable score in the quadratic family, in two invariants
# ---------------------------------------------------------------------------

def plot_quadratic_scores(out: Path):
    # Two panels because the family has two reward-visible degrees of freedom
    # and no single axis holds all seven rows. The split is not presentational:
    # panel A is every row the vertex settles, panel B is the two it cannot.
    #
    # Placement note: members sit at |x_v| = 0, 0.1, 0.25 and 5, so panel A's
    # flat-row labels go at 1.0 -- the one gap wide enough for them. Panel A's
    # band -1 < R^2 < 0 is empty for this family (nothing scores there but the
    # sinusoid), which is where the vertical-line labels live.
    fig = plt.figure(figsize=(14.8, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.24)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ======================= PANEL A: the vertex's five ====================
    xv = np.concatenate([np.linspace(0, 0.05, 300),
                         np.logspace(np.log10(0.05), np.log10(30), 700)])
    LABEL_XA = 1.0   # the gap between quad_A (0.25) and quad_B (5)

    axA.axvspan(-0.01, 1.0, color=BAND, zorder=0)
    axA.axvline(1.0, color=MUTED, lw=1, ls=":", zorder=1)
    xtr = axA.get_xaxis_transform()
    axA.text(1.0, 0.15, " vertex leaves the domain", transform=xtr, rotation=90,
             va="bottom", ha="left", fontsize=8, color=MUTED)
    axA.axvline(np.sqrt(KAPPA), color=MUTED, lw=1, ls=":", zorder=1)
    axA.text(np.sqrt(KAPPA) * 0.93, 0.15, "$\\sqrt{\\kappa}=0.264$", transform=xtr,
             rotation=90, va="bottom", ha="right", fontsize=8, color=MUTED)
    for yline in (1.0, 0.0, -1.0):
        axA.axhline(yline, color=GRID, lw=0.9, zorder=1)

    for key, tex, ntok, flat, dodge, ls, color, lw in QUAD_LADDER:
        axA.plot(xv, quad_ladder_r2(key, xv) + dodge, ls=ls, color=color, lw=lw,
                 zorder=3, solid_capstyle="round")

    # 4 rows x 4 members = 16 markers: section 1.3's quadratic block, minus the
    # two rows panel B carries and the sinusoid, flat at the floor for all four.
    handles = []
    for name, short, color, marker in QUAD_SLOTS:
        t = QUADRATIC_TARGETS[name]
        axv = abs(t.vertex)
        for key, tex, ntok, flat, dodge, ls, ccol, lw in QUAD_LADDER:
            if key == "sin":
                continue
            yv = float(quad_ladder_r2(key, np.array([axv]))[0]) + dodge
            axA.plot([axv], [yv], marker, color=color, ms=8.5, mec="white",
                     mew=1.1, ls="none", zorder=5)
        handles.append(Line2D([], [], color=color, marker=marker, ls="none",
                              ms=8.5, mec="white", mew=1.1,
                              label=f"{name} — {short}"))

    axA.text(0.0042, 0.065, "$\\lambda = C_1x + C_0$  (5 tok)",
             fontsize=8.5, color=INK)
    axA.text(0.0042, 0.925, "$1-\\lambda = C_2x^2 + C_0$  (7 tok)",
             fontsize=8.5, color=STEEL)
    axA.text(0.0042, -0.95, "$C_3\\sin(C_4x)$   (6 tok) — measured",
             fontsize=9, color=TEAL, va="bottom")
    axA.text(LABEL_XA, 1.03, r"$C_0{+}C_1x{+}C_2x^2$   (11 tok)", fontsize=8.5,
             color=STEEL, ha="center", va="bottom")
    axA.text(LABEL_XA, -0.05, r"$C_0$   (1 tok)", fontsize=8.5, color=STEEL,
             ha="center", va="top")

    axA.annotate("quad_B  $\\lambda=0.9972$\nthe $0.9972$ linear cheat",
                 xy=(5, 0.9972), xytext=(2.6, 0.72), fontsize=8.5, color=INK,
                 ha="center", linespacing=1.5,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=9))
    axA.annotate("quad_A  $\\lambda=0.472$\nnear the crossing: neither\ntwo-term row can take it",
                 xy=(0.25, 0.472), xytext=(2.6, 0.30), fontsize=8.5, color=INK,
                 ha="center", linespacing=1.5,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=9))

    axA.set_xscale("symlog", linthresh=0.05, linscale=1.0)
    axA.set_xlim(-0.01, 30)
    axA.set_ylim(-1.34, 1.30)
    axA.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    axA.set_xlabel("$|x_v| = |c_1/2c_2|$ — vertex to the origin   "
                   "(symlog: linear below $0.05$)",
                   fontsize=10, color=INK, labelpad=8)
    axA.set_ylabel("$R^2$ the structure would score on $y = c_0+c_1x+c_2x^2$",
                   fontsize=10.5, color=INK)
    axA.set_title("A   Five rows the vertex alone settles",
                  fontsize=12, color=INK, pad=82, loc="left")
    axA.text(0.0, 1.017,
             "$\\lambda = x_v^2/(x_v^2+\\kappa)$ with $\\kappa = 0.069875$ exactly — arithmetic, not fits. A free $C_0$ absorbs\n"
             "$c_0$, so the fit never sees the offset and the vertex alone settles the row; the two that move\n"
             "sum to $1$ on every member. Each marker is one cell of §1.3's quadratic block. Flat rows are\n"
             "dodged by $\\pm0.02$ (their true values are $0$ and $1$); the two moving rows are undodged.",
             transform=axA.transAxes, fontsize=8.8, color=MUTED, va="bottom",
             linespacing=1.6)
    _style(axA)

    # ================= PANEL B: the two the vertex cannot reach =============
    # Price, not R^2, is what collapses these to ONE curve each: R^2 still needs
    # lambda, but the DIFFERENCE from the counterpart is a pure quadratic in the
    # offset. The floor band is universal: both counterparts lie in [0,1], so a
    # row floors when its price reaches counterpart + 1 -- somewhere in [1,2],
    # whatever the vertex.
    OM_MAX, PRICE_MAX = 2.75, 4.75
    om = np.linspace(0, OM_MAX, 800)

    axB.axhspan(1.0, 2.0, color=BAND, zorder=0)
    axB.axhspan(2.0, PRICE_MAX, color="#e3e8ef", zorder=0)
    for yline in (1.0, 2.0):
        axB.axhline(yline, color=GRID, lw=0.9, zorder=1)

    axB.plot(om, price_linear_term(om), "-", color=INK, lw=2.3, zorder=3,
             solid_capstyle="round")
    axB.plot(om, price_quadratic_term(om), "-", color=STEEL, lw=2.3, zorder=3,
             solid_capstyle="round")

    axB.text(0.05, 2.62,
             "priced out entirely — every row up here reads\n"
             "exactly $-1$, whatever its vertex",
             fontsize=8.5, color=MUTED, ha="left", va="center", linespacing=1.5)
    axB.text(0.05, 1.44,
             "the floor bites in this band: a row\n"
             "reads $-1$ at price $=$ counterpart $+\\,1$",
             fontsize=8.0, color=MUTED, ha="left", va="center", linespacing=1.5)

    axB.annotate(r"$C_1x$   (3 tok):  price $=\omega^2$" "\n"
                 "span $\\{x\\}$ reaches no constant at all,\nso it pays the mean whole",
                 xy=(2.06, price_linear_term(2.06)), xytext=(0.52, 4.22),
                 fontsize=9, color=INK, ha="left", va="center", linespacing=1.5,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=6))
    axB.annotate(r"$C_2x^2$   (5 tok):  price $=\kappa_2\langle x^2\rangle\,\omega_0^2 = 0.444\,\omega_0^2$"
                 "\n" "$x^2$ is not orthogonal to $1$, so it absorbs\n"
                 "part of $c_0$ — the rest is charged at $0.444$",
                 xy=(2.34, price_quadratic_term(2.34)), xytext=(2.71, 0.55),
                 fontsize=8.6, color=STEEL, ha="right", va="center", linespacing=1.5,
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=6))

    # The four members, on whichever curve prices them.
    for name, short, color, marker in QUAD_SLOTS:
        lam, omega, omega0 = reward_coords(QUADRATIC_TARGETS[name])
        for off, price in ((omega, price_linear_term(omega)),
                           (omega0, price_quadratic_term(omega0))):
            if abs(off) <= OM_MAX and price <= PRICE_MAX:
                axB.plot([abs(off)], [price], marker, color=color, ms=9,
                         mec="white", mew=1.1, ls="none", zorder=5)

    # The linear family lands on the SAME two curves -- figure 1b is the
    # lambda = 1 slice. Open markers separate the families by ink, not hue, so
    # each member keeps its identity across all four figures.
    for name, short, color, marker in LIN_SLOTS:
        lam, omega, omega0 = reward_coords(LINEAR_TARGETS[name])
        for off, price in ((omega, price_linear_term(omega)),
                           (omega0, price_quadratic_term(omega0))):
            if abs(off) <= OM_MAX and price <= PRICE_MAX:
                axB.plot([abs(off)], [price], marker, mfc="none", mec=color,
                         mew=1.5, ms=8.5, ls="none", zorder=4)
    lin_handles = [Line2D([], [], color=MUTED, marker="o", mfc="none", mew=1.5,
                          ms=8.5, ls="none",
                          label="linear family (open): the same two curves, at $\\lambda=1$")]

    axB.text(0.05, 3.45,
             "lin_A and lin_D land on the curves at left;\n"
             "lin_B ($\\omega=8.5$) and lin_C ($1.7{\\times}10^{6}$) are\n"
             "off to the right — both floored",
             fontsize=8.0, color=MUTED, ha="left", va="center", linespacing=1.5)

    axB.set_xlim(0, OM_MAX)
    axB.set_ylim(0, PRICE_MAX)
    axB.set_xlabel("normalized offset the row pays for:   $\\omega = \\bar y/s$ for $C_1x$,   "
                   "$\\omega_0 = c_0/s$ for $C_2x^2$",
                   fontsize=10, color=INK, labelpad=8)
    axB.set_ylabel("$R^2$ the row gives up for having no intercept",
                   fontsize=10.5, color=INK)
    axB.set_title("B   The two rows the vertex cannot reach, and what a missing intercept costs",
                  fontsize=12, color=INK, pad=82, loc="left")
    axB.text(0.0, 1.017,
             "A row without an intercept scores what its panel-A counterpart scores, minus a quadratic in\n"
             "the offset, then clips: $R^2 = \\mathrm{clip}(\\lambda - \\omega^2)$ and $\\mathrm{clip}((1-\\lambda) - 0.444\\,\\omega_0^2)$. "
             "The second\ncoordinate enters through the offset and nowhere else, which is what leaves one curve per\n"
             "structure. Exact for lines and parabolas alike — which is why the linear members land here too.",
             transform=axB.transAxes, fontsize=8.8, color=MUTED, va="bottom",
             linespacing=1.6)
    _style(axB)

    fig.suptitle("Every reachable score in the quadratic family — the vertex settles five rows, "
                 "and the offset settles the other two",
                 fontsize=12.5, color=INK, x=0.055, y=1.125, ha="left")
    _member_legend(fig, handles + lin_handles, y=-0.055, ncol=5)
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("Claude-research/figures"))
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_linear_family(a.out_dir / "symreg_linear_family.png")
    plot_linear_scores(a.out_dir / "symreg_linear_scores.png")
    plot_quadratic_family(a.out_dir / "symreg_quadratic_family.png")
    plot_quadratic_scores(a.out_dir / "symreg_quadratic_scores.png")


if __name__ == "__main__":
    main()
