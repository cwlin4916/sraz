"""Figures for the two designed target families of note 02, sections 1.1 and 1.2.

Three figures. The linear family needs two, because its members and its scores
answer different questions and no longer fit one canvas:

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

  symreg_quadratic_family.png  FIGURE 2. quad_A..quad_D, partitioned by the
                               vertex x_v = -c1/2c2 and the discriminant
                               D = c1^2-4c0c2. Members and scores still share a
                               canvas, because that family's scores are two
                               curves rather than seven.

Figure 1a / 2 row A draw the members RAW, one small multiple each, on its own
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
unambiguous where 24 markers land on seven curves.

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
KAPPA = float(np.sum((XS ** 2 - M2) ** 2) / (4 * np.sum(XS ** 2)))  # 0.069875
KAPPA2 = (1.0 - M2 ** 2 / M4) / M2                                  # 1.2685805...

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


def _member_legend(fig, handles, y=-0.02):
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
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
    fig = plt.figure(figsize=(13.6, 8.0))
    ax = fig.add_subplot(1, 1, 1)

    rho = np.concatenate([np.linspace(0, 0.1, 500), np.logspace(-1, 7.2, 1200)])

    # --- domain band and reference lines --------------------------------
    ax.axvspan(-0.02, 1, color=BAND, zorder=0)
    ax.axvline(1.0, color=MUTED, lw=1, ls=":", zorder=1)
    xtr = ax.get_xaxis_transform()
    ax.text(1.35, 0.055, "root leaves the domain", transform=xtr, rotation=90,
            va="bottom", ha="left", fontsize=8, color=MUTED)
    for yline in (1.0, 0.0, -1.0):
        ax.axhline(yline, color=GRID, lw=0.9, ls="-", zorder=1)

    # --- the seven curves -----------------------------------------------
    struct_handles = []
    for key, tex, ntok, flat, dodge, ls, color, lw in LADDER:
        y = ladder_r2(key, rho) + dodge
        ax.plot(rho, y, ls=ls, color=color, lw=lw, zorder=3,
                solid_capstyle="round")
        note = "  measured" if key == "sin" else ""
        struct_handles.append(
            Line2D([], [], color=color, ls=ls, lw=lw,
                   label=f"{tex}   ({ntok} tok){note}"))

    # --- the four members, on every curve -------------------------------
    # 6 curves x 4 members = 24 markers: this is section 1.3's linear block,
    # drawn. The sinusoid row carries no markers -- it is flat at the floor for
    # every member and 4 more markers there would only crowd the pile-up the
    # clip already causes.
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

    # --- direct labels: only the two rows that actually move -------------
    ax.annotate(r"$C_1x$   (3 tok)" "\n" r"$R^2 = \mathrm{clip}(1 - \rho^2/\langle x^2\rangle)$",
                xy=(0.40, r2_linear_term(np.array([0.40]))[0]), xytext=(0.115, -0.42),
                fontsize=9, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=6))
    ax.annotate(r"$C_2x^2$   (5 tok)" "\n" r"$R^2 = \mathrm{clip}(-\kappa_2\,\rho^2)$",
                xy=(0.62, r2_quadratic_term(np.array([0.62]))[0]), xytext=(2.2, -0.50),
                fontsize=9, color=STEEL, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1, shrinkB=6))

    # --- the crossings, which are the family's whole varying content ------
    for xv, lab, col in ((RHO_ZERO, r"$\sqrt{\langle x^2\rangle}=0.592$", INK),
                         (RHO_FLOOR, r"$\sqrt{2\langle x^2\rangle}=0.837$", INK),
                         (RHO_FLOOR_Q, r"$1/\sqrt{\kappa_2}=0.888$", STEEL)):
        ax.plot([xv], [-1.0 if xv != RHO_ZERO else 0.0], "|", color=col, ms=9,
                mew=1.4, zorder=6)
    ax.text(RHO_ZERO, 0.055, r" $\rho=0.592$" "\n" r" $C_1x$ crosses $0$",
            fontsize=7.8, color=INK, ha="left", va="bottom")
    ax.text(0.80, -0.93, r"both intercept-free rows hit the floor" "\n"
                         r"at $\rho = 0.837$ and $0.888$ — inside the domain",
            fontsize=7.8, color=MUTED, ha="right", va="bottom")

    # --- what the flat rows mean -----------------------------------------
    ax.text(1.1e3, 1.075,
            "flat at $1$: exact. The 11-token quadratic buys nothing over the\n"
            "5-token line — on a line, $C_2$ is dead weight.",
            fontsize=8.4, color=INK, ha="center", va="bottom")
    ax.text(1.1e3, 0.075,
            "flat at $0$: no better than $\\bar y$. Adding $x^2$ to $C_0$ buys nothing —\n"
            "$x$ is odd, $x^2$ is even, so the residual is untouched.",
            fontsize=8.4, color=INK, ha="center", va="bottom")
    ax.text(1.1e3, -0.93,
            "flat at $-1$: the sinusoid fails outright on every member.",
            fontsize=8.4, color=TEAL, ha="center", va="bottom")

    ax.annotate("", xy=(5, -1.06), xytext=(1e6, -1.06),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1,
                                shrinkA=7, shrinkB=7))
    ax.text(7e3, -1.135, "lin_B and lin_C: $10^{5}\\times$ apart in $\\rho$, "
                         "identical after clipping",
            ha="center", fontsize=8.4, color=MUTED)

    # --- axes -------------------------------------------------------------
    ax.set_xscale("symlog", linthresh=0.1, linscale=1.0)
    ax.set_xlim(-0.02, 2e7)
    ax.set_ylim(-1.30, 1.30)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xlabel("$\\rho = |c_0/c_1|$ — distance from the root to the origin"
                  "   (symlog: linear below $0.1$, log above)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("$R^2$ the structure would score on $y = c_0 + c_1x$",
                  fontsize=10.5, color=INK)
    ax.set_title("Every reachable score in the linear family, as a closed form in the root",
                 fontsize=12.5, color=INK, pad=30, loc="left")
    ax.text(0.0, 1.035,
            "Curves are exact arithmetic, not fits — except $C_3\\sin(C_4x)$, the one structure "
            "with a constant it is nonlinear in, drawn as measured under the 50-eval cap. "
            "$\\langle x^2\\rangle = 0.35$ and $\\kappa_2 = 1.269$ are exact on the 41-point grid.\n"
            "Only the two intercept-free rows move: a free $C_0$ absorbs $c_0$ whatever it is, so "
            "every row that has one is flat in $\\rho$. Flat rows are dodged by $\\pm0.02$ where they "
            "coincide exactly (true values are $0$ and $1$); the two moving curves are undodged.",
            transform=ax.transAxes, fontsize=8.8, color=MUTED, va="bottom",
            linespacing=1.5)
    _style(ax)

    leg = ax.legend(handles=struct_handles, loc="center left",
                    bbox_to_anchor=(0.545, 0.685), frameon=True, fontsize=9,
                    title="candidate structure", title_fontsize=9,
                    labelspacing=0.55, borderpad=0.8)
    leg.get_frame().set_edgecolor(GRID)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_linewidth(0.8)
    leg.get_title().set_color(MUTED)
    for txt in leg.get_texts():
        txt.set_color(INK)
    ax.add_artist(leg)

    _member_legend(fig, member_handles, y=-0.035)
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: the quadratic family
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

    _member_legend(fig, handles, y=-0.035)
    fig.savefig(out, dpi=200, facecolor="white",
                bbox_inches="tight", pad_inches=0.3)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("docs/notes/figures"))
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    plot_linear_family(a.out_dir / "symreg_linear_family.png")
    plot_linear_scores(a.out_dir / "symreg_linear_scores.png")
    plot_quadratic_family(a.out_dir / "symreg_quadratic_family.png")


if __name__ == "__main__":
    main()
