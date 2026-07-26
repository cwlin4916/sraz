"""Orientation figures for the two decoy scenarios of the controlled study.

One figure per value of the decoy parameter lambda (eq. `eq:lambda-values`):

    scenario_weak.png     lambda = 1/4     (no value-order inversion)
    scenario_strong.png   lambda = 0.997   (inversion at both M)

Each figure has four panels:

    (a) the controlled derivation grammar as a tree, with the decoy branch D
        and the continue branch C colour-separated, and the exact values
        R(D), V*(C), V^q(C), rho_0 printed on the leaves;
    (b) function space: the target y_lambda, the decoy fit yhat_D, and the
        residual that only the continue branch can remove;
    (c) the unit variance ledger of Proposition `prop:target-normalization`,
        i.e. the partition 1 = lambda + 3 (1-lambda)/3;
    (d) the root value line, showing whether the mean-rollout evaluator
        preserves or reverses the correct ordering.

Everything is analytic: no MCTS, no fitting, no randomness. Run

    python3 make_scenario_figures.py

Outputs are written next to this script.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- house style (matches make_figures.py) ---------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "0.90",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.facecolor": "white",
})

HERE = os.path.dirname(os.path.abspath(__file__))

# --- one colour per semantic role, used identically in every panel ---------
C_DECOY = "#c44e52"   # the decoy branch D and the atom psi_1 it can reach
C_DECOY_BG = "#f8e3e2"
C_CONT = "#4c72b0"    # the continue branch C and the atoms psi_2, psi_3, psi_4
C_CONT_BG = "#dfe7f2"
C_DISTR = "#9a9a9a"   # distractor atoms psi_5, ..., psi_M
C_DISTR_BG = "#ececec"
C_TGT = "#222222"     # the target itself
C_OK = "#2e7d5b"      # ordering preserved
C_BAD = "#b0442f"     # ordering inverted


# ===========================================================================
# Empirical function space (writeup section "Empirical function space")
# ===========================================================================
N_GRID = 41
x = np.linspace(-1.0, 1.0, N_GRID)


def ip(f, g):
    """Empirical inner product <f,g>_n = (1/n) sum_i f(x_i) g(x_i)."""
    return float(np.mean(f * g))


def orthonormal_basis(n_basis=12):
    """Centred monomials x, x^2, ..., x^{n_basis}, modified Gram-Schmidt."""
    psis = []
    for j in range(1, n_basis + 1):
        v = x ** j
        v = v - v.mean()
        for _ in range(2):                      # re-orthogonalise for stability
            for p in psis:
                v = v - ip(v, p) * p
        psis.append(v / np.sqrt(ip(v, v)))
    return psis


PSI = orthonormal_basis(12)


def y_lambda(lam):
    """Four-component target, eq. `eq:target-family`."""
    return (np.sqrt(lam) * PSI[0]
            + np.sqrt((1.0 - lam) / 3.0) * (PSI[1] + PSI[2] + PSI[3]))


def p_of_M(M):
    """Mean-rollout value of the continue branch, eq. `eq:continue-vroll`."""
    return 1.0 - (1.0 - 1.0 / M) ** 4


def rho0_of_M(M):
    """Exact-completion mass, eq. `eq:exact-mass`."""
    return 24.0 / M ** 4


# ===========================================================================
# Small drawing helpers for the grammar panel
# ===========================================================================
def box(ax, xc, yc, w, h, text, fc, ec, fontsize=9.0, tc="black", lw=1.1,
        weight="normal", z=3):
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
        clip_on=False))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=fontsize,
            color=tc, zorder=z + 1, weight=weight, linespacing=1.55,
            clip_on=False)


def arrow(ax, p0, p1, color, lw=1.2, rad=0.0, z=2, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=10,
        shrinkA=1.5, shrinkB=1.5, linewidth=lw, linestyle=ls,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=z))


# ===========================================================================
# Panel (a): the controlled derivation grammar
# ===========================================================================
def panel_grammar(ax, lam, lam_name):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    p4, p12 = p_of_M(4), p_of_M(12)
    inverted4, inverted12 = lam > p4, lam > p12

    x_root, x_act, x_exp, x_val = 0.050, 0.215, 0.455, 0.800
    y_d, y_c = 0.82, 0.30

    # -- root ---------------------------------------------------------------
    box(ax, x_root, 0.56, 0.078, 0.105, "$s_0$", "white", "0.35",
        fontsize=11, weight="bold")
    ax.text(x_root, 0.485, "root", ha="center", va="top", fontsize=7.6,
            color="0.45")

    # -- root -> D and root -> C -------------------------------------------
    arrow(ax, (x_root + 0.042, 0.605), (x_act - 0.052, y_d - 0.05), C_DECOY,
          lw=1.5, rad=0.10)
    arrow(ax, (x_root + 0.042, 0.515), (x_act - 0.052, y_c + 0.05), C_CONT,
          lw=1.5, rad=-0.10)
    ax.text(0.118, 0.775, r"$s_0 \to D$", fontsize=8.2, color=C_DECOY,
            ha="center", va="bottom", rotation=30)
    ax.text(0.118, 0.335, r"$s_0 \to C$", fontsize=8.2, color=C_CONT,
            ha="center", va="top", rotation=-30)

    # -- decoy branch -------------------------------------------------------
    box(ax, x_act, y_d, 0.100, 0.150,
        "$D$\nterminate", C_DECOY_BG, C_DECOY, fontsize=9.5, tc=C_DECOY)
    arrow(ax, (x_act + 0.055, y_d), (x_exp - 0.120, y_d), C_DECOY, lw=1.3)
    box(ax, x_exp, y_d, 0.225, 0.185,
        r"$\mathcal{M}_D=\mathrm{span}\{1,\psi_1\}$" "\n"
        r"a single terminal expression",
        C_DECOY_BG, C_DECOY, fontsize=8.8, tc=C_DECOY)
    arrow(ax, (x_exp + 0.117, y_d), (x_val - 0.180, y_d), C_DECOY, lw=1.3)
    box(ax, x_val, y_d, 0.350, 0.300,
        r"$\widehat y_D=\sqrt{\lambda}\,\psi_1$" "\n"
        rf"$R(D)=V^q(D)=V^*(D)=\lambda={lam:.3f}$" "\n"
        r"deterministic: one value, every visit",
        C_DECOY_BG, C_DECOY, fontsize=8.8, tc=C_DECOY)

    # -- continue branch ----------------------------------------------------
    box(ax, x_act, y_c, 0.100, 0.150,
        "$C$\ncontinue", C_CONT_BG, C_CONT, fontsize=9.5, tc=C_CONT)

    slot_y = [0.525, 0.425, 0.325, 0.225]
    for r, yy in enumerate(slot_y, start=1):
        arrow(ax, (x_act + 0.055, y_c), (x_exp - 0.058, yy), C_CONT,
              lw=0.9, rad=0.05 if yy > y_c else -0.05)
        box(ax, x_exp, yy, 0.108, 0.070, f"$A_{r}$", C_CONT_BG, C_CONT,
            fontsize=9.2, tc=C_CONT)
        arrow(ax, (x_exp + 0.058, yy), (x_val - 0.180, y_c), C_CONT, lw=0.9,
              rad=-0.04 if yy > y_c else 0.04)
    ax.text(x_exp, 0.135,
            r"$A_r \to \psi_1\,|\,\psi_2\,|\,\cdots\,|\,\psi_M$",
            ha="center", va="center", fontsize=9.0, color=C_CONT)
    ax.text(x_exp, 0.062,
            r"$\psi_5,\dots,\psi_M$: distractors, no variance",
            ha="center", va="center", fontsize=7.6, color=C_DISTR)

    box(ax, x_val, y_c, 0.350, 0.470,
        r"$\mathcal{M}_{\mathbf{j}}=\mathrm{span}(\{1\}\cup\{\psi_j\}_{j\in J(\mathbf{j})})$"
        "\n"
        r"$V^*(C)=1$  at $\mathbf{j}\sim(1,2,3,4)$;  $M^4$ leaves, $4!$ exact"
        "\n"
        rf"$M=4$:    $V^q(C)=p_4={p4:.3f}$,   $\rho_0={rho0_of_M(4):.4f}$"
        "\n"
        rf"$M=12$:  $V^q(C)=p_{{12}}={p12:.3f}$,   $\rho_0={rho0_of_M(12):.5f}$",
        C_CONT_BG, C_CONT, fontsize=8.4, tc=C_CONT)

    # -- verdict strip ------------------------------------------------------
    ok = not (inverted4 or inverted12)
    strip_c, strip_bg = (C_OK, "#e2f0e9") if ok else (C_BAD, "#fbe6e0")
    if ok:
        msg = (r"mean-rollout ordering $\bf{preserved}$:  "
               rf"$R(D)={lam:.3f} < V^q(C)$ for $M=4$ and $M=12$,  "
               rf"matching $V^*(D)={lam:.3f} < V^*(C)=1$")
    else:
        which = ("both $M=4$ and $M=12$" if (inverted4 and inverted12)
                 else "$M=12$")
        msg = (r"mean-rollout ordering $\bf{inverted}$:  "
               rf"$R(D)={lam:.3f} > V^q(C)$ for {which},  "
               rf"while $V^*(D)={lam:.3f} < V^*(C)=1$")
    box(ax, 0.5, -0.095, 0.99, 0.105, msg, strip_bg, strip_c,
        fontsize=8.8, tc=strip_c)

    ax.set_title(rf"(a) controlled derivation grammar,   "
                 rf"$\lambda_{{\mathrm{{{lam_name}}}}}={lam:g}$",
                 fontsize=10.5, pad=4, loc="left")


# ===========================================================================
# Panel (b): function space -- where the decoy sits
# ===========================================================================
def panel_function_space(ax, lam):
    y = y_lambda(lam)
    fit = np.sqrt(lam) * PSI[0]
    resid = y - fit

    ax.axhline(0.0, color="0.80", lw=0.8)
    ax.fill_between(x, fit, y, color=C_CONT, alpha=0.22, lw=0,
                    label=r"residual: only $C$ removes it")
    ax.plot(x, y, color=C_TGT, lw=2.0, label=r"target $y_\lambda$", zorder=4)
    ax.plot(x, fit, color=C_DECOY, lw=1.8, ls="--", zorder=5,
            label=r"decoy fit $\widehat y_D=\sqrt{\lambda}\,\psi_1$")
    ax.plot(x, resid, color=C_CONT, lw=1.3, ls=":", zorder=3,
            label=r"$y_\lambda-\widehat y_D \in \mathrm{span}\{\psi_2,\psi_3,\psi_4\}$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("value")
    ax.set_xlim(-1.03, 1.03)
    stack = np.concatenate([y, fit, resid])
    lo, hi = stack.min(), stack.max()
    rng = hi - lo
    ax.set_ylim(lo - 0.20 * rng, hi + 0.62 * rng)   # headroom for the legend
    ax.legend(fontsize=7.0, loc="upper left", handlelength=1.9,
              borderaxespad=0.15, labelspacing=0.3)
    ax.text(0.98, 0.02,
            rf"$\|y_\lambda-\widehat y_D\|_n^2=1-\lambda={1-lam:g}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2,
            color=C_CONT)
    ax.set_title("(b) where the decoy sits", fontsize=9.5, pad=5, loc="left")


# ===========================================================================
# Panel (c): the unit variance ledger
# ===========================================================================
def panel_ledger(ax, lam):
    w_rest = (1.0 - lam) / 3.0
    weights = [lam, w_rest, w_rest, w_rest]
    colors = [C_DECOY, C_CONT, C_CONT, C_CONT]
    alphas = [1.0, 1.0, 0.78, 0.56]
    labels = [r"$\psi_1$", r"$\psi_2$", r"$\psi_3$", r"$\psi_4$"]

    left = 0.0
    for w, c, a, lab in zip(weights, colors, alphas, labels):
        ax.barh(0.0, w, left=left, height=0.42, color=c, alpha=a,
                edgecolor="white", linewidth=1.0, zorder=3)
        if w > 0.06:
            ax.text(left + w / 2, 0.0, lab, ha="center", va="center",
                    fontsize=9.5, color="white", weight="bold", zorder=4)
        left += w

    # distractors carry exactly zero
    ax.text(0.0, 0.82, r"distractors $\psi_5,\dots,\psi_M$ carry $w_m=0$",
            ha="left", va="center", fontsize=8.0, color=C_DISTR)

    # brackets: decoy-reachable above the bar, continue-only below it
    ax.plot([0.0, lam], [0.31, 0.31], color=C_DECOY, lw=1.4,
            solid_capstyle="butt")
    ax.text(lam / 2 if lam < 0.9 else 0.5, 0.38,
            rf"captured by $D$:  $\lambda={lam:g}$",
            ha="center", va="bottom", fontsize=8.4, color=C_DECOY)
    ax.plot([lam, 1.0], [-0.31, -0.31], color=C_CONT, lw=1.4,
            solid_capstyle="butt")
    ax.annotate(rf"reachable only through $C$:" "\n"
                rf"$1-\lambda={1-lam:g}=3\times\frac{{1-\lambda}}{{3}}$",
                xy=((1 + lam) / 2, -0.31), xytext=(0.50, -0.88),
                ha="center", va="bottom", fontsize=8.4, color=C_CONT,
                arrowprops=dict(arrowstyle="->", color=C_CONT, lw=0.8))

    ax.set_xlim(-0.01, 1.06)
    ax.set_ylim(-1.02, 1.05)
    ax.set_yticks([])
    ax.set_xlabel("share of the unit target variance\n"
                  r"$w_m=\langle y_\lambda,\psi_m\rangle_n^2$,"
                  r"   $\sum_m w_m=\|y_\lambda\|_n^2=1$", fontsize=8.6)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_title("(c) the unit variance ledger", fontsize=9.5, pad=5,
                 loc="left")


# ===========================================================================
# Panel (d): the root value line
# ===========================================================================
def panel_values(ax, lam):
    rows = [(4, 1.0), (12, 0.0)]
    for M, yy in rows:
        pM = p_of_M(M)
        inverted = lam > pM
        col = C_BAD if inverted else C_OK

        ax.plot([0, 1], [yy, yy], color="0.85", lw=1.0, zorder=1)
        lo, hi = sorted((lam, pM))
        ax.plot([lo, hi], [yy, yy], color=col, lw=3.2, alpha=0.35, zorder=2)

        ax.scatter([pM], [yy], s=62, color=C_CONT, zorder=5,
                   label=r"$V^q(C)=p_M$" if M == 4 else None)
        ax.scatter([lam], [yy], s=62, marker="s", color=C_DECOY, zorder=5,
                   label=r"$V^q(D)=V^*(D)=\lambda$" if M == 4 else None)
        ax.scatter([1.0], [yy], s=95, marker="*", color=C_TGT, zorder=5,
                   label=r"$V^*(C)=1$" if M == 4 else None)

        ax.text(-0.06, yy, f"$M={M}$", ha="right", va="center", fontsize=9.0)
        ax.text(0.44, yy + 0.17,
                "$V^q$ inverts $V^*$" if inverted else "$V^q$ agrees with $V^*$",
                ha="center", va="bottom", fontsize=8.2, color=col)
        # stagger the two value labels when the markers nearly coincide
        drop = 0.40 if abs(lam - pM) < 0.10 else 0.17
        ax.text(pM, yy - 0.17, f"{pM:.3f}", ha="center", va="top",
                fontsize=7.6, color=C_CONT)
        ax.text(lam, yy - drop, f"{lam:g}", ha="center", va="top",
                fontsize=7.6, color=C_DECOY)

    ax.set_xlim(-0.06, 1.10)
    ax.set_ylim(-0.72, 2.35)
    ax.set_yticks([])
    ax.set_xlabel("root value")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7.2, loc="upper center", ncol=3, handletextpad=0.2,
              columnspacing=0.7, borderaxespad=0.1, scatterpoints=1)
    ax.set_title("(d) root value ordering", fontsize=9.5, pad=5, loc="left")


# ===========================================================================
# Assembly
# ===========================================================================
def make_scenario(lam, lam_name, outfile):
    fig = plt.figure(figsize=(10.0, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.42, 1.0],
                          hspace=0.42, wspace=0.30,
                          left=0.055, right=0.985, top=0.945, bottom=0.115)
    panel_grammar(fig.add_subplot(gs[0, :]), lam, lam_name)
    panel_function_space(fig.add_subplot(gs[1, 0]), lam)
    panel_ledger(fig.add_subplot(gs[1, 1]), lam)
    panel_values(fig.add_subplot(gs[1, 2]), lam)

    out = os.path.join(HERE, outfile)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def report():
    print(f"grid: n={N_GRID}, mu2={ip(x, x):.6g}")
    for lam in (0.25, 0.997):
        y = y_lambda(lam)
        print(f"  lambda={lam}: mean={ip(y, np.ones_like(y)):.3e}  "
              f"||y||_n^2={ip(y, y):.12f}  "
              f"w=({ip(y, PSI[0])**2:.6f}, {ip(y, PSI[1])**2:.6f}, "
              f"{ip(y, PSI[2])**2:.6f}, {ip(y, PSI[3])**2:.6f})")
    for M in (4, 12):
        print(f"  M={M}: p_M={p_of_M(M):.6f}  rho_0={rho0_of_M(M):.6f}  "
              f"K_0.95={int(np.ceil(np.log(0.05)/np.log(1-rho0_of_M(M))))}")


if __name__ == "__main__":
    report()
    make_scenario(0.25, "weak", "scenario_weak.png")
    make_scenario(0.997, "strong", "scenario_strong.png")
