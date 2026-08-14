"""Design visualisations for Experiment 1 of ``writeup-milton/writeup.tex``.

Three PNGs, all analytic or exactly enumerated -- no search, no fitting, no
randomness:

    exp1_design.png     the controlled 2x2 design: the inversion boundary
                        lambda = p_M, the rarity curve rho_0 = 4!/M^4 with its
                        sample complexity K_0.95, and the exact terminal-reward
                        distribution of each cell (Experiment 1's plot #1,
                        computed by enumerating all M^4 tuples).

    exp1_pipeline.png   the execution plan: the six stages of Experiment 1,
                        which artefact each stage produces, and the exact gate
                        that must pass before Experiment 2 may run.

    gap_map.png         requirement-by-requirement status of the writeup's two
                        experiments against the code in ``src/sraz``: what
                        exists, what exists but diverges, what is missing.

Run from anywhere:

    python3 Claude-milton-experiments/figures/make_design_figures.py

Outputs are written next to this script.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

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

# colours carried over from writeup-milton/figures/make_scenario_figures.py
C_DECOY = "#c44e52"
C_DECOY_BG = "#f8e3e2"
C_CONT = "#4c72b0"
C_CONT_BG = "#dfe7f2"
C_DISTR = "#9a9a9a"
C_TGT = "#222222"
C_OK = "#2e7d5b"
C_OK_BG = "#e2f0e9"
C_BAD = "#b0442f"
C_BAD_BG = "#fbe6e0"
C_WARN = "#b58900"
C_WARN_BG = "#fdf3d8"

LAMBDAS = (0.25, 0.997)
MS = (4, 12)
DELTA = 0.05


# ===========================================================================
# The closed forms of the controlled family
# ===========================================================================
def p_of_M(M: int) -> float:
    """V^q(C) = 1 - (1 - 1/M)^4, eq. `eq:continue-vroll`."""
    return 1.0 - (1.0 - 1.0 / M) ** 4


def rho0_of_M(M: int) -> float:
    """rho_0(C) = 4!/M^4, eq. `eq:exact-mass`.

    The count 4! is of permutations of four *distinct* atoms, so the identity
    is a probability only for M >= 4; for a smaller vocabulary the four target
    atoms cannot all be selected and the exact mass is zero.
    """
    return 24.0 / M ** 4 if M >= 4 else 0.0


def k_of(rho: float, delta: float = DELTA) -> int:
    """K_{1-delta}, eq. `eq:hit-sample-complexity`."""
    return int(np.ceil(np.log(delta) / np.log1p(-rho)))


def reward_distribution(lam: float, M: int):
    """Exact law of R(j) over the M^4 equally likely tuples.

    Uses eq. `eq:analytic-terminal-reward`: the reward depends on the tuple
    only through which of the four weighted atoms it covers, so enumerating
    ``M**4`` tuples and bucketing by reward is a census.
    """
    counts: dict[float, int] = {}
    w_rest = (1.0 - lam) / 3.0
    for j in itertools.product(range(1, M + 1), repeat=4):
        J = set(j)
        r = lam * (1 in J) + w_rest * sum(m in J for m in (2, 3, 4))
        r = round(r, 12)
        counts[r] = counts.get(r, 0) + 1
    total = float(M ** 4)
    rs = np.array(sorted(counts))
    ps = np.array([counts[r] / total for r in rs])
    return rs, ps


# ===========================================================================
# Drawing helpers
# ===========================================================================
def box(ax, xc, yc, w, h, text, fc, ec, fontsize=9.0, tc="black", lw=1.1,
        weight="normal", z=3, family=None, ha="center", va="center",
        linespacing=1.55):
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, clip_on=False))
    tx = {"center": xc, "left": xc - w / 2 + 0.014,
          "right": xc + w / 2 - 0.014}[ha]
    ty = {"center": yc, "top": yc + h / 2 - 0.014,
          "bottom": yc - h / 2 + 0.014}[va]
    ax.text(tx, ty, text, ha=ha, va=va, fontsize=fontsize, color=tc,
            zorder=z + 1, weight=weight, linespacing=linespacing,
            clip_on=False, family=family)


def arrow(ax, p0, p1, color, lw=1.3, rad=0.0, z=2, scale=10, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=scale, shrinkA=1.5,
        shrinkB=1.5, linewidth=lw, linestyle=ls, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=z))


# ===========================================================================
# exp1_design.png
# ===========================================================================
def panel_cells(ax):
    """(a) The 2x2 design with every exact quantity printed."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    xs = {4: 0.320, 12: 0.740}
    ys = {0.25: 0.665, 0.997: 0.215}

    for M, x in xs.items():
        tag = "low" if M == 4 else "high"
        ax.text(x, 0.945, rf"$M={M}$   ({tag} distraction)", ha="center",
                va="center", fontsize=9.2, color="0.25")
    for lam, y in ys.items():
        name = "weak" if lam < 0.9 else "strong"
        ax.text(0.048, y, rf"$\lambda_{{\mathrm{{{name}}}}}$" "\n"
                rf"$={lam:g}$", ha="center", va="center", fontsize=9.2,
                color="0.25")

    for lam in LAMBDAS:
        for M in MS:
            pM, rho = p_of_M(M), rho0_of_M(M)
            inv = lam > pM
            col, bg = (C_BAD, C_BAD_BG) if inv else (C_OK, C_OK_BG)
            verdict = (r"$\bf{inversion}$:  $V^q$ prefers $D$" if inv
                       else r"no inversion:  $V^q$ prefers $C$")
            box(ax, xs[M], ys[lam], 0.400, 0.400,
                rf"$V^*(D)=\lambda={lam:g}$,    $V^*(C)=1$" "\n"
                rf"$V^q(D)={lam:g}$,    $V^q(C)=p_M={pM:.6f}$" "\n"
                rf"$\rho_0(C)=4!/M^4={rho:.6f}$" "\n"
                rf"$K_{{0.95}}={k_of(rho)}$" "\n\n"
                + verdict,
                bg, col, fontsize=7.8, tc=col, lw=1.3, linespacing=1.45)

    box(ax, 0.5, -0.075, 0.99, 0.100,
        r"the two knobs are orthogonal:  $\lambda$ moves only the "
        r"value-semantic axis (inversion iff $\lambda > p_M$),   "
        r"$M$ moves only the rare-event axis ($\rho_0 = 4!/M^4$)",
        "0.97", "0.55", fontsize=8.4, tc="0.20")
    ax.set_title(r"(a) the controlled $2\times2$ design, exactly",
                 fontsize=10.0, pad=2, loc="left")


def panel_inversion(ax):
    """(b) The inversion boundary lambda = p_M, with the four cells on it."""
    Mgrid = np.arange(4, 41)
    pM = np.array([p_of_M(int(M)) for M in Mgrid])
    ax.plot(Mgrid, pM, "-", lw=2.0, color=C_CONT,
            label=r"$V^q(C)=p_M=1-(1-1/M)^4$")
    ax.fill_between(Mgrid, pM, 1.02, color=C_BAD, alpha=0.09, lw=0)
    ax.fill_between(Mgrid, 0.0, pM, color=C_OK, alpha=0.09, lw=0)
    ax.text(30, 0.90, "inversion region\n" r"$\lambda > p_M$", ha="center",
            va="center", fontsize=8.0, color=C_BAD, linespacing=1.4)
    ax.text(30, 0.16, "no inversion\n" r"$\lambda < p_M$", ha="center",
            va="center", fontsize=8.0, color=C_OK, linespacing=1.4)

    for lam, ls in zip(LAMBDAS, ("--", "-.")):
        ax.axhline(lam, ls=ls, lw=1.4, color=C_DECOY, alpha=0.85)
        ax.text(39.4, lam, rf"$\lambda={lam:g}$", ha="right",
                va="bottom" if lam < 0.9 else "top", fontsize=7.6,
                color=C_DECOY)
        for M in MS:
            inv = lam > p_of_M(M)
            ax.plot([M], [lam], "s" if inv else "o", ms=8.5,
                    color=C_BAD if inv else C_OK, zorder=6,
                    mec="white", mew=1.0)

    ax.set_xlabel(r"vocabulary size $M$")
    ax.set_ylabel(r"root value")
    ax.set_xlim(4, 40)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=7.4, loc="center right")
    ax.set_title(r"(b) where the mean-rollout ordering flips",
                 fontsize=9.5, pad=4, loc="left")


def panel_rarity(ax):
    """(c) rho_0 and the completion budget it forces."""
    Mgrid = np.arange(4, 41)
    rho = np.array([rho0_of_M(int(M)) for M in Mgrid])
    Ks = np.array([k_of(float(r)) for r in rho])

    ax.plot(Mgrid, rho, "-", lw=2.0, color=C_CONT,
            label=r"$\rho_0(C)=4!/M^4$")
    ax.set_yscale("log")
    ax.set_xlabel(r"vocabulary size $M$")
    ax.set_ylabel(r"$\rho_0(C)$", color=C_CONT)
    ax.tick_params(axis="y", labelcolor=C_CONT)

    ax2 = ax.twinx()
    ax2.plot(Mgrid, Ks, "--", lw=1.8, color=C_DECOY,
             label=r"$K_{0.95}=\lceil \log\delta/\log(1-\rho_0)\rceil$")
    ax2.set_yscale("log")
    ax2.set_ylabel(r"$K_{0.95}$", color=C_DECOY)
    ax2.tick_params(axis="y", labelcolor=C_DECOY)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    for M in MS:
        r, K = rho0_of_M(M), k_of(rho0_of_M(M))
        ax.plot([M], [r], "o", ms=7, color=C_CONT, zorder=6, mec="white")
        ax2.plot([M], [K], "s", ms=7, color=C_DECOY, zorder=6, mec="white")
        ax.annotate(rf"$M={M}$: $\rho_0={r:.5f}$, $K_{{0.95}}={K}$",
                    xy=(M, r), xytext=(M + 1.6, r * 0.16),
                    fontsize=7.4, color="0.25",
                    arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.2, loc="upper right")
    ax.set_xlim(4, 40)
    ax.set_title(r"(c) rarity of the exact completion, and its cost",
                 fontsize=9.5, pad=4, loc="left")


def panel_reward_law(ax):
    """(d) Exact reward law per cell, as the survival function.

    A survival curve is comparable across cells even though the four supports
    differ, and it reads off the two design quantities directly: its height at
    ``r = lambda`` is the chance a random completion beats the decoy, and its
    height at ``r = 1`` is ``rho_0``.
    """
    styles = {(0.25, 4): (C_OK, "-"), (0.25, 12): (C_CONT, "-"),
              (0.997, 4): (C_WARN, "--"), (0.997, 12): (C_BAD, "--")}
    for (lam, M), (col, ls) in styles.items():
        rs, ps = reward_distribution(lam, M)
        surv = np.cumsum(ps[::-1])[::-1]                 # Pr[R >= r]
        # Pr[R > lambda]: strictly better than the decoy. The weak inequality
        # would count the ties at R = lambda, which are not improvements.
        beat = float(ps[rs > lam + 1e-12].sum())
        exact = float(ps[rs >= 1 - 1e-12].sum())
        ax.step(np.append(rs, 1.0 + 1e-9), np.append(surv, 0.0), where="post",
                color=col, ls=ls, lw=1.9,
                label=rf"$\lambda={lam:g}$, $M={M}$:   "
                      rf"$\Pr[R>\lambda]={beat:.5f}$,   "
                      rf"$V^q(C)={float(rs @ ps):.5f}$,   "
                      rf"$\rho_0={exact:.5f}$")
        ax.plot([1.0], [exact], "o", ms=6, color=col, zorder=6, mec="white")

    for lam, col in ((0.25, "0.45"), (0.997, "0.45")):
        ax.axvline(lam, ls=":", lw=1.0, color=col)
        ax.text(lam, 1.55, rf"$R(D)=\lambda={lam:g}$", ha="center",
                va="bottom", fontsize=7.2, color="0.35")
    ax.axvline(1.0, ls="-", lw=1.0, color=C_TGT, alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlabel(r"reward level $r$")
    ax.set_ylabel(r"$\Pr_q[R(\mathbf{j}) \geq r]$")
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(5e-4, 3.2)
    ax.legend(fontsize=7.0, loc="lower left", labelspacing=0.30)
    ax.set_title(r"(d) exact completion-reward survival over all $M^4$ "
                 r"leaves.  At $\lambda=0.997$ a sizeable share of completions "
                 r"strictly beats the decoy, yet the $\it{mean}$ does not: "
                 r"$V^q(C)=p_M<\lambda$",
                 fontsize=9.5, pad=16, loc="left")


def figure_design():
    fig = plt.figure(figsize=(11.0, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.42, 1.0, 1.05],
                          hspace=0.52, wspace=0.30,
                          left=0.062, right=0.955, top=0.950, bottom=0.070)
    panel_cells(fig.add_subplot(gs[0, :]))
    panel_inversion(fig.add_subplot(gs[1, 0]))
    panel_rarity(fig.add_subplot(gs[1, 1]))
    panel_reward_law(fig.add_subplot(gs[2, :]))
    out = os.path.join(HERE, "exp1_design.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# exp1_pipeline.png
# ===========================================================================
STAGES = [
    ("S1", "empirical function space",
     r"$x_i$: 41 points on $[-1,1]$;  centre $x,\dots,x^{12}$ and"
     "\n"
     r"Gram--Schmidt in $\langle\cdot,\cdot\rangle_n$ $\Rightarrow$ "
     r"$\psi_1,\dots,\psi_{12}$",
     "new: basis module", C_CONT),
    ("S2", "controlled target",
     r"$y_\lambda=\sqrt{\lambda}\,\psi_1+\sqrt{(1-\lambda)/3}"
     r"(\psi_2+\psi_3+\psi_4)$"
     "\n"
     r"assert $\overline{y}_\lambda=0$ and $\|y_\lambda\|_n^2=1$ "
     r"to $10^{-12}$",
     "new: target family", C_CONT),
    ("S3", "atom-slot environment",
     r"root $\to D$ (terminate) $\mid$ $C$ (open $A_1..A_4$),"
     "\n"
     r"$A_r\to\psi_1\mid\dots\mid\psi_M$,   $M\in\{4,12\}$",
     "new: Game subclass", C_DECOY),
    ("S4", "dual evaluator",
     r"analytic $R(\mathbf{j})=\sum_{m\in J}w_m$   vs."
     "\n"
     r"least squares on the design matrix;  agree to $10^{-12}$",
     "new: evaluator + test", C_DECOY),
    ("S5", "exact census",
     r"enumerate all prefixes of depth $0..4$ and all $M^4$ leaves;"
     "\n"
     r"compute $V^*$, $V^q$, $\rho_0$, $a_q$, $g_q$ at every state",
     "new: census script", C_WARN),
    ("S6", "primary outputs",
     r"reward law $\cdot$ $V^q$ vs $V^*$ by depth $\cdot$ "
     r"$g_q$ by depth $\cdot$ hit curve"
     "\n"
     r"+ the six root numbers of the writeup",
     "new: plotting script", C_WARN),
]


def figure_pipeline():
    fig = plt.figure(figsize=(11.5, 7.4))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    ys = np.linspace(0.855, 0.245, len(STAGES))
    x_tag, x_name, x_body, x_art = 0.028, 0.062, 0.510, 0.878
    for k, (tag, name, body, artefact, col) in enumerate(STAGES):
        box(ax, x_tag, ys[k], 0.046, 0.086, tag, "white", col,
            fontsize=9.0, tc=col, weight="bold")
        ax.text(x_name, ys[k], name, ha="left", va="center", fontsize=8.6,
                color=col, weight="bold")
        box(ax, x_body, ys[k], 0.462, 0.098, body, "0.985", col,
            fontsize=8.0, tc="0.15", ha="left", lw=1.0, linespacing=1.45)
        box(ax, x_art, ys[k], 0.212, 0.086, artefact, "white", col,
            fontsize=8.0, tc=col, lw=1.0)
        if k + 1 < len(STAGES):
            arrow(ax, (x_tag, ys[k] - 0.049), (x_tag, ys[k + 1] + 0.049),
                  "0.50", lw=1.2)

    box(ax, 0.5, 0.100, 0.99, 0.120,
        r"$\bf{Gate}$ before Experiment 2:  the analytic and least-squares "
        r"rewards must agree to $10^{-12}$ on every one of the $M^4$ leaves," "\n"
        r"and the root numbers must reproduce $V^q(C)=p_M$ and "
        r"$\rho_0(C)=4!/M^4$ exactly.  Experiment 1 is a census: "
        r"no seeds, no intervals, no significance tests.",
        C_OK_BG, C_OK, fontsize=8.2, tc=C_OK, linespacing=1.6)

    ax.text(0.006, 0.975,
            "Experiment 1 -- exact value-semantic audit: six stages, "
            "each ending in a checkable artefact",
            ha="left", va="center", fontsize=10.5, color="0.15")
    ax.text(0.006, 0.938,
            "Blue = analytic prerequisites, red = the environment and its "
            "reward, amber = the audit itself.  Every stage is new code: "
            "nothing in src/sraz realises the atom-slot grammar.",
            ha="left", va="center", fontsize=8.2, color="0.45")

    out = os.path.join(HERE, "exp1_pipeline.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# gap_map.png
# ===========================================================================
# status: 2 = runs as specified, 1 = exists but diverges, 0 = absent
GAP_ROWS = [
    # (requirement, where the writeup asks for it, status, code anchor)
    ("state = partial derivation", "MDP", 2, "SymRegGame.state buffer"),
    ("deterministic transition $T(s,a)$", "MDP", 2, "SymRegGame.step splice"),
    ("legal-action set $\\mathcal{A}(s)$", "MDP", 2, "get_action_mask"),
    ("terminal reward $R(\\pi)$", "MDP", 1,
     "fit_expression: lmfit, clipped to $[-1,1]$"),
    ("uniform completion policy $q$", "MDP", 1,
     "MCTS._rollout_value: uniform over $(i,p)$ pairs"),
    ("orthonormal $\\psi_1..\\psi_M$", "Exp 1", 0,
     "only in a figure script"),
    ("target family $y_\\lambda$", "Exp 1", 0, "not in targets.py"),
    ("atom-slot grammar, $M\\in\\{4,12\\}$", "Exp 1", 0, "no such grammar"),
    ("analytic reward $\\sum_{m\\in J} w_m$", "Exp 1", 0, "absent"),
    ("$10^{-12}$ dual-evaluator check", "Exp 1", 0, "absent"),
    ("exact $V^*(s)$ by enumeration", "Exp 1", 2,
     "informativeness_oracle.make_vstar_exact"),
    ("exact $V^q(s)$", "Exp 1", 1,
     "sampled in informativeness.py; exact only in a figure script"),
    ("$\\rho_\\varepsilon(s)$", "Exp 1", 1,
     "exact in make_mdp_figures.Census; nothing in src/"),
    ("sibling regret $g_q(s)$", "Exp 1", 0, "absent"),
    ("uniform random search baseline", "Exp 2", 1,
     "reachable via rollouts; no runner"),
    ("Mean-UCT $\\overline{Q}+\\sqrt{2\\log N/N_a}$", "Exp 2", 1,
     "calc_masked_ucbs is PUCT, min--max normalised"),
    ("unvisited actions score $+\\infty$", "Exp 2", 0,
     "finite PUCT bonus: unvisited can be skipped"),
    ("Max-UCT $Q^{\\max}$ statistic", "Exp 2", 2,
     "backup_rule=\"max\" in update_edge"),
    ("one persistent tree, no commitment", "Exp 2", 0,
     "Agent.policy rebuilds MCTS per move"),
    ("budget = terminal evaluations", "Exp 2", 1,
     "n_simulations is per move; $B{+}1$ per root search"),
    ("$S_B$, $T_{\\mathrm{exact}}$, $r_B$, $F_B$", "Exp 2", 0,
     "not recorded by any runner"),
    ("100 seeds, checkpointed to $B_{\\max}$", "Exp 2", 1,
     "run_uct_sweep sweeps, no checkpoints"),
]

STATUS = {2: ("runs as specified", C_OK, C_OK_BG),
          1: ("exists, diverges", C_WARN, C_WARN_BG),
          0: ("absent", C_BAD, C_BAD_BG)}


def figure_gap_map():
    n = len(GAP_ROWS)
    fig = plt.figure(figsize=(11.0, 0.40 * n + 1.55))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    top, bot = 0.905, 0.075
    ys = np.linspace(top, bot, n)
    dy = (top - bot) / max(1, n - 1)
    h = dy * 0.82

    ax.text(0.004, 0.975, "requirement", fontsize=8.6, color="0.30",
            ha="left", va="center", weight="bold")
    ax.text(0.360, 0.975, "asked in", fontsize=8.6, color="0.30",
            ha="center", va="center", weight="bold")
    ax.text(0.452, 0.975, "status", fontsize=8.6, color="0.30",
            ha="center", va="center", weight="bold")
    ax.text(0.578, 0.975, "anchor in the code", fontsize=8.6, color="0.30",
            ha="left", va="center", weight="bold")

    for k, (req, where, st, anchor) in enumerate(GAP_ROWS):
        label, col, bg = STATUS[st]
        y = ys[k]
        ax.add_patch(FancyBboxPatch(
            (0.0, y - h / 2), 1.0, h,
            boxstyle="round,pad=0.0,rounding_size=0.004",
            facecolor=bg if k % 2 == 0 else "white", edgecolor="none",
            alpha=0.55 if k % 2 == 0 else 1.0, zorder=1))
        ax.text(0.008, y, req, fontsize=8.0, color="0.10", ha="left",
                va="center", zorder=3)
        ax.text(0.360, y, where, fontsize=7.6, color="0.40", ha="center",
                va="center", zorder=3)
        box(ax, 0.452, y, 0.115, h * 0.82, label, bg, col, fontsize=7.2,
            tc=col, lw=1.0, z=3)
        ax.text(0.520, y, anchor, fontsize=7.6, color=col, ha="left",
                va="center", zorder=3)

    counts = {s: sum(1 for r in GAP_ROWS if r[2] == s) for s in (2, 1, 0)}
    ax.text(0.004, 0.020,
            f"{counts[2]} of {n} requirements run as specified,   "
            f"{counts[1]} exist but diverge,   {counts[0]} are absent.   "
            "The absent rows are the writeup's controlled environment and "
            "the Experiment-2 protocol, not the MDP itself.",
            fontsize=8.0, color="0.30", ha="left", va="center")

    out = os.path.join(HERE, "gap_map.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def report():
    print("controlled-family closed forms (writeup Table `tab:controlled-cells`):")
    for M in MS:
        print(f"  M={M:>2}:  p_M={p_of_M(M):.6f}   rho_0={rho0_of_M(M):.6f}   "
              f"K_0.95={k_of(rho0_of_M(M))}")
    for lam in LAMBDAS:
        for M in MS:
            rs, ps = reward_distribution(lam, M)
            print(f"  lambda={lam:<6g} M={M:>2}:  inversion="
                  f"{'yes' if lam > p_of_M(M) else 'no ':<3}  "
                  f"E[R]={float(rs @ ps):.6f}  "
                  f"Pr[R=1]={float(ps[rs >= 1 - 1e-12].sum()):.6f}  "
                  f"support={len(rs)}")


if __name__ == "__main__":
    report()
    figure_design()
    figure_pipeline()
    figure_gap_map()
