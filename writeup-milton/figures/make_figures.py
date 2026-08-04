"""Closed-form figures for the quadratic-family note.

Everything here is analytic: the two figures are computed directly from the
grid moments, with no MCTS or network involved. Run:

    python3 make_figures.py

Outputs (next to this script):
    decomposition.png   -- orthogonal linear/quadratic split of a target
    targets_lambda.png  -- normalized target shapes + the vertex law lambda(x_v)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --- minimal, clean house style -------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
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

# --- the fixed 41-point symmetric grid ------------------------------------
x = np.linspace(-1.0, 1.0, 41)
mu2 = np.mean(x ** 2)                 # 0.35
mu4 = np.mean(x ** 4)                 # 0.220325
nu = mu4 - mu2 ** 2                   # 0.097825
kappa = nu / (4.0 * mu2)             # 0.069875

# muted palette
C_LIN = "#c44e52"   # linear component
C_QUAD = "#4c72b0"  # quadratic component
C_TGT = "#333333"   # target


def target(c0, c1, c2):
    return c0 + c1 * x + c2 * x ** 2


def lam(c1, c2):
    return (c1 ** 2 * mu2) / (c1 ** 2 * mu2 + c2 ** 2 * nu)


TARGETS = {
    "quad_A": (1.0, -1.0, 2.0),
    "quad_B": (6.0, -5.0, 0.5),
    "quad_C": (0.001, 0.001, 1000.0),
    "quad_D": (-0.48, 0.4, 2.0),
}


# ===========================================================================
# Figure 1: orthogonal decomposition of a single (balanced) target
# ===========================================================================
def fig_decomposition():
    c0, c1, c2 = TARGETS["quad_A"]
    y = target(c0, c1, c2)
    ybar = y.mean()
    centered = y - ybar
    lin = c1 * x
    quad = c2 * (x ** 2 - mu2)

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=1)
    ax.plot(x, centered, color=C_TGT, lw=2.0, label=r"$y-\bar y$ (centered target)")
    ax.plot(x, lin, color=C_LIN, lw=1.6, ls="--",
            label=r"$c_1 x$  (odd, linear)")
    ax.plot(x, quad, color=C_QUAD, lw=1.6, ls="-.",
            label=r"$c_2(x^2-\mu_2)$  (even, quadratic)")
    ax.scatter(x, centered, s=8, color=C_TGT, zorder=3)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("centered value")
    ax.set_xlim(-1.03, 1.03)
    ax.legend(loc="upper center", fontsize=8.5, ncol=1, handlelength=2.4)
    ax.set_title(r"quad_A: $\langle x,\,x^2-\mu_2\rangle=0$, "
                 r"so $s_y^2=c_1^2\mu_2+c_2^2\nu$", fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(HERE, "decomposition.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# Figure 2: normalized target shapes (left) + the vertex law (right)
# ===========================================================================
def fig_targets_lambda():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.5))

    # -- left: unit-variance normalized shapes -----------------------------
    colors = {"quad_A": "#4c72b0", "quad_B": "#c44e52",
              "quad_C": "#55a868", "quad_D": "#8172b3"}
    styles = {"quad_A": "-", "quad_B": "--", "quad_C": ":", "quad_D": "-."}
    for name, (c0, c1, c2) in TARGETS.items():
        y = target(c0, c1, c2)
        z = (y - y.mean()) / y.std()
        axL.plot(x, z, color=colors[name], ls=styles[name], lw=1.8,
                 label=name)
    axL.axhline(0.0, color="0.8", lw=0.8)
    axL.set_xlabel(r"$x$")
    axL.set_ylabel(r"$(y-\bar y)/s_y$")
    axL.set_xlim(-1.03, 1.03)
    axL.set_title("normalized target shapes", fontsize=9.5)
    axL.legend(fontsize=8.5, loc="upper center", ncol=2, handlelength=2.2)

    # -- right: lambda(x_v) = x_v^2 / (x_v^2 + kappa) ----------------------
    xv = np.linspace(0.0, 1.5, 400)
    axR.plot(xv, xv ** 2 / (xv ** 2 + kappa), color="#333333", lw=2.0)

    # balanced point |x_v| = sqrt(kappa) -> lambda = 1/2
    xb = np.sqrt(kappa)
    axR.scatter([xb], [0.5], color="0.2", zorder=5, s=28)
    axR.annotate(r"balanced: $|x_v|=\sqrt{\kappa}\approx0.264$",
                 xy=(xb, 0.5), xytext=(0.42, 0.34), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))

    # named targets on the curve
    pts = {"quad_C": 0.0, "quad_D": 0.1, "quad_A": 0.25}
    for name, v in pts.items():
        l = v ** 2 / (v ** 2 + kappa)
        axR.scatter([v], [l], color="#c44e52", zorder=6, s=26)
        axR.annotate(name, xy=(v, l),
                     xytext=(v + 0.03, l + 0.05), fontsize=8, color="#c44e52")
    # quad_B lives off-scale at x_v = 5
    axR.annotate(r"quad_B: $x_v=5 \rightarrow \lambda \approx 0.997$",
                 xy=(1.5, 0.997), xytext=(0.5, 0.9), fontsize=8,
                 color="#c44e52",
                 arrowprops=dict(arrowstyle="->", color="#c44e52", lw=0.8))

    axR.set_xlabel(r"vertex position $x_v$")
    axR.set_ylabel(r"linear variance share $\lambda$")
    axR.set_xlim(-0.03, 1.53)
    axR.set_ylim(-0.02, 1.05)
    axR.set_title(r"$\lambda(x_v)=x_v^2/(x_v^2+\kappa)$", fontsize=9.5)

    fig.tight_layout()
    out = os.path.join(HERE, "targets_lambda.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    print(f"mu2={mu2:.6g}  mu4={mu4:.6g}  nu={nu:.6g}  kappa={kappa:.6g}")
    for name, (c0, c1, c2) in TARGETS.items():
        print(f"  {name}: x_v={-c1/(2*c2):+.4g}  lambda={lam(c1, c2):.4f}")
    fig_decomposition()
    fig_targets_lambda()
