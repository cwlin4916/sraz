"""Figures for the minimal controlled family, and for the action encoding.

Three PNGs, one per definition or claim, all computed by running the real
game -- nothing here is transcribed by hand:

    action_index.png     why an action is an integer in {0, ..., LP-1}: the
                         L x P grid of (pos, prod) pairs, and one decode.

    family_targets.png   the eight named polynomial targets of
                         ``sraz.instances.symreg.targets`` on the plane, and
                         the two invariants that partition them.

    family_difficulty.png  where the difficulty sits for that family, under one
                         fixed MDP: the reward ceiling as a function of
                         expression length, the four root children, and the
                         deception/rarity plane.

The MDP is held fixed (ADDITIVE_GRAMMAR, L = 12): all eight targets share the
same 4,898 reachable forms and the same 247 terminals, so the target is the
only thing that moves. Every number is an exact census, not a sample.

Run from anywhere:

    python3 writeup-milton/figures/make_family_figures.py

Outputs are written next to this script.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_mdp_figures import (  # noqa: E402
    C_CONT, C_CONT_BG, C_DEAD, C_DECOY, C_DECOY_BG, C_OK, C_TGT, HERE, box,
)
from sraz.instances.symreg.game import ADDITIVE_GRAMMAR, SymRegGame  # noqa: E402
from sraz.instances.symreg.targets import TARGETS  # noqa: E402

sys.setrecursionlimit(100_000)

MONO = "DejaVu Sans Mono"
L = 12          # buffer length, as in the MDP section
EPS = 1e-6      # tolerance for "exact"

C_NT = "#8452a8"
C_NT_BG = "#ece2f6"


# ===========================================================================
# The census: one MDP, eight targets
# ===========================================================================
class Family:
    """Exact values for every named target under one fixed derivation MDP."""

    def __init__(self, max_len: int = L):
        self.max_len = max_len
        g0 = SymRegGame(max_len=max_len, grammar_rules=ADDITIVE_GRAMMAR,
                        target="quad_A")
        self.G = g0.grammar
        self.root = (self.G.symdict[self.G.start],)
        self.children = self._build(g0)
        self.terminals = [s for s, c in self.children.items() if not c]
        self.kids = self.children[self.root]           # 4 root children
        self.stats = {name: self._solve(name) for name in TARGETS}

    # -- the reachable DAG, identical for every target ----------------------
    def _build(self, g):
        children, stack, seen = {}, [self.root], set()
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            acts = []
            for i, t in enumerate(s):
                if t in self.G.nonterms:
                    for j in self.G.proddict[t]:
                        if len(s) + len(self.G.productions[j]) - 1 < g.state_len:
                            c = s[:i] + tuple(self.G.productions[j]) + s[i + 1:]
                            acts.append(c)
                            stack.append(c)
            children[s] = acts
        return children

    def decode(self, s):
        return " ".join(self.G.tokenlist[t] for t in s)

    # -- exact V*, V^q, rho_0 by dynamic programming over that DAG ----------
    def _solve(self, name):
        g = SymRegGame(max_len=self.max_len, grammar_rules=ADDITIVE_GRAMMAR,
                       target=name)
        vstar, vq, rho = {}, {}, {}

        def solve(s):
            if s in vq:
                return
            cs = self.children[s]
            if not cs:
                r = g._fit_cached(self.decode(s))
                vstar[s] = vq[s] = r
                rho[s] = 1.0 if r >= 1 - EPS else 0.0
                return
            for c in cs:
                solve(c)
            vstar[s] = max(vstar[c] for c in cs)
            vq[s] = float(np.mean([vq[c] for c in cs]))
            rho[s] = float(np.mean([rho[c] for c in cs]))

        solve(self.root)

        best_len = {}
        for s in self.terminals:
            best_len[len(s)] = max(best_len.get(len(s), -1.0), vstar[s])

        C = self.kids[0]                       # '+ S S', the only continue
        D = self.kids[1:]                      # three one-action terminals
        decoys = [(self.decode(c), vstar[c]) for c in D]
        dbest = max(v for _, v in decoys)
        margin = dbest - vq[C]
        # A positive margin is only a *trap* if the decoy is not itself
        # optimal: value-semantic failure needs V^q(D) > V^q(C) AND
        # V^*(D) < V^*(C).
        solved = dbest >= 1 - EPS
        return dict(
            target=TARGETS[name],
            vstar0=vstar[self.root], vq0=vq[self.root], rho0=rho[self.root],
            vstarC=vstar[C], vqC=vq[C], rhoC=rho[C],
            decoys=decoys, dbest=dbest, margin=margin,
            solved=solved, trap=(margin > 0 and not solved),
            best_len=dict(sorted(best_len.items())),
            n_exact=sum(1 for s in self.terminals if vstar[s] >= 1 - EPS),
        )


# ===========================================================================
# FIGURE -- action_index.png : why a is an integer in {0, ..., LP-1}
# ===========================================================================
def fig_action_index(fam: Family, out="action_index.png"):
    P, a = fam.G.nprods, 2 * fam.G.nprods + 3
    pos, prod = divmod(a, P)
    rhs = [fam.G.tokenlist[t] for t in fam.G.productions[prod]]
    rules = [" ".join(fam.G.tokenlist[t] for t in r) for r in fam.G.productions]

    fig = plt.figure(figsize=(8.6, 4.5))
    gs = GridSpec(1, 2, width_ratios=[1.0, 1.28], wspace=0.10,
                  left=0.075, right=0.985, top=0.845, bottom=0.135)

    # -- (a) the grid of action indices ------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(-0.5, P - 0.5)
    ax.set_ylim(L - 0.5, -0.5)
    ax.grid(False)
    for i in range(L):
        for j in range(P):
            k = i * P + j
            live, demo = i == 0, k == a
            ax.add_patch(plt.Rectangle(
                (j - 0.46, i - 0.44), 0.92, 0.88,
                facecolor="#fff0c8" if demo else
                (C_CONT_BG if live else "#f6f6f6"),
                edgecolor=C_DECOY if demo else (C_CONT if live else "#dddddd"),
                linewidth=1.8 if demo else 0.9, zorder=3 if demo else 2))
            ax.text(j, i, str(k), ha="center", va="center",
                    fontsize=9.4, family=MONO, zorder=4,
                    color=C_DECOY if demo else (C_TGT if live else "#9a9a9a"),
                    weight="bold" if (live or demo) else "normal")
    ax.set_xticks(range(P))
    ax.set_xticklabels([f"prod {j}\n{rules[j]}" for j in range(P)],
                       fontsize=7.6, family=MONO)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f"pos {i}" for i in range(L)], fontsize=8.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"$a=\\mathrm{{pos}}\\cdot P+\\mathrm{{prod}}$\n"
                 f"$L\\times P={L}\\times{P}={L*P}$ cells, "
                 f"numbered $0,\\dots,LP-1={L*P-1}$",
                 fontsize=10.0, pad=8)
    fig.text(0.075, 0.012,
             "shaded row: the only cells that exist at $s_0$, where "
             "$|s_0|=1$;   masking, not the index set, decides which "
             "actions are legal",
             fontsize=8.6, color="#555555")

    # -- (b) one decode, and the splice it names ---------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    box(ax, 0.20, 0.90, 0.26, 0.115, f"a = {a}", "#fff0c8", C_DECOY,
        fontsize=13, family=MONO, weight="bold", tc=C_DECOY)
    ax.annotate("", xy=(0.20, 0.755), xytext=(0.20, 0.838),
                arrowprops=dict(arrowstyle="-|>", color=C_TGT, lw=1.2))
    ax.text(0.235, 0.795, f"divmod({a}, {P})", fontsize=10.0, va="center",
            family=MONO)
    box(ax, 0.20, 0.685, 0.30, 0.115, f"({pos}, {prod})", "white", C_NT,
        fontsize=13, family=MONO, weight="bold", tc=C_NT)
    ax.text(0.39, 0.712, "pos: which cell", fontsize=9.4, va="center")
    ax.text(0.39, 0.652, "prod: which rule", fontsize=9.4, va="center")

    s0 = ("+", "S", "S", "S")
    s1 = s0[:pos] + tuple(rhs) + s0[pos + 1:]
    ch, cw = 0.105, 0.088

    def strip(x0, y, toks, mark, mcol, mbg):
        for i, tok in enumerate(toks):
            hot, nt = mark(i), tok == "S"
            ax.add_patch(plt.Rectangle(
                (x0 + i * cw, y - ch / 2), cw * 0.90, ch,
                facecolor=mbg if hot else (C_NT_BG if nt else "#f2f2f2"),
                edgecolor=mcol if hot else (C_NT if nt else "#c8c8c8"),
                linewidth=1.7 if hot else 1.0))
            ax.text(x0 + i * cw + cw * 0.45, y, tok, ha="center", va="center",
                    fontsize=11.0, family=MONO,
                    color=C_NT if nt else C_TGT,
                    weight="bold" if hot else "normal")

    strip(0.02, 0.44, s0, lambda i: i == pos, C_DECOY, "#fff0c8")
    ax.text(0.40, 0.44, f"cell {pos} holds\na nonterminal", fontsize=9.0,
            color=C_DECOY, va="center")
    ax.annotate("", xy=(0.20, 0.225), xytext=(0.20, 0.365),
                arrowprops=dict(arrowstyle="-|>", color=C_TGT, lw=1.2))
    ax.text(0.235, 0.295, f"write rule {prod}", fontsize=9.0, family=MONO,
            va="center")

    strip(0.02, 0.155, s1, lambda i: pos <= i < pos + len(rhs), C_OK,
          "#e6f2ec")
    ax.text(0.02, 0.155 - ch,
            f"$|s|:\\ {len(s0)}\\to{len(s1)}={len(s0)}+{len(rhs)}-1$, "
            f"legal because ${len(s1)}<L={L}$",
            fontsize=8.8, color=C_OK, va="top")

    fig.savefig(os.path.join(HERE, out), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# FIGURE -- family_targets.png : the family, on the plane
# ===========================================================================
def pretty(coeffs) -> str:
    """'y = 1 - x + 2x^2', the way one would write it by hand."""
    parts = []
    for p, c in enumerate(coeffs):
        if c == 0 and p > 0:
            continue
        mag = f"{abs(c):g}"
        if p == 0:
            body = mag
        else:
            unit = "x" if p == 1 else f"x^{p}"
            body = unit if abs(c) == 1 else f"{mag}{unit}"
        sign = "-" if c < 0 else ("+" if parts else "")
        parts.append(f"{sign}{body}" if not parts else f"{sign} {body}")
    return "$y = " + " ".join(parts) + "$"


def fig_family(fam: Family, out="family_targets.png"):
    fig = plt.figure(figsize=(13.2, 5.8))
    gs = GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 0.18, 1.55],
                  height_ratios=[1, 1], wspace=0.36, hspace=0.62,
                  left=0.045, right=0.985, top=0.88, bottom=0.115)

    names = list(TARGETS)
    for k, name in enumerate(names):
        tgt = TARGETS[name]
        ax = fig.add_subplot(gs[k // 4, k % 4])
        xs = np.linspace(tgt.x_min, tgt.x_max, 400)
        ys = tgt.ys(xs)
        col = C_CONT if tgt.family == "linear" else C_DECOY
        ax.plot(xs, ys, color=col, lw=2.0)
        ax.plot(tgt.xs(), tgt.ys(tgt.xs()), ".", color=col, ms=2.6, alpha=0.55)
        ax.axhline(0, color="0.75", lw=0.8, zorder=1)
        for r in tgt.roots_in_domain():
            ax.plot([r], [0], "o", color=C_TGT, ms=5.0, zorder=5)
        if tgt.vertex_in_domain():
            xv = tgt.vertex
            ax.plot([xv], [tgt.ys(np.array([xv]))[0]], "^", color=C_OK,
                    ms=7.0, zorder=5)
        ax.set_xlim(tgt.x_min, tgt.x_max)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(labelsize=8)
        ax.set_title(name, fontsize=9.4, family=MONO, pad=18, color=col)
        ax.text(0.5, 1.035, pretty(tgt.coeffs), transform=ax.transAxes,
                fontsize=9.0, ha="center", va="bottom")
        lo, hi = float(ys.min()), float(ys.max())
        ax.text(0.03, 0.94, f"spread $\\Delta y = {hi - lo:.3g}$",
                transform=ax.transAxes, fontsize=7.8, color="#666666",
                va="top")

    # -- the invariant plane: quadratics --------------------------------
    ax = fig.add_subplot(gs[0, 5])
    quads = [t for t in TARGETS.values() if t.family == "quadratic"]
    ax.axvspan(-1, 1, color=C_CONT_BG, zorder=0)
    ax.axhline(0, color="0.55", lw=1.0, zorder=1)
    ax.axvline(-1, color=C_CONT, lw=0.9, ls=":", zorder=1)
    ax.axvline(1, color=C_CONT, lw=0.9, ls=":", zorder=1)
    for t in quads:
        xv = float(np.clip(t.vertex, -1.9, 5.6))
        ax.plot([xv], [t.discriminant], "o", color=C_DECOY, ms=8, zorder=4)
        ax.annotate(t.name, (xv, t.discriminant), fontsize=8.4,
                    family=MONO, xytext=(8, 4), textcoords="offset points")
    ax.set_xlim(-2.1, 6.2)
    ax.set_ylim(-9.5, 16)
    ax.set_xlabel("vertex $x_v=-c_1/2c_2$", fontsize=9)
    ax.set_ylabel(r"discriminant $\mathrm{disc}=c_1^2-4c_0c_2$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.0, 15.0, "domain", fontsize=8, color=C_CONT, ha="center")
    ax.text(6.0, 1.0, r"$\mathrm{disc}>0$: two zeros", fontsize=8, color="#666666",
            ha="right", va="bottom")
    ax.text(6.0, -1.0, r"$\mathrm{disc}<0$: none", fontsize=8, color="#666666",
            ha="right", va="top")
    ax.set_title("quadratic family: two invariants", fontsize=9.5, pad=5)

    # -- the invariant line: linears ------------------------------------
    ax = fig.add_subplot(gs[1, 5])
    lins = [t for t in TARGETS.values() if t.family == "linear"]
    ax.axvspan(-1, 1, color=C_CONT_BG, zorder=0)
    ax.axvline(-1, color=C_CONT, lw=0.9, ls=":", zorder=1)
    ax.axvline(1, color=C_CONT, lw=0.9, ls=":", zorder=1)
    for k, t in enumerate(lins):
        r = t.real_roots[0] if t.real_roots else np.nan
        rc = float(np.clip(r, -2.6, 2.6))
        y = 0.55 - 0.30 * k
        ax.plot([rc], [y], "o", color=C_CONT, ms=8, zorder=4)
        lbl = f"{t.name}  zero {r:.3g}" + ("  (clipped)" if abs(r) > 2.6 else "")
        ax.annotate(lbl, (rc, y), fontsize=8.4, family=MONO,
                    xytext=(9, -3), textcoords="offset points")
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.55, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("zero $-c_0/c_1$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title("linear family: the only invariant", fontsize=9.5, pad=5)

    fig.text(0.045, 0.028,
             "each panel: the target on its own $y$-scale, sampled at the "
             "$n=41$ grid points;   $\\bullet$ zero inside the domain,   "
             "$\\blacktriangle$ vertex inside the domain",
             fontsize=8.8, color="#555555")

    fig.savefig(os.path.join(HERE, out), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# FIGURE -- family_difficulty.png : where the difficulty is
# ===========================================================================
def fig_difficulty(fam: Family, out="family_difficulty.png"):
    names = list(TARGETS)
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = GridSpec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.15],
                  wspace=0.22, hspace=0.42,
                  left=0.055, right=0.985, top=0.92, bottom=0.075)

    # -- (a) reward ceiling against expression length -----------------------
    ax = fig.add_subplot(gs[0, 0])
    for name in names:
        st = fam.stats[name]
        ks = sorted(st["best_len"])
        vs = [st["best_len"][k] for k in ks]
        hot = name in ("quad_B", "lin_A")
        ax.plot(ks, vs, "-o", ms=4.2,
                color=(C_DECOY if name == "quad_B" else
                       C_CONT if name == "lin_A" else "#c9c9c9"),
                lw=2.2 if hot else 1.1, zorder=4 if hot else 2,
                label=name if hot else None)
        if hot:                       # label the elbow, not the shared endpoint
            j = 1 if name == "lin_A" else 2
            ax.annotate(name, (ks[j], vs[j]), fontsize=8.8, family=MONO,
                        color=C_DECOY if name == "quad_B" else C_CONT,
                        xytext=(-42, -4) if name == "lin_A" else (10, 4),
                        textcoords="offset points")
    qb = fam.stats["quad_B"]["best_len"]
    ax.annotate(f"{qb[5]:.6f}", (5, qb[5]), fontsize=8.6, family=MONO,
                color=C_DECOY, xytext=(6, -17), textcoords="offset points")
    ax.text(0.99, 0.06, "grey: the other six targets", transform=ax.transAxes,
            fontsize=8.4, color="#9a9a9a", ha="right")
    ax.plot([5, 9], [qb[5], qb[9]], color=C_DECOY, lw=6, alpha=0.18, zorder=1)
    ax.annotate("plateau: 3 more actions buy $0.0028$",
                (7, qb[7]), fontsize=8.6, color=C_DECOY,
                xytext=(-4, -34), textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=C_DECOY, lw=0.8))
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_xlabel("tokens in the terminal expression", fontsize=9.5)
    ax.set_ylabel("best reachable $R^2$", fontsize=9.5)
    ax.set_ylim(-0.06, 1.06)
    ax.tick_params(labelsize=8.5)
    ax.set_title("(a)  the reward ceiling is bought by length", fontsize=10,
                 loc="left")

    # -- (c) deception / rarity plane ---------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axvspan(0, 0.75, color=C_DECOY_BG, zorder=0)
    ax.axvline(0, color=C_DECOY, lw=1.0, zorder=1)
    off = {"lin_D": (-10, 10), "quad_C": (8, -3), "lin_B": (8, -3)}
    for name in names:
        st = fam.stats[name]
        x, y = st["margin"], st["rhoC"]
        col = C_OK if st["solved"] else (C_DECOY if st["trap"] else C_CONT)
        ax.plot([x], [y], "o", ms=9, color=col, zorder=4)
        ax.annotate(name, (x, y), fontsize=8.6, family=MONO, color=col,
                    xytext=off.get(name, (8, -3)), textcoords="offset points")
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=C_DECOY,
                   label="trap: $V^q$ prefers a suboptimal terminal"),
        plt.Line2D([], [], marker="o", ls="", color=C_OK,
                   label="the one-action terminal is already exact"),
        plt.Line2D([], [], marker="o", ls="", color=C_CONT,
                   label="no inversion"),
    ], loc="lower left", fontsize=8.2, handletextpad=0.4,
        bbox_to_anchor=(0.0, -0.44), ncol=1)
    ax.set_xlim(-0.45, 0.75)
    ax.set_yscale("log")
    ax.set_ylim(0.055, 1.15)
    ax.set_xlabel(r"$\max_i R(D_i)-V^q(C)$   (deception margin)", fontsize=9.5)
    ax.set_ylabel(r"$\rho_0(C)$", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.text(0.73, 0.062, "mean rollout prefers a one-action terminal",
            fontsize=8.6, color=C_DECOY, ha="right")
    ax.set_title("(c)  deception against rarity", fontsize=10, loc="left")

    # -- (b) the four root children, per target -----------------------------
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(-1.08, 1.30)
    ax.set_ylim(-0.6, len(names) - 0.35)
    dec_names = [d for d, _ in fam.stats["lin_A"]["decoys"]]
    for k, name in enumerate(names):
        st = fam.stats[name]
        y = len(names) - 1 - k
        inv = st["trap"]
        ax.axhspan(y - 0.42, y + 0.42,
                   color="#fbf1f0" if inv else
                   ("#eef6f1" if st["solved"] else "white"), zorder=0)
        # continue branch: the interval it spans
        ax.plot([st["vqC"], st["vstarC"]], [y, y], color=C_CONT, lw=2.4,
                alpha=0.45, zorder=2, solid_capstyle="round")
        ax.plot([st["vqC"]], [y], "o", ms=8, color=C_CONT, zorder=4)
        ax.plot([st["vstarC"]], [y], "*", ms=13, color=C_CONT, zorder=4)
        for (rule, r), mk in zip(st["decoys"], ("s", "D", "v")):
            ax.plot([r], [y], mk, ms=6.5, color=C_DECOY, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.text(-1.13, y, name, fontsize=9, family=MONO, ha="right", va="center")
        ax.text(1.34, y, f"{st['margin']:+.3f}", fontsize=8.6, family=MONO,
                ha="right", va="center",
                color=C_DECOY if inv else (C_OK if st["solved"] else "#9a9a9a"))
    ax.text(1.34, len(names) - 0.55, "margin", fontsize=8.4, ha="right",
            color="#666666")
    ax.set_yticks([])
    ax.set_xlabel("reward", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.set_title("(b)  the four children of $s_0$: three terminate at once, "
                 "one continues", fontsize=10, loc="left")
    handles = [
        plt.Line2D([], [], marker="s", ls="", color=C_DECOY,
                   label=f"$D_1$ = {dec_names[0]}"),
        plt.Line2D([], [], marker="D", ls="", color=C_DECOY,
                   label=f"$D_2$ = {dec_names[1]}"),
        plt.Line2D([], [], marker="v", ls="", color=C_DECOY,
                   label=f"$D_3$ = {dec_names[2]}"),
        plt.Line2D([], [], marker="o", ls="", color=C_CONT, label="$V^q(C)$"),
        plt.Line2D([], [], marker="*", ls="", ms=12, color=C_CONT,
                   label="$V^*(C)$"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=5, fontsize=8.6,
              bbox_to_anchor=(0.0, -0.30), handletextpad=0.4,
              columnspacing=1.6)

    fig.savefig(os.path.join(HERE, out), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
def report(fam: Family):
    print(f"\nforms {len(fam.children):,}   terminals {len(fam.terminals)}   "
          f"root children {[fam.decode(c) for c in fam.kids]}")
    print(f"\n{'target':9s} {'V*(C)':>7s} {'V^q(C)':>9s} {'rho0(C)':>9s} "
          f"{'D1':>9s} {'D2':>9s} {'D3':>9s} {'margin':>9s}")
    for name, st in fam.stats.items():
        d = [r for _, r in st["decoys"]]
        print(f"{name:9s} {st['vstarC']:7.4f} {st['vqC']:9.6f} "
              f"{st['rhoC']:9.6f} {d[0]:9.6f} {d[1]:9.6f} {d[2]:9.6f} "
              f"{st['margin']:+9.6f}")
    print("\nLaTeX rows:")
    for name, st in fam.stats.items():
        d = [r for _, r in st["decoys"]]
        print(f"    \\texttt{{{name}}} & ${st['target'].infix}$ & "
              f"${d[0]:.4f}$ & ${d[1]:.4f}$ & ${d[2]:.4f}$ & "
              f"${st['vqC']:.4f}$ & ${st['rhoC']:.4f}$ & "
              f"${st['margin']:+.4f}$ \\\\")


if __name__ == "__main__":
    fam = Family()
    report(fam)
    fig_action_index(fam)
    fig_family(fam)
    fig_difficulty(fam)
