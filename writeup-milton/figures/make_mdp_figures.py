"""Figures for the derivation-MDP subsection: how the game actually plays out.

Unlike ``make_scenario_figures.py``, which draws the *idealised* controlled
grammar analytically, this script replays the **real** environment
(``sraz.instances.symreg.game.SymRegGame`` on the ``additive_quadratic``
instance) and the **real** search (``sraz.core.mcts.MCTS`` with a uniform prior
and random rollouts). Nothing is hand-transcribed: every token buffer, visit
count, Q value, V*, V^q and rho_0 printed on the figures is computed here.

    mdp_derivation.png   the derivation MDP: one episode's token buffer, the
                         tree expansion below the root with the exact triple
                         (V*, V^q, rho_0) on each root child, the terminal
                         reward in data space, and the rarity of the optimum.

    mcts_expansion.png   the MCTS tree as it grows over the simulation budget
                         B in {8, 32, 128}, followed by the visit allocation,
                         the exploitation statistic against its exact limit,
                         and the best expression actually observed.

Exact statistics are censuses, not estimates: every sentential form reachable
under the action mask is enumerated (4898 of them at ``MAX_LEN = 12``), and
V^q / rho_0 are computed by exact expectation over the uniform action mask
rather than by sampling.

Run from anywhere:

    python3 writeup-milton/figures/make_mdp_figures.py

Outputs are written next to this script.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from sraz.core.mcts import MCTS  # noqa: E402
from sraz.core.policy_value_net import UniformPolicyValueNet  # noqa: E402
from sraz.instances.symreg.game import SymRegGame, prefix_to_infix  # noqa: E402
from sraz.instances.symreg.problems import get_problem  # noqa: E402

sys.setrecursionlimit(100_000)

# --- house style (matches make_scenario_figures.py) -------------------------
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

# --- one colour per semantic role, as in make_scenario_figures.py -----------
C_DECOY = "#c44e52"     # the shallow branch with the highest reachable reward
C_DECOY_BG = "#f8e3e2"
C_CONT = "#4c72b0"      # the compositional branch that contains the optimum
C_CONT_BG = "#dfe7f2"
C_DEAD = "#9a9a9a"      # shallow branches that are not even competitive
C_DEAD_BG = "#f0f0f0"
C_TGT = "#222222"
C_OK = "#2e7d5b"
C_BAD = "#b0442f"

ROLE_C = {"decoy": C_DECOY, "cont": C_CONT, "dead": C_DEAD}
ROLE_BG = {"decoy": C_DECOY_BG, "cont": C_CONT_BG, "dead": C_DEAD_BG}

# --- the instance ----------------------------------------------------------
# max_len = 12 is the smallest buffer whose mask admits the exact three-term
# expression '+ C0 + * C1 x * C2 * x x' (11 tokens; the mask requires strictly
# fewer than max_len tokens), so V*(s_0) = 1 exactly. At max_len = 11 the
# optimum is excluded and V*(s_0) = 0.998597.
MAX_LEN = 12
LMFIT_MAX_NFEV = 50
PROBLEM = "additive_quadratic"
BUDGETS = (8, 32, 128)
MCTS_SEED = 0


class RecordingGame(SymRegGame):
    """A game that logs every terminal evaluation, across clones.

    ``MCTS`` evaluates rollout leaves on ``self.game.clone()``, i.e. on a
    deepcopy, so a per-instance log would be thrown away with the clone.
    ``LOG`` is a *class* attribute: ``deepcopy`` copies instance dictionaries
    only, so every clone appends to the same list. This is what makes the
    budget accounting honest -- the writeup's budget is one terminal
    evaluation per simulation, and this counts them.
    """

    LOG: list[float] = []

    def _fit_cached(self, rule: str) -> float:
        r = super()._fit_cached(rule)
        RecordingGame.LOG.append(r)
        return r


def build_game(cls=SymRegGame, max_len: int = MAX_LEN):
    prob = get_problem(PROBLEM)
    return cls(max_len=max_len, lmfit_max_nfev=LMFIT_MAX_NFEV,
               **prob.game_kwargs())


# ===========================================================================
# Exact census over sentential forms
# ===========================================================================
class Census:
    """Exact V*, V^q and rho_0 for every mask-reachable sentential form.

    ``V^q`` is taken under the *codebase's* rollout policy, which is uniform
    over the flattened ``(position, production)`` legal-action mask -- not
    uniform over the productions of one designated slot. The two measures
    differ as soon as a form holds more than one nonterminal, so which one is
    meant is load-bearing.
    """

    def __init__(self, game: SymRegGame):
        self.game = game
        self.G = game.grammar
        self._vstar: dict[tuple, float] = {}
        self._vq: dict[tuple, float] = {}
        self._rho: dict[tuple, float] = {}
        self._nleaf: dict[tuple, int] = {}
        self.root = (self.G.symdict[self.G.start],)
        self.vstar_root = self.vstar(self.root)
        self._walk(self.root)

    # -- grammar mechanics, mirroring get_action_mask / step ----------------
    def legal(self, state: tuple) -> list[tuple[int, int]]:
        out = []
        for i, t in enumerate(state):
            if t in self.G.nonterms:
                for j in self.G.proddict[t]:
                    if len(state) + len(self.G.productions[j]) - 1 < self.game.state_len:
                        out.append((i, j))
        return out

    def child(self, state: tuple, pos: int, prod: int) -> tuple:
        rhs = self.G.productions[prod]
        return state[:pos] + tuple(rhs) + state[pos + 1:]

    def rule(self, state: tuple) -> str:
        return " ".join(self.G.tokenlist[t] for t in state)

    def reward(self, state: tuple) -> float:
        return self.game._fit_cached(self.rule(state))

    # -- the three exact statistics ----------------------------------------
    def vstar(self, state: tuple) -> float:
        hit = self._vstar.get(state)
        if hit is not None:
            return hit
        acts = self.legal(state)
        v = (self.reward(state) if not acts
             else max(self.vstar(self.child(state, i, j)) for i, j in acts))
        self._vstar[state] = v
        return v

    def _walk(self, state: tuple) -> tuple[float, float]:
        hit = self._vq.get(state)
        if hit is not None:
            return hit, self._rho[state]
        acts = self.legal(state)
        if not acts:
            r = self.reward(state)
            vq, rho = r, float(r >= self.vstar_root - 1e-12)
        else:
            vq = rho = 0.0
            for i, j in acts:
                a, b = self._walk(self.child(state, i, j))
                vq += a
                rho += b
            vq /= len(acts)
            rho /= len(acts)
        self._vq[state] = vq
        self._rho[state] = rho
        return vq, rho

    def vq(self, state: tuple) -> float:
        return self._walk(state)[0]

    def rho0(self, state: tuple) -> float:
        return self._walk(state)[1]

    def n_terminals(self, state: tuple) -> int:
        hit = self._nleaf.get(state)
        if hit is not None:
            return hit
        acts = self.legal(state)
        n = (1 if not acts
             else sum(self.n_terminals(self.child(state, i, j)) for i, j in acts))
        self._nleaf[state] = n
        return n

    def n_forms(self) -> int:
        return len(self._vq)

    def argmax_terminal(self, state: tuple) -> tuple:
        """A terminal attaining V*(state); ties broken first-found."""
        cur = state
        while True:
            acts = self.legal(cur)
            if not acts:
                return cur
            target = self.vstar(cur)
            for i, j in acts:
                ch = self.child(cur, i, j)
                if abs(self.vstar(ch) - target) < 1e-12:
                    cur = ch
                    break
            else:                                      # pragma: no cover
                return cur


# ===========================================================================
# Drawing helpers
# ===========================================================================
def box(ax, xc, yc, w, h, text, fc, ec, fontsize=9.0, tc="black", lw=1.1,
        weight="normal", z=3, family=None, ha="center"):
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z, clip_on=False))
    tx = xc if ha == "center" else xc - w / 2 + 0.012
    ax.text(tx, yc, text, ha=ha, va="center", fontsize=fontsize, color=tc,
            zorder=z + 1, weight=weight, linespacing=1.5, clip_on=False,
            family=family)


def arrow(ax, p0, p1, color, lw=1.2, rad=0.0, z=2, ls="-", scale=9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=scale,
        shrinkA=1.0, shrinkB=1.0, linewidth=lw, linestyle=ls,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=z))


# ===========================================================================
# FIGURE 1 -- the derivation MDP
# ===========================================================================
def panel_trace(ax, cen: Census, trace):
    """(a) One episode: the token buffer rewritten action by action."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    n = len(trace)
    ys = np.linspace(0.93, 0.07, n)
    x_state, x_arrow, x_act = 0.410, 0.132, 0.680

    for k, step in enumerate(trace):
        toks = step["tokens"]
        pad = cen.game.state_len - len(toks)
        buf = ("  ".join(cen.G.tokenlist[t] for t in toks)
               + ("   " + "·  " * pad if pad else ""))
        term = step["terminal"]
        fc, ec, tc = ((C_CONT_BG, C_CONT, C_CONT) if term
                      else ("0.97", "0.55", "0.15"))
        box(ax, x_state, ys[k], 0.50, 0.088, buf, fc, ec, fontsize=8.2,
            family="monospace", tc=tc, ha="left")
        ax.text(0.004, ys[k], rf"$s_{{{k}}}$", ha="left", va="center",
                fontsize=9.2, color="0.25")
        ax.text(0.040, ys[k], rf"$|\mathcal{{A}}(s_{{{k}}})|={step['n_actions']}$",
                ha="left", va="center", fontsize=7.2, color="0.50")

        if step["action"] is not None:
            pos, prod = step["action"]
            rhs = " ".join(cen.G.tokenlist[t] for t in cen.G.productions[prod])
            flat = pos * cen.G.nprods + prod
            ymid = (ys[k] + ys[k + 1]) / 2
            arrow(ax, (x_arrow, ys[k] - 0.044),
                  (x_arrow, ys[k + 1] + 0.044), "0.50", lw=1.0)
            ax.text(x_act, ymid,
                    rf"$a_{{{k}}}={flat}$:  $(\mathrm{{pos}},\mathrm{{prod}})"
                    rf"=({pos},{prod})$,   splice  ",
                    ha="right", va="center", fontsize=7.6, color="0.30")
            ax.text(x_act, ymid, rf"$S\to$ {rhs}", ha="left", va="center",
                    fontsize=7.6, color="0.15", family="monospace")

    last = trace[-1]
    ax.text(x_act + 0.010, ys[-1],
            rf"no nonterminal left:  $R(\pi)={last['reward']:.6f}$",
            ha="left", va="center", fontsize=8.4, color=C_CONT)
    ax.set_title(r"(a) one episode of the derivation MDP: the length-"
                 rf"${cen.game.state_len}$ token buffer, rewritten action by "
                 r"action  ($\cdot$ = pad)",
                 fontsize=10.0, pad=2, loc="left")


def panel_tree(ax, cen: Census, kids):
    """(b) The tree below the root, with the exact triple on each child."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    x_root, x_kid, x_val = 0.038, 0.205, 0.700
    ys = np.linspace(0.90, 0.14, len(kids))

    box(ax, x_root, 0.52, 0.060, 0.105, "$s_0$", "white", "0.35",
        fontsize=10.5, weight="bold")
    ax.text(x_root, 0.445, "S", ha="center", va="top", fontsize=8.0,
            color="0.45", family="monospace")

    for k, kid in enumerate(kids):
        col, bg = ROLE_C[kid["role"]], ROLE_BG[kid["role"]]
        arrow(ax, (x_root + 0.032, 0.52), (x_kid - 0.088, ys[k]), col,
              lw=1.7 if kid["role"] != "dead" else 0.9,
              rad=0.09 * np.sign(ys[k] - 0.52))
        box(ax, x_kid, ys[k], 0.165, 0.098, f"S $\\to$ {kid['rhs']}", bg, col,
            fontsize=8.2, tc=col)
        tag = {"decoy": "shallow decoy",
               "cont": "compositional",
               "dead": "shallow, not competitive"}[kid["role"]]
        ax.text(x_kid + 0.093, ys[k] + 0.022, tag, ha="left", va="center",
                fontsize=7.2, color=col)
        ax.text(x_kid + 0.093, ys[k] - 0.026,
                f"{kid['n_terminals']:,} terminal"
                f"{'s' if kid['n_terminals'] != 1 else ''}",
                ha="left", va="center", fontsize=6.8, color=col, alpha=0.8)
        box(ax, x_val, ys[k], 0.480, 0.084,
            rf"$V^*={kid['vstar']:+.6f}$        "
            rf"$V^q={kid['vq']:+.6f}$        "
            rf"$\rho_0={kid['rho0']:.5f}$",
            "white", col, fontsize=8.2, tc=col, lw=0.8)

    d = max((k for k in kids if k["role"] == "decoy"), key=lambda k: k["vstar"])
    c = next(k for k in kids if k["role"] == "cont")
    inverted = d["vq"] > c["vq"] and d["vstar"] < c["vstar"]
    col, bg = (C_BAD, "#fbe6e0") if inverted else (C_OK, "#e2f0e9")
    verb = "inverted" if inverted else "preserved"
    rel = ">" if d["vq"] > c["vq"] else "<"
    box(ax, 0.5, -0.02, 0.99, 0.092,
        rf"root ordering $\bf{{{verb}}}$:   "
        rf"$V^q(D)={d['vq']:.4f} {rel} V^q(C)={c['vq']:.4f}$,   while   "
        rf"$V^*(D)={d['vstar']:.4f} < V^*(C)={c['vstar']:.4f}$",
        bg, col, fontsize=8.6, tc=col)

    ax.set_title(rf"(b) exact census below the root: "
                 rf"${cen.n_forms():,}$ sentential forms enumerated, "
                 rf"$V^*(s_0)={cen.vstar_root:.6f}$",
                 fontsize=10.0, pad=2, loc="left")


def panel_reward(ax, cen: Census, kids):
    """(c) The terminal reward in data space: what each branch can fit."""
    g = cen.game
    xs, ys = g.xs, g.exact_ys
    d = max((k for k in kids if k["role"] == "decoy"), key=lambda k: k["vstar"])
    c = next(k for k in kids if k["role"] == "cont")

    ax.plot(xs, ys, "o", ms=3.6, color=C_TGT, zorder=5,
            label=f"target: {g.target_infix}")
    ax.plot(xs, c["fit"], "-", lw=1.6, color=C_CONT, zorder=3,
            label=rf"best under $C$ (`{c['rhs']}`): $R^2={c['vstar']:.4f}$")
    ax.plot(xs, d["fit"], "--", lw=1.9, color=C_DECOY, zorder=4,
            label=rf"best under $D$ (`{d['rhs']}`): $R^2={d['vstar']:.4f}$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    lo, hi = float(ys.min()), float(ys.max())
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.46 * (hi - lo))
    ax.legend(fontsize=6.9, loc="upper left", handlelength=1.8,
              labelspacing=0.30, borderaxespad=0.15)
    ax.text(0.985, 0.03,
            rf"$D$ leaves $1-R^2={1 - d['vstar']:.4f}$ unexplained,"
            "\n" r"but is structurally wrong",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
            color=C_DECOY, linespacing=1.4)
    ax.set_title("(c) the terminal reward: a near-perfect wrong form",
                 fontsize=9.5, pad=4, loc="left")


def panel_rarity(ax, cen: Census, kids):
    """(d) Hit curve: how many uniform completions the optimum costs."""
    c = next(k for k in kids if k["role"] == "cont")
    rho_root = cen.rho0(cen.root)

    K = np.unique(np.round(np.logspace(0, 3.0, 240)).astype(int))
    rows = ((c["rho0"], C_CONT, r"after committing to $C$", 0.0),
            (rho_root, "0.40", r"from the root $s_0$", -0.19))
    for rho, col, lab, dy in rows:
        ax.plot(K, 1.0 - (1.0 - rho) ** K, color=col, lw=1.9,
                label=rf"{lab}:  $\rho_0={rho:.4f}$")
        k95 = int(np.ceil(np.log(0.05) / np.log1p(-rho)))
        ax.plot([k95], [0.95], "o", ms=5.5, color=col, zorder=5)
        ax.annotate(rf"$K_{{0.95}}={k95}$", xy=(k95, 0.95),
                    xytext=(k95 * 1.75, 0.87 + dy), fontsize=7.6, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=0.7))

    ax.axhline(0.95, ls=":", lw=0.9, color="0.65")
    ax.set_xscale("log")
    ax.set_xlabel(r"independent uniform completions $K$")
    ax.set_ylabel(r"$P_{\mathrm{hit}}(K)=1-(1-\rho_0)^K$")
    ax.set_ylim(0, 1.06)
    ax.legend(fontsize=7.2, loc="lower right", labelspacing=0.30)
    ax.set_title(r"(d) rarity of the optimum under uniform completion",
                 fontsize=9.5, pad=4, loc="left")


def figure_derivation(cen: Census, trace, kids):
    fig = plt.figure(figsize=(11.0, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.94, 1.04, 1.00],
                          hspace=0.26, wspace=0.24,
                          left=0.055, right=0.985, top=0.955, bottom=0.075)
    panel_trace(fig.add_subplot(gs[0, :]), cen, trace)
    panel_tree(fig.add_subplot(gs[1, :]), cen, kids)
    panel_reward(fig.add_subplot(gs[2, 0]), cen, kids)
    panel_rarity(fig.add_subplot(gs[2, 1]), cen, kids)
    out = os.path.join(HERE, "mdp_derivation.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# FIGURE 2 -- the MCTS tree as the budget grows
# ===========================================================================
def pad_key(tokens, game):
    buf = np.full(game.state_len, game.grammar.pad_tok, dtype=np.int64)
    buf[:len(tokens)] = np.array(tokens, dtype=np.int64)
    return buf.tobytes()


def flat_of(edge):
    return int(edge[0] if isinstance(edge, tuple) else edge)


def run_search(budget: int, seed: int = MCTS_SEED):
    """One real ``perform_simulations`` from the root, with evaluations logged."""
    game = build_game(RecordingGame)
    net = UniformPolicyValueNet(
        n_actions=game.state_len * game.grammar.nprods, value=0.0)
    game.reset_wrapper()
    RecordingGame.LOG = []
    mcts = MCTS(game, net, n_simulations=budget, temperature=1.0,
                c_exploration=1.0, rollout_n=1, rollout_blend=0.0,
                rollout_budget=10 ** 7, backup_rule="mean",
                rng=np.random.default_rng(seed))
    mcts.perform_simulations(None, add_noise=False)
    return mcts, game, list(RecordingGame.LOG)


def harvest(mcts, game, cen: Census, roles, max_depth=3, min_visits=2):
    """Read the built tree into a node/edge list with a tidy layered layout."""
    nodes, edges = [], []

    def add(tokens, depth, role):
        nid = len(nodes)
        nd = mcts.nodes.get(pad_key(tokens, game))
        nodes.append({"id": nid, "tokens": tokens, "depth": depth,
                      "role": role, "kids": [],
                      "total_N": 0 if nd is None else nd.total_N,
                      "terminal": True if nd is None else nd.is_terminal_state})
        if nd is None or nd.is_terminal_state or depth >= max_depth:
            return nid
        for edge, n in sorted(nd.action_N.items(), key=lambda kv: -kv[1]):
            if depth > 0 and n < min_visits:
                continue
            pos, prod = divmod(flat_of(edge), cen.G.nprods)
            ct = cen.child(tokens, pos, prod)
            if pad_key(ct, game) not in mcts.nodes:
                continue
            cid = add(ct, depth + 1, roles.get(prod, role) if depth == 0 else role)
            nodes[nid]["kids"].append(cid)
            edges.append({"parent": nid, "child": cid, "N": n,
                          "Q": nd.action_Q.get(edge, 0.0),
                          "role": nodes[cid]["role"], "depth": depth})
        return nid

    add(cen.root, 0, "cont")

    # tidy layout: leaves take consecutive slots, parents centre on their kids
    counter = [0.0]

    def place(nid):
        nd = nodes[nid]
        if not nd["kids"]:
            nd["_y"] = counter[0]
            counter[0] += 1.0
        else:
            for cid in nd["kids"]:
                place(cid)
            nd["_y"] = float(np.mean([nodes[c]["_y"] for c in nd["kids"]]))

    place(0)
    span = max(1.0, counter[0] - 1.0)
    maxd = max(nd["depth"] for nd in nodes)
    for nd in nodes:
        nd["x"] = 0.11 + 0.78 * (nd["depth"] / max(1, maxd))
        nd["y"] = 0.93 - 0.86 * (nd["_y"] / span)
    return nodes, edges


def panel_mcts_tree(ax, nodes, edges, cen: Census, budget, total_N, n_evals):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.03)
    ax.axis("off")
    ax.grid(False)
    pos = {nd["id"]: (nd["x"], nd["y"]) for nd in nodes}

    for e in edges:
        p0, p1 = pos[e["parent"]], pos[e["child"]]
        col = ROLE_C[e["role"]]
        arrow(ax, p0, p1, col, lw=0.7 + 3.6 * (e["N"] / max(1, total_N)),
              rad=0.0, z=2, scale=7)

    stats = {e["child"]: e for e in edges if e["depth"] == 0}
    for nd in nodes:
        xc, yc = pos[nd["id"]]
        col = ROLE_C[nd["role"]]
        ms = 4.5 + 9.0 * np.sqrt(nd["total_N"] / max(1, total_N))
        ax.plot([xc], [yc], "o", ms=ms, color=col, zorder=5,
                mfc=col if nd["terminal"] else "white", mew=1.4)
        if nd["depth"] == 1:
            e = stats.get(nd["id"])
            lab = " ".join(cen.G.tokenlist[t] for t in nd["tokens"])
            # a node with children continues rightwards, so label above it;
            # a terminal has clear space below.
            # one line per node keeps siblings' labels from colliding; an
            # internal node is labelled above so the label clears its chain.
            up = bool(nd["kids"])
            dy, va = ((0.050, "bottom") if up else (-0.050, "top"))
            if e is not None:
                lab = f"{lab}   N={e['N']}, Q={e['Q']:+.3f}"
            ax.text(xc + 0.022, yc + dy, lab, ha="left", va=va,
                    fontsize=6.0, color=col, family="monospace")

    ax.set_title(rf"$B={budget}$:  {len(nodes)} tree nodes, "
                 rf"{n_evals} terminal evaluations",
                 fontsize=9.0, pad=3, loc="left")


def panel_allocation(ax, kids, series):
    Bs = [s["budget"] for s in series]
    for kid in sorted(kids, key=lambda k: -k["vstar"]):
        frac = [s["share"].get(kid["flat"], 0.0) for s in series]
        col = ROLE_C[kid["role"]]
        ax.plot(Bs, frac, "o-", ms=4.5, lw=1.8, color=col,
                alpha=1.0 if kid["role"] != "dead" else 0.45,
                ls="-" if kid["role"] != "dead" else ":",
                label=rf"`{kid['rhs']}`  ({kid['role']})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs)
    ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("simulation budget $B$")
    ax.set_ylabel(r"visit share  $N_B(s_0,a)/N_B(s_0)$")
    ax.set_ylim(-0.06, 1.34)
    ax.legend(fontsize=6.6, loc="upper center", ncol=2, columnspacing=0.7,
              handlelength=1.5, labelspacing=0.22, borderaxespad=0.05)
    ax.set_title("(d) where the budget goes", fontsize=9.5, pad=4, loc="left")


def panel_statistic(ax, kids, series):
    Bs = [s["budget"] for s in series]
    shown = [max((k for k in kids if k["role"] == "decoy"),
                 key=lambda k: k["vstar"]),
             next(k for k in kids if k["role"] == "cont")]
    for kid in shown:
        col = ROLE_C[kid["role"]]
        name = "D" if kid["role"] == "decoy" else "C"
        ax.plot(Bs, [s["Q"].get(kid["flat"], np.nan) for s in series],
                "o-", ms=4.5, lw=1.8, color=col,
                label=rf"$\overline{{Q}}_B(s_0,{name})$")
        x_lab = Bs[0] * 0.92
        if abs(kid["vq"] - kid["vstar"]) < 1e-9:
            # deterministic branch: the two coincide, so draw one line
            ax.axhline(kid["vq"], ls="--", lw=1.2, color=col, alpha=0.85)
            ax.text(x_lab, kid["vq"] - 0.006,
                    rf"$V^q({name})=V^*({name})={kid['vq']:.3f}$",
                    fontsize=6.9, color=col, va="top", ha="left")
        else:
            ax.axhline(kid["vq"], ls="--", lw=1.1, color=col, alpha=0.85)
            ax.axhline(kid["vstar"], ls=":", lw=1.3, color=col, alpha=0.85)
            ax.text(x_lab, kid["vq"] + 0.006, rf"$V^q({name})={kid['vq']:.3f}$",
                    fontsize=6.9, color=col, va="bottom", ha="left")
            ax.text(x_lab, kid["vstar"] + 0.006,
                    rf"$V^*({name})={kid['vstar']:.3f}$",
                    fontsize=6.9, color=col, va="bottom", ha="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs)
    ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("simulation budget $B$")
    ax.set_ylabel(r"$\overline{Q}_B(s_0,\cdot)$")
    ax.set_xlim(Bs[0] * 0.88, Bs[-1] * 1.15)
    ax.set_ylim(0.55, 1.06)
    ax.legend(fontsize=7.2, loc="lower right", labelspacing=0.25)
    ax.set_title(r"(e) $\overline{Q}_B \to V^q$, not $V^*$",
                 fontsize=9.5, pad=4, loc="left")


def panel_best(ax, cen: Census, kids, series):
    Bs = [s["budget"] for s in series]
    d = max((k for k in kids if k["role"] == "decoy"), key=lambda k: k["vstar"])
    ax.plot(Bs, [s["best_seen"] for s in series], "o-", ms=5, lw=1.9,
            color=C_TGT, label=r"best observed $R^{\max}_B$")
    ax.axhline(cen.vstar_root, ls=":", lw=1.4, color=C_OK)
    ax.text(Bs[0] * 0.92, cen.vstar_root - 0.0004,
            rf"$V^*(s_0)={cen.vstar_root:.4f}$", fontsize=7.0, color=C_OK,
            va="top", ha="left")
    ax.axhline(d["vstar"], ls="--", lw=1.3, color=C_DECOY)
    ax.text(Bs[0] * 0.92, d["vstar"] + 0.0004,
            rf"decoy ceiling  ${d['vstar']:.4f}$", fontsize=7.0,
            color=C_DECOY, va="bottom", ha="left")
    gap = cen.vstar_root - max(s["best_seen"] for s in series)
    ax.text(0.04, 0.30,
            "exact recovery never attained:\n"
            rf"simple regret $r_B={gap:.6f}$ at every $B$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0,
            color="0.30", linespacing=1.4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Bs)
    ax.set_xticklabels([str(b) for b in Bs])
    ax.set_xlabel("simulation budget $B$")
    ax.set_ylabel(r"$R^{\max}_B$")
    ax.set_xlim(Bs[0] * 0.88, Bs[-1] * 1.15)
    ax.set_ylim(d["vstar"] - 0.0035, cen.vstar_root + 0.0025)
    ax.legend(fontsize=7.2, loc="center right")
    ax.set_title("(f) best expression actually observed",
                 fontsize=9.5, pad=4, loc="left")


def figure_mcts(cen: Census, kids):
    roles = {k["prod"]: k["role"] for k in kids}
    series, trees = [], []
    for B in BUDGETS:
        mcts, g, log = run_search(B)
        root = mcts.nodes[pad_key(cen.root, g)]
        total = max(1, root.total_N)
        share = {flat_of(e): n / total for e, n in root.action_N.items()}
        qs = {flat_of(e): q for e, q in root.action_Q.items()}
        series.append({"budget": B, "share": share, "Q": qs,
                       "best_seen": max(log) if log else float("nan"),
                       "n_evals": len(log), "total_N": total})
        trees.append((B, *harvest(mcts, g, cen, roles), total, len(log)))
        print(f"  B={B:>4}: tree nodes={len(mcts.nodes):>3}  "
              f"terminal evals={len(log):>4}  best={max(log):+.6f}  "
              f"share={ {k: round(v, 3) for k, v in sorted(share.items())} }")

    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.96, 1.0],
                          hspace=0.30, wspace=0.40,
                          left=0.062, right=0.965, top=0.880, bottom=0.085)
    for k, (B, nodes, edges, total, n_ev) in enumerate(trees):
        panel_mcts_tree(fig.add_subplot(gs[0, k]), nodes, edges, cen, B,
                        total, n_ev)
    panel_allocation(fig.add_subplot(gs[1, 0]), kids, series)
    panel_statistic(fig.add_subplot(gs[1, 1]), kids, series)
    panel_best(fig.add_subplot(gs[1, 2]), cen, kids, series)

    fig.text(0.062, 0.972,
             r"(a)–(c)  the tree built by one persistent root search; edge "
             r"width $\propto N(s,a)$, hollow node = expanded nonterminal, "
             r"filled = terminal.",
             fontsize=8.4, color="0.30", ha="left", va="center")
    fig.text(0.062, 0.938,
             r"Red = the shallow decoy branch, blue = the compositional "
             r"branch containing the optimum, grey = the rest.  Deeper edges "
             r"with $N<2$ are pruned from the drawing only.",
             fontsize=8.0, color="0.45", ha="left", va="center")
    out = os.path.join(HERE, "mcts_expansion.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# Data collection
# ===========================================================================
def collect_trace(cen: Census):
    """The V*-greedy episode, for panel (a)."""
    cur, trace = cen.root, []
    while True:
        acts = cen.legal(cur)
        step = {"tokens": cur, "n_actions": len(acts),
                "terminal": not acts, "action": None,
                "reward": cen.reward(cur) if not acts else None}
        if not acts:
            trace.append(step)
            return trace
        for i, j in acts:
            if abs(cen.vstar(cen.child(cur, i, j)) - cen.vstar(cur)) < 1e-12:
                step["action"] = (i, j)
                break
        trace.append(step)
        cur = cen.child(cur, *step["action"])


def collect_kids(cen: Census):
    """Root children with role, exact triple, and the V*-attaining fit curve.

    The role is decided by what the child can reach, not by its name:
    ``cont`` attains V*(s_0); among the rest, the single best shallow branch is
    the ``decoy`` (it is what a mean-based statistic will prefer), and the
    remainder are ``dead``.
    """
    out = []
    for j in cen.G.proddict[cen.G.symdict[cen.G.start]]:
        rhs = cen.G.productions[j]
        if len(cen.root) + len(rhs) - 1 >= cen.game.state_len:
            continue
        ch = cen.child(cen.root, 0, j)
        out.append({
            "flat": j, "prod": j,
            "rhs": " ".join(cen.G.tokenlist[t] for t in rhs),
            "vstar": cen.vstar(ch), "vq": cen.vq(ch), "rho0": cen.rho0(ch),
            "n_terminals": cen.n_terminals(ch), "fit": _best_fit_curve(cen, ch),
        })
    for kid in out:
        kid["role"] = ("cont" if abs(kid["vstar"] - cen.vstar_root) < 1e-12
                       else "dead")
    rest = [k for k in out if k["role"] == "dead"]
    if rest:
        max(rest, key=lambda k: k["vstar"])["role"] = "decoy"
    return out


def _best_fit_curve(cen: Census, state: tuple) -> np.ndarray:
    """Re-fit the V*-attaining terminal below ``state`` and return its curve."""
    import lmfit
    import sympy
    from sympy.parsing import sympy_parser
    from sraz.instances.symreg.game import C_INIT

    tokens = cen.rule(cen.argmax_terminal(state)).split()
    param_values, ind_vars = {}, ()
    if "x" in tokens:
        param_values["x"] = cen.game.xs
        ind_vars = ("x",)
    for t in tokens:
        if "C" in t:
            param_values[t] = C_INIT
    model = sympy_parser.parse_expr(prefix_to_infix(tokens), evaluate=False)
    fn = sympy.lambdify(list(model.free_symbols), model)
    res = lmfit.Model(fn, independent_vars=ind_vars).fit(
        data=cen.game.exact_ys, max_nfev=LMFIT_MAX_NFEV, **param_values)
    out = np.asarray(res.best_fit, dtype=float)
    return (out if out.shape == cen.game.xs.shape
            else np.full_like(cen.game.xs, float(out.ravel()[0])))


def report(cen: Census, kids):
    g = cen.game
    print(f"instance: {PROBLEM}   max_len={g.state_len}   "
          f"lmfit_max_nfev={LMFIT_MAX_NFEV}")
    print(f"target:   {g.target_infix}   "
          f"constants={ {k: round(v, 6) for k, v in g.constants.items()} }")
    print(f"grid:     x in [{g.xs.min():g}, {g.xs.max():g}], n={len(g.xs)}")
    print(f"census:   {cen.n_forms():,} sentential forms,  "
          f"{cen.n_terminals(cen.root):,} terminals")
    print(f"  root  V*={cen.vstar_root:.12f}  V^q={cen.vq(cen.root):.6f}  "
          f"rho_0={cen.rho0(cen.root):.6e}")
    for kid in kids:
        print(f"  S -> {kid['rhs']:<14} [{kid['role']:>5}]  "
              f"V*={kid['vstar']:+.6f}  V^q={kid['vq']:+.6f}  "
              f"rho_0={kid['rho0']:.6e}  terminals={kid['n_terminals']:,}")


if __name__ == "__main__":
    _cen = Census(build_game())
    _kids = collect_kids(_cen)
    report(_cen, _kids)
    figure_derivation(_cen, collect_trace(_cen), _kids)
    figure_mcts(_cen, _kids)
