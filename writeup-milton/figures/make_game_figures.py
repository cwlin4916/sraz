"""Figures for the section that specifies the game as the *code* plays it.

Two questions, one figure each, both answered by executing
``sraz.instances.symreg.game.SymRegGame`` -- nothing here is transcribed by
hand:

    game_mdp.png          (i)  what the MDP is: the state, the action, the mask
                               that decides which actions are legal, the
                               transition, and the reward, each panel labelled
                               with the value the code actually produces.

    game_termination.png  (ii) when an episode ends: the three exits in
                               ``step``, the length ledger that bounds every
                               terminal, and the escape-production argument
                               that rules out the third exit entirely.

Every count is an exhaustive census over the mask-reachable sentential forms
(4,898 of them for ``additive_quadratic`` at ``MAX_LEN = 12``), so the numbers
on the figures are exact rather than sampled. The census machinery, the house
palette and the drawing helpers are shared with ``make_mdp_figures.py``.

Run from anywhere:

    python3 writeup-milton/figures/make_game_figures.py

Outputs are written next to this script.
"""

from __future__ import annotations

import collections
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_mdp_figures import (  # noqa: E402
    C_BAD, C_CONT, C_CONT_BG, C_DEAD, C_DEAD_BG, C_DECOY, C_DECOY_BG, C_OK,
    C_TGT, HERE, MAX_LEN, Census, arrow, box, build_game,
)

MONO = "DejaVu Sans Mono"

C_PAD = "#cfcfcf"        # pad token: present in the observation, not in the form
C_NT = "#8452a8"         # the nonterminal S
C_NT_BG = "#ece2f6"


# ===========================================================================
# Facts, all read off the running game
# ===========================================================================
class Facts:
    """Everything the two figures assert, computed once from the real game."""

    def __init__(self, cen: Census):
        self.cen = cen
        self.game = cen.game
        self.G = cen.G
        self.L = cen.game.state_len

        # -- exhaustive census, classifying every mask-reachable form --------
        self.forms: set[tuple] = set()
        self.terminals: list[tuple] = []
        self.deadends: list[tuple] = []
        self.n_legal: list[int] = []
        self.depth: dict[tuple, int] = {cen.root: 0}
        stack = [cen.root]
        while stack:
            s = stack.pop()
            if s in self.forms:
                continue
            self.forms.add(s)
            acts = cen.legal(s)
            if not acts:
                (self.deadends if self._has_nt(s) else self.terminals).append(s)
                continue
            self.n_legal.append(len(acts))
            for i, j in acts:
                c = cen.child(s, i, j)
                self.depth.setdefault(c, self.depth[s] + 1)
                stack.append(c)

        self.term_len = collections.Counter(len(s) for s in self.terminals)
        self.max_form_len = max(len(s) for s in self.forms)
        self.max_depth = max(self.depth.values())
        self.nominal_A = self.L * self.G.nprods

        # -- the escape productions: the reason exit 3 cannot fire -----------
        self.escapes = [k for k, r in enumerate(self.G.productions) if len(r) == 1]

        # -- exit 2, provoked on the real game ------------------------------
        self.game.reset_wrapper()
        illegal = int(np.flatnonzero(~self.game.get_action_mask())[0])
        _, r, term, _, info = self.game.step(illegal)
        self.exit2 = (illegal, r, term, info)

        # -- uniform-mask episodes, the policy MCTS rolls out with ----------
        rng = np.random.default_rng(0)
        self.ep_len, self.ep_reward = [], []
        for _ in range(2000):
            self.game.reset_wrapper()
            n = 0
            while True:
                v = np.flatnonzero(self.game.get_action_mask())
                _, r, term, _, _ = self.game.step_wrapper(int(rng.choice(v)))
                n += 1
                if term:
                    self.ep_len.append(n)
                    self.ep_reward.append(r)
                    break

        self.trace = self._trace_to_optimum()

    def _has_nt(self, s: tuple) -> bool:
        return any(t in self.G.nonterms for t in s)

    def sym(self, tok: int) -> str:
        return self.G.tokenlist[tok]

    def rhs_str(self, k: int) -> str:
        return " ".join(self.sym(t) for t in self.G.productions[k])

    def mask_matrix(self, s: tuple) -> np.ndarray:
        m = np.zeros((self.L, self.G.nprods), dtype=bool)
        for i, j in self.cen.legal(s):
            m[i, j] = True
        return m

    def _trace_to_optimum(self):
        """The V*-greedy episode, recorded step by step."""
        cen, cur, out = self.cen, self.cen.root, []
        while True:
            acts = cen.legal(cur)
            if not acts:
                out.append(dict(state=cur, act=None, n_legal=0))
                return out
            tgt = cen.vstar(cur)
            pick = next((i, j) for i, j in acts
                        if abs(cen.vstar(cen.child(cur, i, j)) - tgt) < 1e-12)
            out.append(dict(state=cur, act=pick, n_legal=len(acts)))
            cur = cen.child(cur, *pick)


# ===========================================================================
# Shared drawing primitive: one token buffer
# ===========================================================================
def draw_buffer(ax, x0, y, s: tuple, F: Facts, cw=0.030, ch=0.052,
                hi=None, show_pad=True, fs=7.6):
    """Draw the observation: the form, then pad up to ``state_len``.

    ``hi`` is a (start, stop) span drawn in the continue colour -- the cells
    the last action wrote.
    """
    n = F.L if show_pad else len(s)
    for k in range(n):
        inside = k < len(s)
        tok = s[k] if inside else F.G.pad_tok
        txt = F.sym(tok) if inside else "·"
        if not inside:
            fc, ec, tc = "white", C_PAD, C_PAD
        elif hi is not None and hi[0] <= k < hi[1]:
            fc, ec, tc = C_CONT_BG, C_CONT, "black"
        elif tok in F.G.nonterms:
            fc, ec, tc = C_NT_BG, C_NT, C_NT
        else:
            fc, ec, tc = "white", "0.55", "black"
        box(ax, x0 + (k + 0.5) * cw, y, cw * 0.92, ch, txt, fc, ec,
            fontsize=fs, tc=tc, lw=0.9, family=MONO)
    return x0 + n * cw


# ===========================================================================
# FIGURE 1 -- the MDP the code implements
# ===========================================================================
def panel_spec(ax, F: Facts):
    """(a) The tuple, with the grammar and production tables beside it."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a)  the MDP as implemented in "
                 "sraz/instances/symreg/game.py — every value below is read "
                 "off the running game", loc="left", fontsize=11, pad=6)

    G, L = F.G, F.L
    rows = [
        ("state  $s$",
         f"token buffer of fixed length {L}, plus real_state_len; "
         f"observation space MultiDiscrete([{G.nsym + 1}] * {L})"),
        ("start  $s_0$",
         f"[{G.start}] then {L - 1} pad — a single nonterminal, "
         f"real_state_len = 1"),
        ("action  $a$",
         f"Discrete({F.nominal_A}) = state_len × nprods = {L} × {G.nprods}; "
         f"decoded (pos, prod) = divmod(a, {G.nprods})"),
        ("legal set  $\\mathcal{A}(s)$",
         "pos < real_state_len,  buffer[pos] is a nonterminal,  and "
         "len(s) + |rhs| − 1 < state_len"),
        ("transition  $T$",
         "deterministic: splice the production's rhs over buffer[pos], "
         "shift the tail right, re-pad"),
        ("reward  $R$",
         "0 at every nonterminal state; at a terminal, clip($R^2$ of the "
         "lmfit constant fit, −1, 1)"),
    ]
    y = 0.955
    for name, body in rows:
        ax.text(0.005, y, name, fontsize=9.2, va="center", weight="bold",
                color=C_TGT)
        ax.text(0.135, y, body, fontsize=8.8, va="center", color="0.15")
        y -= 0.086

    ax.text(0.005, 0.375, "the four productions, one nonterminal $S$",
            fontsize=9.2, weight="bold", color=C_TGT, va="center")
    y = 0.300
    for k in range(G.nprods):
        rhs = G.productions[k]
        d = len(rhs) - 1
        esc = k in F.escapes
        ax.text(0.02, y, f"p{k}:  S →  {F.rhs_str(k)}", fontsize=8.8,
                family=MONO, va="center",
                color=C_OK if esc else "0.15")
        ax.text(0.30, y, f"|rhs| = {len(rhs)}", fontsize=8.6, family=MONO,
                va="center", color="0.45")
        ax.text(0.385, y, f"Δlen = {d:+d}", fontsize=8.6, family=MONO,
                va="center", color=C_OK if d == 0 else "0.45")
        if esc:
            ax.text(0.47, y, "escape production: never lengthens the buffer, "
                             "so it is never masked out", fontsize=8.4,
                    va="center", color=C_OK, style="italic")
        y -= 0.062

    ax.text(0.005, 0.005,
            f"tokens {list(G.tokenlist)} + pad({G.pad_tok});   "
            f"the target is {F.game.target_infix};   "
            f"γ = 1, and the only nonzero reward is terminal.",
            fontsize=8.4, color="0.35", va="bottom", family=MONO)


def panel_episode(ax, F: Facts):
    """(b) The V*-greedy episode, buffer by buffer."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b)  one episode, replayed from the game: the action "
                 "(pos, prod) rewrites one nonterminal cell into the "
                 "production's right-hand side", loc="left", fontsize=11,
                 pad=6)

    tr = F.trace
    n = len(tr)
    ys = np.linspace(0.88, 0.14, n)
    x0 = 0.135
    for k, (step, y) in enumerate(zip(tr, ys)):
        s = step["state"]
        hi = None
        if k > 0:
            pos, prod = tr[k - 1]["act"]
            hi = (pos, pos + len(F.G.productions[prod]))
        ax.text(0.005, y, f"$s_{{{k}}}$", fontsize=9, va="center",
                color=C_TGT)
        ax.text(0.045, y, f"|A| = {step['n_legal']:>2}", fontsize=8.2,
                family=MONO, va="center",
                color=C_BAD if step["n_legal"] == 0 else "0.45")
        draw_buffer(ax, x0, y, s, F, hi=hi)
        if step["act"] is not None:
            pos, prod = step["act"]
            ax.text(x0 + F.L * 0.030 + 0.018, y,
                    f"a = (pos {pos}, p{prod}):  S → {F.rhs_str(prod)}",
                    fontsize=8.4, family=MONO, va="center", color=C_CONT)
        else:
            r = F.cen.reward(s)
            ax.text(x0 + F.L * 0.030 + 0.018, y,
                    f"no nonterminal left → terminal,  R = {r:.6f}",
                    fontsize=8.4, family=MONO, va="center", color=C_OK,
                    weight="bold")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, (x0 - 0.012, y0 - 0.012), (x0 - 0.012, y1 + 0.012),
              "0.6", lw=0.9, scale=7)

    ax.text(0.005, 0.03,
            "cells: purple = the nonterminal $S$,  blue = written by the last "
            "action,  grey dot = pad.  The buffer is length "
            f"{F.L} throughout; only real_state_len grows.",
            fontsize=8.2, color="0.35", va="bottom")


def panel_mask(ax, F: Facts):
    """(c) The legal-action mask as the code returns it."""
    tr = F.trace
    s = tr[min(2, len(tr) - 2)]["state"]
    m = F.mask_matrix(s)
    ax.imshow(m.T, cmap=matplotlib.colors.ListedColormap(["#f4f4f4", C_CONT_BG]),
              vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(F.L))
    ax.set_xticklabels(range(F.L), fontsize=7)
    ax.set_yticks(range(F.G.nprods))
    ax.set_yticklabels([f"p{k}" for k in range(F.G.nprods)], fontsize=7.5)
    ax.set_xticks(np.arange(-0.5, F.L, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, F.G.nprods, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("position in the buffer", fontsize=9)
    ax.set_ylabel("production", fontsize=9)
    ax.grid(False, which="major")
    for i in range(F.L):
        for j in range(F.G.nprods):
            ax.text(i, j, "✓" if m[i, j] else "·", ha="center", va="center",
                    fontsize=8 if m[i, j] else 7,
                    color=C_CONT if m[i, j] else "0.72",
                    weight="bold" if m[i, j] else "normal")
    # the cells the form actually occupies, so "why these columns" is visible
    ax.set_title(f"(c)  the mask at  {F.cen.rule(s)}   "
                 f"(real_state_len = {len(s)})\n"
                 f"{m.sum()} legal of {F.nominal_A} nominal actions: only the "
                 f"cells holding $S$", fontsize=9.5, loc="left")


def panel_legal_hist(ax, F: Facts):
    """(d) How much of the nominal action space is ever legal."""
    cnt = collections.Counter(F.n_legal)
    ks = sorted(cnt)
    ax.bar(ks, [cnt[k] for k in ks], color=C_CONT, width=0.8)
    ax.axvline(F.nominal_A, color=C_BAD, lw=1.2, ls="--")
    ax.text(F.nominal_A - 1, max(cnt.values()) * 0.6,
            f"nominal |A| = {F.nominal_A}", rotation=90, fontsize=8,
            color=C_BAD, ha="right", va="center")
    ax.set_xlim(0, F.nominal_A + 2)
    ax.set_yscale("log")
    ax.set_xlabel("$|\\mathcal{A}(s)|$, legal actions at $s$", fontsize=9)
    ax.set_ylabel("nonterminal forms", fontsize=9)
    ax.set_title(f"(d)  the mask is the game: mean "
                 f"{np.mean(F.n_legal):.2f}, max {max(F.n_legal)}\n"
                 f"over all {len(F.n_legal):,} nonterminal forms",
                 fontsize=9.5, loc="left")


def panel_census(ax, F: Facts):
    """(e) The reachable state space, by depth."""
    by_depth = collections.Counter(F.depth.values())
    ks = sorted(by_depth)
    ax.bar(ks, [by_depth[k] for k in ks], color=C_CONT, width=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("depth: actions taken from $s_0$", fontsize=9)
    ax.set_ylabel("distinct forms", fontsize=9)
    ax.set_title(f"(e)  the whole reachable space: {len(F.forms):,} forms, "
                 f"{len(F.terminals):,} terminal\n"
                 f"deepest episode {F.max_depth} actions — small enough to "
                 f"enumerate exactly", fontsize=9.5, loc="left")


def figure_mdp(F: Facts):
    fig = plt.figure(figsize=(15.0, 12.2))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.80, 1.10, 0.80],
                  hspace=0.34, wspace=0.30,
                  left=0.045, right=0.985, top=0.965, bottom=0.055)
    panel_spec(fig.add_subplot(gs[0, :]), F)
    panel_episode(fig.add_subplot(gs[1, :]), F)
    panel_mask(fig.add_subplot(gs[2, 0]), F)
    panel_legal_hist(fig.add_subplot(gs[2, 1]), F)
    panel_census(fig.add_subplot(gs[2, 2]), F)
    out = os.path.join(HERE, "game_mdp.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# FIGURE 2 -- termination
# ===========================================================================
def panel_exits(ax, F: Facts):
    """(a) The three ways step() can end an episode."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(a)  termination: the three exits in SymRegGame.step, and "
                 "which of them can actually fire", loc="left", fontsize=11,
                 pad=6)

    box(ax, 0.105, 0.62, 0.185, 0.15,
        "apply $a$ to $s$\nsplice rhs over buffer[pos]", "white", "0.4",
        fontsize=8.8)

    exits = [
        (0.885, C_OK, "#e2f0e8",
         "EXIT 1  normal",
         "not _has_nonterms(state[:real_state_len])",
         f"R = clip($R^2$, −1, 1) via lmfit;  terminated = True\n"
         f"fires on every one of the {len(F.terminals):,} terminals",
         "reachable: this is how episodes end"),
        (0.575, C_BAD, C_DECOY_BG,
         "EXIT 2  invalid action",
         "pos ≥ real_state_len,  buffer[pos] not a nonterminal,\n"
         "prod ∉ proddict[tok],  or new_len > state_len",
         f"R = {F.exit2[1]:.1f};  terminated = {F.exit2[2]};  rule = None\n"
         f"verified by calling step({F.exit2[0]}) on $s_0$ off-mask",
         "unreachable under masked play — defensive only"),
        (0.265, C_DEAD, C_DEAD_BG,
         "EXIT 3  dead end",
         "nonterminals remain, but the mask is all zero\n ",
         "step is never called; a rollout hits `break` in\n"
         "MCTS._rollout_value and the sample is discarded",
         f"cannot occur: 0 of {len(F.forms):,} forms — see (c)"),
    ]
    for y, col, bg, title, cond, eff, note in exits:
        arrow(ax, (0.20, 0.62), (0.325, y), col, lw=1.3, rad=0.10)
        box(ax, 0.60, y, 0.545, 0.255, "", bg, col, lw=1.2)
        ax.text(0.340, y + 0.095, title, fontsize=9.2, weight="bold",
                color=col, va="center")
        ax.text(0.340, y + 0.022, cond, fontsize=8.0, family=MONO,
                color="0.25", va="center", linespacing=1.6)
        ax.text(0.340, y - 0.078, eff, fontsize=8.0, color="0.25",
                va="center", linespacing=1.6)
        ax.text(0.882, y, note, fontsize=8.2, color=col, va="center",
                style="italic")

    ax.text(0.005, 0.012,
            "There is no step limit, no truncation and no stop action: the "
            "episode ends when the sentential form runs out of nonterminals, "
            "and not before.\n"
            "Intermediate reward is exactly 0, so the return of an episode is "
            "the terminal $R^2$ alone.",
            fontsize=8.6, color="0.3", va="bottom", linespacing=1.6)


def panel_ledger(ax, F: Facts):
    """(b) Why every terminal is short: the length ledger."""
    G = F.G
    ks = range(G.nprods)
    d = [len(G.productions[k]) - 1 for k in ks]
    cols = [C_OK if x == 0 else C_CONT for x in d]
    ax.bar(list(ks), d, color=cols, width=0.7)
    for k, v in zip(ks, d):
        ax.text(k, v + 0.12, f"{v:+d}", ha="center", fontsize=8.5,
                color=C_OK if v == 0 else C_CONT)
    ax.set_xticks(list(ks))
    ax.set_xticklabels([f"p{k}\n{F.rhs_str(k)}" for k in ks], fontsize=7,
                       family=MONO)
    ax.set_ylabel("Δ real_state_len  =  |rhs| − 1", fontsize=9)
    ax.set_ylim(-0.4, max(d) + 0.9)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_title("(b)  the length ledger\n"
                 "each action adds |rhs| − 1 cells; the mask admits an action "
                 "only if\nthe result stays strictly below state_len",
                 fontsize=9.5, loc="left")


def panel_no_deadend(ax, F: Facts):
    """(c) The terminal length distribution and the escape-production bound."""
    ks = sorted(F.term_len)
    ax.bar(ks, [F.term_len[k] for k in ks], color=C_CONT, width=1.2)
    for k in ks:
        ax.text(k, F.term_len[k] * 1.12, str(F.term_len[k]), ha="center",
                fontsize=7.5, color="0.35")
    ax.axvline(F.L, color=C_BAD, lw=1.3, ls="--")
    ax.text(F.L + 0.35, 1.6, f"state_len = {F.L}\nno terminal reaches it",
            rotation=90, fontsize=7.6, color=C_BAD, ha="left", va="bottom")
    ax.set_yscale("log")
    ax.set_xlim(0, F.L + 2.2)
    ax.set_xlabel("tokens in the terminal expression", fontsize=9)
    ax.set_ylabel("terminals", fontsize=9)
    ax.set_title(f"(c)  no dead ends, and a hard length bound\n"
                 f"longest terminal {max(ks)} = state_len − 1; the escape "
                 f"production\n"
                 f"p{F.escapes[0]} (S → {F.rhs_str(F.escapes[0])}, Δ = 0) is "
                 f"legal at every $S$", fontsize=9.5, loc="left")


def panel_uniform(ax, F: Facts):
    """(d) What the rollout policy actually terminates with."""
    cnt = collections.Counter(F.ep_len)
    ks = sorted(cnt)
    n = len(F.ep_len)
    ax.bar(ks, [cnt[k] / n for k in ks], color=C_DEAD, width=0.75,
           label="episode length (actions)")
    ax.set_xlabel("actions taken before termination", fontsize=9)
    ax.set_ylabel("fraction of episodes", fontsize=9)
    ax.set_xlim(0, F.max_depth + 1)

    exact = float(np.mean([r >= 1 - 1e-9 for r in F.ep_reward]))
    ax.set_title(f"(d)  uniform play over the mask, {n:,} episodes\n"
                 f"mean length {np.mean(F.ep_len):.2f}, max {max(F.ep_len)};  "
                 f"mean R = {np.mean(F.ep_reward):.4f}\n"
                 f"P(R = 1) = {exact:.4f} — termination is cheap, the "
                 f"optimum is not", fontsize=9.5, loc="left")


def figure_termination(F: Facts):
    fig = plt.figure(figsize=(15.0, 9.0))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.10, 0.90],
                  hspace=0.34, wspace=0.30,
                  left=0.045, right=0.985, top=0.955, bottom=0.075)
    panel_exits(fig.add_subplot(gs[0, :]), F)
    panel_ledger(fig.add_subplot(gs[1, 0]), F)
    panel_no_deadend(fig.add_subplot(gs[1, 1]), F)
    panel_uniform(fig.add_subplot(gs[1, 2]), F)
    out = os.path.join(HERE, "game_termination.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
def report(F: Facts):
    print("=" * 72)
    print(f"instance additive_quadratic, state_len = {F.L}, "
          f"nprods = {F.G.nprods}, nominal |A| = {F.nominal_A}")
    print(f"forms {len(F.forms):,}   terminals {len(F.terminals):,}   "
          f"dead ends {len(F.deadends)}   max depth {F.max_depth}")
    print(f"terminal lengths {dict(sorted(F.term_len.items()))}   "
          f"longest {max(F.term_len)} = state_len - 1")
    print(f"escape productions {F.escapes} "
          f"({[F.rhs_str(k) for k in F.escapes]}), max form length "
          f"{F.max_form_len} < state_len {F.L}")
    print(f"|A(s)|: min {min(F.n_legal)} mean {np.mean(F.n_legal):.2f} "
          f"max {max(F.n_legal)}")
    print(f"exit 2 probe: step({F.exit2[0]}) -> reward {F.exit2[1]}, "
          f"terminated {F.exit2[2]}, info {F.exit2[3]}")
    print(f"uniform play: mean len {np.mean(F.ep_len):.2f}, "
          f"mean R {np.mean(F.ep_reward):.4f}, "
          f"P(R=1) {np.mean([r >= 1 - 1e-9 for r in F.ep_reward]):.4f}")
    print(f"V*(s_0) = {F.cen.vstar_root:.12f}   "
          f"V^q(s_0) = {F.cen.vq(F.cen.root):.6f}   "
          f"rho_0(s_0) = {F.cen.rho0(F.cen.root):.6e}")
    print("=" * 72)


if __name__ == "__main__":
    _F = Facts(Census(build_game()))
    report(_F)
    figure_mdp(_F)
    figure_termination(_F)
