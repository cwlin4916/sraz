"""Figure for Exit 3 and prop:no-dead-ends.

One PNG, two claims, both counterfactual -- the shipped grammars have zero
dead ends (prop:no-dead-ends), so both panels turn a knob the shipped system
never turns and let the real mask logic show what breaks:

    deadend_causes.png
        (a) drop the escape production S -> C0 from ADDITIVE_GRAMMAR and
            greedily expand the leftmost S; the mask goes empty at length 11
            with six nonterminals still open.
        (b) keep the real grammar but cap the buffer at L = 1; even the
            zero-growth escape production no longer fits at s0 itself.

Every buffer shown and every legal/illegal call is the real ``compile_grammar``
mask (eq:action-mask), evaluated on the stated grammar and L -- no hand-picked
numbers.

Run from anywhere:

    python3 writeup-milton/figures/make_deadend_figures.py

Output is written next to this script.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_mdp_figures import C_BAD, C_OK, HERE  # noqa: E402
from make_game_figures import C_NT, C_NT_BG, C_PAD  # noqa: E402
from sraz.instances.symreg.game import ADDITIVE_GRAMMAR, compile_grammar  # noqa: E402

MONO = "DejaVu Sans Mono"

# (a) the same grammar with the one-token escape production removed
NO_ESCAPE_GRAMMAR = {"S": [w for w in ADDITIVE_GRAMMAR["S"] if w != ["C0"]]}


def legal(G, L: int, s: tuple, pos: int, prod: int) -> bool:
    """eq:action-mask, checked rather than assumed."""
    return (pos < len(s) and s[pos] in G.nonterms
            and prod in G.proddict[s[pos]]
            and len(s) + len(G.productions[prod]) - 1 < L)


def any_legal(G, L: int, s: tuple) -> bool:
    return any(legal(G, L, s, pos, prod)
               for pos in range(len(s)) for prod in range(G.nprods))


def apply_leftmost(G, s: tuple, prod: int) -> tuple:
    pos = next(i for i, t in enumerate(s) if t in G.nonterms)
    rhs = tuple(G.productions[prod])
    return s[:pos] + rhs + s[pos + 1:]


def cell(ax, x0, y, cw, ch, tok, text, tokenlist, pad_tok, nonterms, fs):
    pad = tok == pad_tok
    nt = tok in nonterms
    fc = C_NT_BG if nt else "white"
    ec = C_PAD if pad else (C_NT if nt else "0.55")
    tc = C_PAD if pad else (C_NT if nt else "black")
    ax.add_patch(plt.Rectangle(
        (x0, y - ch / 2), cw * 0.90, ch,
        facecolor=fc, edgecolor=ec, linewidth=1.1, clip_on=True))
    ax.text(x0 + cw * 0.45, y, text, ha="center", va="center",
            fontsize=fs, family=MONO, color=tc,
            weight="bold" if nt else "normal", clip_on=True)


# ===========================================================================
def panel_no_escape(ax):
    """(a) a nonterminal with no length-1 production runs out of room."""
    G = compile_grammar(NO_ESCAPE_GRAMMAR)
    L = 12  # same buffer length as def:derivation-mdp

    s = (G.symdict[G.start],)
    trace = [s]
    while True:
        pos = next((i for i, t in enumerate(s) if t in G.nonterms), None)
        if pos is None or not any_legal(G, L, s):
            break
        s = apply_leftmost(G, s, 0)  # always take p0 = "+ S S"
        trace.append(s)
    assert not any_legal(G, L, s), "expected a dead end"
    n_nt = sum(1 for t in s if t in G.nonterms)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(
        "(a)  no escape production: drop $S\\to$ C0 from the grammar",
        loc="left", fontsize=10.6, pad=6)

    x0, cw, ch = 0.03, 0.066, 0.125
    rows = [trace[0], trace[2], trace[-1]]
    labels = ["s0", "after 2 actions",
              f"after {len(trace) - 1} actions, greedy leftmost S -> +SS"]
    ys = [0.86, 0.605, 0.35]
    for row, y, label in zip(rows, ys, labels):
        ids = list(row) + [G.pad_tok] * (L - len(row))
        stuck = row is rows[-1]
        for i, tok in enumerate(ids):
            cell(ax, x0 + i * cw, y, cw, ch, tok,
                 "·" if tok == G.pad_tok else G.tokenlist[tok],
                 G.tokenlist, G.pad_tok, G.nonterms, 9.5)
        nt_here = sum(1 for t in row if t in G.nonterms)
        tail = f"{nt_here} open S" + (", A(s) = ∅" if stuck else "")
        ax.text(x0, y - ch / 2 - 0.04, f"{label} — {tail}",
                fontsize=8.4, color=C_BAD if stuck else "0.3", va="top",
                clip_on=True)

    ax.text(0.02, 0.02,
            f"every remaining production has |w| ≥ 3: at length 11, "
            f"|s|+|w|-1 ≥ 13, not < {L} = L, for any of them.\n"
            f"{n_nt} nonterminals are stuck at once; a rollout through here "
            "discards the sample instead of scoring it.",
            fontsize=8.6, color="0.3", va="bottom", linespacing=1.4,
            clip_on=True)


def panel_small_l(ax):
    """(b) the escape production exists but L is capped below what it needs."""
    G = compile_grammar(ADDITIVE_GRAMMAR)
    L = 1  # capped below the shortest completion of s0 (1 token, needs L >= 2)
    s = (G.symdict[G.start],)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"(b)  buffer too small: the real grammar at L = {L}",
                 loc="left", fontsize=10.6, pad=6)

    cell(ax, 0.05, 0.88, 0.09, 0.15, s[0], "S", G.tokenlist, G.pad_tok,
         G.nonterms, 11)
    ax.text(0.17, 0.88, "s0, |s0| = 1 = L: no cell is free", fontsize=8.8,
            color="0.25", va="center", clip_on=True)

    y0 = 0.68
    for k, prod in enumerate(range(G.nprods)):
        w = G.productions[prod]
        new_len = len(s) + len(w) - 1
        ok = legal(G, L, s, 0, prod)
        y = y0 - 0.135 * k
        rhs = " ".join(G.tokenlist[t] for t in w)
        ax.text(0.05, y, f"$S\\to$ {rhs}", fontsize=9.0, family=MONO,
                color="0.25", va="center", clip_on=True)
        ax.text(0.46, y, f"|s0|+|w|-1 = {new_len} < {L}", fontsize=9.0,
                family=MONO, color="0.25", va="center", clip_on=True)
        mark, col = ("legal", C_OK) if ok else ("illegal", C_BAD)
        ax.text(0.86, y, mark, fontsize=9.2, color=col, va="center",
                weight="bold", clip_on=True)
    assert not any_legal(G, L, s), "expected a dead end"

    ax.text(0.02, 0.02,
            f"even the zero-growth escape production needs |s0| < L; at "
            f"L = {L} it fails like every other row.\n"
            "A(s0) = ∅ while s0 still holds a nonterminal: the root itself "
            "is a dead end.",
            fontsize=8.6, color="0.3", va="bottom", linespacing=1.4,
            clip_on=True)


def figure_deadend_causes():
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.06,
                  left=0.01, right=0.99, top=0.93, bottom=0.04)
    panel_no_escape(fig.add_subplot(gs[0, 0]))
    panel_small_l(fig.add_subplot(gs[0, 1]))
    out = os.path.join(HERE, "deadend_causes.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    figure_deadend_causes()
