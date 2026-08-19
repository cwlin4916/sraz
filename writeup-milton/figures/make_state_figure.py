"""Figure for the State bullet of def:derivation-mdp.

One PNG, one claim: a sentential form is stored as a fixed-length array of
L tokens, real_state_len records |s|, the cells beyond it hold a pad token,
and that pad token is part of the MultiDiscrete observation the agent
actually sees. Everything drawn is read off ``compile_grammar`` and the
eq:action-mask formula -- no hand-picked numbers.

    state_buffer.png   one mid-derivation form, drawn twice: as the
                       symbolic buffer and as the integer array
                       MultiDiscrete returns for it.

Run from anywhere:

    python3 writeup-milton/figures/make_state_figure.py

Output is written next to this script.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_mdp_figures import C_OK, C_TGT, HERE  # noqa: E402
from make_game_figures import C_NT, C_NT_BG, C_PAD  # noqa: E402
from sraz.instances.symreg.game import ADDITIVE_GRAMMAR, compile_grammar  # noqa: E402

MONO = "DejaVu Sans Mono"
L = 12  # buffer length, as in def:derivation-mdp and action_index.png

G = compile_grammar(ADDITIVE_GRAMMAR)


def legal(s: tuple, pos: int, prod: int) -> bool:
    """eq:action-mask, checked rather than assumed."""
    return (pos < len(s) and s[pos] in G.nonterms
            and prod in G.proddict[s[pos]]
            and len(s) + len(G.productions[prod]) - 1 < L)


def apply_action(s: tuple, pos: int, prod: int) -> tuple:
    assert legal(s, pos, prod), (s, pos, prod)
    rhs = tuple(G.productions[prod])
    return s[:pos] + rhs + s[pos + 1:]


def derive() -> tuple:
    """s0 -(0,+SS)-> +SS -(1,C0)-> +C0S -(2,+SS)-> +C0+SS, three legal steps."""
    s = (G.symdict[G.start],)
    for pos, prod in [(0, 0), (1, 1), (2, 0)]:
        s = apply_action(s, pos, prod)
    return s


# ===========================================================================
def fig_state(out="state_buffer.png"):
    s = derive()
    n = len(s)
    ids = list(s) + [G.pad_tok] * (L - n)

    fig, ax = plt.subplots(figsize=(8.6, 3.05))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    x0, cw, ch = 0.150, 0.0705, 0.170
    y_sym, y_id, y_brace = 0.845, 0.535, 0.260

    def cell(y, i, tok, text, fs):
        pad = tok == G.pad_tok
        nt = tok in G.nonterms
        fc = C_NT_BG if nt else "white"
        ec = C_PAD if pad else (C_NT if nt else "0.55")
        tc = C_PAD if pad else (C_NT if nt else "black")
        ax.add_patch(plt.Rectangle(
            (x0 + i * cw, y - ch / 2), cw * 0.90, ch,
            facecolor=fc, edgecolor=ec, linewidth=1.1))
        ax.text(x0 + i * cw + cw * 0.45, y, text, ha="center", va="center",
                fontsize=fs, family=MONO, color=tc,
                weight="bold" if nt else "normal")

    for i, tok in enumerate(ids):
        cell(y_sym, i, tok, "·" if tok == G.pad_tok else G.tokenlist[tok],
             11.5)
    for i, tok in enumerate(ids):
        cell(y_id, i, tok, str(tok), 10.0)

    ax.text(0.0, y_sym, "buffer\n(symbol)", fontsize=9.0, ha="left",
            va="center", weight="bold", color=C_TGT)
    ax.text(0.0, y_id, "observation\n(token id)", fontsize=9.0, ha="left",
            va="center", weight="bold", color="0.35")

    xa, xm, xb = x0, x0 + n * cw, x0 + L * cw
    ax.plot([xa, xm - cw * 0.06], [y_brace, y_brace], color=C_OK, lw=1.6,
            solid_capstyle="butt")
    ax.text((xa + xm) / 2, y_brace - 0.035,
            f"|s| = real_state_len = {n}", ha="center", va="top",
            fontsize=9.0, family=MONO, color=C_OK)
    ax.plot([xm + cw * 0.06, xb], [y_brace, y_brace], color=C_PAD, lw=1.6,
            solid_capstyle="butt")
    ax.text((xm + xb) / 2, y_brace - 0.035,
            f"L - |s| = {L - n} pad cells, id = pad_tok = {G.pad_tok}",
            ha="center", va="top", fontsize=8.6, family=MONO, color="0.45")

    ax.set_title(
        f"L = {L} fixed-length buffer   |   observation_space = "
        f"MultiDiscrete([{G.nsym + 1}] * {L})",
        fontsize=10.6, family=MONO, pad=10)

    legend = "  ".join(f"{i}:{G.tokenlist[i]}" for i in range(G.nsym))
    ax.text(0.0, 0.03, f"token id -> symbol:  {legend}  {G.pad_tok}:pad",
            fontsize=8.4, family=MONO, color="0.35", va="bottom")

    fig.savefig(os.path.join(HERE, out), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_state()
