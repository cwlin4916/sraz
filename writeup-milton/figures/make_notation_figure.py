"""Figure for subsec:notation: the four named children of s_0.

One PNG, one claim: |A(s_0)| = 4, one legal action per production, so s_0 has
four distinct successors. Three productions carry no nonterminal on the
right and are already terminal (D_1, D_2, D_3); the fourth, C, is the only
child that continues. Names and strings match eq:root-decoys exactly -- read
off ADDITIVE_GRAMMAR via compile_grammar, not transcribed by hand. D_1, D_2,
D_3 are ordered by production length, which is exactly the C0 / * C1 x /
* C2 * x x order the equation gives them.

    root_children.png   s_0 branching into D_1, D_2, D_3, C.

Run from anywhere:

    python3 writeup-milton/figures/make_notation_figure.py

Output is written next to this script.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_mdp_figures import (  # noqa: E402
    C_CONT, C_CONT_BG, C_DECOY, C_DECOY_BG, HERE, arrow, box,
)
from sraz.instances.symreg.game import ADDITIVE_GRAMMAR, compile_grammar  # noqa: E402

MONO = "DejaVu Sans Mono"

G = compile_grammar(ADDITIVE_GRAMMAR)


def root_children() -> list[dict]:
    """The four children of s_0, in eq:root-decoys order: D_1, D_2, D_3, C."""
    start = G.symdict[G.start]
    kids = []
    for j in G.proddict[start]:
        rhs = G.productions[j]
        kids.append({
            "prod": j,
            "decoded": " ".join(G.tokenlist[t] for t in rhs),
            "continues": any(t in G.nonterms for t in rhs),
        })
    cont = [k for k in kids if k["continues"]]
    term = sorted((k for k in kids if not k["continues"]),
                  key=lambda k: len(k["decoded"]))
    assert len(cont) == 1 and len(term) == 3, \
        "grammar no longer has the root-decoys shape assumed here"
    for i, k in enumerate(term, start=1):
        k["name"] = f"D{i}"
    cont[0]["name"] = "C"
    return term + cont


# ===========================================================================
def fig_root_children(out="root_children.png"):
    kids = root_children()

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    x_root, x_kid, x_tag = 0.075, 0.400, 0.700
    ys = [0.85, 0.615, 0.38, 0.145]

    box(ax, x_root, 0.50, 0.110, 0.160, "s0", "white", "0.35",
        fontsize=13.0, weight="bold", family=MONO)
    ax.text(x_root, 0.50 - 0.115, "(S)", ha="center", va="top",
            fontsize=8.4, color="0.45", family=MONO)

    for kid, y in zip(kids, ys):
        col, bg = (C_CONT, C_CONT_BG) if kid["continues"] else (C_DECOY, C_DECOY_BG)
        arrow(ax, (x_root + 0.058, 0.50), (x_kid - 0.205, y), col, lw=1.7,
              rad=0.09 * (1 if y > 0.50 else -1))
        box(ax, x_kid, y, 0.410, 0.145, f"{kid['name']} = {kid['decoded']}",
            bg, col, fontsize=11.5, tc=col, family=MONO, weight="bold")
        tag = ("continues -- only child with a nonterminal" if kid["continues"]
               else "terminal -- one-symbol derivation")
        ax.text(x_tag, y, tag, ha="left", va="center", fontsize=8.8, color=col)

    ax.set_title(
        f"root $s_0$: {len(kids)} legal actions, one per production, "
        f"{len(kids) - 1} terminal + 1 that continues",
        fontsize=11.0, pad=10)
    fig.text(0.075, 0.005,
             "from ADDITIVE_GRAMMAR via compile_grammar -- no hand-picked "
             "numbers", fontsize=8.0, family=MONO, color="0.45")

    fig.savefig(os.path.join(HERE, out), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for kid in root_children():
        print(f"  {kid['name']} = {kid['decoded']!r:20s} "
              f"continues={kid['continues']}  prod={kid['prod']}")
    fig_root_children()
