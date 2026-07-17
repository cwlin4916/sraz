"""Figure for Claude-research/notes/01-game-spec.md — symbolic regression as a grammar game.

Produces one PNG by replaying the REAL environment
(`sraz.instances.symreg.game.SymRegGame`); nothing is hand-transcribed:

  sr_game_progression.png
      Panel A: derivation trace — the 15-slot token buffer rewritten
               production-by-production from S to a terminal prefix string.
      Panel B: terminal reward — the target data and the lmfit best-fit of
               the chosen structure's constants, with R².

Run from repo root:
    python scripts/plotting/plot_sr_game.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import lmfit  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sympy  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from sympy.parsing import sympy_parser  # noqa: E402

from sraz.instances.symreg.game import C_INIT, SymRegGame, prefix_to_infix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

DPI = 160
PROBLEM_SEED = 42  # the instance used by the tests and by run_symreg.py --seed 42

# Scripted, mask-legal derivation: (position in buffer, global production index).
# This is the canonical 14-token frontier episode from tests/test_grammar_env.py.
SCRIPTED_ACTIONS = [(0, 0), (1, 6), (7, 0), (8, 1), (9, 3)]
EXPECTED_STATES = [
    "S",
    "+ S S",
    "+ * C3 sin * C4 x S",
    "+ * C3 sin * C4 x + S S",
    "+ * C3 sin * C4 x + C0 S",
    "+ * C3 sin * C4 x + C0 * C2 * x x",
]

# Palette (matches the AlphaZero_PP derivation-diagram family).
C_NONTERM_FACE, C_NONTERM_EDGE = "#e8f0fe", "#1a73e8"
C_TERM_FACE, C_TERM_EDGE = "#ffffff", "#9aa0a6"
C_PAD_FACE, C_PAD_EDGE = "#f1f3f4", "#dadce0"
C_FRESH_FACE, C_FRESH_EDGE = "#e6f4ea", "#188038"
C_PICKED_EDGE = "#d93025"


def replay_episode(problem_seed: int):
    """Replay the scripted derivation in the real env; return everything the
    figure needs: the game, per-step state copies, the emitted rule, and the
    episode's terminal reward."""
    game = SymRegGame(problem_seed=problem_seed)  # constructor resets once
    g = game.grammar

    def decode(state) -> list[str]:
        return [g.tokenlist[int(t)] for t in state if int(t) != g.pad_tok]

    states = [game.state.copy()]
    assert decode(states[0]) == EXPECTED_STATES[0].split()

    done, reward, info = False, 0.0, {}
    for t, (i, j) in enumerate(SCRIPTED_ACTIONS, start=1):
        mask = game.get_action_mask()  # flat (105,)
        assert mask[g.nprods * i + j], f"step {t}: action ({i},{j}) is masked"
        _, reward, done, _, info = game.step_wrapper(g.nprods * i + j)
        states.append(game.state.copy())
        got = decode(game.state)
        assert got == EXPECTED_STATES[t].split(), f"step {t}: {got}"
    assert done and info["rule"] is not None
    assert game.real_state_len == 14  # exactly at the sentence cap
    return game, states, info["rule"], reward


def fit_details(game: SymRegGame, rule: str, episode_reward: float):
    """Refit the emitted rule mirroring `fit_expression` (which returns only a
    scalar) to recover the lmfit ModelResult for Panel B; check the recomputed
    clipped R² matches the episode reward."""
    tokens = rule.strip().split()
    param_values = {"x": game.xs}
    for term in tokens:
        if "C" in term:
            param_values[term] = C_INIT
    infix = prefix_to_infix(tokens)
    model = sympy_parser.parse_expr(infix, evaluate=False)
    fn = sympy.lambdify(list(model.free_symbols), model)
    lm_mod = lmfit.Model(fn, independent_vars=("x",))
    res = lm_mod.fit(data=game.exact_ys, **param_values)
    ss_res = np.sum((res.data - res.best_fit) ** 2)
    ss_tot = np.sum((res.data - np.mean(res.data)) ** 2)
    r2 = 1 - ss_res / ss_tot
    assert np.isclose(np.clip(r2, -1.0, 1.0), episode_reward, atol=1e-7)
    return res, r2


def _slot(ax, x, y, text, face, edge, *, lw=1.2, bold=False):
    ax.add_patch(FancyBboxPatch((x - 0.42, y - 0.38), 0.84, 0.76,
                                boxstyle="round,pad=0.02", fc=face, ec=edge, lw=lw,
                                zorder=2))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=9, family="monospace",
                fontweight="bold" if bold else "normal", zorder=3)


def draw_derivation(ax, game: SymRegGame, states, rule: str) -> None:
    g = game.grammar
    tokenlist, pad_tok, nonterms = g.tokenlist, g.pad_tok, g.nonterms
    n_slots, row_dy = game.state_len, 2.0

    for t, state in enumerate(states):
        y = -t * row_dy
        act = SCRIPTED_ACTIONS[t - 1] if t >= 1 else None
        fresh = set(range(act[0], act[0] + len(g.productions[act[1]]))) if act else set()
        picked = SCRIPTED_ACTIONS[t][0] if t < len(SCRIPTED_ACTIONS) else None
        ax.text(-1.4, y, f"$s_{t}$", ha="center", va="center", fontsize=11)
        for i in range(n_slots):
            tok = int(state[i])
            if tok == pad_tok:
                face, edge, text, bold, lw = C_PAD_FACE, C_PAD_EDGE, "", False, 1.0
            elif tok in nonterms:
                face, edge, text, bold, lw = (C_NONTERM_FACE, C_NONTERM_EDGE,
                                              tokenlist[tok], True, 1.4)
            else:
                face, edge, text, bold, lw = (C_TERM_FACE, C_TERM_EDGE,
                                              tokenlist[tok], False, 1.2)
            if i in fresh:
                face, edge = C_FRESH_FACE, C_FRESH_EDGE
            if picked is not None and i == picked:
                edge, lw = C_PICKED_EDGE, 2.4
            _slot(ax, i, y, text, face, edge, lw=lw, bold=bold)

        if t < len(SCRIPTED_ACTIONS):
            i, j = SCRIPTED_ACTIONS[t]
            rhs = " ".join(tokenlist[s] for s in g.productions[j])
            ax.annotate("", xy=(-1.4, y - row_dy + 0.55), xytext=(-1.4, y - 0.55),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4))
            ax.text(-0.7, y - row_dy / 2,
                    f"a = (i={i}, j={j})   flat {g.nprods}·{i}+{j} = "
                    f"{g.nprods * i + j}   apply S -> {rhs}",
                    ha="left", va="center", fontsize=8.5, family="monospace",
                    color="#202124")

    y_end = -len(SCRIPTED_ACTIONS) * row_dy - 1.15
    ax.text(-0.7, y_end,
            f"no nonterminal left -> done; emitted rule (prefix, 14 tokens = cap): "
            f"{' '.join(rule.split())}\n"
            f"infix (prefix_to_infix): {prefix_to_infix(rule.split())}",
            ha="left", va="center", fontsize=9, family="monospace", color="#188038",
            fontweight="bold")

    handles = [
        Patch(fc=C_NONTERM_FACE, ec=C_NONTERM_EDGE, label="nonterminal S"),
        Patch(fc=C_TERM_FACE, ec=C_TERM_EDGE, label="terminal token"),
        Patch(fc=C_PAD_FACE, ec=C_PAD_EDGE, label="pad (empty slot)"),
        Patch(fc="white", ec=C_PICKED_EDGE, lw=2.0, label="slot picked by action"),
        Patch(fc=C_FRESH_FACE, ec=C_FRESH_EDGE, label="freshly spliced RHS"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95,
              bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Panel A — derivation trace: the 15-slot state buffer, rewritten "
                 "production-by-production", fontsize=11, loc="left")
    ax.set_xlim(-2.2, n_slots - 0.3)
    ax.set_ylim(y_end - 0.9, 1.0)
    ax.axis("off")


def draw_fit(ax, game: SymRegGame, res, r2: float, reward: float,
             problem_seed: int) -> None:
    xs, ys = game.xs, game.exact_ys
    x_dense = np.linspace(xs.min(), xs.max(), 400)
    ax.scatter(xs, ys, s=18, color="#202124", zorder=3,
               label="target data (41 pts)")
    ax.plot(x_dense, res.eval(x=x_dense), color="#1a73e8", lw=2.0, zorder=2,
            label="lmfit best fit of chosen structure")
    fitted = "   ".join(f"{k}={res.params[k].value:.2f}" for k in sorted(res.params))
    ax.text(0.02, 0.97,
            f"fit C3*sin(C4*x) + C0 + C2*x*x   (init all C=2.5)\n"
            f"{fitted}\n"
            f"$R^2$ = {r2:.4f}   reward = clip($R^2$, -1, 1) = {reward:.4f}\n"
            f"target constants fixed by problem_seed={problem_seed}\n"
            f"full target form needs 18 tokens > 14-token cap -> $R^2=1$ unreachable",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fef7e0", ec="#f9ab00", lw=0.8))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Panel B — terminal reward: fit the constants of the chosen "
                 "structure to the target data", fontsize=11, loc="left")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "Claude-research" / "figures",
                        help="Where to write the PNG.")
    parser.add_argument("--problem-seed", type=int, default=PROBLEM_SEED,
                        help="Seed fixing the target constants C*.")
    args = parser.parse_args()

    game, states, rule, reward = replay_episode(args.problem_seed)
    res, r2 = fit_details(game, rule, reward)

    fig = plt.figure(figsize=(12.5, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.9], hspace=0.16)
    draw_derivation(fig.add_subplot(gs[0]), game, states, rule)
    draw_fit(fig.add_subplot(gs[1]), game, res, r2, reward, args.problem_seed)
    fig.suptitle("Symbolic regression as a grammar game: one scripted episode",
                 fontsize=13, y=0.985)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "sr_game_progression.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    params = "  ".join(f"{k}={res.params[k].value:.4f}" for k in sorted(res.params))
    print(f"wrote {out}")
    print(f"R^2={r2:.4f}  reward={reward:.4f}  target constants="
          + str({k: round(v, 3) for k, v in game.constants.items()}))
    print(f"fitted params: {params}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
