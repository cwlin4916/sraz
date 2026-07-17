"""R^2 of every additive-grammar expression against the true target.

The companion to plot_pure_mcts_sweep.py. That figure shows *what score* pure
MCTS reaches at each simulation budget; this one shows *what scores exist* --
the reward landscape the search is moving over -- so a plateau in the sweep can
be read off as a named expression rather than a number.

ADDITIVE_GRAMMAR builds sums of monomials drawn from {C0, C1*x, C2*x**2}, and
sums commute, so the semantically distinct expressions are exactly the seven
non-empty subsets of those three terms. Repeated terms are omitted: C0 + C0 is
a reparameterization of C0 and scores identically, so it would add bars without
adding structure. Each subset is folded into a prefix rule and scored through
the game's own fit_expression, on the game's own (xs, exact_ys) at
problem_seed=42 -- the same oracle and the same target the sweep searches
against, not a reimplementation of them.

Run from repo root:
    python scripts/plotting/plot_expression_r2.py --problem additive_quadratic
"""

from sraz.utils import disable_numpy_multithreading, use_deterministic_cuda

disable_numpy_multithreading()
use_deterministic_cuda()

import argparse  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sympy  # noqa: E402
from sympy.parsing import sympy_parser  # noqa: E402

from sraz.instances.symreg.game import (  # noqa: E402
    SymRegGame,
    fit_expression,
    prefix_to_infix,
)
from sraz.instances.symreg.problems import get_problem  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# House hues, carried over from plot_target_families.py for continuity.
BLUE, ORANGE = "#3b7dd8", "#e8710a"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d8"

# The three monomials of ADDITIVE_GRAMMAR, as prefix token lists.
TERMS = [
    ("C0", ["C0"]),
    ("C1*x", ["*", "C1", "x"]),
    ("C2*x**2", ["*", "C2", "*", "x", "x"]),
]


def prefix_sum(term_tokens):
    """Fold term token lists into one prefix rule: [a, b, c] -> '+ a + b c'."""
    tokens = list(term_tokens[-1])
    for part in reversed(term_tokens[:-1]):
        tokens = ["+"] + list(part) + tokens
    return " ".join(tokens)


def enumerate_expressions():
    """Every non-empty subset of the monomials, as (rule, label, n_terms).

    `label` is built from the term names rather than from prefix_to_infix,
    which fully parenthesizes: it renders the 3-term sum as
    '(C0+((C1*x)+(C2*(x*x))))', unreadable on an axis and not the notation the
    target itself is written in. The rule that gets SCORED is still the prefix
    string, and its round-trip through prefix_to_infix is asserted below, so
    the pretty label cannot drift from the expression it names.
    """
    rows = []
    for k in range(1, len(TERMS) + 1):
        for subset in combinations(TERMS, k):
            rule = prefix_sum([toks for _, toks in subset])
            rows.append({
                "rule": rule,
                "label": " + ".join(name for name, _ in subset),
                "infix": prefix_to_infix(rule.split()),
                "n_terms": k,
            })
    return rows


def check_labels(rows):
    """The axis label must denote the same function as the scored prefix rule."""
    for r in rows:
        pretty = sympy_parser.parse_expr(r["label"].replace("**2", "**2"))
        scored = sympy_parser.parse_expr(r["infix"])
        if sympy.simplify(pretty - scored) != 0:
            raise AssertionError(
                f"label {r['label']!r} is not the expression scored as "
                f"{r['infix']!r} (rule {r['rule']!r})")


def score_expressions(rows, game):
    """Score each rule through the game's own oracle, on the game's own data."""
    for r in rows:
        r["r2"] = fit_expression(r["rule"], game.xs, game.exact_ys,
                                 max_nfev=game.lmfit_max_nfev)
        print(f"  R2={r['r2']:+.4f}  n_terms={r['n_terms']}  {r['label']}")
    return rows


def plot_expressions(rows, out_path, game, problem, seed, marks=()):
    """Sorted horizontal bars: one series, identity carried by the axis labels.

    Bars start at zero and the axis is not truncated, so length stays
    proportional to R^2. The consequence is honest and worth stating: on this
    domain every non-constant structure already explains >= 90% of the
    variance, so the top bars genuinely are near-identical and the printed
    values, not the lengths, carry the last two digits.

    `marks` are (R^2, note) pairs -- measurements from elsewhere (the pure-MCTS
    sweep) tied to whichever bar they land on. They are annotation, not data:
    nothing in this script produces them.
    """
    rows = sorted(rows, key=lambda r: r["r2"])
    labels = [r["label"] for r in rows]
    vals = [r["r2"] for r in rows]
    # The target is exactly expressible, so R^2 = 1 names a structure rather
    # than a maximum: the bar that attains it takes the second hue.
    colors = [ORANGE if v >= 1.0 - 1e-9 else BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.barh(labels, vals, color=colors, height=0.62)
    ax.axvline(1.0, ls=":", c=MUTED, lw=1, zorder=0)  # reachable optimum
    for y, v in enumerate(vals):
        ax.text(v + 0.012, y, f"{v:.4f}", va="center", ha="left",
                fontsize=9, color=INK)

    # Annotation column, left-aligned clear of the longest bar.
    for value, note in marks:
        y = min(range(len(vals)), key=lambda i: abs(vals[i] - value))
        if abs(vals[y] - value) > 5e-4:
            raise ValueError(
                f"mark R²={value} matches no expression (nearest "
                f"{labels[y]!r} at {vals[y]:.4f})")
        ax.annotate(note, xy=(vals[y] + 0.075, y), xytext=(1.22, y),
                    va="center", ha="left", fontsize=8.5, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=1))

    # The right margin is an annotation gutter, not data. Ticks, grid and the
    # axis line all stop at 1.0 so the empty span cannot read as headroom.
    ax.set_xlim(min(0.0, min(vals)), 1.95 if marks else 1.15)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    ax.set_xlabel("R² of the least-squares fit against the target")

    # Title and subtitle are placed by hand: set_title would stack them at one
    # y and they would overprint.
    consts = ", ".join(f"{k}={v:.3f}" for k, v in game.constants.items())
    ax.text(0.0, 1.12, f"R² of every expression the {problem} grammar can build",
            transform=ax.transAxes, fontsize=11, color=INK)
    ax.text(0.0, 1.04,
            f"target {game.target_infix}  ({consts}),   "
            f"x ∈ [{game.xs[0]:g}, {game.xs[-1]:g}], {len(game.xs)} points, "
            f"seed {seed}",
            transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.xaxis.grid(alpha=0.3, color=GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", type=str, default="additive_quadratic")
    ap.add_argument("--seed", type=int, default=42,
                    help="problem_seed; fixes the target constants (default 42, "
                         "matching plot_pure_mcts_sweep.py).")
    ap.add_argument("--mark", type=str, action="append", default=[],
                    metavar="R2=NOTE",
                    help="Annotate the bar at R2 with NOTE, e.g. "
                         "--mark '0.9287=pure-MCTS plateau'. Errors if no "
                         "expression scores R2, so a stale mark cannot "
                         "silently label the wrong bar.")
    ap.add_argument("--out", type=str, default=str(
        REPO_ROOT / "Claude-research" / "figures" / "expression_r2.png"))
    args = ap.parse_args()

    marks = []
    for spec in args.mark:
        value, _, note = spec.partition("=")
        marks.append((float(value), note))

    problem = get_problem(args.problem)
    game = SymRegGame(problem_seed=args.seed, **problem.game_kwargs())
    print(f"expression R² on {args.problem}   target: {game.target_infix}")
    print(f"constants: {game.constants}")
    rows = enumerate_expressions()
    check_labels(rows)
    score_expressions(rows, game)
    plot_expressions(rows, Path(args.out), game, args.problem, args.seed, marks)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
