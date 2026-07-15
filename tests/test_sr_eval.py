"""SR evaluator: prefix->infix conversion and constant fitting."""

import numpy as np

from sraz.instances.symreg.game import (
    SymRegGame, fit_expression, prefix_to_infix, target_ys,
)


def _problem(seed=42):
    game = SymRegGame(problem_seed=seed)
    return game.xs, game.exact_ys


def test_prefix_to_infix():
    assert prefix_to_infix("+ C0 * C1 x".split()) == "(C0+(C1*x))"
    assert prefix_to_infix("* C3 sin * C4 x".split()) == "(C3*sin((C4*x)))"
    assert prefix_to_infix(["C0"]) == "C0"


def test_fit_true_form():
    xs, ys = _problem()
    r2 = fit_expression("+ * C3 sin * C4 x + C0 + * C1 x * C2 * x x", xs, ys)
    assert r2 > 0.999  # measured 0.99957 (lmfit's sin-frequency fit plateaus < 1)


def test_fit_partial_forms():
    xs, ys = _problem()
    assert fit_expression("+ C0 + * C1 x * C2 * x x", xs, ys) > 0.96  # quad-only, measured 0.973
    assert fit_expression("+ * C3 sin * C4 x + C0 * C2 * x x", xs, ys) > 0.98  # measured 0.9874


def test_fit_constant_only():
    xs, ys = _problem()
    # a lone constant fits the mean: R^2 == 0 by definition
    assert fit_expression("C0", xs, ys) == 0.0


def test_fit_garbage_returns_minus_one():
    xs, ys = _problem()
    assert fit_expression("+ +", xs, ys) == -1.0     # unparseable
    assert fit_expression("", xs, ys) == -1.0


def test_fit_always_clipped():
    xs, ys = _problem()
    exprs = [
        "/ C0 C0",              # constant 1: R^2 very negative before clipping
        "/ C0 * x x",
        "* C1 x",
        "/ * C1 x * C2 * x x",
    ]
    for e in exprs:
        r2 = fit_expression(e, xs, ys)
        assert -1.0 <= r2 <= 1.0, e


def test_target_ys_matches_definition():
    xs = np.linspace(1.0, 3.0, 41)
    consts = {"C0": 1.5, "C1": 2.0, "C2": 3.0}
    expected = 4 * np.sin(4 * xs) + 1.5 + 2.0 * xs + 3.0 * xs ** 2
    assert np.allclose(target_ys(xs, consts), expected)
