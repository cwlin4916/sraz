"""Parameterized SR instances: the problem registry + the additive best-corner game.

The additive_quadratic instance is the study's positive control (informative +
non-deceptive). These tests pin the properties that make it best-corner:
R^2 = 1 is reachable, adding a term never lowers the fit, and no completion hits
the -1 failure floor. See Claude-research/02-informativeness-and-deception.md.
"""

import numpy as np
import pytest

from sraz.instances.symreg.game import (
    ADDITIVE_GRAMMAR,
    ADDITIVE_INFIX,
    SymRegGame,
    fit_expression,
    target_ys_additive,
)
from sraz.instances.symreg.problems import PROBLEMS, SRProblem, get_problem


def _additive_game(seed=42):
    return SymRegGame(problem_seed=seed, grammar_rules=ADDITIVE_GRAMMAR,
                      target_ys_fn=target_ys_additive,
                      constant_names=("C0", "C1", "C2"), target_infix=ADDITIVE_INFIX)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_has_expected_problems():
    assert set(PROBLEMS) >= {"sine", "additive_quadratic"}
    for name, prob in PROBLEMS.items():
        assert isinstance(prob, SRProblem)
        assert prob.name == name
        assert set(prob.game_kwargs()) == {
            "grammar_rules", "target_ys_fn", "constant_names", "target_infix"}


def test_get_problem_unknown_raises():
    with pytest.raises(KeyError):
        get_problem("does_not_exist")


def test_problem_game_kwargs_build_a_game():
    for name in PROBLEMS:
        prob = get_problem(name)
        g = SymRegGame(problem_seed=0, **prob.game_kwargs())
        assert g.target_infix == prob.infix
        assert g.constant_names == prob.constant_names


# ---------------------------------------------------------------------------
# Parameterization / default unchanged
# ---------------------------------------------------------------------------

def test_default_game_is_the_sine_instance():
    g = SymRegGame()
    assert g.grammar.nprods == 7          # full grammar (+,C0,C1x,C2x2,*SS,/SS,sin)
    assert int(g.action_space.n) == g.state_len * 7
    assert "sin" in g.target_infix


def test_additive_game_grammar_shape():
    g = _additive_game()
    assert g.grammar.nprods == 4
    assert int(g.action_space.n) == g.state_len * 4
    assert g.grammar.nsym == 7             # S + C0 * C1 x C2
    assert set(g.constant_names) == {"C0", "C1", "C2"}


def test_additive_target_matches_formula_and_has_no_sine():
    g = _additive_game(seed=7)
    c = g.constants
    expected = c["C0"] + c["C1"] * g.xs + c["C2"] * g.xs ** 2
    np.testing.assert_allclose(g.exact_ys, expected)
    # distinct from the sine instance's target
    assert not np.allclose(g.exact_ys, 4 * np.sin(4 * g.xs) + expected)


# ---------------------------------------------------------------------------
# Best-corner properties (informative + non-deceptive)
# ---------------------------------------------------------------------------

def test_additive_perfect_expression_fits_to_one():
    g = _additive_game()
    r2 = fit_expression("+ + C0 * C1 x * C2 * x x", g.xs, g.exact_ys)
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_additive_is_non_deceptive_adding_a_term_never_lowers_r2():
    g = _additive_game()
    r2 = lambda e: fit_expression(e, g.xs, g.exact_ys)
    r_lin = r2("* C1 x")
    r_lin_quad = r2("+ * C1 x * C2 * x x")
    r_full = r2("+ + C0 * C1 x * C2 * x x")
    # each added term weakly improves the least-squares fit
    assert r_lin_quad >= r_lin - 1e-9
    assert r_full >= r_lin_quad - 1e-9
    assert r_full == pytest.approx(1.0, abs=1e-6)


def test_additive_has_no_fit_failures():
    # informativeness precondition: no division / sine => no -1 clip-floor.
    g = _additive_game()
    exprs = ["C0", "* C1 x", "* C2 * x x", "+ C0 C0", "+ * C1 x C0",
             "+ * C2 * x x * C1 x", "+ + C0 * C1 x * C2 * x x"]
    for e in exprs:
        r = fit_expression(e, g.xs, g.exact_ys)
        assert r > -1.0 + 1e-9, f"{e!r} hit the -1 failure floor: {r}"
        assert -1.0 <= r <= 1.0


def test_additive_game_derives_and_scores_two_term_sum():
    g = _additive_game()
    g.reset_wrapper()
    nprods = g.grammar.nprods
    # + S S ; fill left S with C1*x ; fill the remaining S with C2*x^2
    seq = [(0, 0), (1, 2), (4, 3)]
    terminated = False
    info = {}
    for pos, prod in seq:
        _, reward, terminated, _, info = g.step_wrapper(pos * nprods + prod)
    assert terminated
    assert info["rule"] == "+ * C1 x * C2 * x x"
    assert reward > 0.99
