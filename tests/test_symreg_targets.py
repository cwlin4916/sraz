"""Target families and the game's fixed-target mode.

Two things are pinned here. First, that each family member really is the shape
its design claims -- the vertex/discriminant geometry is asserted, not trusted,
because the whole point of choosing coefficients over sampling them is that the
shapes are known. Second, that adding targets did not disturb the default
instance the earlier note documents.

Complements tests/test_symreg_game_unit.py (game internals on the default
target) and tests/test_sr_eval.py (the evaluator).
"""

import numpy as np
import pytest

from sraz.instances.symreg.game import SymRegGame, fit_expression
from sraz.instances.symreg.targets import (
    LINEAR_TARGETS, QUADRATIC_TARGETS, TARGETS, Target,
    family_targets, get_target,
)


def monotonic(t: Target) -> bool:
    dy = np.diff(t.ys(t.xs()))
    return bool((dy > 0).all() or (dy < 0).all())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_partitions_into_the_two_families():
    assert set(TARGETS) == set(LINEAR_TARGETS) | set(QUADRATIC_TARGETS)
    assert not set(LINEAR_TARGETS) & set(QUADRATIC_TARGETS)
    assert {t.name for t in family_targets("linear")} == set(LINEAR_TARGETS)
    assert {t.name for t in family_targets("quadratic")} == set(QUADRATIC_TARGETS)


def test_every_target_is_self_consistent():
    for name, t in TARGETS.items():
        assert t.name == name  # the key is the target's own name
        assert len(t.coeffs) == (2 if t.family == "linear" else 3)
        xs = t.xs()
        assert xs.shape == (t.n_x,)
        assert (t.x_min, t.x_max) == (xs[0], xs[-1]) == (-1.0, 1.0)
        assert np.isfinite(t.ys(xs)).all()


def test_unknown_names_raise():
    with pytest.raises(KeyError, match="Unknown target"):
        get_target("nope")
    with pytest.raises(KeyError, match="Unknown family"):
        family_targets("cubic")


@pytest.mark.parametrize("kwargs, match", [
    (dict(name="t", family="linear", coeffs=(1.0,), label=""), "needs 2 coeffs"),
    (dict(name="t", family="quadratic", coeffs=(1.0, 2.0), label=""), "needs 3 coeffs"),
    (dict(name="t", family="cubic", coeffs=(1.0, 2.0), label=""), "unknown family"),
    (dict(name="t", family="linear", coeffs=(1.0, 2.0), label="", x_min=1.0, x_max=1.0),
     "empty domain"),
])
def test_target_validates_its_own_shape(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Target(**kwargs)


def test_ys_evaluates_the_polynomial_it_names():
    # ascending-power Horner must agree with the literal polynomial
    t = get_target("quad_A")
    c0, c1, c2 = t.coeffs
    xs = t.xs()
    assert np.allclose(t.ys(xs), c0 + c1 * xs + c2 * xs ** 2)

    lin = get_target("lin_A")
    d0, d1 = lin.coeffs
    assert np.allclose(lin.ys(lin.xs()), d0 + d1 * lin.xs())


# ---------------------------------------------------------------------------
# Family geometry: each member is the shape its design claims
# ---------------------------------------------------------------------------

def test_quad_A_interior_extremum():
    """Vertex strictly inside the domain, no real roots: a genuine turnaround."""
    t = get_target("quad_A")
    assert t.vertex == pytest.approx(0.25)
    assert t.x_min < t.vertex < t.x_max and t.vertex_in_domain()
    assert t.discriminant == pytest.approx(-7.0)
    assert t.discriminant < 0 and t.real_roots == []
    assert not monotonic(t)  # the turnaround is what forbids a linear cheat


def test_quad_B_monotonic_exterior():
    """Vertex far outside, and -- unlike a naive c0 -- no root inside either.

    A root in the domain would make this a sign-changing curve and collide with
    Family D. Both properties are needed for 'pseudo-linear'.
    """
    t = get_target("quad_B")
    assert t.vertex == pytest.approx(5.0)
    assert not t.vertex_in_domain() and t.vertex > t.x_max
    assert t.roots_in_domain() == []
    assert monotonic(t)
    ys = t.ys(t.xs())
    assert (ys > 0).all()  # sign-stable across the domain

    # It really does look like a line here: a least-squares line explains
    # essentially all of its variance, which is exactly the trap being set.
    xs = t.xs()
    resid = ys - np.polyval(np.polyfit(xs, ys, 1), xs)
    r2_line = 1 - resid.var() / ys.var()
    assert r2_line > 0.99


def test_quad_C_ill_conditioned_scales():
    """A 1e6 spread between the leading and linear coefficients."""
    t = get_target("quad_C")
    c0, c1, c2 = t.coeffs
    assert c2 / c1 == pytest.approx(1e6)
    assert t.vertex_in_domain()
    # c0 != 0 keeps it off a degenerate double root at the origin
    assert c0 != 0.0 and t.discriminant < 0 and t.roots_in_domain() == []
    assert not monotonic(t)


def test_quad_C_does_not_stall_the_inner_optimizer():
    """Records that the ill-conditioning hypothesis does not hold here.

    Family C is designed on the expectation that extreme coefficient scales
    make the inner Levenberg-Marquardt fit stall and return a false-negative
    R^2. It does not: the grammar's constants enter every candidate structure
    *linearly*, so the fit is a linear least-squares problem in disguise and
    the scale spread costs it nothing. Even a 50-evaluation cap reaches R^2 = 1.

    A stall needs a constant the model is nonlinear in -- the sinusoid's
    frequency C4 is the one structure in this grammar that has one.
    """
    t = get_target("quad_C")
    xs, ys = t.xs(), t.ys(t.xs())
    assert fit_expression("+ C0 + * C1 x * C2 * x x", xs, ys, max_nfev=50) == \
        pytest.approx(1.0, abs=1e-6)

    # The linear term is 1e-6 of the leading one, so dropping it costs nothing
    # measurable: the ill-conditioned target is effectively two-parameter, and
    # therefore *easier* than the balanced Family A rather than harder.
    assert fit_expression("+ * C2 * x x C0", xs, ys, max_nfev=50) == \
        pytest.approx(1.0, abs=1e-6)
    a = get_target("quad_A")
    assert fit_expression("+ * C2 * x x C0", a.xs(), a.ys(a.xs()), max_nfev=50) < 0.99


def test_quad_D_root_crossing():
    """Positive discriminant with both roots inside the domain, and -- unlike
    symmetric roots -- a linear term that is actually needed.

    Roots at +/-0.5 would force c1 = 0, and a target with no linear term is fitted
    exactly by the 7-token 'C2 x^2 + C0'. Asymmetric roots keep all three terms
    load-bearing.
    """
    t = get_target("quad_D")
    assert t.discriminant == pytest.approx(4.0) and t.discriminant > 0
    assert t.roots_in_domain() == pytest.approx([-0.6, 0.4])
    assert len(t.real_roots) == 2
    ys = t.ys(t.xs())
    assert ys.min() < 0 < ys.max()  # the curve genuinely crosses y = 0
    assert not monotonic(t)

    c0, c1, c2 = t.coeffs
    assert c1 != 0.0
    # the two-term structure must NOT suffice, or the family tests nothing
    assert fit_expression("+ * C2 * x x C0", t.xs(), ys) < 0.99


def test_quadratic_family_members_are_topologically_distinct():
    """No two members share a (vertex-inside, roots-inside, monotonic) signature.

    This is the property that makes the family a partition rather than four
    arbitrary parabolas.
    """
    sigs = {
        name: (t.vertex_in_domain(), bool(t.roots_in_domain()), monotonic(t))
        for name, t in QUADRATIC_TARGETS.items()
    }
    assert sigs["quad_A"] != sigs["quad_B"]
    assert sigs["quad_A"] != sigs["quad_D"]
    assert sigs["quad_B"] != sigs["quad_D"]
    # A and C share a signature by design; the scale ratio separates them.
    assert sigs["quad_A"] == sigs["quad_C"]
    a, c = get_target("quad_A"), get_target("quad_C")
    assert max(abs(x) for x in a.coeffs) / max(abs(x) for x in c.coeffs) < 1e-2


def test_linear_family_is_monotonic_and_spans_its_design_axes():
    for t in family_targets("linear"):
        assert monotonic(t)          # a line always is
        assert t.vertex is None      # and has no stationary point
        assert t.discriminant is None

    assert get_target("lin_A").roots_in_domain() == pytest.approx([-0.25])
    assert get_target("lin_B").roots_in_domain() == []      # root pushed to x = -5
    assert get_target("lin_D").roots_in_domain() == pytest.approx([0.0])
    c0, c1 = get_target("lin_C").coeffs
    assert c0 / c1 == pytest.approx(1e6)                    # ill-conditioned


# ---------------------------------------------------------------------------
# The game in fixed-target mode
# ---------------------------------------------------------------------------

def test_target_replaces_drawn_constants_and_grid():
    t = get_target("quad_A")
    game = SymRegGame(target=t)
    assert game.target is t
    assert np.array_equal(game.xs, t.xs())
    assert np.array_equal(game.exact_ys, t.ys(t.xs()))
    assert game.constants == {"c0": 1.0, "c1": -1.0, "c2": 2.0}
    # the domain really moved off the default instance's [1, 3]
    assert (game.xs[0], game.xs[-1]) == (-1.0, 1.0)


def test_target_accepts_a_name():
    by_name = SymRegGame(target="quad_D")
    by_obj = SymRegGame(target=get_target("quad_D"))
    assert by_name.target is by_obj.target
    assert np.array_equal(by_name.exact_ys, by_obj.exact_ys)


def test_target_is_seed_independent():
    """A chosen target has nothing left to draw, so the seed cannot move it."""
    a = SymRegGame(target="quad_B", problem_seed=0)
    b = SymRegGame(target="quad_B", problem_seed=999)
    assert a.constants == b.constants
    assert np.array_equal(a.exact_ys, b.exact_ys)


def test_redraw_with_fixed_target_is_rejected():
    with pytest.raises(ValueError, match="incompatible with a fixed target"):
        SymRegGame(target="lin_A", redraw_constants=True)


def test_exact_structures_score_one_on_their_family():
    """The family's own form is reachable and fits perfectly, so R^2 = 1 is the
    ceiling every sweep is measured against."""
    for t in family_targets("linear"):
        r2 = fit_expression("+ C0 * C1 x", t.xs(), t.ys(t.xs()))
        assert r2 == pytest.approx(1.0, abs=1e-6), t.name
    for t in family_targets("quadratic"):
        r2 = fit_expression("+ C0 + * C1 x * C2 * x x", t.xs(), t.ys(t.xs()))
        assert r2 == pytest.approx(1.0, abs=1e-6), t.name


def test_lin_D_is_hit_exactly_by_the_three_token_sentence():
    """y = 2x is exactly '* C1 x' -- the shortest non-trivial derivation."""
    t = get_target("lin_D")
    game = SymRegGame(target=t)
    game.reset_wrapper()
    g = game.grammar
    _, r, term, _, info = game.step_wrapper(0 * g.nprods + 2)  # S -> * C1 x
    assert term and info["rule"] == "* C1 x"
    assert r == pytest.approx(1.0, abs=1e-9)


def test_constant_sentence_scores_zero_on_every_target():
    """R^2 = 0 is 'no better than the mean' -- the floor a search must beat."""
    for name in TARGETS:
        t = get_target(name)
        assert fit_expression("C0", t.xs(), t.ys(t.xs())) == pytest.approx(0.0, abs=1e-9), name


# ---------------------------------------------------------------------------
# lmfit iteration cap
# ---------------------------------------------------------------------------

def test_max_nfev_is_forwarded_to_the_fit():
    """A cap of 1 cannot converge, so it must change the score -- proving the
    argument reaches lmfit rather than being silently dropped."""
    t = get_target("quad_A")
    xs, ys = t.xs(), t.ys(t.xs())
    rule = "+ C0 * C1 x"
    assert fit_expression(rule, xs, ys, max_nfev=None) == pytest.approx(0.4721, abs=5e-4)
    # a starved fit fails and falls to the clip floor rather than merely
    # scoring worse: the reward is a property of the oracle, not the structure
    assert fit_expression(rule, xs, ys, max_nfev=1) == -1.0


def test_game_uses_its_configured_max_nfev():
    game = SymRegGame(target="quad_A", lmfit_max_nfev=50)
    assert game.lmfit_max_nfev == 50
    game.reset_wrapper()
    g = game.grammar
    for pos, prod in [(0, 0), (1, 1), (2, 3)]:  # -> "+ C0 * C2 * x x"
        _, r, term, _, info = game.step_wrapper(pos * g.nprods + prod)
    assert term and info["rule"] == "+ C0 * C2 * x x"
    assert r == fit_expression(info["rule"], game.xs, game.exact_ys, max_nfev=50)


def test_max_nfev_50_still_fits_the_convex_families():
    """<b> caps the inner optimizer at 50 on the grounds that linear and
    quadratic fits are convex. That must not cost the exact structures their
    perfect score."""
    for family, rule in (("linear", "+ C0 * C1 x"),
                         ("quadratic", "+ C0 + * C1 x * C2 * x x")):
        for t in family_targets(family):
            r2 = fit_expression(rule, t.xs(), t.ys(t.xs()), max_nfev=50)
            assert r2 == pytest.approx(1.0, abs=1e-6), f"{t.name} stalled under a 50-eval cap"


# ---------------------------------------------------------------------------
# The default instance must not have moved
# ---------------------------------------------------------------------------

def test_default_game_is_untouched_by_the_target_machinery():
    """Regression guard for the numbers in the first note: no target argument
    must reproduce the sinusoid instance on [1, 3] with drawn constants."""
    game = SymRegGame(problem_seed=42)
    assert game.target is None
    assert game.lmfit_max_nfev is None
    assert (game.xs[0], game.xs[-1], len(game.xs)) == (1.0, 3.0, 41)
    assert set(game.constants) == {"C0", "C1", "C2"}
    assert game.constants["C0"] == pytest.approx(3.322, abs=5e-4)
    assert game.constants["C1"] == pytest.approx(2.317, abs=5e-4)
    assert game.constants["C2"] == pytest.approx(3.576, abs=5e-4)

    # the documented score ladder, unchanged
    for rule, expected in [
        ("C0", 0.0000),
        ("* C2 * x x", 0.8409),
        ("* C1 x", 0.8729),
        ("+ * C1 x * C2 * x x", 0.9316),
        ("+ C0 + * C1 x * C2 * x x", 0.9731),
        ("+ * C3 sin * C4 x + C0 * C2 * x x", 0.9873),
    ]:
        assert fit_expression(rule, game.xs, game.exact_ys) == pytest.approx(expected, abs=5e-5), rule


def test_documented_token_counts():
    """The first note's ladder mis-stated two of these; pin the real counts."""
    assert len("* C1 x".split()) == 3
    assert len("* C2 * x x".split()) == 5
    assert len("+ * C1 x * C2 * x x".split()) == 9          # note said 7
    assert len("+ C0 + * C1 x * C2 * x x".split()) == 11    # note said 9
    assert len("+ * C3 sin * C4 x + C0 * C2 * x x".split()) == 14
    # and the 11-token quadratic really does fit under the 14-token cap
    assert len("+ C0 + * C1 x * C2 * x x".split()) <= 14


# ---------------------------------------------------------------------------
# Closed forms behind the section 3.1 / 3.2 figures
#
# The ladder of section 3.3 reads as eight unrelated numbers per column. It is
# not: on the symmetric grid the basis {1, x, x^2 - <x^2>} is orthogonal, so
# every score that varies within a family is a closed form in that family's own
# invariant. These pin the curves that scripts/plotting/plot_target_families.py
# draws -- without them the figures assert a theory the repo never checks.
# ---------------------------------------------------------------------------

XS_SYM = np.linspace(-1.0, 1.0, 41)
MEAN_X2 = float(np.mean(XS_SYM ** 2))                     # 0.35, exactly
MEAN_X4 = float(np.mean(XS_SYM ** 4))                     # 0.220325, exactly
KAPPA = float(np.sum((XS_SYM ** 2 - MEAN_X2) ** 2)
              / (4 * np.sum(XS_SYM ** 2)))                # 0.069875, exactly
KAPPA2 = (1.0 - MEAN_X2 ** 2 / MEAN_X4) / MEAN_X2         # 1.2685805..., exactly


def test_the_grid_constants_are_the_values_the_figures_hardcode():
    """The closed forms are only closed because these numbers are exact."""
    assert MEAN_X2 == pytest.approx(0.35, abs=1e-12)
    assert MEAN_X4 == pytest.approx(0.220325, abs=1e-12)
    assert KAPPA == pytest.approx(0.069875, abs=1e-12)
    assert KAPPA2 == pytest.approx(1.269, abs=5e-4)


def test_linear_score_is_a_closed_form_in_the_root():
    """R^2 of the 3-token '* C1 x' is clip(1 - rho^2/<x^2>), rho = |c0/c1|.

    One of the two curves that move in figure 1b. With no free constant, the
    structure pays for c0 out of its residual; x _|_ 1 on the symmetric grid
    fixes the slope at c1, so the residual is exactly c0.
    """
    for name, t in LINEAR_TARGETS.items():
        c0, c1 = t.coeffs
        rho = abs(c0 / c1)
        predicted = float(np.clip(1 - rho ** 2 / MEAN_X2, -1, 1))
        measured = fit_expression("* C1 x", t.xs(), t.ys(t.xs()))
        assert measured == pytest.approx(predicted, abs=1e-6), name


def test_quadratic_term_score_is_also_a_closed_form_in_the_root():
    """R^2 of the 5-token '* C2 * x x' is clip(-kappa2 * rho^2).

    The linear family's *second* moving row, and the other curve figure 1b
    draws. It is intercept-free too, so rho enters the same way -- but it pays
    twice: x is odd and x^2 even, so the odd component of the target is
    unreachable as well. Hence the score is <= 0 for every rho, and 0 only at
    rho = 0, where it is merely useless rather than harmful.
    """
    for name, t in LINEAR_TARGETS.items():
        c0, c1 = t.coeffs
        rho = abs(c0 / c1)
        predicted = float(np.clip(-KAPPA2 * rho ** 2, -1, 1))
        measured = fit_expression("* C2 * x x", t.xs(), t.ys(t.xs()))
        assert measured == pytest.approx(predicted, abs=1e-6), name


def test_a_free_constant_is_exactly_what_makes_a_ladder_row_flat():
    """The rule figure 1b is organised by, asserted rather than listed.

    A structure carrying a free C0 absorbs c0 whatever it is, so its R^2 cannot
    depend on rho at all; a structure without one must pay for c0 out of its
    residual, so rho enters. Sweep rho at fixed c1 and check which rows move.
    This is why exactly two of the seven rows are curves and five are flat.
    """
    rhos = [0.0, 0.1, 0.25, 0.5]
    for rule in ("C0", "+ * C1 x C0", "+ * C2 * x x C0",
                 "+ C0 + * C1 x * C2 * x x"):
        vals = [fit_expression(rule, XS_SYM, r + XS_SYM) for r in rhos]
        assert max(vals) - min(vals) < 1e-6, f"{rule} has an intercept but moved"
    for rule in ("* C1 x", "* C2 * x x"):
        vals = [fit_expression(rule, XS_SYM, r + XS_SYM) for r in rhos]
        assert max(vals) - min(vals) > 0.1, f"{rule} is intercept-free but did not move"


def test_the_three_crossings_figure_1b_labels():
    """The curves' landmarks, all three inside the domain -- which is the point:
    the reward stops discriminating before the root has even left [-1, 1]."""
    assert np.sqrt(MEAN_X2) == pytest.approx(0.592, abs=5e-4)       # C1x hits 0
    assert np.sqrt(2 * MEAN_X2) == pytest.approx(0.837, abs=5e-4)   # C1x floors
    assert 1 / np.sqrt(KAPPA2) == pytest.approx(0.888, abs=5e-4)    # C2x^2 floors
    assert max(np.sqrt(2 * MEAN_X2), 1 / np.sqrt(KAPPA2)) < 1.0     # all inside
    assert fit_expression("* C1 x", XS_SYM, np.sqrt(MEAN_X2) + XS_SYM) == \
        pytest.approx(0.0, abs=1e-6)
    assert fit_expression("* C1 x", XS_SYM, np.sqrt(2 * MEAN_X2) + XS_SYM) == \
        pytest.approx(-1.0, abs=1e-6)
    assert fit_expression("* C2 * x x", XS_SYM, (1 / np.sqrt(KAPPA2)) + XS_SYM) == \
        pytest.approx(-1.0, abs=1e-6)


def test_the_other_linear_ladder_rows_are_constant_across_the_family():
    """Only the intercept-free rows vary with the root; the rest are flat.

    Figure 1b draws five flat rows and two curves. That split is only honest if
    these four really are constant down the whole column (the fifth flat row,
    the sinusoid, is a measured fit failure and is pinned separately).
    """
    for rule, expected in (("C0", 0.0),
                           ("+ * C1 x C0", 1.0),
                           ("+ * C2 * x x C0", 0.0),
                           ("+ C0 + * C1 x * C2 * x x", 1.0)):
        for name, t in LINEAR_TARGETS.items():
            got = fit_expression(rule, t.xs(), t.ys(t.xs()))
            assert got == pytest.approx(expected, abs=1e-6), f"{rule} on {name}"


def test_the_sinusoid_row_is_a_measured_failure_not_a_closed_form():
    """Figure 1b draws six curves as exact arithmetic and one as measured. The
    sinusoid is the exception because C4 is the one constant in this grammar
    that its structure is *nonlinear* in: every other candidate is linear
    least-squares in disguise and fits regardless of scale, while this one
    stalls under the cap and returns the clip floor. The -1 is a property of the
    oracle, not of the structure.
    """
    for name, t in LINEAR_TARGETS.items():
        got = fit_expression("* C3 sin * C4 x", t.xs(), t.ys(t.xs()), max_nfev=50)
        assert got == pytest.approx(-1.0, abs=1e-9), name


def test_quadratic_two_term_rows_are_a_closed_form_in_the_vertex():
    """R^2 of '+ * C1 x C0' is lambda = x_v^2/(x_v^2+kappa), and R^2 of
    '+ * C2 * x x C0' is 1 - lambda. One number per target, set by the vertex
    alone -- the two curves figure 2b's panel A draws."""
    for name, t in QUADRATIC_TARGETS.items():
        lam = t.vertex ** 2 / (t.vertex ** 2 + KAPPA)
        lin = fit_expression("+ * C1 x C0", t.xs(), t.ys(t.xs()))
        quad = fit_expression("+ * C2 * x x C0", t.xs(), t.ys(t.xs()))
        assert lin == pytest.approx(lam, abs=1e-6), name
        assert quad == pytest.approx(1 - lam, abs=1e-6), name


def test_quadratic_two_term_rows_sum_to_one():
    """The orthogonal split, stated without reference to the closed form: the
    linear share and the curvature share exhaust the variance."""
    for name, t in QUADRATIC_TARGETS.items():
        lin = fit_expression("+ * C1 x C0", t.xs(), t.ys(t.xs()))
        quad = fit_expression("+ * C2 * x x C0", t.xs(), t.ys(t.xs()))
        assert lin + quad == pytest.approx(1.0, abs=1e-6), name


def test_quad_B_trap_depth_is_lambda_of_its_vertex():
    """Section 6.1's 0.9972 deceptive cheat is not a measured coincidence: it is
    lambda(x_v=5), forced by where quad_B's vertex was placed."""
    t = get_target("quad_B")
    assert t.vertex == pytest.approx(5.0)
    lam = 25.0 / (25.0 + KAPPA)
    assert lam == pytest.approx(0.9972, abs=5e-5)
    assert fit_expression("+ * C1 x C0", t.xs(), t.ys(t.xs())) == pytest.approx(lam, abs=1e-6)


# ---------------------------------------------------------------------------
# Figure 2b: the quadratic family needs TWO coordinates, not one
#
# The linear family has one reward-visible degree of freedom, so figure 1b puts
# all seven rows on a single rho axis. This family has two. The seam is the same
# one -- whether the structure carries a free C0 -- with the two sets swapped:
# here the intercept-carrying rows are the ones the vertex settles, and the two
# without an intercept must also pay for an offset the vertex cannot see.
# ---------------------------------------------------------------------------

KAPPA2_M2 = KAPPA2 * MEAN_X2                              # 0.444003..., exactly


def reward_coords(c0, c1, c2):
    """(lambda, omega, omega0): a target's reward-visible coordinates.

    Mirrors reward_coords() in scripts/plotting/plot_target_families.py. lambda
    is the linear share of the variance; omega = ybar/s and omega0 = c0/s are
    the mean and the intercept in units of the target's own spread, which is
    what makes them scale-free. Lines and parabolas alike -- a line is lambda=1.
    """
    y = c0 + c1 * XS_SYM + c2 * XS_SYM ** 2
    s = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    var_x2 = MEAN_X4 - MEAN_X2 ** 2
    lam = c1 ** 2 * MEAN_X2 / (c1 ** 2 * MEAN_X2 + c2 ** 2 * var_x2)
    return lam, float(y.mean()) / s, c0 / s


def predicted_intercept_free(lam, omega, omega0):
    """What figure 2b's panel B claims the two intercept-free rows score."""
    return (float(np.clip(lam - omega ** 2, -1.0, 1.0)),
            float(np.clip((1.0 - lam) - KAPPA2_M2 * omega0 ** 2, -1.0, 1.0)))


def test_the_rate_figure_2b_hardcodes_is_exact():
    """0.444 is the fraction of the intercept `* C2 * x x` still pays after x^2
    has absorbed what it can. `* C1 x` pays the whole thing, at a rate of 1."""
    assert KAPPA2_M2 == pytest.approx(0.444, abs=5e-4)


def test_the_vertex_alone_does_not_settle_the_intercept_free_rows():
    """Why figure 2b needs two panels where figure 1b needed one axis.

    Hold the vertex fixed and move the offset. The rows WITH a free C0 do not
    budge -- the constant absorbs c0, whatever it is -- while the two without
    one swing across most of the reward's range. Drawing those two against |x_v|
    would assert a function that does not exist.
    """
    settled, moved = [], []
    for c0 in (0.0, 0.5, 1.0, 3.0, 10.0):
        ys = c0 - 0.5 * XS_SYM + XS_SYM ** 2      # x_v = -c1/2c2 = 0.25 throughout
        settled.append(fit_expression("+ * C1 x C0", XS_SYM, ys))
        moved.append(fit_expression("* C1 x", XS_SYM, ys))
    lam = 0.25 ** 2 / (0.25 ** 2 + KAPPA)
    assert settled == pytest.approx([lam] * 5, abs=1e-6)
    assert max(moved) - min(moved) > 0.8


def test_intercept_free_rows_are_a_closed_form_in_the_normalized_offset():
    """The law figure 2b's panel B draws, on every member of BOTH families:

        R^2(`* C1 x`)     = clip(lambda - omega^2)
        R^2(`* C2 * x x`) = clip((1-lambda) - kappa2<x^2> * omega0^2)

    Read against panel A -- where lambda is exactly what `+ * C1 x C0` scores
    and 1-lambda what `+ * C2 * x x C0` scores -- each line says the same thing:
    a row without an intercept scores what its counterpart scores, minus a
    quadratic in the normalized offset. That is why the second coordinate costs
    a panel and not a curve family: it enters through the offset and nowhere
    else.
    """
    for name, t in TARGETS.items():
        c0, c1, c2 = (list(t.coeffs) + [0.0, 0.0, 0.0])[:3]
        assert t.xs() == pytest.approx(XS_SYM), name   # the closed forms need this grid
        want_lin, want_quad = predicted_intercept_free(*reward_coords(c0, c1, c2))
        ys = t.ys(t.xs())
        assert fit_expression("* C1 x", t.xs(), ys) == pytest.approx(want_lin, abs=1e-6), name
        assert fit_expression("* C2 * x x", t.xs(), ys) == pytest.approx(want_quad, abs=1e-6), name


def test_figure_1b_is_the_lambda_equals_one_slice_of_figure_2b():
    """The two figures draw ONE law, not two. A line is lambda = 1, and there
    the panel-B laws collapse to figure 1b's two falling curves exactly:
    omega^2 = rho^2/<x^2> and kappa2<x^2>*omega0^2 = kappa2*rho^2."""
    for name, t in LINEAR_TARGETS.items():
        c0, c1 = t.coeffs
        rho = abs(c0 / c1)
        lam, omega, omega0 = reward_coords(c0, c1, 0.0)
        assert lam == pytest.approx(1.0, abs=1e-12), name
        assert omega ** 2 == pytest.approx(rho ** 2 / MEAN_X2, rel=1e-9), name
        assert KAPPA2_M2 * omega0 ** 2 == pytest.approx(KAPPA2 * rho ** 2, rel=1e-9), name
        # ...which is to say: figure 1b's closed forms, recovered from 2b's.
        want_lin, want_quad = predicted_intercept_free(lam, omega, omega0)
        assert want_lin == pytest.approx(np.clip(1 - rho ** 2 / MEAN_X2, -1, 1), abs=1e-9), name
        assert want_quad == pytest.approx(np.clip(-KAPPA2 * rho ** 2, -1, 1), abs=1e-9), name


def test_the_law_holds_away_from_the_eight_designed_members():
    """Not an artifact of the coefficients section 1 happens to choose: random
    lines and parabolas obey it too."""
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(120):
        c0, c1, c2 = rng.normal(0.0, 3.0, 3)
        if rng.random() < 0.2:
            c2 = 0.0                                   # keep some pure lines in the draw
        if abs(c1) < 1e-6 and abs(c2) < 1e-6:
            continue                                   # a constant has no variance to share
        ys = c0 + c1 * XS_SYM + c2 * XS_SYM ** 2
        want_lin, want_quad = predicted_intercept_free(*reward_coords(c0, c1, c2))
        assert fit_expression("* C1 x", XS_SYM, ys) == pytest.approx(want_lin, abs=1e-6)
        assert fit_expression("* C2 * x x", XS_SYM, ys) == pytest.approx(want_quad, abs=1e-6)
        checked += 1
    assert checked > 100


def test_reward_is_invariant_to_rescaling_but_not_to_shifting():
    """Why the figures draw the members RAW on per-panel axes.

    Every ladder structure is closed under y -> a*y, so rescaling is free and a
    normalized panel would preserve the ladder. Shifting is not free -- and the
    1e6 spread these families are built around lives in the *offset*. A
    normalization that shifts therefore erases the family.
    """
    rules = ["C0", "* C1 x", "* C2 * x x", "+ * C1 x C0",
             "+ * C2 * x x C0", "+ C0 + * C1 x * C2 * x x"]
    shift_changed = 0
    for name, t in TARGETS.items():
        xs, ys = t.xs(), t.ys(t.xs())
        for rule in rules:
            base = fit_expression(rule, xs, ys)
            assert fit_expression(rule, xs, 137.0 * ys) == pytest.approx(base, abs=1e-6), \
                f"rescaling changed {rule} on {name}"
            if abs(fit_expression(rule, xs, ys + 3.0) - base) > 1e-3:
                shift_changed += 1
    assert shift_changed > 0, "a shift must not be free, or the offset carries no signal"


def test_z_scoring_collapses_the_linear_family_but_not_the_quadratic():
    """The corollary that rules out the obvious shared-axis fix.

    Every non-degenerate line is the same line after z-scoring, so a z-scored
    panel would draw lin_A..lin_D as one curve and claim it had shown four. The
    quadratics survive, which is exactly why the two figures may share a layout
    but not a normalization.
    """
    def zs(d):
        return np.array([(t.ys(t.xs()) - t.ys(t.xs()).mean()) / t.ys(t.xs()).std()
                         for t in d.values()])

    lin = zs(LINEAR_TARGETS)
    assert float(np.max(lin.max(0) - lin.min(0))) < 1e-8
    quad = zs(QUADRATIC_TARGETS)
    assert float(np.max(quad.max(0) - quad.min(0))) > 1.0
