"""Target families for the symbolic-regression grammar game.

A :class:`Target` fixes a hidden function and the grid it is sampled on. It
replaces the game's default behaviour of drawing C* from a problem seed: a
target's coefficients are *chosen*, not sampled, so each family member is a
named, reproducible point in function space rather than a lottery ticket.

Two families are defined, both on the symmetric domain x in [-1, 1]:

- ``linear``    -- y = c0 + c1 x
- ``quadratic`` -- y = c0 + c1 x + c2 x^2

The quadratic members partition the space by the two geometric invariants that
matter relative to the domain: the vertex ``x_v = -c1 / (2 c2)`` (inside or
outside the domain) and the discriminant ``D = c1^2 - 4 c0 c2`` (real roots or
none). The linear members mirror the same logic with the only invariants a
line has: its root ``-c0 / c1`` and its coefficient scaling.

Both families are scored by the same 7-production grammar as the default
target, sinusoid production included. The sinusoid is a *distractor* here: no
member needs it, so a search that reaches for it is overfitting structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


X_MIN_SYM, X_MAX_SYM, N_X_SYM = -1.0, 1.0, 41


@dataclass(frozen=True)
class Target:
    """A hidden function plus the grid it is sampled on.

    Coefficients are ordered by ascending power: ``(c0, c1)`` for a linear
    target, ``(c0, c1, c2)`` for a quadratic one.
    """

    name: str
    family: str          # "linear" | "quadratic"
    coeffs: tuple[float, ...]
    label: str           # one-line description of the design intent
    x_min: float = X_MIN_SYM
    x_max: float = X_MAX_SYM
    n_x: int = N_X_SYM

    def __post_init__(self):
        if self.family == "linear" and len(self.coeffs) != 2:
            raise ValueError(f"{self.name}: linear target needs 2 coeffs, got {len(self.coeffs)}")
        if self.family == "quadratic" and len(self.coeffs) != 3:
            raise ValueError(f"{self.name}: quadratic target needs 3 coeffs, got {len(self.coeffs)}")
        if self.family not in ("linear", "quadratic"):
            raise ValueError(f"{self.name}: unknown family {self.family!r}")
        if self.x_min >= self.x_max:
            raise ValueError(f"{self.name}: empty domain [{self.x_min}, {self.x_max}]")

    def xs(self) -> np.ndarray:
        return np.linspace(self.x_min, self.x_max, self.n_x)

    def ys(self, xs: np.ndarray) -> np.ndarray:
        """Evaluate the target. Horner over ascending-power coefficients."""
        out = np.zeros_like(xs, dtype=np.float64)
        for power, c in enumerate(self.coeffs):
            out = out + c * xs ** power
        return out

    @property
    def short_label(self) -> str:
        """The design-intent phrase alone, without the coefficient detail.

        Figure legends name a target by its shape, not by its dict key.
        """
        return self.label.split(":")[0]

    @property
    def infix(self) -> str:
        terms = []
        for power, c in enumerate(self.coeffs):
            if power == 0:
                terms.append(f"{c!r}")
            elif power == 1:
                terms.append(f"{c!r}*x")
            else:
                terms.append(f"{c!r}*x**{power}")
        return " + ".join(terms)

    @property
    def vertex(self) -> float | None:
        """Stationary point x_v = -c1 / (2 c2); None for a linear target."""
        if self.family != "quadratic":
            return None
        c0, c1, c2 = self.coeffs
        if c2 == 0.0:
            return None
        return -c1 / (2.0 * c2)

    @property
    def discriminant(self) -> float | None:
        """D = c1^2 - 4 c0 c2; None for a linear target."""
        if self.family != "quadratic":
            return None
        c0, c1, c2 = self.coeffs
        return c1 ** 2 - 4.0 * c0 * c2

    @property
    def real_roots(self) -> list[float]:
        """Real roots of the target, ascending."""
        # np.roots wants descending powers
        coeffs_desc = list(reversed(self.coeffs))
        if all(c == 0.0 for c in coeffs_desc[:-1]):
            return []
        r = np.roots(coeffs_desc)
        return sorted(float(z.real) for z in np.atleast_1d(r) if abs(z.imag) < 1e-12)

    def vertex_in_domain(self) -> bool:
        xv = self.vertex
        return xv is not None and self.x_min <= xv <= self.x_max

    def roots_in_domain(self) -> list[float]:
        return [r for r in self.real_roots if self.x_min <= r <= self.x_max]


# ---------------------------------------------------------------------------
# The linear family: y = c0 + c1 x on [-1, 1]
#
# A line has no vertex, so the family varies the only invariants it has --
# where its root sits relative to the domain, and how its two coefficients are
# scaled against each other.
# ---------------------------------------------------------------------------

LINEAR_TARGETS = {
    "lin_A": Target(
        name="lin_A", family="linear", coeffs=(0.5, 2.0),
        label="Root interior: root at x=-0.25, both coefficients O(1)",
    ),
    "lin_B": Target(
        name="lin_B", family="linear", coeffs=(5.0, 1.0),
        label="Root exterior: root at x=-5, offset dominates over the domain",
    ),
    "lin_C": Target(
        name="lin_C", family="linear", coeffs=(1000.0, 0.001),
        label="Ill-conditioned: c0/c1 = 1e6, mirrors quadratic Family C",
    ),
    "lin_D": Target(
        name="lin_D", family="linear", coeffs=(0.0, 2.0),
        label="Pure proportional: y = 2x, exactly the 3-token '* C1 x'",
    ),
}


# ---------------------------------------------------------------------------
# The quadratic family: y = c0 + c1 x + c2 x^2 on [-1, 1]
#
# Partitioned by vertex position and discriminant, per the design in the
# note's Section 3. Each member is a topologically distinct shape over the
# domain, not a random draw.
# ---------------------------------------------------------------------------

QUADRATIC_TARGETS = {
    "quad_A": Target(
        name="quad_A", family="quadratic", coeffs=(1.0, -1.0, 2.0),
        label="Interior extremum: vertex x_v=0.25 inside the domain, D=-7<0 (arch)",
    ),
    # c0 = 6 (not 1) keeps both roots outside [-1, 1]: with c0 = 1 the smaller
    # root falls at x = 0.204, which would make this a root-crossing target and
    # collide with Family D. Family B must be monotonic *and* sign-stable.
    "quad_B": Target(
        name="quad_B", family="quadratic", coeffs=(6.0, -5.0, 0.5),
        label="Monotonic exterior: vertex x_v=5 far outside, pseudo-linear on the domain",
    ),
    # c0 = 0.001 (not 0) avoids a degenerate double root at the origin, which
    # would make this a root-crossing target too. Every coefficient but c2 is
    # 1e-3, so the 1e6 scale spread is the only thing being tested.
    "quad_C": Target(
        name="quad_C", family="quadratic", coeffs=(0.001, 0.001, 1000.0),
        label="Ill-conditioned scales: c2=1000 vs c1=0.001, stresses the inner optimizer",
    ),
    # Roots at -0.6 and 0.4 rather than a symmetric +/-0.5: symmetric roots
    # force c1 = 0, and a target with no linear term is fitted exactly by the
    # 7-token 'C2 x^2 + C0'. That would make Family D *easier* than Family A
    # and stop it testing the three-term derivation at all.
    "quad_D": Target(
        name="quad_D", family="quadratic", coeffs=(-0.48, 0.4, 2.0),
        label="Root crossing: D=4>0, roots at x=-0.6 and 0.4 both inside the domain",
    ),
}


TARGETS: dict[str, Target] = {**LINEAR_TARGETS, **QUADRATIC_TARGETS}


def get_target(name: str) -> Target:
    if name not in TARGETS:
        raise KeyError(
            f"Unknown target {name!r}. Available: {', '.join(sorted(TARGETS))}"
        )
    return TARGETS[name]


def family_targets(family: str) -> list[Target]:
    """All targets in a family, in declaration order."""
    out = [t for t in TARGETS.values() if t.family == family]
    if not out:
        raise KeyError(f"Unknown family {family!r}. Available: linear, quadratic")
    return out
