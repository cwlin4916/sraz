"""Named symbolic-regression problem instances.

An *instance* pairs a grammar (what expressions can be built) with a target
(what the emitted expression is fitted against). The engine and the game are
instance-agnostic; this registry is where we define concrete instances to place
on the informativeness x deception 2x2 (see
Claude-research/02-informativeness-and-deception.md).

- "sine"              : the original deceptive + uninformative game
                        (4*sin(4x) + C0 + C1*x + C2*x**2, full grammar with
                        division/product/sine). The shipped default.
- "additive_quadratic": the best-corner positive control (informative +
                        non-deceptive): sums of monomials only, target exactly
                        expressible, so R^2 = 1 is reachable and adding a term
                        never lowers the fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sraz.instances.symreg.game import (
    ADDITIVE_GRAMMAR,
    ADDITIVE_INFIX,
    SR_GRAMMAR,
    TARGET_INFIX,
    target_ys,
    target_ys_additive,
)


@dataclass(frozen=True)
class SRProblem:
    """A concrete SR instance: grammar + target + the constants to fit."""
    name: str
    grammar_rules: dict
    target_ys_fn: Callable
    constant_names: tuple
    infix: str

    def game_kwargs(self) -> dict:
        """The keyword arguments SymRegGame needs to realize this instance."""
        return {
            "grammar_rules": self.grammar_rules,
            "target_ys_fn": self.target_ys_fn,
            "constant_names": self.constant_names,
            "target_infix": self.infix,
        }


PROBLEMS: dict[str, SRProblem] = {
    "sine": SRProblem(
        name="sine",
        grammar_rules=SR_GRAMMAR,
        target_ys_fn=target_ys,
        constant_names=("C0", "C1", "C2"),
        infix=TARGET_INFIX,
    ),
    "additive_quadratic": SRProblem(
        name="additive_quadratic",
        grammar_rules=ADDITIVE_GRAMMAR,
        target_ys_fn=target_ys_additive,
        constant_names=("C0", "C1", "C2"),
        infix=ADDITIVE_INFIX,
    ),
}


def get_problem(name: str) -> SRProblem:
    if name not in PROBLEMS:
        raise KeyError(
            f"unknown problem {name!r}; available: {sorted(PROBLEMS)}")
    return PROBLEMS[name]
