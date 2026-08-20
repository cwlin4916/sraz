#!/usr/bin/env python3
"""Exact root quantities for the target family under one fixed MDP.

Reproduces Table 2 of writeup-milton (the R(D_i), V^q(C), rho(C), margin
columns) by exhaustive enumeration and backward induction over the whole
reachable state graph -- no search, no sampling, no MCTS. This is
Experiment 1 of the writeup, and it is the prerequisite for every later
experiment: if the MDP here is not the MDP the writeup specifies, every
downstream search number is measuring the wrong environment.

Two things are deliberately re-derived rather than trusted:

1.  The MDP itself. `SymRegGame` defaults to the *sine* grammar at
    max_len=15. The controlled study needs ADDITIVE_GRAMMAR at L=12. The
    script asserts the resulting root has exactly 4 legal actions and that
    the exact expression scores 1.
2.  The transition rule. Enumeration is done symbolically (tuples of
    tokens) because cloning a Game per state is far too slow, so the
    symbolic legal-action set is cross-checked against the real
    `game.get_action_mask()` on randomly drawn reachable states.

Outputs (to --out-dir):
    07-exact-audit.json    all computed quantities, machine-readable
    07-exact-audit.md      Table 2 comparison + census
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sraz.instances.symreg.game import (  # noqa: E402
    ADDITIVE_GRAMMAR, SymRegGame, fit_expression,
)
from sraz.instances.symreg.targets import family_targets, get_target  # noqa: E402

# ---------------------------------------------------------------------------
# The MDP, stated once, symbolically.
#
# Production order follows ADDITIVE_GRAMMAR["S"] exactly, so index i here is
# production i in the real grammar. The writeup's names:
#   C  = "+ S S"          the only continuing action
#   D1 = "C0"             the three D_i are immediately terminal
#   D2 = "* C1 x"
#   D3 = "* C2 * x x"
# ---------------------------------------------------------------------------
PRODS = [tuple(rhs) for rhs in ADDITIVE_GRAMMAR["S"]]
PROD_NAMES = ["C", "D1", "D2", "D3"]
START = ("S",)
L_DEFAULT = 12
TAU_DEFAULT = 1e-6

# Table 2 of the writeup, verbatim, for the rows this script can check.
# (R(D1), R(D2), R(D3), V^q(C), rho(C), margin, note)
TABLE2 = {
    "lin_A":  (0.0,  0.8214, -0.0793, 0.5646, 0.3945,  0.2569, "trap"),
    "lin_B":  (0.0, -1.0,    -1.0,    0.1341, 0.3945, -0.1341, "---"),
    "lin_C":  (0.0, -1.0,    -1.0,    0.1341, 0.3945, -0.1341, "---"),
    "lin_D":  (0.0,  1.0,     0.0,    0.5924, 0.5924,  0.4076, "solved at once"),
    "quad_A": (0.0, -1.0,    -0.0711, 0.3540, 0.1128, -0.3540, "---"),
    "quad_B": (0.0, -1.0,    -1.0,    0.1563, 0.1128, -0.1563, "plateau"),
    "quad_C": (0.0, -1.0,     1.0,    0.4840, 0.5569,  0.5160, "solved at once"),
    "quad_D": (0.0,  0.0170,  0.6461, 0.5106, 0.1128,  0.1355, "trap"),
}
# Census the writeup reports for this MDP (L=12, 4 productions).
CENSUS_EXPECTED = {"reachable": 4898, "terminal": 247, "nonterminal": 4651}
PARITY_EXPECTED = {1: 1, 3: 2, 5: 5, 7: 14, 9: 48, 11: 177}
EXACT_SENTENCE = "+ C0 + * C1 x * C2 * x x"


def legal_actions(form: tuple[str, ...], L: int) -> list[tuple[int, int]]:
    """(position, production) pairs the game would unmask at `form`.

    Mirrors `SymRegGame.get_action_mask` (game.py:301): production j is legal
    at nonterminal position i iff `len + len(rhs_j) - 1 < L`. The bound does
    not involve i, so every nonterminal position offers the same production
    set -- hence |A(s)| = (#nonterminals) * (#productions that fit).
    """
    fits = [j for j, rhs in enumerate(PRODS) if len(form) + len(rhs) - 1 < L]
    return [(i, j) for i, tok in enumerate(form) if tok == "S" for j in fits]


def apply_action(form: tuple[str, ...], pos: int, prod: int) -> tuple[str, ...]:
    return form[:pos] + PRODS[prod] + form[pos + 1:]


def is_terminal(form: tuple[str, ...]) -> bool:
    return "S" not in form


def enumerate_reachable(L: int) -> tuple[list, set]:
    """Every form reachable from S. Returns (all_forms, terminal_forms)."""
    seen, terminals, frontier = {START}, set(), [START]
    while frontier:
        form = frontier.pop()
        if is_terminal(form):
            terminals.add(form)
            continue
        for pos, prod in legal_actions(form, L):
            child = apply_action(form, pos, prod)
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return sorted(seen, key=lambda f: (len(f), f)), terminals


def validate_against_game(game: SymRegGame, forms: list, L: int, n: int, rng) -> dict:
    """Cross-check the symbolic transition against the real game's mask.

    Drives the real game down a random derivation and, at every state along
    the way, compares its unmasked (pos, prod) set to `legal_actions`.
    """
    nprods = game.grammar.nprods
    checked = mismatches = 0
    for _ in range(n):
        game.reset_wrapper()
        form = START
        while not is_terminal(form):
            mask = game.get_action_mask()
            real = {divmod(int(a), nprods) for a in np.flatnonzero(mask)}
            if real != set(legal_actions(form, L)):
                mismatches += 1
            checked += 1
            pos, prod = random.Random(int(rng.integers(1 << 30))).choice(
                sorted(real))
            game.step_wrapper(pos * nprods + prod)
            form = apply_action(form, pos, prod)
        # the decoded sentence must agree too
        if " ".join(form) != game._decode_state():
            mismatches += 1
        checked += 1
    return {"states_checked": checked, "mismatches": mismatches}


def score_terminals(terminals: set, xs, ys, cache_path: Path) -> dict:
    """R^2 for every terminal sentence, memoised on disk across runs."""
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    fresh = 0
    scores = {}
    for form in terminals:
        sent = " ".join(form)
        if sent not in cache:
            cache[sent] = float(fit_expression(sent, xs, ys))
            fresh += 1
        scores[form] = cache[sent]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return scores, fresh


def backward_induction(forms: list, scores: dict, L: int, tau: float) -> dict:
    """V*, V^q and rho at every reachable form.

    V*(s)   = max_a V*(T(s,a))                     ... optimal value
    V^q(s)  = mean_a V^q(T(s,a))                   ... value of the uniform
                                                       completion policy q
    rho(s)  = mean_a rho(T(s,a))                   ... prob. q reaches an
                                                       exact terminal
    with V* = V^q = R and rho = 1{R >= 1-tau} at terminals. q is uniform over
    the *flattened* (pos, prod) mask, which is what the writeup specifies
    (sec. 2) and what the repo's rollouts actually sample.

    Children never have a shorter form than their parent, and the one
    length-preserving action ("C0") strictly drops the nonterminal count, so
    processing by (-len, #nonterminals) visits every child before its parent.
    """
    order = sorted(forms, key=lambda f: (-len(f), sum(t == "S" for t in f)))
    Vstar, Vq, rho = {}, {}, {}
    for form in order:
        if is_terminal(form):
            r = scores[form]
            Vstar[form] = Vq[form] = r
            rho[form] = 1.0 if r >= 1.0 - tau else 0.0
            continue
        kids = [apply_action(form, p, j) for p, j in legal_actions(form, L)]
        Vstar[form] = max(Vstar[k] for k in kids)
        Vq[form] = float(np.mean([Vq[k] for k in kids]))
        rho[form] = float(np.mean([rho[k] for k in kids]))
    return {"Vstar": Vstar, "Vq": Vq, "rho": rho}


def length_ceiling(terminals: set, scores: dict) -> dict:
    """Best R^2 attainable by a terminal of each length (writeup fig. panel a)."""
    out: dict[int, float] = {}
    for form in terminals:
        out[len(form)] = max(out.get(len(form), -np.inf), scores[form])
    return {k: float(v) for k, v in sorted(out.items())}


def audit(name: str, forms: list, terminals: set, L: int, tau: float,
          cache_dir: Path, verbose: bool = True) -> dict:
    target = get_target(name)
    xs = target.xs()
    ys = target.ys(xs)
    game = SymRegGame(target=name, grammar_rules=ADDITIVE_GRAMMAR, max_len=L)

    # --- setup assertions: is this actually the writeup's MDP? -------------
    root_actions = len(legal_actions(START, L))
    assert root_actions == 4, f"{name}: root has {root_actions} actions, want 4"
    assert int(game.get_action_mask().sum()) == 4, \
        f"{name}: real game root mask has {int(game.get_action_mask().sum())}, want 4"
    r_exact = float(fit_expression(EXACT_SENTENCE, xs, ys))
    assert r_exact >= 1.0 - tau, \
        f"{name}: exact expression scores {r_exact!r}, not exact to tau={tau}"

    scores, fresh = score_terminals(terminals, xs, ys, cache_dir / f"fit_{name}.json")
    vals = backward_induction(forms, scores, L, tau)
    Vstar, Vq, rho = vals["Vstar"], vals["Vq"], vals["rho"]

    # --- root decomposition ------------------------------------------------
    children = {PROD_NAMES[j]: apply_action(START, p, j)
                for p, j in legal_actions(START, L)}
    R = {nm: scores[f] for nm, f in children.items() if is_terminal(f)}
    C = children["C"]
    best_Di = max(R.values())
    margin = best_Di - Vq[C]
    trap = (best_Di > Vq[C]) and (best_Di < Vstar[C] - tau)

    row = {
        "target": name,
        "infix": target.infix,
        "coeffs": list(target.coeffs),
        "support": sorted(p for p, c in enumerate(target.coeffs) if c != 0),
        "R_D1": R["D1"], "R_D2": R["D2"], "R_D3": R["D3"],
        "V_star_C": Vstar[C], "V_q_C": Vq[C], "rho_C": rho[C],
        "margin": margin, "inversion_trap": bool(trap),
        "V_star_s0": Vstar[START], "V_q_s0": Vq[START], "rho_s0": rho[START],
        "R_exact_sentence": r_exact,
        "length_ceiling": length_ceiling(terminals, scores),
        "n_exact_terminals": sum(1 for f in terminals if scores[f] >= 1.0 - tau),
        "fresh_fits": fresh,
    }

    # rho(s0) vs rho(C)/4 -- the three D_i are terminal and (unless the target
    # is solved by one atom) inexact, so the exact mass at the root should be
    # exactly a quarter of that at C. Recorded, not asserted: it fails by
    # design on lin_D / quad_C, where a D_i *is* exact.
    row["rho_s0_over_rho_C"] = (rho[START] / rho[C]) if rho[C] else None

    if verbose:
        print(f"\n--- {name}: {target.infix}")
        print(f"    support P = {row['support']}   "
              f"exact-expr R = {r_exact:.12f}   fresh fits = {fresh}")
        print(f"    R(D1)={R['D1']:+.4f}  R(D2)={R['D2']:+.4f}  "
              f"R(D3)={R['D3']:+.4f}")
        print(f"    V*(C)={Vstar[C]:.6f}  Vq(C)={Vq[C]:.4f}  "
              f"rho(C)={rho[C]:.4f}  margin={margin:+.4f}  "
              f"{'TRAP' if trap else '---'}")
        print(f"    root: V*={Vstar[START]:.6f}  Vq={Vq[START]:.4f}  "
              f"rho={rho[START]:.4f}  (rho(s0)/rho(C)={row['rho_s0_over_rho_C']})")
    return row


def check_table2(row: dict, tol: float = 5e-4) -> list[str]:
    """Compare a computed row against the writeup's Table 2. Returns failures."""
    name = row["target"]
    if name not in TABLE2:
        return [f"{name}: no Table 2 entry to check against"]
    exp = TABLE2[name]
    fields = [("R_D1", exp[0]), ("R_D2", exp[1]), ("R_D3", exp[2]),
              ("V_q_C", exp[3]), ("rho_C", exp[4]), ("margin", exp[5])]
    fails = []
    for key, want in fields:
        got = row[key]
        if abs(got - want) > tol:
            fails.append(f"{name}.{key}: got {got:.6f}, Table 2 says {want:.4f}")
    want_trap = exp[6] == "trap"
    if row["inversion_trap"] != want_trap:
        fails.append(f"{name}.inversion_trap: got {row['inversion_trap']}, "
                     f"Table 2 says {exp[6]!r}")
    if abs(row["V_star_C"] - 1.0) > tol:
        fails.append(f"{name}.V_star_C: got {row['V_star_C']:.6f}, want 1")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=None,
                    help="target names (default: the linear family)")
    ap.add_argument("--family", default="linear",
                    help="family to use when --targets is omitted")
    ap.add_argument("--max-len", type=int, default=L_DEFAULT)
    ap.add_argument("--tau", type=float, default=TAU_DEFAULT)
    ap.add_argument("--validate-n", type=int, default=200,
                    help="random derivations used to check the symbolic mask")
    ap.add_argument("--out-dir", default=str(REPO / "Claude-experiments" / "8-17"))
    args = ap.parse_args()

    names = args.targets or [t.name for t in family_targets(args.family)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L, tau = args.max_len, args.tau

    print(f"MDP: ADDITIVE_GRAMMAR ({len(PRODS)} productions), L={L}, tau={tau:g}")
    print(f"targets: {', '.join(names)}")

    # --- the state graph is target-independent: enumerate once -------------
    forms, terminals = enumerate_reachable(L)
    census = {"reachable": len(forms), "terminal": len(terminals),
              "nonterminal": len(forms) - len(terminals)}
    parity: dict[int, int] = {}
    for f in terminals:
        parity[len(f)] = parity.get(len(f), 0) + 1
    parity = dict(sorted(parity.items()))
    print(f"\ncensus: {census}   (writeup: {CENSUS_EXPECTED})")
    print(f"terminals by length: {parity}")
    print(f"           (writeup: {PARITY_EXPECTED})")

    problems = []
    if census != CENSUS_EXPECTED:
        problems.append(f"census {census} != writeup {CENSUS_EXPECTED}")
    if parity != PARITY_EXPECTED:
        problems.append(f"parity {parity} != writeup {PARITY_EXPECTED}")

    # --- symbolic transition vs the real game ------------------------------
    rng = np.random.default_rng(0)
    ref = SymRegGame(target=names[0], grammar_rules=ADDITIVE_GRAMMAR, max_len=L)
    val = validate_against_game(ref, forms, L, args.validate_n, rng)
    print(f"\nmask cross-check vs real game: {val['states_checked']} states, "
          f"{val['mismatches']} mismatches")
    if val["mismatches"]:
        problems.append(f"symbolic mask disagrees with game on "
                        f"{val['mismatches']} states")

    cache_dir = out_dir / "cache"
    rows = [audit(n, forms, terminals, L, tau, cache_dir) for n in names]

    # --- validation against Table 2 ---------------------------------------
    print("\n" + "=" * 72)
    fails = [f for row in rows for f in check_table2(row, tol=5e-4)]
    if fails:
        print("TABLE 2 MISMATCHES:")
        for f in fails:
            print("  " + f)
    else:
        print(f"Table 2 reproduced exactly for {len(rows)} target(s) "
              f"(tol 5e-4 on every checked field).")
    problems.extend(fails)

    result = {
        "mdp": {"grammar": "ADDITIVE_GRAMMAR", "productions": [list(p) for p in PRODS],
                "max_len": L, "tau": tau, "root_actions": len(legal_actions(START, L)),
                "exact_sentence": EXACT_SENTENCE},
        "census": census, "terminals_by_length": parity,
        "census_expected": CENSUS_EXPECTED, "parity_expected": PARITY_EXPECTED,
        "mask_validation": val,
        "rows": rows,
        "table2_failures": fails,
        "ok": not problems,
    }
    (out_dir / "07-exact-audit.json").write_text(json.dumps(result, indent=2))

    # --- markdown report ---------------------------------------------------
    hdr = ("| target | y(x) | R(D1) | R(D2) | R(D3) | V*(C) | Vq(C) | rho(C) "
           "| margin | trap | rho(s0) |")
    sep = "|" + "---|" * 11
    lines = [
        f"# Exact root quantities, {'/'.join(sorted({get_target(n).family for n in names}))} family",
        "",
        f"`ADDITIVE_GRAMMAR` ({len(PRODS)} productions), $L = {L}$, "
        f"$\\tau = 10^{{{int(round(np.log10(tau)))}}}$, grid $x \\in [-1,1]$, "
        f"$n = 41$.",
        "",
        f"Reachable forms {census['reachable']}, terminal {census['terminal']}, "
        f"nonterminal {census['nonterminal']}. Terminals by length: "
        + ", ".join(f"$\\ell={k}$: {v}" for k, v in parity.items()) + ".",
        "",
        "Computed by exhaustive backward induction, then checked field-by-field "
        "against Table 2 of the writeup: "
        + ("**all fields agree**." if not fails
           else f"**{len(fails)} mismatch(es)**, listed below."),
        "", hdr, sep,
    ]
    for r in rows:
        lines.append(
            f"| `{r['target']}` | ${r['infix']}$ | {r['R_D1']:+.4f} | "
            f"{r['R_D2']:+.4f} | {r['R_D3']:+.4f} | {r['V_star_C']:.6f} | "
            f"{r['V_q_C']:.4f} | {r['rho_C']:.4f} | {r['margin']:+.4f} | "
            f"{'trap' if r['inversion_trap'] else '---'} | {r['rho_s0']:.4f} |")
    if fails:
        lines += ["", "## Mismatches", ""] + [f"- {f}" for f in fails]
    lines += ["", "## Reward ceiling by terminal length", "",
              "| target | " + " | ".join(f"$\\ell={k}$" for k in parity) + " |",
              "|" + "---|" * (len(parity) + 1)]
    for r in rows:
        lc = r["length_ceiling"]
        lines.append(f"| `{r['target']}` | "
                     + " | ".join(f"{lc.get(k, float('nan')):.6f}" for k in parity)
                     + " |")
    (out_dir / "07-exact-audit.md").write_text("\n".join(lines) + "\n")

    print(f"\nwrote {out_dir/'07-exact-audit.json'}")
    print(f"wrote {out_dir/'07-exact-audit.md'}")
    if problems:
        print(f"\nFAILED with {len(problems)} problem(s).")
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
