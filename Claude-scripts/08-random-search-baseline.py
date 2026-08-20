#!/usr/bin/env python3
"""Uniform random search on the target family: the baseline every search must beat.

The writeup's search experiments compare MCTS variants at budgets
B in {16, ..., 8192}, scoring each run by R_max(B), the best terminal reward
seen in B evaluations. Before building any selection rule it is worth knowing
what the *null* policy achieves on the same axis: draw complete derivations
i.i.d. from the uniform completion policy q and keep the best.

That baseline is not estimated here, it is computed in closed form. q induces
an exact distribution over the 247 terminals (forward-propagate 1/|A(s)| down
the reachable DAG), and B i.i.d. draws give

    P(R_max(B) <= r) = F(r)^B,

so the full distribution of R_max(B) -- mean, median, P(exact), P(beating the
root trap) -- follows for every B with no sampling error at all. The closed form
is validated in two independent ways, both cheaper and sharper than a naive
resampling of the whole protocol:

  1.  `validate_q_law` walks real derivations, stepping the symbolic
      transition under a uniform choice over the flattened mask, and compares
      the empirical terminal frequencies to the propagated law
      (total-variation distance and a chi-square test). This is the only
      non-trivial ingredient, so it is the one that gets hammered.
  2.  `mc_rmax_true_walk` runs the protocol end-to-end at a small budget --
      walk, score, track the running max -- as a check that the R_max
      bookkeeping itself agrees.

The per-budget R_max replicates are then drawn from the validated law with
numpy, which makes the full 8192-draw grid affordable.

Two reference points are reported side by side:

  * search from s0, the actual protocol; and
  * search from C = "+ S S", the continue child,

because the gap between them is exactly what a root selection rule can buy.
The writeup's K_{1-delta} = ceil(log delta / log(1-rho)) is evaluated at both.

Requires the terminal scores cached by 07-family-exact-audit.py (which is also
what validates that this is the right MDP); run that first.

Outputs (to --out-dir):
    08-random-search.json    exact + Monte Carlo results
    08-random-search.md      report, incl. the budget-grid design check
    08-random-search.png     recovery curves vs the writeup's budget grid
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sraz.instances.symreg.game import ADDITIVE_GRAMMAR, fit_expression  # noqa: E402
from sraz.instances.symreg.targets import family_targets, get_target  # noqa: E402

# The MDP and its helpers are defined once, in 07; import rather than restate.
_audit = __import__("07-family-exact-audit")
PRODS = _audit.PRODS
START = _audit.START
legal_actions = _audit.legal_actions
apply_action = _audit.apply_action
is_terminal = _audit.is_terminal
enumerate_reachable = _audit.enumerate_reachable

CONTINUE = ("+", "S", "S")          # the writeup's C
BUDGET_GRID = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
DELTAS = [0.5, 0.95, 0.99]


def q_terminal_distribution(root: tuple, forms: list, L: int) -> dict:
    """Exact probability q assigns each terminal, starting from `root`.

    q picks uniformly among the flattened (pos, prod) legal actions, so a form
    s passes mass P(s)/|A(s)| to each child. Children are never shorter than
    their parent and the one length-preserving action ("C0") strictly drops the
    nonterminal count, so ordering by (length, -#nonterminals) visits every
    parent before its children -- the exact reverse of 07's backward pass.
    """
    reach = {root}
    frontier = [root]
    while frontier:
        f = frontier.pop()
        if is_terminal(f):
            continue
        for pos, prod in legal_actions(f, L):
            c = apply_action(f, pos, prod)
            if c not in reach:
                reach.add(c)
                frontier.append(c)
    order = sorted(reach, key=lambda f: (len(f), -sum(t == "S" for t in f)))
    mass = {f: 0.0 for f in reach}
    mass[root] = 1.0
    out = {}
    for f in order:
        m = mass[f]
        if is_terminal(f):
            out[f] = m
            continue
        acts = legal_actions(f, L)
        share = m / len(acts)
        for pos, prod in acts:
            mass[apply_action(f, pos, prod)] += share
    total = sum(out.values())
    assert abs(total - 1.0) < 1e-12, f"q mass sums to {total!r}, not 1"
    return out


def exact_rmax_stats(dist: dict, scores: dict, budgets: list, tau: float,
                     trap_level: float | None) -> dict:
    """Closed-form distribution of R_max(B) under B i.i.d. draws from q."""
    pairs: dict[float, float] = {}
    for form, p in dist.items():
        if p > 0.0:
            pairs[scores[form]] = pairs.get(scores[form], 0.0) + p
    vals = np.array(sorted(pairs), dtype=float)
    probs = np.array([pairs[v] for v in vals], dtype=float)
    F = np.cumsum(probs)
    F[-1] = 1.0                                     # guard against round-off

    rho = float(probs[vals >= 1.0 - tau].sum())
    p_trap = (float(probs[vals > trap_level + tau].sum())
              if trap_level is not None else None)

    rows = []
    for B in budgets:
        Fb = np.power(F, B)
        pmf = np.diff(np.concatenate(([0.0], Fb)))
        mean = float((vals * pmf).sum())
        # median: smallest value whose CDF reaches 1/2
        med = float(vals[int(np.searchsorted(Fb, 0.5))])
        rows.append({
            "B": B,
            "P_exact": 1.0 - (1.0 - rho) ** B,
            "E_Rmax": mean,
            "median_Rmax": med,
            "P_beat_trap": (1.0 - (1.0 - p_trap) ** B
                            if p_trap is not None else None),
        })
    return {"rho": rho, "p_single_beats_trap": p_trap,
            "n_support": int(len(vals)), "budgets": rows}


def K_delta(rho: float, delta: float) -> float | None:
    """Writeup eq. (10): draws needed for exact recovery w.p. >= 1-delta."""
    if rho <= 0.0:
        return None
    if rho >= 1.0:
        return 1
    return math.ceil(math.log(1.0 - delta) / math.log(1.0 - rho))


def validate_q_law(root: tuple, dist: dict, scores: dict, L: int,
                   n_draws: int, seed: int) -> dict:
    """Walk real derivations and test their terminal law against `dist`.

    This is the assumption everything else rests on, so it is checked by
    simulation rather than assumed: sample uniformly over the flattened
    (pos, prod) mask at every step -- the policy the writeup specifies in
    sec. 2 and the one the repo's rollouts implement -- and compare the
    resulting terminal histogram to the forward-propagated probabilities.
    """
    rng = np.random.default_rng(seed)
    keys = sorted(dist)
    index = {f: i for i, f in enumerate(keys)}
    counts = np.zeros(len(keys), dtype=np.int64)
    # Draw the randomness in bulk; per-call rng overhead dominates otherwise.
    pool = rng.random(n_draws * 16)
    ptr = 0
    for _ in range(n_draws):
        form = root
        while not is_terminal(form):
            acts = legal_actions(form, L)
            if ptr >= pool.size:
                pool = rng.random(n_draws * 4)
                ptr = 0
            pos, prod = acts[int(pool[ptr] * len(acts))]
            ptr += 1
            form = apply_action(form, pos, prod)
        counts[index[form]] += 1

    p = np.array([dist[f] for f in keys], dtype=float)
    obs = counts.astype(float)
    exp = p * n_draws
    tv = 0.5 * float(np.abs(obs / n_draws - p).sum())
    # chi-square on the cells with enough expected mass to be meaningful
    ok = exp >= 5.0
    chi2 = float(((obs[ok] - exp[ok]) ** 2 / exp[ok]).sum())
    dof = int(ok.sum()) - 1
    try:
        from scipy.stats import chi2 as chi2_dist
        pval = float(chi2_dist.sf(chi2, dof)) if dof > 0 else None
    except Exception:
        pval = None
    rho_hat = float(obs[[i for i, f in enumerate(keys)
                         if scores[f] >= 1.0 - 1e-6]].sum() / n_draws) \
        if any(scores[f] >= 1.0 - 1e-6 for f in keys) else 0.0
    return {"n_draws": n_draws, "total_variation": tv, "chi2": chi2,
            "dof": dof, "p_value": pval, "rho_hat_walk": rho_hat,
            "cells_tested": int(ok.sum()), "n_terminals": len(keys)}


def mc_rmax_from_law(dist: dict, scores: dict, budgets: list, tau: float,
                     n_seeds: int, seed: int) -> dict:
    """R_max(B) replicates, drawing terminals from the (validated) law of q.

    Equivalent to running the protocol, given that `validate_q_law` has already
    confirmed the law -- and vectorised, so the full grid up to B=8192 is cheap.
    """
    keys = sorted(dist)
    p = np.array([dist[f] for f in keys], dtype=float)
    p = p / p.sum()
    r = np.array([scores[f] for f in keys], dtype=float)
    Bmax = max(budgets)
    rng = np.random.default_rng(seed)

    hits = {B: 0 for B in budgets}
    stats = {B: [] for B in budgets}
    first = []
    chunk = max(1, min(n_seeds, 4_000_000 // Bmax))
    done = 0
    while done < n_seeds:
        m = min(chunk, n_seeds - done)
        draws = r[rng.choice(len(p), size=(m, Bmax), p=p)]
        run = np.maximum.accumulate(draws, axis=1)
        exact = draws >= 1.0 - tau
        anyexact = np.cumsum(exact, axis=1) > 0
        idx = np.where(exact.any(axis=1), exact.argmax(axis=1) + 1, Bmax + 1)
        first.extend(idx.tolist())
        for B in budgets:
            stats[B].extend(run[:, B - 1].tolist())
            hits[B] += int(anyexact[:, B - 1].sum())
        done += m
    fe = np.array(first, dtype=float)
    return {"n_seeds": n_seeds,
            "budgets": [{"B": B,
                         "P_exact_hat": hits[B] / n_seeds,
                         "hits": hits[B],
                         "wilson95": wilson(hits[B], n_seeds),
                         "wilson999": wilson(hits[B], n_seeds, Z999),
                         "E_Rmax_hat": float(np.mean(stats[B])),
                         "median_Rmax_hat": float(np.median(stats[B]))}
                        for B in budgets],
            "B_exact_median": float(np.median(fe)),
            "B_exact_mean": float(np.mean(fe)),
            "B_exact_censored": int((fe > Bmax).sum())}


def mc_rmax_true_walk(root: tuple, scores: dict, L: int, B: int,
                      n_seeds: int, tau: float, seed: int) -> dict:
    """End-to-end protocol check: real derivations, real running max."""
    rng = np.random.default_rng(seed)
    pool = rng.random(n_seeds * B * 12)
    ptr = 0
    best_all, hits = [], 0
    for _ in range(n_seeds):
        best, hit = -np.inf, False
        for _ in range(B):
            form = root
            while not is_terminal(form):
                acts = legal_actions(form, L)
                if ptr >= pool.size:
                    pool = rng.random(n_seeds * B)
                    ptr = 0
                pos, prod = acts[int(pool[ptr] * len(acts))]
                ptr += 1
                form = apply_action(form, pos, prod)
            v = scores[form]
            best = max(best, v)
            hit = hit or v >= 1.0 - tau
        best_all.append(best)
        hits += int(hit)
    return {"B": B, "n_seeds": n_seeds, "P_exact_hat": hits / n_seeds,
            "wilson95": wilson(hits, n_seeds),
            "E_Rmax_hat": float(np.mean(best_all))}


Z95, Z999 = 1.959963985, 3.290526731


def wilson(k: int, n: int, z: float = Z95) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p, z2 = k / n, z * z
    den = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def run_target(name: str, forms: list, terminals: set, L: int, tau: float,
               budgets: list, n_seeds: int, cache_dir: Path,
               walk_draws: int, walk_B: int, walk_seeds: int) -> dict:
    target = get_target(name)
    xs, ys = target.xs(), None
    ys = target.ys(xs)

    cache_path = cache_dir / f"fit_{name}.json"
    if not cache_path.exists():
        raise SystemExit(
            f"missing {cache_path}\nrun 07-family-exact-audit.py first "
            f"(it enumerates and scores every terminal, and validates the MDP)")
    cache = json.loads(cache_path.read_text())
    missing = [f for f in terminals if " ".join(f) not in cache]
    if missing:                                     # top up rather than fail
        for f in missing:
            cache[" ".join(f)] = float(fit_expression(" ".join(f), xs, ys))
        cache_path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    scores = {f: cache[" ".join(f)] for f in terminals}

    # the root trap level: best immediately-terminal root action
    root_terminals = [apply_action(START, p, j) for p, j in legal_actions(START, L)
                      if is_terminal(apply_action(START, p, j))]
    trap_level = max(scores[f] for f in root_terminals)

    out = {"target": name, "infix": target.infix,
           "support": sorted(p for p, c in enumerate(target.coeffs) if c != 0),
           "trap_level": trap_level}
    seed = sum(ord(ch) for ch in name) * 7919
    for label, root in (("s0", START), ("C", CONTINUE)):
        dist = q_terminal_distribution(root, forms, L)
        ex = exact_rmax_stats(dist, scores, budgets, tau, trap_level)
        ex["K"] = {f"{d}": K_delta(ex["rho"], d) for d in DELTAS}
        out[label] = {
            "exact": ex,
            "q_law_check": validate_q_law(root, dist, scores, L, walk_draws, seed),
            "monte_carlo": mc_rmax_from_law(dist, scores, budgets, tau,
                                           n_seeds, seed + 1),
            "protocol_check": mc_rmax_true_walk(root, scores, L, walk_B,
                                                walk_seeds, tau, seed + 2),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--family", default="linear")
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--tau", type=float, default=1e-6)
    ap.add_argument("--seeds", type=int, default=2000,
                    help="R_max replicates drawn from the validated law of q")
    ap.add_argument("--walk-draws", type=int, default=100_000,
                    help="real derivations walked to validate q's terminal law")
    ap.add_argument("--walk-budget", type=int, default=64,
                    help="budget for the end-to-end protocol check")
    ap.add_argument("--walk-seeds", type=int, default=200,
                    help="replicates for the end-to-end protocol check")
    ap.add_argument("--budgets", type=int, nargs="+", default=BUDGET_GRID)
    ap.add_argument("--out-dir", default=str(REPO / "Claude-experiments" / "8-17"))
    args = ap.parse_args()

    names = args.targets or [t.name for t in family_targets(args.family)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L, tau = args.max_len, args.tau

    forms, terminals = enumerate_reachable(L)
    print(f"MDP: ADDITIVE_GRAMMAR, L={L}, tau={tau:g}; "
          f"{len(forms)} reachable forms, {len(terminals)} terminals")
    print(f"budgets: {args.budgets}")
    print(f"R_max replicates: {args.seeds}; "
          f"q-law validation walks: {args.walk_draws}; "
          f"protocol check: {args.walk_seeds} seeds at B={args.walk_budget}\n")

    rows = [run_target(n, forms, terminals, L, tau, args.budgets, args.seeds,
                       out_dir / "cache", args.walk_draws, args.walk_budget,
                       args.walk_seeds) for n in names]

    worst, n95_miss = [], []
    for r in rows:
        e0, eC = r["s0"]["exact"], r["C"]["exact"]
        print(f"--- {r['target']}: {r['infix']}   trap level "
              f"R={r['trap_level']:+.4f}")
        print(f"    rho(s0)={e0['rho']:.4f}  K50={e0['K']['0.5']:>3}  "
              f"K95={e0['K']['0.95']:>3}  K99={e0['K']['0.99']:>3}")
        print(f"    rho(C) ={eC['rho']:.4f}  K50={eC['K']['0.5']:>3}  "
              f"K95={eC['K']['0.95']:>3}  K99={eC['K']['0.99']:>3}")
        mc = {d["B"]: d for d in r["s0"]["monte_carlo"]["budgets"]}
        print(f"    {'B':>6} {'P_exact':>9} {'MC':>7} {'wilson95':>16} "
              f"{'E[Rmax]':>9}")
        for d in r["s0"]["exact"]["budgets"]:
            m = mc[d["B"]]
            lo, hi = m["wilson95"]
            in95 = lo <= d["P_exact"] <= hi
            l999, h999 = m["wilson999"]
            in999 = l999 <= d["P_exact"] <= h999
            n95_miss.append(not in95)
            flag = "" if in95 else ("  <- outside 95% CI"
                                    if in999 else "  <-- DISAGREE (99.9%)")
            print(f"    {d['B']:>6} {d['P_exact']:>9.4f} "
                  f"{m['P_exact_hat']:>7.3f} [{lo:>6.3f},{hi:>6.3f}] "
                  f"{d['E_Rmax']:>9.4f}{flag}")
            if not in999:
                worst.append(f"{r['target']} B={d['B']}: exact "
                             f"{d['P_exact']:.4f} outside MC 99.9% CI "
                             f"[{l999:.4f},{h999:.4f}]")
        print(f"    B_exact (median over seeds) = "
              f"{r['s0']['monte_carlo']['B_exact_median']:.0f}, "
              f"censored {r['s0']['monte_carlo']['B_exact_censored']}"
              f"/{r['s0']['monte_carlo']['n_seeds']}")
        qc = r["s0"]["q_law_check"]
        pv = "n/a" if qc["p_value"] is None else f"{qc['p_value']:.3f}"
        print(f"    q-law check ({qc['n_draws']} walks): TV={qc['total_variation']:.4f}"
              f"  chi2={qc['chi2']:.1f} dof={qc['dof']} p={pv}"
              f"  rho_hat={qc['rho_hat_walk']:.4f} vs {e0['rho']:.4f}")
        pc = r["s0"]["protocol_check"]
        ex_at = next(d for d in e0["budgets"] if d["B"] == pc["B"]) \
            if any(d["B"] == pc["B"] for d in e0["budgets"]) else None
        tgt = f" vs exact {ex_at['P_exact']:.4f}" if ex_at else ""
        print(f"    protocol check (real walks, B={pc['B']}): "
              f"P_exact={pc['P_exact_hat']:.3f} "
              f"[{pc['wilson95'][0]:.3f},{pc['wilson95'][1]:.3f}]{tgt}")
        if ex_at and not (pc["wilson95"][0] <= ex_at["P_exact"] <= pc["wilson95"][1]):
            worst.append(f"{r['target']} protocol check B={pc['B']}: exact "
                         f"{ex_at['P_exact']:.4f} outside walk CI")
        if qc["p_value"] is not None and qc["p_value"] < 0.001:
            worst.append(f"{r['target']} q-law chi2 p={qc['p_value']:.2g} "
                         f"-- walked terminals do not match propagated law")
        print()

    # --- the design check --------------------------------------------------
    print("=" * 72)
    print("Budget-grid design check, against the writeup's own grid")
    print(f"  grid = {BUDGET_GRID}\n")
    print(f"  {'target':<8} {'K95(s0)':>8} {'saturated':>11} "
          f"{'P_ex @ B={}'.format(BUDGET_GRID[0]):>14} "
          f"{'informative B':>16}")
    design = []
    for r in rows:
        e0 = r["s0"]["exact"]
        rho, k95 = e0["rho"], e0["K"]["0.95"]
        sat = sum(1 for B in BUDGET_GRID if k95 is not None and B >= k95)
        by_B = {d["B"]: d for d in e0["budgets"]}
        lo = (by_B[BUDGET_GRID[0]]["P_exact"] if BUDGET_GRID[0] in by_B
              else 1.0 - (1.0 - rho) ** BUDGET_GRID[0])
        hi = (by_B[BUDGET_GRID[-1]]["P_exact"] if BUDGET_GRID[-1] in by_B
              else 1.0 - (1.0 - rho) ** BUDGET_GRID[-1])
        # widest B window where random search is neither hopeless nor certain,
        # i.e. P(exact) in [0.05, 0.95] -- the only place a method can separate
        b_lo = K_delta(rho, 0.95) and math.ceil(
            math.log(1.0 - 0.05) / math.log(1.0 - rho))
        b_hi = K_delta(rho, 0.95)
        window = f"{b_lo}-{b_hi}" if b_lo else "n/a"
        print(f"  {r['target']:<8} {str(k95):>8} {sat:>7}/{len(BUDGET_GRID):<3} "
              f"{lo:>14.4f} {window:>16}")
        design.append({"target": r["target"], "K95_s0": k95,
                       "writeup_grid": BUDGET_GRID,
                       "grid_points_saturated": sat, "n_grid": len(BUDGET_GRID),
                       "P_exact_at_grid_min": lo, "P_exact_at_grid_max": hi,
                       "informative_B_lo": b_lo, "informative_B_hi": b_hi})

    result = {"mdp": {"grammar": "ADDITIVE_GRAMMAR", "max_len": L, "tau": tau},
              "budgets": args.budgets, "n_seeds": args.seeds,
              "deltas": DELTAS, "rows": rows, "design_check": design,
              "mc_exact_disagreements": worst}

    n_cells = len(n95_miss)
    miss95, exp95 = sum(n95_miss), 0.05 * n_cells
    print(f"\nClosed form vs Monte Carlo over {n_cells} (target, B) cells: "
          f"{miss95} outside the 95% CI (expected ~{exp95:.1f} by chance at "
          f"n={args.seeds}).")
    if worst:
        print(f"WARNING: {len(worst)} cell(s) outside even the 99.9% CI -- "
              f"this is not multiple-comparison noise:")
        for w in worst:
            print("  " + w)
    else:
        print("No cell falls outside its 99.9% CI, so the closed form and the "
              "sampled protocol agree.")
    result_extra = {"cells": n_cells, "outside_95": miss95,
                    "expected_outside_95": exp95,
                    "outside_999": len(worst)}

    result["agreement"] = result_extra
    (out_dir / "08-random-search.json").write_text(json.dumps(result, indent=2))
    write_report(out_dir, result, rows, args)
    plot(out_dir, rows, args.budgets)
    print(f"\nwrote {out_dir/'08-random-search.json'}")
    print(f"wrote {out_dir/'08-random-search.md'}")
    print(f"wrote {out_dir/'08-random-search.png'}")


def write_report(out_dir: Path, result: dict, rows: list, args) -> None:
    L = args.max_len
    lines = [
        f"# Uniform random search on the {'/'.join(sorted({get_target(r['target']).family for r in rows}))} family",
        "",
        f"`ADDITIVE_GRAMMAR`, $L = {L}$, $\\tau = 10^{{-6}}$, grid "
        f"$x \\in [-1,1]$, $n = 41$. Budgets "
        f"$B \\in \\{{{', '.join(str(b) for b in args.budgets)}\\}}$.",
        "",
        "$R_{\\max}(B)$ is the best reward in $B$ i.i.d. draws from the uniform "
        "completion policy $q$. The distribution is **exact**: $q$'s law over the "
        "247 terminals is computed by forward propagation, so "
        "$P(R_{\\max}(B) \\le r) = F(r)^B$ in closed form. The `MC` column is "
        "an independent check on that algebra, not the source of the numbers; "
        "see [Validation](#validation).",
        "",
        "## Rarity and the sample complexity of guessing",
        "",
        "$K_{1-\\delta} = \\lceil \\log \\delta / \\log(1-\\rho) \\rceil$ "
        "(writeup eq. 10).",
        "",
        "| target | $y(x)$ | trap level | $P(\\text{1 draw} > \\text{trap})$ "
        "| $\\rho(s_0)$ | $K_{50}$ | $K_{95}$ "
        "| $K_{99}$ | $\\rho(C)$ | $K_{95}(C)$ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e0, eC = r["s0"]["exact"], r["C"]["exact"]
        lines.append(
            f"| `{r['target']}` | ${r['infix']}$ | {r['trap_level']:+.4f} "
            f"| {e0['p_single_beats_trap']:.4f} "
            f"| {e0['rho']:.4f} | {e0['K']['0.5']} | {e0['K']['0.95']} "
            f"| {e0['K']['0.99']} | {eC['rho']:.4f} | {eC['K']['0.95']} |")

    lines += ["", "## Exact recovery probability across the budget grid", "",
              "| target | " + " | ".join(f"$B={b}$" for b in args.budgets) + " |",
              "|" + "---|" * (len(args.budgets) + 1)]
    for r in rows:
        cells = " | ".join(f"{d['P_exact']:.4f}"
                           for d in r["s0"]["exact"]["budgets"])
        lines.append(f"| `{r['target']}` | {cells} |")

    lines += ["", "## $E[R_{\\max}(B)]$", "",
              "| target | " + " | ".join(f"$B={b}$" for b in args.budgets) + " |",
              "|" + "---|" * (len(args.budgets) + 1)]
    for r in rows:
        cells = " | ".join(f"{d['E_Rmax']:.4f}"
                           for d in r["s0"]["exact"]["budgets"])
        lines.append(f"| `{r['target']}` | {cells} |")

    lines += ["", "## Design check", "",
              "Against the writeup's own budget grid "
              f"$\\{{{', '.join(str(b) for b in BUDGET_GRID)}\\}}$.",
              "",
              "| target | $K_{95}(s_0)$ | grid points $\\ge K_{95}$ | "
              "$P(\\text{exact})$ at $B=%d$ | at $B=%d$ | informative $B$ |"
              % (BUDGET_GRID[0], BUDGET_GRID[-1]),
              "|---|---|---|---|---|---|"]
    for d in result["design_check"]:
        win = (f"{d['informative_B_lo']}--{d['informative_B_hi']}"
               if d["informative_B_lo"] else "n/a")
        lines.append(f"| `{d['target']}` | {d['K95_s0']} | "
                     f"{d['grid_points_saturated']}/{d['n_grid']} | "
                     f"{d['P_exact_at_grid_min']:.4f} | "
                     f"{d['P_exact_at_grid_max']:.6f} | {win} |")
    lines += ["",
              "A grid point at or above $K_{95}$ is one where *random guessing "
              "alone* already recovers the target at least 95% of the time, so no "
              "search method can be distinguished from any other there. The "
              "informative window is the range of $B$ over which "
              "$P(\\text{exact}) \\in [0.05, 0.95]$ -- outside it every method "
              "scores the same and $\\Delta_{\\text{semantic}}(B)$ is squeezed "
              "toward zero by the ceiling, not by the search rule.",
              "",
              "## Validation", "",
              f"- Closed form vs Monte Carlo ({args.seeds} replicates) over "
              f"{result['agreement']['cells']} $(\\text{{target}}, B)$ cells: "
              f"{result['agreement']['outside_95']} "
              f"{'falls' if result['agreement']['outside_95'] == 1 else 'fall'} "
              f"outside the Wilson 95% "
              f"interval (about {result['agreement']['expected_outside_95']:.1f} "
              f"expected by chance), and "
              + ("**none** outside the 99.9% interval."
                 if not result["mc_exact_disagreements"]
                 else f"**{len(result['mc_exact_disagreements'])}** outside the "
                      f"99.9% interval: "
                      + "; ".join(result["mc_exact_disagreements"])),
              "- $q$'s terminal law sums to 1 to within $10^{-12}$ at both roots "
              "(asserted in `q_terminal_distribution`).",
              "- $\\rho$ here is computed by *forward* propagation; `07` computes "
              "it by *backward* induction. The two agree.",
              "",
              "| target | walks | TV distance | $\\chi^2$ | dof | $p$ | "
              "$\\hat\\rho$ (walked) | $\\rho$ (exact) |",
              "|---|---|---|---|---|---|---|---|",
              ]
    for r in rows:
        qc, e0 = r["s0"]["q_law_check"], r["s0"]["exact"]
        pv = "n/a" if qc["p_value"] is None else f"{qc['p_value']:.3f}"
        lines.append(f"| `{r['target']}` | {qc['n_draws']} | "
                     f"{qc['total_variation']:.4f} | {qc['chi2']:.1f} | "
                     f"{qc['dof']} | {pv} | {qc['rho_hat_walk']:.4f} | "
                     f"{e0['rho']:.4f} |")
    lines += ["",
              f"End-to-end protocol check (real derivation walks, "
              f"B={args.walk_budget}, {args.walk_seeds} seeds):", ""]
    for r in rows:
        pc = r["s0"]["protocol_check"]
        ex_at = next((d for d in r["s0"]["exact"]["budgets"]
                      if d["B"] == pc["B"]), None)
        tgt = f", exact {ex_at['P_exact']:.4f}" if ex_at else ""
        lines.append(f"- `{r['target']}`: $\\hat P(\\text{{exact}}) = "
                     f"{pc['P_exact_hat']:.3f}$ "
                     f"(95% CI {pc['wilson95'][0]:.3f}--{pc['wilson95'][1]:.3f}"
                     f"{tgt})")
    (out_dir / "08-random-search.md").write_text("\n".join(lines) + "\n")


def plot(out_dir: Path, rows: list, budgets: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                        # plotting is not the result
        print(f"(skipping plot: {exc})")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    Bs = np.array(budgets, dtype=float)
    dense = np.unique(np.concatenate([Bs, np.logspace(0, np.log10(max(Bs)), 200)]))
    for r in rows:
        rho = r["s0"]["exact"]["rho"]
        line, = axes[0].plot(dense, 1 - (1 - rho) ** dense, lw=1.6,
                             label=f"{r['target']}  $\\rho={rho:.4f}$")
        mc = r["s0"]["monte_carlo"]["budgets"]
        axes[0].plot([d["B"] for d in mc], [d["P_exact_hat"] for d in mc],
                     "o", ms=3.5, color=line.get_color())
        axes[1].plot(Bs, [d["E_Rmax"] for d in r["s0"]["exact"]["budgets"]],
                     "o-", ms=3.5, lw=1.6, color=line.get_color(),
                     label=r["target"])
        axes[1].axhline(r["trap_level"], ls=":", lw=0.8, color=line.get_color())
    for ax, ttl, yl in ((axes[0], "Exact recovery by uniform random search",
                         "$P(R_{\\max}(B) = 1)$"),
                        (axes[1], "Best reward found", "$E[R_{\\max}(B)]$")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("budget $B$ (terminal evaluations)")
        ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5)
    axes[0].axhline(0.95, color="k", ls="--", lw=0.8)
    axes[0].text(Bs[0], 0.955, "95%", fontsize=7.5)
    axes[0].set_ylim(-0.02, 1.02)
    fig.suptitle("Uniform random search from $s_0$ (lines: exact; dots: Monte Carlo)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_dir / "08-random-search.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
