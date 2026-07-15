"""Exact-V* informativeness (the non-circular version of scripts/analysis/informativeness.py).

Estimates each intermediate state's best-achievable return by EXACT enumeration
of all reachable terminal expressions (leftmost derivation, memoized) instead of
the max over random completions. Enumeration is exponential in the token budget,
so this uses a reduced --max-len (default 11) for tractability; the grammar and
target are unchanged.

On the SAME sampled states with the SAME mean-random-return, it reports the
correlation of mean-random-return with BOTH estimators of best-achievable
return, isolating the estimator's effect:
  - proxy : max over random completions (the circular estimator)
  - exact : enumeration oracle (the clean estimator)

Run from repo root:
    python scripts/analysis/informativeness_oracle.py
    python scripts/analysis/informativeness_oracle.py --max-len 11 --problems additive_quadratic sine
"""

from sraz.utils import disable_numpy_multithreading

disable_numpy_multithreading()

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from sraz.instances.symreg.game import SymRegGame  # noqa: E402
from sraz.instances.symreg.problems import get_problem  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from informativeness import _random_complete, sample_states  # noqa: E402

sys.setrecursionlimit(100000)


def make_vstar_exact(game):
    """Return vstar(state_tuple) = exact max R^2 over reachable terminals.

    Expands the leftmost nonterminal (a leftmost derivation reaches exactly the
    terminals of length <= max_len-1, since length is non-decreasing), memoized
    on the sentential form. Correct because R^2 is a global property of the whole
    expression, so V* cannot be factored per-subtree.
    """
    g = game.grammar
    memo = {}

    def vstar(state):
        cached = memo.get(state)
        if cached is not None:
            return cached
        idx = -1
        for i, t in enumerate(state):
            if t in g.nonterms:
                idx = i
                break
        if idx == -1:  # terminal: score it
            rule = " ".join(g.tokenlist[t] for t in state)
            v = game._fit_cached(rule)
            memo[state] = v
            return v
        best = -np.inf
        for j in g.proddict[state[idx]]:
            rhs = g.productions[j]
            if len(state) + len(rhs) - 1 < game.state_len:  # matches the mask
                child = state[:idx] + tuple(rhs) + state[idx + 1:]
                cv = vstar(child)
                if cv > best:
                    best = cv
        memo[state] = best
        return best

    return vstar, memo


def _corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan"), float("nan")
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y).correlation)


def compute(problem, seed=0, max_len=11, n_states=100, n_best=80, k_roll=24):
    prob = get_problem(problem)
    game = SymRegGame(problem_seed=42, max_len=max_len, **prob.game_kwargs())
    rng = np.random.default_rng(seed)
    states = sample_states(game, rng, n_states)
    vstar, memo = make_vstar_exact(game)

    roll, v_proxy, v_exact = [], [], []
    for snap in states:
        rs = np.array([_random_complete(game, snap, rng)
                       for _ in range(n_best + k_roll)])
        st = tuple(int(t) for t in snap["state"][:snap["real_state_len"]])
        v_proxy.append(float(rs[:n_best].max()))
        roll.append(float(rs[n_best:].mean()))
        v_exact.append(float(vstar(st)))   # game state is unaffected by fits
    p_proxy = _corr(roll, v_proxy)
    p_exact = _corr(roll, v_exact)
    return {
        "problem": problem, "n_states": len(states),
        "n_enumerated_forms": len(memo),
        "corr_proxy": p_proxy, "corr_exact": p_exact,
        "mean_roll": float(np.mean(roll)),
        "mean_vstar_proxy": float(np.mean(v_proxy)),
        "mean_vstar_exact": float(np.mean(v_exact)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", nargs="+",
                    default=["additive_quadratic", "sine"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=11)
    ap.add_argument("--n-states", type=int, default=100)
    args = ap.parse_args()

    print(f"exact-V* informativeness  (max_len={args.max_len})\n")
    for name in args.problems:
        r = compute(name, seed=args.seed, max_len=args.max_len,
                    n_states=args.n_states)
        print(f"[{r['problem']}]  ({r['n_states']} states, "
              f"{r['n_enumerated_forms']} sentential forms enumerated)")
        print(f"    corr(mean_roll, V*_proxy=max-of-random) : "
              f"pearson={r['corr_proxy'][0]:+.3f}  spearman={r['corr_proxy'][1]:+.3f}   [circular]")
        print(f"    corr(mean_roll, V*_exact=enumeration)   : "
              f"pearson={r['corr_exact'][0]:+.3f}  spearman={r['corr_exact'][1]:+.3f}   [clean]")
        print(f"    mean_roll={r['mean_roll']:+.3f}  "
              f"mean_V*_proxy={r['mean_vstar_proxy']:+.3f}  "
              f"mean_V*_exact={r['mean_vstar_exact']:+.3f}\n")


if __name__ == "__main__":
    main()
