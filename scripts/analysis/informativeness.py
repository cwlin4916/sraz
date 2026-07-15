"""Policy-free informativeness metric (Claude-research/02).

For a set of intermediate (non-terminal) partial derivations, correlate each
state's best-achievable return (estimated by the MAX over many random
completions) with its MEAN random-completion return. High correlation => the
value of a half-built expression is readable under weak (random) play, i.e. the
instance is 'informative'. Also reports the fraction of random completions that
hit the -1 failure floor, which is itself an (un)informativeness signal.

Run from repo root:
    python scripts/analysis/informativeness.py
    python scripts/analysis/informativeness.py --problems additive_quadratic sine
"""

from sraz.utils import disable_numpy_multithreading

disable_numpy_multithreading()

import argparse  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from sraz.instances.symreg.game import SymRegGame  # noqa: E402
from sraz.instances.symreg.problems import get_problem  # noqa: E402


def _random_complete(game, snap, rng):
    """From partial state `snap`, play uniform-random legal actions to a
    terminal; return the terminal reward."""
    game.unstash_state(snap)
    for _ in range(game.state_len + 5):
        if game.terminated or game.truncated:
            break
        valid = np.flatnonzero(game.get_action_mask().ravel())
        if len(valid) == 0:
            break
        game.step_wrapper(int(valid[rng.integers(len(valid))]))
    return float(game.reward) if game.reward is not None else -1.0


def sample_states(game, rng, n_states, max_trajectories=4000):
    """Collect distinct non-terminal partial states via random legal play."""
    seen = {}
    trajs = 0
    while len(seen) < n_states and trajs < max_trajectories:
        trajs += 1
        game.reset_wrapper()
        for _ in range(game.state_len + 5):
            if game.terminated or game.truncated:
                break
            key = game.state[:game.real_state_len].tobytes()
            if key not in seen:
                seen[key] = game.stash_state()
            valid = np.flatnonzero(game.get_action_mask().ravel())
            if len(valid) == 0:
                break
            game.step_wrapper(int(valid[rng.integers(len(valid))]))
    return list(seen.values())


def informativeness(problem_name, seed=0, n_states=100, n_best=80, k_roll=24):
    prob = get_problem(problem_name)
    game = SymRegGame(problem_seed=42, **prob.game_kwargs())
    rng = np.random.default_rng(seed)
    states = sample_states(game, rng, n_states)

    best, roll, fail = [], [], []
    for snap in states:
        rs = np.array([_random_complete(game, snap, rng)
                       for _ in range(n_best + k_roll)])
        best.append(rs[:n_best].max())          # V*(s) estimate
        roll.append(rs[n_best:].mean())         # E[return | random completion]
        fail.append(float(np.mean(rs <= -1.0 + 1e-9)))
    best, roll = np.array(best), np.array(roll)
    # guard against zero-variance (degenerate correlation)
    pear = float(pearsonr(roll, best)[0]) if roll.std() > 1e-9 and best.std() > 1e-9 else float("nan")
    spear = float(spearmanr(roll, best).correlation) if roll.std() > 1e-9 and best.std() > 1e-9 else float("nan")
    return {
        "problem": problem_name, "n_states": len(states),
        "pearson": pear, "spearman": spear,
        "mean_rollout": float(roll.mean()), "mean_best": float(best.mean()),
        "fail_pct": 100.0 * float(np.mean(fail)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problems", nargs="+",
                    default=["additive_quadratic", "sine"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-states", type=int, default=100)
    ap.add_argument("--n-best", type=int, default=80)
    ap.add_argument("--k-roll", type=int, default=24)
    args = ap.parse_args()

    hdr = (f"{'problem':20s} {'n_st':>4s} {'pearson':>8s} {'spearman':>8s} "
           f"{'mean_roll':>9s} {'mean_best':>9s} {'fail%':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for name in args.problems:
        r = informativeness(name, seed=args.seed, n_states=args.n_states,
                            n_best=args.n_best, k_roll=args.k_roll)
        print(f"{r['problem']:20s} {r['n_states']:4d} {r['pearson']:8.3f} "
              f"{r['spearman']:8.3f} {r['mean_rollout']:9.3f} "
              f"{r['mean_best']:9.3f} {r['fail_pct']:6.1f}")


if __name__ == "__main__":
    main()
