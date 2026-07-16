"""Pure-MCTS metric vs MCTS budget for the symbolic-regression grammar game.

Runs net-free classic MCTS (uniform prior + random rollouts) over K independent
seeded episodes at several n_simulations values, and plots the R^2 distribution
(mean +/- std) and best-of-K against the simulation budget -- the pure-MCTS
analogue of plot_additive_sweep.py, so net vs no-net can be compared.

Run from repo root:
    python scripts/plotting/plot_pure_mcts_sweep.py --problem additive_quadratic \
        --sims 10 25 50 100 200 400 --eval-episodes 12
"""

from sraz.utils import disable_numpy_multithreading, use_deterministic_cuda

disable_numpy_multithreading()
use_deterministic_cuda()

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sraz.instances.symreg.config import SymRegConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run"))
from eval_pure_mcts import greedy_episode  # noqa: E402


def run_sweep(problem, sim_budgets, n_episodes, base_seed, rollout_n):
    """Return {n_sim: [per-episode R^2]} for net-free MCTS at each budget."""
    results = {}
    for ns in sim_budgets:
        cfg = SymRegConfig(problem=problem, pure_mcts=True)
        cfg.agent.mcts_params["n_simulations"] = ns
        if rollout_n is not None:
            cfg.agent.mcts_params["rollout_n"] = rollout_n
        cfg.game.kwargs["problem_seed"] = base_seed
        cfg.agent.random_seeds = {"mcts": base_seed, "train": base_seed + 1,
                                  "eval": base_seed + 2,
                                  "external_policy": base_seed + 3}
        game, net, agent, _ = cfg.build()
        r2s = [greedy_episode(agent, game, seed=base_seed + k)[0]
               for k in range(n_episodes)]
        results[ns] = r2s
        print(f"  n_sim={ns:4d}: mean={np.mean(r2s):+.4f} +/- {np.std(r2s):.4f}  "
              f"best-of-{n_episodes}={np.max(r2s):+.4f}")
    return results


def plot_sweep(results, out_path, problem, n_episodes):
    sims = sorted(results)
    means = np.array([np.mean(results[ns]) for ns in sims])
    stds = np.array([np.std(results[ns]) for ns in sims])
    bestk = [np.max(results[ns]) for ns in sims]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(sims, means, yerr=stds, fmt="o-", capsize=3,
                label="mean +/- std per-episode R²")
    ax.plot(sims, bestk, "s--", label=f"best-of-{n_episodes} R²")
    ax.axhline(1.0, ls=":", c="gray", lw=1)  # reachable optimum
    ax.set_xscale("log")
    ax.set_xlabel("MCTS simulations per move (log scale)")
    ax.set_ylabel("R²")
    ax.set_title(f"Pure MCTS on {problem} — metric vs MCTS budget "
                 f"({n_episodes} episodes/budget, seed 42)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", type=str, default="additive_quadratic")
    ap.add_argument("--sims", type=int, nargs="+",
                    default=[10, 25, 50, 100, 200, 400])
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rollout-n", type=int, default=None)
    ap.add_argument("--out", type=str, default=str(
        REPO_ROOT / "Claude-research" / "figures" / "pure_mcts_sim_vs_metric.png"))
    args = ap.parse_args()
    print(f"pure MCTS sweep on {args.problem} "
          f"({args.eval_episodes} episodes/budget)")
    results = run_sweep(args.problem, args.sims, args.eval_episodes,
                        args.seed, args.rollout_n)
    plot_sweep(results, Path(args.out), args.problem, args.eval_episodes)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
