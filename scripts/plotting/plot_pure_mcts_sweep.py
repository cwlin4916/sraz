"""Pure-MCTS metric vs MCTS budget for the symbolic-regression grammar game.

Runs net-free classic MCTS (uniform prior + random rollouts) over K independent
seeded episodes at several n_simulations values, and plots the R^2 distribution
(mean +/- std) and best-of-K against the simulation budget -- the pure-MCTS
analogue of plot_additive_sweep.py, so net vs no-net can be compared.

Every episode's terminal expression is recorded alongside its R^2 and written to
a JSON sidecar next to the figure. The scores here are strongly quantized -- the
grammar has few distinct terminals, so a "mean" is a mixture over a handful of
expressions rather than a spread around a centre -- and the sidecar is what lets
a plateau be read back as a named expression instead of a number. It also feeds
--from-raw, which replots without re-running the sweep.

Run from repo root:
    python scripts/plotting/plot_pure_mcts_sweep.py --problem additive_quadratic \
        --sims 10 25 50 100 200 400 --eval-episodes 12
    python scripts/plotting/plot_pure_mcts_sweep.py --from-raw \
        Claude-research/figures/pure_mcts_sim_vs_metric.json
"""

from sraz.utils import disable_numpy_multithreading, use_deterministic_cuda

disable_numpy_multithreading()
use_deterministic_cuda()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullLocator, ScalarFormatter  # noqa: E402

from sraz.instances.symreg.config import SymRegConfig  # noqa: E402
from sraz.instances.symreg.game import prefix_to_infix  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run"))
from eval_pure_mcts import greedy_episode  # noqa: E402


def run_sweep(problem, sim_budgets, n_episodes, base_seed, rollout_n):
    """Return {n_sim: {"r2s": [...], "exprs": [...]}} for net-free MCTS."""
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
        episodes = [greedy_episode(agent, game, seed=base_seed + k)
                    for k in range(n_episodes)]
        r2s = [r2 for r2, _ in episodes]
        exprs = [prefix_to_infix(rule.split()) if rule else None
                 for _, rule in episodes]
        results[ns] = {"r2s": r2s, "exprs": exprs}
        # The expression mix, not just the moments: a mean below a plateau means
        # some episodes terminated on a strictly worse expression, and only the
        # counts say which.
        mix = ", ".join(f"{n}x {e}" for e, n in Counter(exprs).most_common())
        print(f"  n_sim={ns:4d}: mean={np.mean(r2s):+.4f} +/- {np.std(r2s):.4f}  "
              f"best-of-{n_episodes}={np.max(r2s):+.4f}")
        print(f"             found: {mix}")
    return results


def save_raw(results, out_path, problem, n_episodes, seed, rollout_n):
    """Persist every episode's score and expression beside the figure."""
    payload = {
        "problem": problem, "seed": seed, "n_episodes": n_episodes,
        "rollout_n": rollout_n,
        "budgets": [
            {"n_simulations": ns,
             "r2s": results[ns]["r2s"],
             "expressions": results[ns]["exprs"],
             "mean_r2": float(np.mean(results[ns]["r2s"])),
             "std_r2": float(np.std(results[ns]["r2s"])),
             "best_r2": float(np.max(results[ns]["r2s"]))}
            for ns in sorted(results)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_raw(path):
    """Rebuild (results, problem, n_episodes, seed) from a sidecar written above."""
    with open(path) as f:
        payload = json.load(f)
    results = {b["n_simulations"]: {"r2s": b["r2s"], "exprs": b["expressions"]}
               for b in payload["budgets"]}
    return results, payload["problem"], payload["n_episodes"], payload["seed"]


def plot_sweep(results, out_path, problem, n_episodes, seed):
    sims = sorted(results)
    means = np.array([np.mean(results[ns]["r2s"]) for ns in sims])
    stds = np.array([np.std(results[ns]["r2s"]) for ns in sims])
    bestk = [np.max(results[ns]["r2s"]) for ns in sims]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(sims, means, yerr=stds, fmt="o-", capsize=3,
                label="mean +/- std per-episode R²")
    ax.plot(sims, bestk, "s--", label=f"best-of-{n_episodes} R²")
    ax.axhline(1.0, ls=":", c="gray", lw=1)  # reachable optimum
    ax.set_xscale("log")
    # Tick every budget actually sampled: the decade ticks a log axis defaults to
    # label neither the dense low-end cluster nor the top of the range, and which
    # budgets were run is the point of the figure. Rotated upright because 10..15
    # occupy under a tenth of a log axis reaching 800, and horizontal labels there
    # would overprint. The minor decade ticks go: they mark budgets never run.
    ax.set_xticks(sims)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.set_xlabel("MCTS simulations per move (log scale)")
    ax.set_ylabel("R²")
    # Two lines: the one-line form overruns the canvas and loses "seed 42".
    ax.set_title(f"Pure MCTS on {problem} — metric vs MCTS budget\n"
                 f"({n_episodes} episodes/budget, seed {seed})", fontsize=11)
    ax.grid(alpha=0.3, which="major")
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
    ap.add_argument("--save-raw", type=str, default=None,
                    help="JSON sidecar for the per-episode scores and "
                         "expressions (default: <out> with a .json suffix).")
    ap.add_argument("--from-raw", type=str, default=None, metavar="JSON",
                    help="Replot from an existing sidecar instead of re-running "
                         "the sweep. --problem/--sims/--eval-episodes/--seed are "
                         "then taken from the file, so the title cannot drift "
                         "from the data it describes.")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.from_raw:
        results, problem, n_episodes, seed = load_raw(args.from_raw)
        print(f"replotting {args.from_raw} ({len(results)} budgets, "
              f"{n_episodes} episodes/budget)")
        plot_sweep(results, out_path, problem, n_episodes, seed)
        print(f"figure: {out_path}")
        return

    print(f"pure MCTS sweep on {args.problem} "
          f"({args.eval_episodes} episodes/budget)")
    results = run_sweep(args.problem, args.sims, args.eval_episodes,
                        args.seed, args.rollout_n)
    plot_sweep(results, out_path, args.problem, args.eval_episodes, args.seed)
    raw_path = Path(args.save_raw) if args.save_raw else out_path.with_suffix(".json")
    save_raw(results, raw_path, args.problem, args.eval_episodes, args.seed,
             args.rollout_n)
    print(f"figure: {out_path}")
    print(f"raw:    {raw_path}")


if __name__ == "__main__":
    main()
