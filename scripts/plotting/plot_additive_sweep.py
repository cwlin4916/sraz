"""Learning curves for the additive_quadratic instance across MCTS sim budgets.

Runs AlphaZero on the additive_quadratic problem at several n_simulations values
(seed 42, so results are reproducible) and plots greedy R^2 and average self-play
R^2 versus training iteration. Visualizes the shallow-trap finding: 25 sims
stalls at C2*x^2 (~0.93) while 100/400 sims climb toward 1.0.
See Claude-research/03-additive-instance-results.md.

Run from repo root:
    python scripts/plotting/plot_additive_sweep.py
    python scripts/plotting/plot_additive_sweep.py --sims 25 50 100 400
Writes Claude-research/figures/additive_sim_sweep.png.
"""

from sraz.utils import disable_numpy_multithreading, use_deterministic_cuda

disable_numpy_multithreading()
use_deterministic_cuda()

import argparse  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sraz.instances.symreg.config import SymRegConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
# run_single_seed lives in the (non-package) run script; import it by path.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "run"))
from run_symreg import run_single_seed  # noqa: E402


def run_sweep(sim_budgets, seed=42, problem="additive_quadratic",
              n_iterations=10, n_games=20):
    """Return {n_sims: iteration_log} for each MCTS budget."""
    results = {}
    for ns in sim_budgets:
        cfg = SymRegConfig(problem=problem)
        cfg.run.n_iterations = n_iterations
        cfg.agent.mcts_params["n_simulations"] = ns
        cfg.trainer.n_games_per_train = n_games
        cfg.game.kwargs["problem_seed"] = seed
        cfg.agent.random_seeds = {
            "mcts": seed, "train": seed + 1, "eval": seed + 2,
            "external_policy": seed + 3,
        }
        cfg.net.kwargs["random_seed"] = seed
        with tempfile.TemporaryDirectory() as td:
            results[ns] = run_single_seed(cfg, Path(td))
    return results


def plot_curves(results, out_path, problem="additive_quadratic"):
    fig, (ax_g, ax_s) = plt.subplots(1, 2, figsize=(12, 4.5),
                                     sharex=True, sharey=True)
    for ns, log in sorted(results.items()):
        iters = [e["iteration"] for e in log]
        ax_g.plot(iters, [e["greedy_r2"] for e in log], "o-", label=f"{ns} sims")
        ax_s.plot(iters, [e["avg_train_reward"] for e in log], "s-",
                  label=f"{ns} sims")
    for ax, title in ((ax_g, "greedy R²"), (ax_s, "avg self-play R²")):
        ax.axhline(1.0, ls=":", c="gray", lw=1)  # reachable optimum
        ax.set_xlabel("iteration")
        ax.set_ylabel("R²")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(title="MCTS budget")
    fig.suptitle(f"AlphaZero on {problem} — learning curves by MCTS budget "
                 f"(seed 42)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary(results, out_path, problem="additive_quadratic"):
    """Summary view: the metric as a function of the MCTS budget (x = n_sim)."""
    sims = sorted(results)
    best_greedy = [max(e["greedy_r2"] for e in results[ns]) for ns in sims]
    final_greedy = [results[ns][-1]["greedy_r2"] for ns in sims]
    final_selfplay = [results[ns][-1]["avg_train_reward"] for ns in sims]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sims, best_greedy, "o-", label="best greedy R²")
    ax.plot(sims, final_greedy, "s--", label="final greedy R²")
    ax.plot(sims, final_selfplay, "^:", label="final avg self-play R²")
    ax.axhline(1.0, ls=":", c="gray", lw=1)  # reachable optimum
    ax.set_xscale("log")
    ax.set_xlabel("MCTS simulations per move (log scale)")
    ax.set_ylabel("R²")
    ax.set_title(f"AlphaZero on {problem} — metric vs MCTS budget (seed 42)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, nargs="+", default=[25, 100, 400])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--problem", type=str, default="additive_quadratic")
    ap.add_argument("--out", type=str, default=str(
        REPO_ROOT / "Claude-research" / "figures" / "additive_sim_sweep.png"))
    args = ap.parse_args()

    results = run_sweep(args.sims, seed=args.seed, problem=args.problem)
    plot_curves(results, Path(args.out), problem=args.problem)
    stem = Path(args.out).stem
    summary_stem = (stem.replace("_sweep", "_vs_metric")
                    if "_sweep" in stem else stem + "_vs_metric")
    summary_out = Path(args.out).with_name(summary_stem + ".png")
    plot_summary(results, summary_out, problem=args.problem)

    print("\n=== summary vs MCTS budget ===")
    print(f"  {'n_sim':>6} {'best_greedy':>12} {'final_greedy':>13} {'final_selfplay':>15}")
    for ns in sorted(results):
        log = results[ns]
        best = max(e["greedy_r2"] for e in log)
        print(f"  {ns:6d} {best:12.4f} {log[-1]['greedy_r2']:13.4f} "
              f"{log[-1]['avg_train_reward']:15.4f}")
    print(f"\ncurves : {args.out}")
    print(f"summary: {summary_out}")


if __name__ == "__main__":
    main()
