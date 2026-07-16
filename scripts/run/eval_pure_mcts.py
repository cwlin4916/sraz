"""Pure (net-free) MCTS evaluation for the symbolic-regression grammar game.

Runs classic MCTS -- uniform prior + random rollouts, via
SymRegConfig(pure_mcts=True) -- as a FIXED search procedure over K independent
seeded episodes, and reports the distribution of the terminal R^2 it finds.

Unlike run_symreg.py there is NO training loop: the policy never changes, so
"iterations" would be meaningless here. Each episode is an independent search
whose only variation is the rollout RNG (seeded per episode for
reproducibility). This is the "how much does the network add?" baseline to
compare against the AlphaZero runs from run_symreg.py.

Usage:
    python scripts/run/eval_pure_mcts.py --problem additive_quadratic --eval-episodes 20
    python scripts/run/eval_pure_mcts.py --problem sine --n-simulations 100 --rollout-n 30
"""

from sraz.utils import disable_numpy_multithreading, use_deterministic_cuda

disable_numpy_multithreading()
use_deterministic_cuda()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sraz.instances.symreg.config import SymRegConfig  # noqa: E402
from sraz.instances.symreg.game import prefix_to_infix  # noqa: E402


def greedy_episode(agent, game, seed):
    """Play one greedy MCTS episode (argmax moves, no root noise) and return
    (terminal R^2, prefix rule). The only randomness is the injected, seeded
    rollout RNG, so the episode is reproducible from `seed`.
    """
    g = game.clone()
    g.reset_wrapper(seed=seed)
    rng = np.random.default_rng(seed)
    reward, rule = 0.0, None
    for _ in range(g.state_len + 5):
        probs = agent.policy(g, "", add_noise=False,
                             temperature_override=0.01, rng=rng)
        _, reward, terminated, truncated, info = g.step_wrapper(int(np.argmax(probs)))
        if terminated or truncated:
            rule = info.get("rule")
            break
    return float(reward), rule


def _plot(rows, path, problem):
    ks = [r["episode"] + 1 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [r["r2"] for r in rows], "o", alpha=0.5, label="per-episode R²")
    ax.plot(ks, [r["best_so_far"] for r in rows], "-", label="best-so-far R²")
    ax.axhline(1.0, ls=":", c="gray", lw=1)  # reachable optimum
    ax.set_xlabel("search episode")
    ax.set_ylabel("R²")
    ax.set_title(f"Pure MCTS on {problem} — per-episode + best-so-far")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_pure_mcts_eval(cfg, n_episodes, base_seed, exp_dir):
    game, net, agent, _trainer = cfg.build()   # trainer is built but never used
    n_sims = agent.mcts_params["n_simulations"]
    rollout_n = agent.mcts_params.get("rollout_n")
    print(f"\n{'='*72}")
    print(f"  Pure MCTS (net-free) on '{cfg.problem}'   target: {game.target_infix}")
    print(f"  n_simulations={n_sims}  rollout_n={rollout_n}  episodes={n_episodes}")
    print(f"  action_space=Discrete({int(game.action_space.n)})  base_seed={base_seed}")
    print(f"{'='*72}\n")

    rows = []
    best_r2, best_rule = -np.inf, None
    for k in range(n_episodes):
        t0 = time.time()
        r2, rule = greedy_episode(agent, game, seed=base_seed + k)
        if r2 > best_r2:
            best_r2, best_rule = r2, rule
        infix = prefix_to_infix(rule.split()) if rule else None
        rows.append({"episode": k, "seed": base_seed + k, "r2": r2,
                     "rule_prefix": rule, "rule_infix": infix,
                     "best_so_far": best_r2,
                     "wall_clock_s": round(time.time() - t0, 2)})
        print(f"[EP {k+1:>3}/{n_episodes}] R2={r2:+.4f}  best={best_r2:+.4f}  expr={infix}")

    r2s = np.array([r["r2"] for r in rows])
    summary = {
        "problem": cfg.problem, "n_simulations": n_sims, "rollout_n": rollout_n,
        "n_episodes": n_episodes, "base_seed": base_seed,
        "mean_r2": float(r2s.mean()), "std_r2": float(r2s.std()),
        "min_r2": float(r2s.min()), "max_r2": float(r2s.max()),
        "best_of_k_r2": float(best_r2),
        "best_expression_prefix": best_rule,
        "best_expression_infix": (prefix_to_infix(best_rule.split())
                                  if best_rule else None),
    }

    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "pure_mcts_eval.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(exp_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _plot(rows, exp_dir / "pure_mcts_eval.png", cfg.problem)

    print(f"\nmean R2 = {summary['mean_r2']:+.4f} ± {summary['std_r2']:.4f}"
          f"  (min {summary['min_r2']:+.4f}, max {summary['max_r2']:+.4f})")
    print(f"best-of-{n_episodes} R2 = {best_r2:+.4f}   "
          f"expr: {summary['best_expression_infix']}")
    print(f"results: {exp_dir}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure (net-free) MCTS evaluation for symbolic regression")
    p.add_argument("--problem", type=str, default="additive_quadratic",
                   help="Named SR instance (default: additive_quadratic).")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="Independent search episodes to run (default 20).")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; episode k uses seed+k.")
    p.add_argument("--n-simulations", type=int, default=None,
                   help="MCTS simulations per move (default: config's 25).")
    p.add_argument("--rollout-n", type=int, default=None,
                   help="Random rollouts per leaf (default: 20).")
    p.add_argument("--max-len", type=int, default=None,
                   help="Token-buffer length (default 15).")
    p.add_argument("--out", type=str, default=None,
                   help="Output root (default: experiments/symreg_puremcts).")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()

    cfg = SymRegConfig(problem=args.problem, pure_mcts=True)
    if args.n_simulations is not None:
        cfg.agent.mcts_params["n_simulations"] = args.n_simulations
    if args.rollout_n is not None:
        cfg.agent.mcts_params["rollout_n"] = args.rollout_n
    if args.max_len is not None:
        cfg.game.kwargs["max_len"] = args.max_len
    cfg.game.kwargs["problem_seed"] = args.seed
    cfg.agent.random_seeds = {
        "mcts": args.seed, "train": args.seed + 1,
        "eval": args.seed + 2, "external_policy": args.seed + 3,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.out) if args.out else Path("experiments") / "symreg_puremcts"
    exp_dir = root / f"{ts}_{args.problem}_puremcts_ep{args.eval_episodes}"
    cfg.save(str(exp_dir / "config.json"))
    run_pure_mcts_eval(cfg, args.eval_episodes, args.seed, exp_dir)


if __name__ == "__main__":
    main()
