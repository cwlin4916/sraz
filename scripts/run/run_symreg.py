"""Symbolic-regression grammar game -- AlphaZero training.

MCTS + MLP policy/value net on the grammar-derivation MDP; terminal reward is
the R² of an lmfit constant fit against the hidden target
4*sin(4*x) + C0 + C1*x + C2*x**2.

Usage:
    python scripts/run/run_symreg.py --seed 42
"""

from sraz.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sraz.instances.symreg.config import SymRegConfig
from sraz.instances.symreg.game import prefix_to_infix


def _save_final_checkpoint(ckpt_path: Path, trainer, agent, net):
    """Save network weights + replay buffer + RNG state.

    Bypasses `Trainer.save_checkpoint` because `CheckpointManager` looks for
    `agent.all_training_examples` (which actually lives on `trainer`).
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(net.model.state_dict(), ckpt_path / "network.pt")
    with open(ckpt_path / "trainer_state.pkl", "wb") as f:
        pickle.dump({
            "all_training_examples": trainer.all_training_examples,
            "rngs": agent.rngs,
            "random_seeds": agent.random_seeds,
        }, f)


# ---------------------------------------------------------------------------
# Greedy evaluation
# ---------------------------------------------------------------------------

def compute_greedy_eval(agent, n_episodes: int = 1):
    """Play greedy episodes (no noise, near-argmax temperature); return
    (avg_r2, best_r2, best_rule_prefix)."""
    r2s = []
    best_r2, best_rule = -np.inf, None
    for i in range(n_episodes):
        game = agent.game.clone()
        game.reset_wrapper(seed=1000 + i)
        reward, rule = 0.0, None
        for _ in range(game.state_len + 5):
            probs = agent.policy(
                game, "", add_noise=False, temperature_override=0.01)
            action = int(np.argmax(probs))
            _, reward, terminated, truncated, info = game.step_wrapper(action)
            if terminated or truncated:
                rule = info.get("rule")
                break
        r2s.append(reward)
        if reward > best_r2:
            best_r2, best_rule = reward, rule
    return float(np.mean(r2s)), float(best_r2), best_rule


# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

def setup_experiment_dir(cfg, seed: int, out_root: Path | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_sims = cfg.agent.mcts_params["n_simulations"]
    n_iters = cfg.run.n_iterations
    dirname = f"{timestamp}_{cfg.problem}_seed{seed}_mcts{n_sims}_iter{n_iters}"
    root = Path(out_root) if out_root is not None else Path("experiments") / "symreg"
    exp_dir = root / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.trainer.checkpoint_dir = str(exp_dir / "checkpoints")
    cfg.run.plot_path = str(exp_dir / "training_metrics.png")
    return exp_dir


def plot_reward_curve(iteration_log: list[dict], path: Path):
    iters = [e["iteration"] for e in iteration_log]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iters, [e["avg_train_reward"] for e in iteration_log],
            "o-", label="avg self-play reward (R²)")
    ax.plot(iters, [e["greedy_r2"] for e in iteration_log],
            "s-", label="greedy R²")
    ax.set_xlabel("iteration")
    ax.set_ylabel("R²")
    ax.set_title("AlphaZero on the SR grammar game")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_single_seed(cfg, exp_dir: Path):
    game, net, agent, trainer = cfg.build()

    iteration_log: list[dict] = []
    best_r2_overall = -np.inf
    best_rule_overall = None
    best_iter = 0

    n_iters = cfg.run.n_iterations
    n_sims = cfg.agent.mcts_params["n_simulations"]
    print(f"\n{'='*78}")
    print(f"  Symbolic regression -- AlphaZero on the derivation grammar game")
    print(f"  target: {game.target_infix}  constants: "
          + str({k: round(v, 3) for k, v in game.constants.items()}))
    print(f"  n_simulations={n_sims}  n_iterations={n_iters}  "
          f"n_games_per_train={cfg.trainer.n_games_per_train}")
    print(f"  action_space=Discrete({game.action_space.n})  "
          f"obs=MultiDiscrete({game.grammar.nsym + 1}^{game.state_len})")
    print(f"  Experiment dir: {exp_dir}")
    print(f"{'='*78}\n")

    for i in range(n_iters):
        t0 = time.time()

        trainer.train_iteration()

        greedy_r2, ep_best_r2, ep_best_rule = compute_greedy_eval(
            agent, n_episodes=cfg.evaluator.n_games)
        if ep_best_r2 > best_r2_overall:
            best_r2_overall = ep_best_r2
            best_rule_overall = ep_best_rule
            best_iter = i + 1
        elapsed = time.time() - t0

        train_records = trainer.statistics_manager.to_list()
        last = train_records[-1] if train_records else {}
        infix = prefix_to_infix(ep_best_rule.split()) if ep_best_rule else None
        entry = {
            "iteration": i + 1,
            "greedy_r2": float(greedy_r2),
            "best_r2": float(best_r2_overall),
            "best_iter": int(best_iter),
            "greedy_expression_prefix": ep_best_rule,
            "greedy_expression_infix": infix,
            "train_loss": float(last.get("train_loss", float("nan"))),
            "train_loss_policy": float(last.get("train_loss_policy", float("nan"))),
            "train_loss_value": float(last.get("train_loss_value", float("nan"))),
            "avg_train_reward": float(last.get("avg_reward", 0.0)),
            "num_examples": int(last.get("num_examples", 0)),
            "fit_cache_size": len(game._fit_cache),
            "wall_clock_s": round(elapsed, 2),
        }
        iteration_log.append(entry)

        print(
            f"[ITER {i+1}/{n_iters}] "
            f"greedy_R2={greedy_r2:+.4f} best={best_r2_overall:+.4f}@{best_iter} "
            f"| train_R2={entry['avg_train_reward']:+.4f} "
            f"| loss={entry['train_loss']:.3f} "
            f"(P:{entry['train_loss_policy']:.3f} V:{entry['train_loss_value']:.3f}) "
            f"| {elapsed:.1f}s"
        )
        if infix:
            print(f"           greedy expr: {infix}")

        trainer.statistics_manager.save_jsonl(str(exp_dir / "train_stats.jsonl"))
        with open(exp_dir / "iteration_log.jsonl", "w") as f:
            for e in iteration_log:
                f.write(json.dumps(e) + "\n")

    _save_final_checkpoint(exp_dir / "checkpoints" / "final", trainer, agent, net)
    plot_reward_curve(iteration_log, exp_dir / "reward_curve.png")

    print(f"\nBest greedy R²: {best_r2_overall:+.4f} at iteration {best_iter}")
    if best_rule_overall:
        print(f"Best expression (prefix): {best_rule_overall}")
        print(f"Best expression (infix):  {prefix_to_infix(best_rule_overall.split())}")
    print(f"Logs:        {exp_dir / 'iteration_log.jsonl'}")
    print(f"Curve:       {exp_dir / 'reward_curve.png'}")
    print(f"Checkpoint:  {exp_dir / 'checkpoints' / 'final'}")

    return iteration_log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Symbolic regression -- AlphaZero")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--problem", type=str, default="sine",
                   help="Named SR instance (default: sine; e.g. additive_quadratic). "
                        "See src/sraz/instances/symreg/problems.py")
    p.add_argument("--n-iterations", type=int, default=None,
                   help="Override config default (10)")
    p.add_argument("--n-simulations", type=int, default=None,
                   help="Override config default (25)")
    p.add_argument("--n-games", type=int, default=None,
                   help="Self-play games per iteration (default: 20)")
    p.add_argument("--max-len", type=int, default=None,
                   help="Token-buffer length (default: 15; the full target needs 19)")
    p.add_argument("--redraw-constants", action="store_true",
                   help="Redraw target constants each episode (original behavior)")
    p.add_argument("--out", type=str, default=None,
                   help="Output root dir (default: experiments/symreg)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    cfg = SymRegConfig(problem=args.problem)
    if args.n_iterations is not None:
        cfg.run.n_iterations = args.n_iterations
    if args.n_simulations is not None:
        cfg.agent.mcts_params["n_simulations"] = args.n_simulations
    if args.n_games is not None:
        cfg.trainer.n_games_per_train = args.n_games
    if args.max_len is not None:
        cfg.game.kwargs["max_len"] = args.max_len
    cfg.game.kwargs["redraw_constants"] = args.redraw_constants
    cfg.game.kwargs["problem_seed"] = args.seed

    s = args.seed
    cfg.agent.random_seeds = {
        "mcts": s, "train": s + 1, "eval": s + 2, "external_policy": s + 3,
    }
    cfg.net.kwargs["random_seed"] = s

    out_root = Path(args.out) if args.out else None
    exp_dir = setup_experiment_dir(cfg, seed=s, out_root=out_root)
    cfg.save(str(exp_dir / "config.json"))

    run_single_seed(cfg, exp_dir)


if __name__ == "__main__":
    main()
