"""Pure-UCT simulation sweep on the SR grammar game.

Runs the derivation game with the network disabled (uniform prior, value 0),
Dirichlet noise off, and the inner optimizer capped -- i.e. AlphaZero reduced
to Upper Confidence Bounds applied to Trees. Sweeps the MCTS simulation budget
and the backup rule over a family of fixed polynomial targets.

Because nothing is learned, an "iteration" here batches self-play games
without carrying anything forward. That is the point: it is the null curve a
learned run has to beat.

Usage:
    python scripts/run/run_uct_sweep.py --family linear
    python scripts/run/run_uct_sweep.py --family quadratic --n-simulations 4 8 12
"""

from sraz.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sraz.instances.symreg.config import SymRegConfig
from sraz.instances.symreg.evaluate import greedy_eval
from sraz.instances.symreg.game import prefix_to_infix
from sraz.instances.symreg.targets import TARGETS, family_targets, get_target


# The <b> hyperparameter block, in one place.
LMFIT_MAX_NFEV = 50          # inner Levenberg-Marquardt evaluation cap
SELF_PLAY_TEMPERATURE = 1.0  # temperature during search
GREEDY_TEMPERATURE = 0.0     # temperature at action selection
C_EXPLORATION = 1.0
DIRICHLET_NOISE = False
ROLLOUT_BUDGET = 20000       # rollout steps shared across one search; 500 starves
                             # anything past ~7 leaf expansions at rollout_n=5


def build_config(target, backup_rule: str, n_simulations: int, seed: int,
                 n_iterations: int, n_games: int, rollout_n: int = 0,
                 rollout_budget: int = ROLLOUT_BUDGET,
                 rollout_mode: str = "mean") -> SymRegConfig:
    cfg = SymRegConfig().use_uniform_net(value=0.0)
    cfg.game.kwargs |= {
        "target": target,
        "lmfit_max_nfev": LMFIT_MAX_NFEV,
        "problem_seed": seed,
        "redraw_constants": False,
    }
    cfg.agent.mcts_params = {
        "n_simulations": n_simulations,
        "temperature": SELF_PLAY_TEMPERATURE,
        "c_exploration": C_EXPLORATION,
        "backup_rule": backup_rule,
        "rollout_n": rollout_n,
        "rollout_budget": rollout_budget,
        "rollout_mode": rollout_mode,
    }
    cfg.agent.random_seeds = {
        "mcts": seed, "train": seed + 1, "eval": seed + 2, "external_policy": seed + 3,
    }
    cfg.trainer.n_games_per_train = n_games
    cfg.trainer.self_play_add_noise = DIRICHLET_NOISE
    cfg.trainer.self_play_temperature = SELF_PLAY_TEMPERATURE
    cfg.run.n_iterations = n_iterations
    cfg.evaluator.n_games = 1
    return cfg


def run_one(target, backup_rule: str, n_simulations: int, seed: int,
            n_iterations: int, n_games: int, run_dir: Path,
            rollout_n: int = 0,
            rollout_budget: int = ROLLOUT_BUDGET) -> list[dict]:
    """One (target, backup_rule, n_simulations, rollout_n) cell of the sweep."""
    cfg = build_config(target, backup_rule, n_simulations, seed, n_iterations,
                       n_games, rollout_n, rollout_budget)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.trainer.checkpoint_dir = str(run_dir / "checkpoints")
    cfg.save(str(run_dir / "config.json"))

    game, net, agent, trainer = cfg.build()

    log: list[dict] = []
    best_r2, best_rule, best_iter = -np.inf, None, 0

    for i in range(n_iterations):
        t0 = time.time()
        trainer.train_iteration()
        greedy_r2, ep_best_r2, ep_best_rule = greedy_eval(
            agent, n_episodes=cfg.evaluator.n_games, temperature=GREEDY_TEMPERATURE)

        # Best-so-far tracks the best structure *any* self-play game emitted,
        # not just the greedy one: with no learning the greedy episode is
        # frozen, so it can never report search's cumulative discoveries.
        sp_best_r2, sp_best_rule = _best_self_play(trainer)
        for r2, rule in ((ep_best_r2, ep_best_rule), (sp_best_r2, sp_best_rule)):
            if rule is not None and r2 > best_r2:
                best_r2, best_rule, best_iter = r2, rule, i + 1

        records = trainer.statistics_manager.to_list()
        last = records[-1] if records else {}
        log.append({
            "iteration": i + 1,
            "target": target.name,
            "backup_rule": backup_rule,
            "n_simulations": n_simulations,
            "rollout_n": rollout_n,
            "greedy_r2": float(greedy_r2),
            "greedy_expression_prefix": ep_best_rule,
            "greedy_expression_infix": (
                prefix_to_infix(ep_best_rule.split()) if ep_best_rule else None),
            "avg_train_reward": float(last.get("avg_reward", 0.0)),
            "best_r2_so_far": float(best_r2),
            "best_expression_prefix": best_rule,
            "best_iter": int(best_iter),
            "n_distinct_sentences": len(game._fit_cache),
            "wall_clock_s": round(time.time() - t0, 3),
        })

    with open(run_dir / "iteration_log.jsonl", "w") as f:
        for e in log:
            f.write(json.dumps(e) + "\n")
    return log


def _best_self_play(trainer) -> tuple[float, str | None]:
    """Best (reward, rule) over the step infos of the last iteration's games."""
    best_r, best_rule = -np.inf, None
    for infos in trainer._last_diagnostics:
        if not infos:
            continue
        rule = infos[-1].get("rule") if isinstance(infos[-1], dict) else None
        if rule is None:
            continue
        r = trainer.game._fit_cached(rule)
        if r > best_r:
            best_r, best_rule = r, rule
    return best_r, best_rule


def parse_args():
    p = argparse.ArgumentParser(description="Pure-UCT simulation sweep on the SR game")
    p.add_argument("--family", choices=["linear", "quadratic"], default=None,
                   help="Run every target in this family")
    p.add_argument("--targets", nargs="+", default=None,
                   choices=sorted(TARGETS), help="Explicit target names")
    p.add_argument("--backup-rules", nargs="+", default=["mean", "max"],
                   choices=["mean", "max", "topk", "softmax"])
    p.add_argument("--n-simulations", nargs="+", type=int,
                   default=[4, 6, 8, 10, 12, 20],
                   help="MCTS simulation budgets to overlay (default: 4..12 by 2, plus 20)")
    p.add_argument("--rollout-n", type=int, default=0,
                   help="Random rollouts per nonterminal leaf (default: 0 = pure UCT). "
                        "Gives unexpanded leaves a non-zero value estimate.")
    p.add_argument("--rollout-budget", type=int, default=ROLLOUT_BUDGET,
                   help="Rollout steps shared across one search; once spent, "
                        "leaves silently fall back to the network value")
    p.add_argument("--n-iterations", type=int, default=10)
    p.add_argument("--n-games", type=int, default=20,
                   help="Self-play games per iteration (default: 20)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None,
                   help="Output root (default: experiments/symreg_uct/<timestamp>)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    args = parse_args()

    if args.targets:
        targets = [get_target(n) for n in args.targets]
    elif args.family:
        targets = family_targets(args.family)
    else:
        raise SystemExit("Pass --family or --targets")

    root = Path(args.out) if args.out else (
        Path("experiments") / "symreg_uct"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.seed}")
    root.mkdir(parents=True, exist_ok=True)

    n_cells = len(targets) * len(args.backup_rules) * len(args.n_simulations)
    print(f"\n{'='*78}")
    print("  Pure UCT on the SR grammar game -- network disabled (uniform prior, value 0)")
    print(f"  targets:       {', '.join(t.name for t in targets)}")
    print(f"  backup rules:  {', '.join(args.backup_rules)}")
    print(f"  simulations:   {args.n_simulations}")
    print(f"  rollout_n:     {args.rollout_n}"
          f"{'' if args.rollout_n == 0 else f' (budget {args.rollout_budget} steps/search)'}")
    print(f"  noise: {DIRICHLET_NOISE}  tau_search: {SELF_PLAY_TEMPERATURE}  "
          f"tau_select: {GREEDY_TEMPERATURE}  lmfit max_nfev: {LMFIT_MAX_NFEV}")
    print(f"  {n_cells} cells x {args.n_iterations} iterations x {args.n_games} games")
    print(f"  out: {root}")
    print(f"{'='*78}\n")

    all_rows: list[dict] = []
    t_start = time.time()
    for target in targets:
        for backup_rule in args.backup_rules:
            for n_sims in args.n_simulations:
                run_dir = root / target.name / backup_rule / f"mcts{n_sims}"
                if args.rollout_n > 0:
                    run_dir = run_dir.with_name(f"{run_dir.name}_rollout{args.rollout_n}")
                log = run_one(target, backup_rule, n_sims, args.seed,
                              args.n_iterations, args.n_games, run_dir,
                              args.rollout_n, args.rollout_budget)
                all_rows.extend(log)
                final = log[-1]
                print(f"[{target.name:7s} {backup_rule:4s} sims={n_sims:2d}] "
                      f"greedy={final['greedy_r2']:+.4f} "
                      f"self-play={final['avg_train_reward']:+.4f} "
                      f"best={final['best_r2_so_far']:+.4f} "
                      f"({final['best_expression_prefix']}) "
                      f"| {sum(e['wall_clock_s'] for e in log):.1f}s")

    with open(root / "sweep.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    meta = {
        "seed": args.seed,
        "targets": [
            {"name": t.name, "family": t.family, "coeffs": list(t.coeffs),
             "label": t.label, "domain": [t.x_min, t.x_max], "n_x": t.n_x,
             "vertex": t.vertex, "discriminant": t.discriminant,
             "roots_in_domain": t.roots_in_domain()}
            for t in targets
        ],
        "backup_rules": args.backup_rules,
        "n_simulations": args.n_simulations,
        "rollout_n": args.rollout_n,
        "rollout_budget": args.rollout_budget,
        "n_iterations": args.n_iterations,
        "n_games_per_iteration": args.n_games,
        "dirichlet_noise": DIRICHLET_NOISE,
        "temperature_search": SELF_PLAY_TEMPERATURE,
        "temperature_select": GREEDY_TEMPERATURE,
        "c_exploration": C_EXPLORATION,
        "lmfit_max_nfev": LMFIT_MAX_NFEV,
        "network": "disabled (uniform prior, value 0)",
        "total_wall_clock_s": round(time.time() - t_start, 1),
    }
    with open(root / "sweep_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSweep complete in {meta['total_wall_clock_s']:.1f}s")
    print(f"Rows:  {root / 'sweep.jsonl'}")
    print(f"Meta:  {root / 'sweep_meta.json'}")


if __name__ == "__main__":
    main()
