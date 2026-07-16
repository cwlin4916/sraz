"""Pure MCTS with random-rollout leaf evaluation: R^2 against simulation budget.

The simplest search this repo can run. Everything that could learn, remember,
or explore off-policy is switched off, leaving exactly three moving parts:

  1. a uniform prior over the legal productions      (no policy network)
  2. a leaf value that is the *mean* of `ROLLOUT_N` random completions
  3. a visit-count argmax at each move               (no sampling)

so the emitted expression is a function of the simulation budget and the
rollout draws alone. Nothing carries across episodes -- there is no trainer
here on purpose. With the network disabled an "iteration" would batch episodes
without propagating anything, so this driver calls `greedy_episode` directly
(the same choice `docs/notes/figures/rollout_experiments.json` records).

Determinism. `rollout_n = 0` consults no RNG at all and would make every
episode identical (tests/test_uct_pure_search.py). Rollouts are the *only*
source of randomness, and MCTS._rollout_value draws them from the **global**
numpy RNG (mcts.py:322 calls np.random.randint, not a seeded stream), so an
episode is pinned by seeding np.random and nothing else. That is why episodes
are seeded here rather than through `Agent.random_seeds`, which the rollout
path never consults.

Usage:
    python scripts/run/run_pure_mcts_targets.py
    python scripts/run/run_pure_mcts_targets.py --n-episodes 15 --out /tmp/probe.json
"""

from sraz.utils import disable_numpy_multithreading, use_deterministic_cuda
disable_numpy_multithreading()
use_deterministic_cuda()

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sraz.core.agent import Agent
from sraz.core.policy_value_net import UniformPolicyValueNet
from sraz.instances.symreg.evaluate import greedy_episode
from sraz.instances.symreg.game import SymRegGame, prefix_to_infix
from sraz.instances.symreg.targets import get_target


# ---------------------------------------------------------------------------
# The hyperparameter block, in one place. Every knob this experiment fixes,
# and why fixing it is what makes the experiment simple.
# ---------------------------------------------------------------------------

TARGETS = ("lin_D", "quad_B")     # one panel each: saturated vs deceptive
N_SIMULATIONS = (5, 9, 13, 17)    # 5..20 in steps of 4; 21 would leave the range
N_EPISODES = 5                    # repeats per cell; they differ only by rollout draws

# -- what makes it "pure" MCTS: the network is disabled, not merely untrained --
NET_VALUE = 0.0                   # UniformPolicyValueNet: uniform prior, value 0
DIRICHLET_NOISE = False           # no root exploration noise

# -- the leaf evaluator: average of random completions --
ROLLOUT_N = 5                     # random completions averaged at each new leaf
ROLLOUT_MODE = "mean"             # average them (vs "max", which behaves differently)
ROLLOUT_BLEND = 0.0               # 0 => leaf value is the rollout alone, no net value
ROLLOUT_BUDGET = 10 ** 6          # rollout steps per search; set so it never binds

# -- search and selection --
BACKUP_RULE = "mean"              # single rule: no mean-vs-max comparison here
C_EXPLORATION = 1.0               # UCB exploration constant
SEARCH_TEMPERATURE = 1.0          # temperature inside search
SELECT_TEMPERATURE = 0.0          # argmax over visit counts at each move

# -- the reward pipeline --
LMFIT_MAX_NFEV = 50               # inner Levenberg-Marquardt evaluation cap

BASE_SEED = 1234                  # episode e uses np.random.seed(BASE_SEED + e)

# Root actions are flat index 7*slot + j; at the start only slot 0 is legal, so
# the seven root actions are exactly the seven productions.
PRODUCTION_LABELS = {
    0: "S -> + S S",
    1: "S -> C0",
    2: "S -> * C1 x",
    3: "S -> * C2 * x x",
    4: "S -> * S S",
    5: "S -> / S S",
    6: "S -> * C3 sin * C4 x",
}


def build_agent(target_name: str, n_simulations: int) -> Agent:
    """A pure-MCTS agent: uniform net, rollout leaves, no learning."""
    game = SymRegGame(target=target_name, lmfit_max_nfev=LMFIT_MAX_NFEV)
    n_actions = game.state_len * game.grammar.nprods
    net = UniformPolicyValueNet(n_actions=n_actions, value=NET_VALUE)
    return Agent(
        game=game,
        net=net,
        mcts_params={
            "n_simulations": n_simulations,
            "temperature": SEARCH_TEMPERATURE,
            "c_exploration": C_EXPLORATION,
            "backup_rule": BACKUP_RULE,
            "rollout_n": ROLLOUT_N,
            "rollout_mode": ROLLOUT_MODE,
            "rollout_blend": ROLLOUT_BLEND,
            "rollout_budget": ROLLOUT_BUDGET,
        },
    )


def run_cell(target_name: str, n_simulations: int, n_episodes: int) -> dict:
    """One (target, budget) cell: `n_episodes` episodes differing only by rollout draws."""
    episodes = []
    for e in range(n_episodes):
        agent = build_agent(target_name, n_simulations)
        np.random.seed(BASE_SEED + e)          # the only randomness: rollout draws
        t0 = time.time()
        out = greedy_episode(agent, temperature=SELECT_TEMPERATURE)
        episodes.append({
            "episode": e,
            "seed": BASE_SEED + e,
            "r2": float(out["reward"]),
            "prefix": out["rule"],
            "infix": prefix_to_infix(out["rule"].split()) if out["rule"] else None,
            "n_moves": len(out["actions"]),
            "wall_clock_s": round(time.time() - t0, 3),
        })

    r2 = np.array([ep["r2"] for ep in episodes], dtype=np.float64)
    return {
        "target": target_name,
        "n_simulations": n_simulations,
        "episodes": episodes,
        "r2_mean": float(r2.mean()),
        "r2_std": float(r2.std(ddof=0)),
        "r2_min": float(r2.min()),
        "r2_max": float(r2.max()),
        "root": root_table(target_name, n_simulations),
    }


def root_table(target_name: str, n_simulations: int) -> list[dict]:
    """Visit counts and backed-up Q at the root, for episode 0's first search.

    This is the diagnostic that explains the curve rather than restating it:
    the emitted expression is decided by which root edge wins the visit-count
    argmax, and Q is what drives that.
    """
    from sraz.core.mcts import MCTS

    agent = build_agent(target_name, n_simulations)
    agent.game.reset_wrapper()
    np.random.seed(BASE_SEED)
    mcts = MCTS(agent.game.clone(), agent.net, **agent.mcts_params)
    mcts.temperature = SELECT_TEMPERATURE
    mcts.perform_simulations("", add_noise=DIRICHLET_NOISE)

    node = mcts.nodes[mcts.game.hashable_obs]

    # MCTS keys its edges by whatever np.unravel_index returns, which for the
    # flat 105-action space is a 1-tuple `(a,)` rather than the int `a`
    # (mcts.py:407). Indexing these dicts with a bare int silently misses every
    # edge, so flatten the keys once here rather than guessing the form.
    counts = {int(np.ravel(a)[0]): n for a, n in node.action_N.items()}
    qs = {int(np.ravel(a)[0]): q for a, q in node.action_Q.items()}

    rows = []
    for j, label in PRODUCTION_LABELS.items():
        rows.append({
            "action": j,
            "production": label,
            "N": int(counts.get(j, 0)),
            "Q": (round(float(qs[j]), 6) if j in qs else None),
        })
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Pure MCTS (no network) with mean random-rollout leaves: R^2 vs budget")
    p.add_argument("--targets", nargs="+", default=list(TARGETS))
    p.add_argument("--n-simulations", nargs="+", type=int, default=list(N_SIMULATIONS))
    p.add_argument("--n-episodes", type=int, default=N_EPISODES)
    p.add_argument("--out", type=str,
                   default="docs/notes/figures/pure_mcts_targets.json")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{'=' * 78}")
    print("  Pure MCTS on the SR grammar game -- network disabled (uniform prior, value 0)")
    print(f"  targets:      {', '.join(args.targets)}")
    print(f"  simulations:  {args.n_simulations}")
    print(f"  episodes:     {args.n_episodes} per cell (seeds {BASE_SEED}..{BASE_SEED + args.n_episodes - 1})")
    print(f"  leaf value:   mean of {ROLLOUT_N} random rollouts (blend {ROLLOUT_BLEND}, "
          f"budget {ROLLOUT_BUDGET})")
    print(f"  backup: {BACKUP_RULE}   c: {C_EXPLORATION}   tau_search: {SEARCH_TEMPERATURE}   "
          f"tau_select: {SELECT_TEMPERATURE}")
    print(f"  noise: {DIRICHLET_NOISE}   lmfit max_nfev: {LMFIT_MAX_NFEV}")
    print(f"{'=' * 78}\n")

    t_start = time.time()
    cells = []
    for target_name in args.targets:
        for n_sims in args.n_simulations:
            cell = run_cell(target_name, n_sims, args.n_episodes)
            cells.append(cell)
            modal = max(set(ep["prefix"] for ep in cell["episodes"]),
                        key=lambda r: sum(ep["prefix"] == r for ep in cell["episodes"]))
            print(f"[{target_name:7s} sims={n_sims:2d}] "
                  f"R2 mean={cell['r2_mean']:+.4f} sd={cell['r2_std']:.4f} "
                  f"range=[{cell['r2_min']:+.3f}, {cell['r2_max']:+.3f}]  modal: {modal}")

    out = {
        "_provenance": {
            "what": (f"Greedy R^2 vs MCTS simulation budget on {', '.join(args.targets)}, "
                     f"pure MCTS: network disabled (uniform prior, value {NET_VALUE}), "
                     f"leaf value = mean of {ROLLOUT_N} random rollouts, Dirichlet noise off, "
                     f"tau={SELECT_TEMPERATURE} at selection, backup_rule={BACKUP_RULE}, "
                     f"c_exploration={C_EXPLORATION}, lmfit_max_nfev={LMFIT_MAX_NFEV}."),
            "why_no_trainer": (
                "With the network disabled nothing is learned, so an episode is a pure "
                "function of (n_simulations, rollout draws). Training iterations cannot "
                "move it; this driver calls greedy_episode directly."),
            "determinism": (
                "rollout_n>0 draws from the GLOBAL numpy RNG (mcts.py:322 uses "
                "np.random.randint, not a seeded stream), so each episode is pinned by "
                f"np.random.seed({BASE_SEED} + episode_index) and by nothing else."),
            "reproduce": "python scripts/run/run_pure_mcts_targets.py",
        },
        "hyperparameters": {
            "targets": list(args.targets),
            "n_simulations": list(args.n_simulations),
            "n_episodes": args.n_episodes,
            "net": f"UniformPolicyValueNet (uniform prior, value {NET_VALUE})",
            "dirichlet_noise": DIRICHLET_NOISE,
            "rollout_n": ROLLOUT_N,
            "rollout_mode": ROLLOUT_MODE,
            "rollout_blend": ROLLOUT_BLEND,
            "rollout_budget": ROLLOUT_BUDGET,
            "backup_rule": BACKUP_RULE,
            "c_exploration": C_EXPLORATION,
            "temperature_search": SEARCH_TEMPERATURE,
            "temperature_select": SELECT_TEMPERATURE,
            "lmfit_max_nfev": LMFIT_MAX_NFEV,
            "base_seed": BASE_SEED,
        },
        "targets": {
            name: {
                "coeffs": list(get_target(name).coeffs),
                "family": get_target(name).family,
                "label": get_target(name).label,
                "infix": get_target(name).infix,
            }
            for name in args.targets
        },
        "cells": cells,
        "total_wall_clock_s": round(time.time() - t_start, 1),
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nDone in {out['total_wall_clock_s']:.1f}s -> {path}")


if __name__ == "__main__":
    main()
