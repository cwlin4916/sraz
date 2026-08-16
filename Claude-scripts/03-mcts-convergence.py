#!/usr/bin/env python
"""Which equation does the MCTS search actually converge to on sine?

Runs *pure* MCTS (net-free: uniform prior + random rollouts + mean backup --
exactly the leaf signal measured in 01/02) as the planner, and reports the
equation greedy play converges to, swept over a range of simulation budgets.
This is the counterpart to the policy-free V*/V~ trees: those show the search
is *fed* a deceptive rollout signal on the multiplicative/division branches;
this shows where that signal actually *lands* the search.

Greedy play mirrors run_symreg.compute_greedy_eval (fresh MCTS per move,
temperature 0.01, argmax over visit counts) but with a *seeded* rng per episode
so the converged equation is reproducible.

Speed: MCTS rollouts fit constants at every terminal they hit. Every reachable
terminal is already priced in the seed-matched fit cache built by 01/02, so we
memoise fit_expression at module level from that cache -> rollouts become dict
lookups and the R^2 values match the V* analysis exactly. The memo lives at
module scope (not on the game), so game.clone()'s deepcopy stays cheap.

Run from repo root:
    .venv/bin/python Claude-scripts/03-mcts-convergence.py \
        --problem sine --problem-seed 42 --episodes 24
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sraz.instances.symreg.game as game_mod
from sraz.instances.symreg.game import prefix_to_infix
from sraz.instances.symreg.config import SymRegConfig

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "Claude-experiments" / "8-5"
CACHE_DIR = OUT_DIR / "cache"


# --------------------------------------------------------------------------
# Memoise fits from the seed-matched cache so rollouts are ~free & exact
# --------------------------------------------------------------------------
def install_fit_memo(cache_key: str) -> dict:
    memo: dict[str, float] = {}
    cpath = CACHE_DIR / f"fit_cache_{cache_key}.json"
    if cpath.exists():
        memo.update(json.loads(cpath.read_text()))
        print(f"[memo] preloaded {len(memo)} fits from {cpath.name}")
    else:
        print(f"[memo] no cache at {cpath.name}; fitting live (slower)")
    orig = game_mod.fit_expression

    def memoised(rule, xs, exact_ys, max_nfev=None):
        v = memo.get(rule)
        if v is None:
            v = orig(rule, xs, exact_ys, max_nfev=max_nfev)
            memo[rule] = v
        return v

    game_mod.fit_expression = memoised          # game._fit_cached calls this
    return memo


# --------------------------------------------------------------------------
# One greedy MCTS episode -> (r2, prefix_rule)
# --------------------------------------------------------------------------
def greedy_episode(agent, seed: int) -> tuple[float, str | None]:
    g = agent.game.clone()
    g.reset_wrapper()
    rng = np.random.default_rng(seed)
    reward, rule = 0.0, None
    for _ in range(g.state_len + 5):
        probs = agent.policy(g, "", add_noise=False,
                             temperature_override=0.01, rng=rng)
        action = int(np.argmax(probs))
        _, reward, terminated, truncated, info = g.step_wrapper(action)
        if terminated or truncated:
            rule = info.get("rule")
            break
    return float(reward), rule


# --------------------------------------------------------------------------
# Sweep simulation budgets
# --------------------------------------------------------------------------
def run_sweep(problem, problem_seed, sims_list, episodes, rollout_n, base_seed):
    cfg = SymRegConfig(problem=problem, pure_mcts=True)
    cfg.game.kwargs["problem_seed"] = problem_seed
    cfg.game.kwargs["redraw_constants"] = False
    cfg.agent.mcts_params["rollout_n"] = rollout_n
    cfg.agent.random_seeds = {"mcts": base_seed, "train": base_seed + 1,
                              "eval": base_seed + 2, "external_policy": base_seed + 3}
    game, net, agent, trainer = cfg.build()
    print(f"target: {game.target_infix}   constants: "
          + str({k: round(v, 3) for k, v in game.constants.items()}))
    print(f"pure MCTS: uniform prior + random rollouts (rollout_n={rollout_n}, "
          f"mean backup); greedy = argmax visit counts\n")

    results = []
    for sims in sims_list:
        agent.mcts_params["n_simulations"] = sims
        t0 = time.time()
        episodes_out = [greedy_episode(agent, base_seed + 100 * len(results) + e)
                        for e in range(episodes)]
        dt = time.time() - t0
        r2s = np.array([r for r, _ in episodes_out])
        infixes = [prefix_to_infix(rule.split()) if rule else None
                   for _, rule in episodes_out]
        counts = Counter(infixes)
        best_i = int(np.argmax(r2s))
        frac_sin = float(np.mean([("sin" in (rule or ""))
                                  for _, rule in episodes_out]))
        rec = {
            "n_simulations": sims,
            "episodes": episodes,
            "mean_r2": float(r2s.mean()),
            "best_r2": float(r2s.max()),
            "worst_r2": float(r2s.min()),
            "std_r2": float(r2s.std()),
            "frac_reaching_sin": frac_sin,
            "best_infix": infixes[best_i],
            "best_prefix": episodes_out[best_i][1],
            "converged_counts": [(k, c) for k, c in counts.most_common()],
            "r2_per_episode": r2s.tolist(),
            "wall_clock_s": round(dt, 1),
        }
        results.append(rec)
        print(f"[sims={sims:>4}] mean R2={rec['mean_r2']:+.4f}  "
              f"best={rec['best_r2']:+.4f}  sin-rate={frac_sin:.2f}  ({dt:.1f}s)")
        top = counts.most_common(3)
        for expr, c in top:
            print(f"           {c:>2}/{episodes}  {expr}")
    return game, results


def draw(results, game, out_path, problem, problem_seed):
    sims = [r["n_simulations"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    # per-episode R^2 as jittered points, plus mean/best lines
    rng = np.random.default_rng(0)
    for i, r in enumerate(results):
        ys = np.array(r["r2_per_episode"])
        xs = i + (rng.random(len(ys)) - 0.5) * 0.3
        ax.scatter(xs, ys, s=18, alpha=0.5, color="tab:blue", zorder=2)
        ax.plot([i - 0.25, i + 0.25], [r["mean_r2"]] * 2, color="black", lw=2,
                zorder=3)
        ax.plot([i - 0.25, i + 0.25], [r["best_r2"]] * 2, color="tab:green",
                lw=2, zorder=3)
    # reference ceilings from the exact-enumeration V* analysis
    ax.axhline(0.9966, color="tab:green", ls=":", lw=1,
               label="V* (global optimum, exact) = 0.997")
    ax.axhline(0.873, color="tab:red", ls=":", lw=1,
               label="honest linear  C1*x = 0.873")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(sims)
    ax.set_xlabel("MCTS simulations per move")
    ax.set_ylabel("greedy terminal R2")
    ax.set_title(f"Pure MCTS convergence on {problem} (seed {problem_seed})\n"
                 f"points = episodes; black = mean, green = best")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", default="sine")
    ap.add_argument("--problem-seed", type=int, default=42)
    ap.add_argument("--sims", type=int, nargs="+",
                    default=[10, 25, 50, 100, 200, 400])
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--rollout-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42, help="base RNG seed")
    args = ap.parse_args()

    cache_key = (f"{args.problem}_seed{args.problem_seed}"
                 if args.problem == "sine" else args.problem)
    install_fit_memo(cache_key)

    print(f"=== MCTS convergence: {args.problem} (problem_seed={args.problem_seed}) ===")
    game, results = run_sweep(args.problem, args.problem_seed, args.sims,
                              args.episodes, args.rollout_n, args.seed)

    tag = cache_key
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"mcts-convergence_{tag}.json").write_text(json.dumps({
        "problem": args.problem, "problem_seed": args.problem_seed,
        "target_infix": game.target_infix, "target_constants": game.constants,
        "rollout_n": args.rollout_n, "episodes": args.episodes,
        "sims": args.sims, "results": results}, indent=2))
    png = OUT_DIR / f"mcts-convergence_{tag}.png"
    draw(results, game, png, args.problem, args.problem_seed)

    print("\n--- summary -------------------------------------------------")
    print(f"  target: {game.target_infix}")
    print(f"  V* (exact, from tree analysis): 0.997")
    for r in results:
        print(f"  sims={r['n_simulations']:>4}: best R2={r['best_r2']:+.4f}  "
              f"mean={r['mean_r2']:+.4f}  sin-rate={r['frac_reaching_sin']:.2f}  "
              f"| modal: {r['converged_counts'][0][0]}")
    print(f"[saved] {(OUT_DIR / f'mcts-convergence_{tag}.json').relative_to(REPO)}")
    print(f"[saved] {png.relative_to(REPO)}")


if __name__ == "__main__":
    main()
