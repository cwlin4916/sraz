#!/usr/bin/env python
"""Optimal value vs random-rollout value of a grammar-derivation state.

For a single state ``s`` of the symbolic-regression grammar game (a partial
derivation), this computes two policy-free quantities and the gap between
them --- *no MCTS, no network*:

    V*(s)      the OPTIMAL value: the best clipped-R^2 reachable from s, taken
               over ALL legal terminal completions (exhaustive enumeration under
               the game's 14-token length cap; fits are parallelised + disk
               cached). Under this game the reward is terminal-only and
               undiscounted (reward_discount = 1.0), so the return from a state
               *is* the R^2 of whatever expression it completes to --- hence
               V*(s) = max over completions.

    V_rand(s)  the RANDOM-ROLLOUT value: complete s with uniform-random *legal
               actions* until termination and take the terminal R^2. This is a
               random variable, so we draw K rollouts and report the whole
               distribution. The action sampling is byte-for-byte the same as
               MCTS._rollout_value (uniform over the flattened legal-action
               mask; mcts.py:329-338) but seeded here, so it is reproducible ---
               sidestepping the known _rollout_value non-reproducibility defect.

    gap        V*(s) - E[V_rand(s)].  This is exactly the Axis-1
               *informativeness* signal from
               Claude-research/snapshots/02-informativeness-and-deception.md,
               measured on the task alone with no policy: a large gap means the
               state is promising but random completion almost never finds it,
               i.e. the value is hidden.

Usage
-----
    # the root [S] on the sine instance (the documented failure), K=1000
    .venv/bin/python Claude-scripts/01-state-value-gap.py --state "S" --rollouts 1000

    # any other partial derivation (space-separated prefix tokens, >=1 S)
    .venv/bin/python Claude-scripts/01-state-value-gap.py --state "+ S S"

Results (JSON + a printed summary) land in Claude-experiments/8-5/; the fit
cache lives in Claude-experiments/8-5/cache/ so re-runs are instant.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from sraz.instances.symreg.game import (ADDITIVE_GRAMMAR, SymRegGame,
                                        fit_expression)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "Claude-experiments" / "8-5"
CACHE_DIR = OUT_DIR / "cache"


# --------------------------------------------------------------------------
# Grammar helpers
# --------------------------------------------------------------------------
def build_game(problem: str, problem_seed: int) -> SymRegGame:
    """Build an instance.

    The sine instance is the shipped grammar/target; its target CONSTANTS
    (C0, C1, C2) are drawn from ``problem_seed`` -- and the documented baseline
    runs use ``problem_seed = 42`` (run_symreg.py maps --seed straight onto
    problem_seed), NOT the shipped default of 0. The clipped-R^2 of every
    sentence depends on those constants, so the seed is load-bearing here.
    """
    if problem == "sine":
        return SymRegGame(problem_seed=problem_seed)
    # A named family target fixes its own coefficients AND belongs to the
    # controlled study's MDP: ADDITIVE_GRAMMAR at L=12. Without these two
    # kwargs SymRegGame would silently fall back to the sine grammar at
    # max_len=15 (game.py:216,228) -- the same tree, wrong environment.
    return SymRegGame(target=problem, grammar_rules=ADDITIVE_GRAMMAR,
                      max_len=12)


def leftmost_nt(state: tuple[int, ...], nonterms: set[int]) -> int:
    """Index of the leftmost nonterminal token, or -1 if the state is terminal."""
    for i, t in enumerate(state):
        if t in nonterms:
            return i
    return -1


def decode(state: tuple[int, ...], tokenlist: list[str]) -> str:
    return " ".join(tokenlist[t] for t in state)


def parse_state(spec: str, game: SymRegGame) -> tuple[int, ...]:
    """Turn a space-separated symbol string (e.g. 'S' or '+ S S') into tokens."""
    g = game.grammar
    toks = []
    for sym in spec.strip().split():
        if sym not in g.symdict:
            raise ValueError(
                f"unknown symbol {sym!r}; known: {sorted(g.symdict)}")
        toks.append(g.symdict[sym])
    if len(toks) == 0:
        raise ValueError("empty state")
    if len(toks) > game.state_len:
        raise ValueError(f"state longer than max_len={game.state_len}")
    return tuple(toks)


# --------------------------------------------------------------------------
# Exhaustive completion enumeration (for V*)
# --------------------------------------------------------------------------
def enumerate_completions(state0: tuple[int, ...], game: SymRegGame):
    """Yield every legal terminal completion of ``state0`` as a token tuple.

    Expands the LEFTMOST nonterminal at each step, so each yielded sentence is
    produced by exactly one leftmost derivation (prefix notation is
    unambiguous). Because derivation length is order-independent and grows
    monotonically, the set of sentences reached this way is *identical* to the
    game's reachable-terminal set, and the per-step length guard reproduces
    get_action_mask's rule (finished sentences carry <= max_len - 1 tokens).
    """
    g = game.grammar
    maxlen = game.state_len
    nonterms = g.nonterms
    stack = [state0]
    while stack:
        state = stack.pop()
        i = leftmost_nt(state, nonterms)
        if i == -1:
            yield state
            continue
        tok = state[i]
        for j in g.proddict[tok]:
            rhs = tuple(g.productions[j])
            new_len = len(state) + len(rhs) - 1
            if new_len < maxlen:                      # ll + |rhs| - 1 < max_len
                stack.append(state[:i] + rhs + state[i + 1:])


# --------------------------------------------------------------------------
# Parallel fitting with a disk cache (keyed by sentence string)
# --------------------------------------------------------------------------
_XS = None
_YS = None
_MAX_NFEV = None


def _fit_init(xs, ys, max_nfev):
    global _XS, _YS, _MAX_NFEV
    _XS, _YS, _MAX_NFEV = xs, ys, max_nfev


def _fit_one(rule: str) -> float:
    return fit_expression(rule, _XS, _YS, max_nfev=_MAX_NFEV)


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"fit_cache_{cache_key}.json"


def fit_all(rules: list[str], game: SymRegGame, cache_key: str,
            workers: int, use_cache: bool = True) -> dict[str, float]:
    """Return {sentence: clipped_R2} for every rule, fitting only cache misses.

    ``cache_key`` must identify the TARGET (e.g. 'sine_seed42'): the fitted R^2
    depends on the target constants, so caches from different seeds must not mix.
    """
    cache: dict[str, float] = {}
    cpath = _cache_path(cache_key)
    if use_cache and cpath.exists():
        cache = json.loads(cpath.read_text())

    todo = [r for r in rules if r not in cache]
    if todo:
        t0 = time.time()
        args = (game.xs, game.exact_ys, game.lmfit_max_nfev)
        if workers == 1:
            _fit_init(*args)
            for k, r in enumerate(todo):
                cache[r] = _fit_one(r)
        else:
            import multiprocessing as mp
            with mp.Pool(workers, initializer=_fit_init, initargs=args) as pool:
                for r, v in zip(todo, pool.imap(_fit_one, todo, chunksize=256)):
                    cache[r] = v
        dt = time.time() - t0
        print(f"[fit] {len(todo)} new fits in {dt:.1f}s "
              f"({1000 * dt / max(len(todo), 1):.2f} ms/fit); "
              f"{len(rules) - len(todo)} cache hits")
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(cache))
    else:
        print(f"[fit] all {len(rules)} rules served from cache")
    return {r: cache[r] for r in rules}


# --------------------------------------------------------------------------
# V*(s): optimal value by exhaustive enumeration
# --------------------------------------------------------------------------
def optimal_value(state: tuple[int, ...], game: SymRegGame, cache_key: str,
                  workers: int, max_enumerate: int | None = None) -> dict:
    g = game.grammar
    print("[V*] enumerating completions ...")
    t0 = time.time()
    seen: set[str] = set()
    order: list[str] = []
    n_total = 0
    truncated = False
    for comp in enumerate_completions(state, game):
        n_total += 1
        s = decode(comp, g.tokenlist)
        if s not in seen:
            seen.add(s)
            order.append(s)
        if max_enumerate is not None and n_total >= max_enumerate:
            truncated = True
            break
    print(f"[V*] {n_total} completions ({len(order)} distinct sentences) "
          f"in {time.time() - t0:.2f}s"
          + ("  [TRUNCATED -- V* is a lower bound]" if truncated else ""))

    fits = fit_all(order, game, cache_key, workers)
    vals = np.array([fits[s] for s in order], dtype=float)
    argmax = order[int(np.argmax(vals))]
    return {
        "n_completions": n_total,
        "n_distinct": len(order),
        "truncated": truncated,
        "v_star": float(vals.max()),
        "argmax_sentence": argmax,
        "reward_hist": _hist(vals),
        "frac_clip_floor": float(np.mean(vals <= -0.9999)),
    }


# --------------------------------------------------------------------------
# V_rand(s): random-rollout value distribution
# --------------------------------------------------------------------------
def _game_at(state: tuple[int, ...], game: SymRegGame) -> SymRegGame:
    """A clone of the game whose derivation buffer is set to ``state``."""
    gg = game.clone()
    gg.reset_wrapper()
    g = gg.grammar
    buf = np.full(gg.state_len, g.pad_tok, dtype=np.int64)
    buf[:len(state)] = state
    gg.state = buf
    gg.real_state_len = len(state)
    gg.obs = buf.copy()
    gg.terminated = not gg._has_nonterms()
    gg.reward = gg._fit_cached(gg._decode_state()) if gg.terminated else 0.0
    return gg


def rollout_values(state: tuple[int, ...], game: SymRegGame, k: int,
                   seed: int) -> dict:
    """K uniform-random legal-action completions of ``state``; return their R^2s.

    Sampling matches MCTS._rollout_value exactly (uniform over the flattened
    legal-action mask), but driven by a local seeded RNG so the result is
    reproducible.
    """
    base = _game_at(state, game)
    rng = np.random.default_rng(seed)
    rewards = np.empty(k, dtype=float)
    sentences: list[str] = []
    t0 = time.time()
    for n in range(k):
        gg = base.clone()
        while not (gg.terminated or gg.truncated):
            mask = gg.get_action_mask()
            valid = np.flatnonzero(mask)
            if len(valid) == 0:                       # dead end (unreachable here)
                break
            action = int(valid[rng.integers(len(valid))])
            gg.step_wrapper(action)
        rewards[n] = gg.reward
        sentences.append(gg.info.get("rule") or gg._decode_state())
    dt = time.time() - t0
    print(f"[V_rand] {k} rollouts in {dt:.1f}s ({1000 * dt / k:.2f} ms each)")

    uniq = sorted(set(sentences))
    return {
        "k": k,
        "seed": seed,
        "mean": float(rewards.mean()),
        "std": float(rewards.std()),
        "min": float(rewards.min()),
        "max": float(rewards.max()),
        "quantiles": {q: float(np.quantile(rewards, q))
                      for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "frac_clip_floor": float(np.mean(rewards <= -0.9999)),
        "n_unique_sentences": len(uniq),
        "reward_hist": _hist(rewards),
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _hist(vals: np.ndarray) -> dict:
    """Coarse histogram over the reward range [-1, 1]."""
    edges = np.array([-1.0, -0.5, 0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0001])
    counts, _ = np.histogram(vals, bins=edges)
    labels = ["[-1,-.5)", "[-.5,0)", "[0,.5)", "[.5,.8)", "[.8,.9)",
              "[.9,.95)", "[.95,.99)", "[.99,1]"]
    return dict(zip(labels, counts.tolist()))


def analyze(spec: str, problem: str, problem_seed: int, k: int, seed: int,
            workers: int, max_enumerate: int | None, save: bool) -> dict:
    game = build_game(problem, problem_seed)
    state = parse_state(spec, game)
    # named targets fix their own coefficients, so the seed only labels sine
    cache_key = f"{problem}_seed{problem_seed}" if problem == "sine" else problem
    print(f"\n=== state {spec!r}  (problem={problem}, problem_seed={problem_seed}, "
          f"target: {game.target_infix}) ===")
    print(f"    target constants: {game.constants}")

    opt = optimal_value(state, game, cache_key, workers, max_enumerate)
    rnd = rollout_values(state, game, k, seed)
    gap = opt["v_star"] - rnd["mean"]

    result = {
        "problem": problem,
        "problem_seed": problem_seed,
        "target_infix": game.target_infix,
        "target_constants": game.constants,
        "state": spec,
        "optimal": opt,
        "rollout": rnd,
        "gap_vstar_minus_mean_rollout": gap,
        "p_rollout_reaches_vstar": float(
            (rnd["max"] >= opt["v_star"] - 1e-4)),
    }

    print("\n--- summary "
          "---------------------------------------------------------")
    print(f"  V*(s)                    = {opt['v_star']:.4f}   "
          f"({opt['argmax_sentence']})")
    print(f"  E[V_rand(s)]             = {rnd['mean']:.4f}  "
          f"(std {rnd['std']:.4f})")
    print(f"  gap = V* - E[V_rand]     = {gap:.4f}")
    print(f"  rollout max              = {rnd['max']:.4f}  "
          f"(reaches V*? {'yes' if result['p_rollout_reaches_vstar'] else 'no'})")
    print(f"  rollout frac at -1 floor = {rnd['frac_clip_floor']:.3f}")
    print(f"  rollout median / p95     = {rnd['quantiles'][0.5]:.4f} "
          f"/ {rnd['quantiles'][0.95]:.4f}")
    print(f"  completions enumerated   = {opt['n_completions']} "
          f"({opt['n_distinct']} distinct)")
    print("  rollout reward histogram : "
          + "  ".join(f"{kk}:{vv}" for kk, vv in rnd["reward_hist"].items()))
    print("-------------------------------------------------------------"
          "------")

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        safe = spec.strip().replace(" ", "_").replace("/", "div") or "root"
        tag = f"{problem}_seed{problem_seed}" if problem == "sine" else problem
        path = OUT_DIR / f"state-value_{tag}_{safe}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"[saved] {path.relative_to(REPO)}")
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", default="S",
                    help="space-separated prefix tokens; default the root 'S'")
    ap.add_argument("--problem", default="sine",
                    help="instance name (default 'sine')")
    ap.add_argument("--problem-seed", type=int, default=42,
                    help="target-constant seed for sine; the documented baseline "
                         "is 42 (run_symreg maps --seed onto problem_seed), "
                         "not the shipped default of 0")
    ap.add_argument("--rollouts", type=int, default=1000, help="K (default 1000)")
    ap.add_argument("--seed", type=int, default=0, help="rollout RNG seed")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2),
                    help="parallel fit workers (default cpu_count-2)")
    ap.add_argument("--max-enumerate", type=int, default=None,
                    help="cap on completions; exceeding it makes V* a lower bound")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    analyze(args.state, args.problem, args.problem_seed, args.rollouts,
            args.seed, args.workers, args.max_enumerate, save=not args.no_save)


if __name__ == "__main__":
    main()
