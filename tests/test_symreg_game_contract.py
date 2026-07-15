"""SymRegGame's contract with the AlphaZero engine (Game ABC semantics)."""

import numpy as np

from sraz.instances.symreg.game import SymRegGame


BEST_SEQ = [(0, 0), (1, 6), (7, 0), (8, 1), (9, 3)]  # 14-token sin+C0+quad form


def _play(game, seq):
    rewards = []
    for pos, prod in seq:
        _, r, term, _, _ = game.step_wrapper(pos * game.grammar.nprods + prod)
        rewards.append(r)
    return rewards, term


def test_seeded_reset_deterministic():
    game = SymRegGame(problem_seed=7)
    obs1, _ = game.reset_wrapper(seed=0)
    h1 = game.hashable_obs
    obs2, _ = game.reset_wrapper(seed=0)
    assert (obs1 == obs2).all()
    assert game.hashable_obs == h1


def test_stash_unstash_exact():
    game = SymRegGame()
    game.reset_wrapper()
    stash = game.stash_state()
    state0 = game.state.copy()
    rsl0, sc0 = game.real_state_len, game.step_count

    _play(game, BEST_SEQ[:3])
    assert game.real_state_len != rsl0

    game.unstash_state(stash)
    assert (game.state == state0).all()
    assert game.real_state_len == rsl0
    assert game.step_count == sc0
    assert game.terminated is None
    # the stash must be reusable (MCTS may unstash from the same object again)
    _play(game, BEST_SEQ[:2])
    game.unstash_state(stash)
    assert (game.state == state0).all()


def test_clone_independence():
    game = SymRegGame()
    game.reset_wrapper()
    clone = game.clone()
    _play(clone, BEST_SEQ)
    assert game.real_state_len == 1
    assert clone.real_state_len == 14


def test_fixed_constants_reward_deterministic():
    game = SymRegGame(problem_seed=3)
    game.reset_wrapper(seed=0)
    r1, term1 = _play(game, BEST_SEQ)
    game.reset_wrapper(seed=99)  # seed must not matter in fixed mode
    r2, term2 = _play(game, BEST_SEQ)
    assert term1 and term2
    assert r1[-1] == r2[-1]


def test_redraw_constants_varies_problem():
    game = SymRegGame(redraw_constants=True, problem_seed=3)
    game.reset_wrapper(seed=1)
    ys1 = game.exact_ys.copy()
    game.reset_wrapper(seed=2)
    ys2 = game.exact_ys.copy()
    assert not np.allclose(ys1, ys2)
    # cache is invalidated on redraw
    assert game._fit_cache == {}


def test_fit_cache_reused():
    game = SymRegGame(problem_seed=5)
    game.reset_wrapper()
    _play(game, BEST_SEQ)
    assert len(game._fit_cache) == 1
    game.reset_wrapper()
    _play(game, BEST_SEQ)
    assert len(game._fit_cache) == 1  # same rule, cached, not refit
