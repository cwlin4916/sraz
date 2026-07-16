"""Unit-level SymRegGame internals: stepping edge cases, mask/grammar
consistency, stash/clone fidelity mid-episode, reward determinism, and
hashable_obs stability.

Complements tests/test_grammar_env.py (grammar compilation + basic MDP
mechanics), tests/test_symreg_game_contract.py (Game ABC contract), and
tests/test_sr_eval.py (the SR evaluator itself).
"""

import numpy as np
import pytest

from sraz.instances.symreg.game import (
    C_MAX, C_MIN, SymRegGame, fit_expression, target_ys,
)


def act(g, pos, prod):
    return pos * g.nprods + prod


def first_nonterm_pos(game):
    g = game.grammar
    return next(i for i in range(game.real_state_len)
                if int(game.state[i]) in g.nonterms)


def grow_to(game, target_len):
    """Apply S -> + S S at the first nonterminal until the buffer holds
    target_len tokens (each application adds 2)."""
    g = game.grammar
    while game.real_state_len < target_len:
        game.step_wrapper(act(g, first_nonterm_pos(game), 0))
    assert game.real_state_len == target_len
    return game


# ---------------------------------------------------------------------------
# Stepping edge cases
# ---------------------------------------------------------------------------

def test_reset_and_step_return_state_copies():
    game = SymRegGame()
    g = game.grammar
    obs, _ = game.reset_wrapper()
    obs[0] = 99  # mutating the returned observation must not corrupt the game
    assert game.state[0] == g.symdict["S"]

    obs2, *_ = game.step_wrapper(act(g, 0, 0))  # S -> + S S
    obs2[0] = 99
    assert game.state[0] == g.symdict["+"]


def test_illegal_step_preserves_state():
    game = SymRegGame()
    game.reset_wrapper()
    before = game.state.copy()
    # position 1 holds the pad token at reset -> guard path
    obs, r, term, trunc, info = game.step_wrapper(act(game.grammar, 1, 0))
    assert r == -1.0 and term is True and trunc is False
    assert info == {"rule": None}
    assert np.array_equal(obs, before)
    assert np.array_equal(game.state, before)
    assert game.real_state_len == 1


def test_step_after_terminal_fails_closed():
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    _, r, term, _, info = game.step_wrapper(act(g, 0, 1))  # S -> C0
    assert term and info["rule"] == "C0"
    # constant fits the mean: R^2 = 0 up to fit round-off
    assert r == pytest.approx(0.0, abs=1e-12)
    after = game.state.copy()
    assert not game.get_action_mask().any()  # terminal states offer no action

    _, r2, term2, trunc2, info2 = game.step_wrapper(act(g, 0, 1))
    assert r2 == -1.0 and term2 is True and trunc2 is False
    assert info2 == {"rule": None}
    assert np.array_equal(game.state, after)
    assert game.real_state_len == 1  # guard path leaves the derivation intact
    assert game.terminated is True


def test_truncated_flag_lifecycle():
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    assert game.truncated is None  # reset_wrapper clears the flag to None
    _, _, term, trunc, _ = game.step_wrapper(act(g, 0, 0))   # [+ S S]
    assert term is False and trunc is False and game.truncated is False
    _, _, term, trunc, _ = game.step_wrapper(act(g, 1, 1))   # [+ C0 S]
    assert trunc is False
    _, _, term, trunc, _ = game.step_wrapper(act(g, 2, 1))   # [+ C0 C0]
    assert term is True and trunc is False and game.truncated is False


# ---------------------------------------------------------------------------
# Action mask consistency with the grammar state
# ---------------------------------------------------------------------------

def test_action_mask_flattening_matches_step_decoding():
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    game.step_wrapper(act(g, 0, 0))  # [+ S S]
    game.step_wrapper(act(g, 1, 2))  # [+ * C1 x S]: single S at position 4
    mask = game.get_action_mask()
    # exactly the flat indices of position 4 are legal (all 7 productions fit)
    assert set(np.nonzero(mask)[0]) == set(range(4 * g.nprods, 5 * g.nprods))
    # stepping one of those flat indices really applies at position 4
    _, r, term, _, info = game.step_wrapper(4 * g.nprods + 1)  # S -> C0
    assert term and info["rule"] == "+ * C1 x C0"
    assert r == fit_expression("+ * C1 x C0", game.xs, game.exact_ys)


def test_mask_length_boundary_per_production():
    game = SymRegGame()  # max_len = 15
    g = game.grammar
    game.reset_wrapper()
    for pos, prod in [(0, 0), (1, 6), (7, 0), (8, 0)]:
        game.step_wrapper(act(g, pos, prod))
    assert game.real_state_len == 12  # S at positions 9, 10, 11
    assert [len(g.productions[j]) for j in range(g.nprods)] == [3, 1, 3, 5, 3, 3, 6]
    mask = game.get_action_mask()
    # at 12 tokens: RHS of length 1 or 3 lands at <= 14 tokens (allowed),
    # length 5 or 6 would need 16/17 (masked)
    for j in (0, 1, 2, 4, 5):
        assert mask[act(g, 9, j)]
    for j in (3, 6):
        assert not mask[act(g, 9, j)]

    game.step_wrapper(act(g, 9, 0))  # -> 14 tokens, S at positions 10..13
    assert game.real_state_len == 14
    on = np.nonzero(game.get_action_mask())[0]
    assert len(on) == 4
    assert all(a % g.nprods == 1 for a in on)  # only the unit production S -> C0
    assert sorted(int(a) // g.nprods for a in on) == [10, 11, 12, 13]

    for pos in (10, 11, 12, 13):
        _, r, term, _, info = game.step_wrapper(act(g, pos, 1))
        assert term is (pos == 13)  # only the last rewrite finishes the form
        if not term:
            assert r == 0.0  # intermediate reward is exactly zero
    assert game.real_state_len == game.state_len - 1  # finished forms cap at max_len-1
    assert info["rule"] == "+ * C3 sin * C4 x + + + C0 C0 C0 C0"
    assert r == fit_expression(info["rule"], game.xs, game.exact_ys)


def test_max_length_growth_episode():
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    grow_to(game, 13)  # + + + + + + S S S S S S S
    on = np.nonzero(game.get_action_mask())[0]
    assert len(on) == 7
    assert all(a % g.nprods == 1 for a in on)  # only S -> C0 still fits
    assert sorted(int(a) // g.nprods for a in on) == list(range(6, 13))

    term = False
    for pos in range(6, 13):
        _, r, term, trunc, info = game.step_wrapper(act(g, pos, 1))
        assert game.real_state_len == 13  # unit productions keep the length
        assert trunc is False
        assert term is (pos == 12)  # nonterminals remain until the last rewrite
        if not term:
            assert r == 0.0  # intermediate reward is exactly zero
    assert term
    assert info["rule"] == "+ + + + + + C0 C0 C0 C0 C0 C0 C0"
    # a sum of copies of C0 is a constant model: R^2 = 0 up to fit round-off
    assert r == pytest.approx(0.0, abs=1e-12)


def test_masked_boundary_action_is_accepted_by_step():
    # The mask forbids derivations reaching exactly max_len tokens (`<` in
    # get_action_mask), but step's guard only rejects true buffer overflows
    # (`>` state_len). Forcing the boundary action is therefore accepted and
    # lands in a state whose mask is all-False — a dead end that masked play
    # can never reach. Documented here as current behavior.
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    grow_to(game, 13)
    a = act(g, 6, 0)  # + S S at an S position: 13 -> 15 tokens
    assert not game.get_action_mask()[a]
    _, r, term, trunc, info = game.step_wrapper(a)
    assert r == 0.0 and term is False and trunc is False
    assert info == {"rule": None}
    assert game.real_state_len == game.state_len == 15
    assert not game.get_action_mask().any()  # no production fits any more


# ---------------------------------------------------------------------------
# Stash / unstash / clone fidelity mid-episode
# ---------------------------------------------------------------------------

def test_stash_unstash_mid_episode_full_fidelity():
    game = SymRegGame(problem_seed=1)
    g = game.grammar
    game.reset_wrapper()
    game.step_wrapper(act(g, 0, 0))  # [+ S S]
    game.step_wrapper(act(g, 1, 1))  # [+ C0 S]
    stash = game.stash_state()
    state_mid = game.state.copy()
    obs_mid = np.array(game.obs, copy=True)
    mask_mid = game.get_action_mask().copy()

    _, r_end, term, _, info = game.step_wrapper(act(g, 2, 1))  # [+ C0 C0]
    assert term and info["rule"] == "+ C0 C0"
    assert game.step_count == 3

    ret = game.unstash_state(stash)
    assert ret is game  # MCTS chains on the return value
    assert np.array_equal(game.state, state_mid)
    assert game.real_state_len == 3
    assert np.array_equal(game.obs, obs_mid)
    assert game.reward == 0.0
    assert game.terminated is False
    assert game.truncated is False
    assert game.info == {"rule": None}
    assert game.step_count == 2
    assert np.array_equal(game.get_action_mask(), mask_mid)

    # replaying the same continuation reproduces the terminal transition
    _, r2, term2, _, info2 = game.step_wrapper(act(g, 2, 1))
    assert term2 and info2["rule"] == "+ C0 C0"
    assert r2 == r_end


def test_stash_buffers_isolated_from_game():
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    game.step_wrapper(act(g, 0, 0))
    stash = game.stash_state()
    snap = stash["state"].copy()
    assert stash["state"] is not game.state  # stash copies, no aliasing

    game.step_wrapper(act(g, 1, 0))  # mutate the live game in place
    assert np.array_equal(stash["state"], snap)

    game.unstash_state(stash)
    assert game.state is not stash["state"]  # unstash copies too
    game.step_wrapper(act(g, 1, 0))
    assert np.array_equal(stash["state"], snap)
    game.unstash_state(stash)  # the stash survives a second unstash
    assert np.array_equal(game.state, snap)
    assert game.real_state_len == 3


def test_clone_mid_episode_independence_and_cache():
    game = SymRegGame(problem_seed=4)
    g = game.grammar
    game.reset_wrapper()
    game.step_wrapper(act(g, 0, 0))
    game.step_wrapper(act(g, 1, 2))  # [+ * C1 x S]
    clone = game.clone()
    assert np.array_equal(clone.state, game.state)
    assert clone.real_state_len == game.real_state_len == 5
    assert clone.step_count == game.step_count == 2
    assert clone.hashable_obs == game.hashable_obs

    _, r_clone, term, _, info = clone.step_wrapper(act(g, 4, 1))
    assert term and info["rule"] == "+ * C1 x C0"
    # the original is untouched: derivation state, flags, and fit cache
    assert game.real_state_len == 5
    assert game.terminated is False
    assert game.step_count == 2
    assert game._fit_cache == {}
    assert "+ * C1 x C0" in clone._fit_cache

    # the original playing the same continuation refits and agrees exactly
    _, r_orig, term2, _, _ = game.step_wrapper(act(g, 4, 1))
    assert term2
    assert r_orig == r_clone


# ---------------------------------------------------------------------------
# Reward determinism / problem-instance handling
# ---------------------------------------------------------------------------

def test_reward_deterministic_across_instances():
    g1 = SymRegGame(problem_seed=11)
    g2 = SymRegGame(problem_seed=11)
    assert g1.constants == g2.constants
    assert np.array_equal(g1.exact_ys, g2.exact_ys)

    seq = [(0, 0), (1, 1), (2, 2)]  # -> "+ C0 * C1 x"
    rewards = []
    for game in (g1, g2):
        game.reset_wrapper()
        for pos, prod in seq:
            _, r, term, _, info = game.step_wrapper(act(game.grammar, pos, prod))
        assert term and info["rule"] == "+ C0 * C1 x"
        rewards.append(r)
    # fresh caches on both instances: the lmfit fit itself is deterministic
    assert rewards[0] == rewards[1]
    # and the value is exactly the evaluator's, not some other constant
    assert rewards[0] == fit_expression("+ C0 * C1 x", g1.xs, g1.exact_ys)


def test_unseeded_redraw_advances_rng():
    game = SymRegGame(redraw_constants=True, problem_seed=9)
    c0 = dict(game.constants)
    game.reset_wrapper()  # no seed -> draws from the persistent problem rng
    c1 = dict(game.constants)
    game.reset_wrapper()
    c2 = dict(game.constants)
    assert c1 != c0 and c2 != c1
    for c in (c0, c1, c2):
        assert all(C_MIN <= v <= C_MAX for v in c.values())
    # exact_ys tracks the freshly drawn constants (same function, same inputs:
    # the arrays are bitwise identical, not merely close)
    assert np.array_equal(target_ys(game.xs, game.constants), game.exact_ys)

    # replaying the same number of unseeded resets on a same-seed game
    # reproduces the constant sequence exactly
    other = SymRegGame(redraw_constants=True, problem_seed=9)
    other.reset_wrapper()
    other.reset_wrapper()
    assert other.constants == c2


# ---------------------------------------------------------------------------
# hashable_obs stability
# ---------------------------------------------------------------------------

def test_hashable_obs_stability_and_keying():
    g1 = SymRegGame(problem_seed=0)
    g2 = SymRegGame(problem_seed=0)
    g1.reset_wrapper()
    g2.reset_wrapper()
    h0 = g1.hashable_obs
    assert g1.hashable_obs == h0  # stable across repeated accesses
    assert g2.hashable_obs == h0  # equal states -> equal keys
    hash(h0)  # actually hashable
    table = {h0: "root"}
    assert table[g2.hashable_obs] == "root"

    a = act(g1.grammar, 0, 0)
    g1.step_wrapper(a)
    h1 = g1.hashable_obs
    assert h1 != h0  # state change -> key change
    g2.step_wrapper(a)
    assert g2.hashable_obs == h1  # same trajectory -> same key

    stash = g1.stash_state()
    g1.step_wrapper(act(g1.grammar, 1, 1))
    assert g1.hashable_obs != h1
    g1.unstash_state(stash)
    assert g1.hashable_obs == h1  # unstash restores the key


# ---------------------------------------------------------------------------
# Non-default max_len
# ---------------------------------------------------------------------------

def test_small_max_len_game():
    game = SymRegGame(max_len=4, problem_seed=2)
    g = game.grammar
    assert game.action_space.n == 4 * g.nprods == 28
    assert list(game.observation_space.nvec) == [g.nsym + 1] * 4
    game.reset_wrapper()
    mask = game.get_action_mask()
    assert mask.shape == (28,)
    # RHS of length 5 or 6 can never fit in a 4-slot buffer
    assert [int(a) for a in np.nonzero(mask)[0]] == [act(g, 0, j) for j in (0, 1, 2, 4, 5)]

    _, r, term, trunc, info = game.step_wrapper(act(g, 0, 2))  # S -> * C1 x
    assert term and trunc is False
    assert info["rule"] == "* C1 x"
    assert game.real_state_len == 3 == game.state_len - 1
    assert r == fit_expression("* C1 x", game.xs, game.exact_ys)
