"""Grammar compilation and derivation-MDP mechanics (no torch needed)."""

import numpy as np
import pytest

from sraz.instances.symreg import game as symreg_game
from sraz.instances.symreg.game import SymRegGame, compile_grammar, SR_GRAMMAR


def act(g, pos, prod):
    return pos * g.nprods + prod


def test_compile_grammar_artifacts():
    g = compile_grammar(SR_GRAMMAR)
    assert g.nsym == 11
    assert g.pad_tok == 11
    assert g.nprods == 7
    assert g.tokenlist[0] == "S"
    assert g.nonterms == {g.symdict["S"]}
    assert g.proddict[g.symdict["S"]] == list(range(7))
    # every non-S symbol is a terminal
    for sym, tok in g.symdict.items():
        if sym != "S":
            assert tok not in g.nonterms


def test_reset_obs_in_space():
    game = SymRegGame()
    obs, info = game.reset_wrapper()
    assert obs in game.observation_space
    assert obs[0] == game.grammar.symdict["S"]
    assert (obs[1:] == game.grammar.pad_tok).all()
    assert game.real_state_len == 1


def test_mask_at_reset():
    game = SymRegGame()
    game.reset_wrapper()
    mask = game.get_action_mask()
    assert mask.shape == (105,)
    assert mask.sum() == 7
    # all valid actions are at position 0
    assert set(np.nonzero(mask)[0] // game.grammar.nprods) == {0}


def test_splice_step():
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    obs, r, term, trunc, info = game.step_wrapper(act(g, 0, 0))  # S -> + S S
    S, plus = g.symdict["S"], g.symdict["+"]
    assert list(obs[:3]) == [plus, S, S]
    assert game.real_state_len == 3
    assert r == 0.0 and not term


def test_full_derivation_of_best_reachable_form():
    # + [* C3 sin * C4 x] [+ C0 [* C2 * x x]] -- 14 tokens, the best form
    # reachable under max_len=15 (the true 18-token target does not fit)
    game = SymRegGame(problem_seed=42)
    g = game.grammar
    game.reset_wrapper()
    seq = [(0, 0), (1, 6), (7, 0), (8, 1), (9, 3)]
    for pos, prod in seq:
        a = act(g, pos, prod)
        assert game.get_action_mask()[a], f"action {(pos, prod)} unexpectedly masked"
        obs, r, term, trunc, info = game.step_wrapper(a)
    assert term
    assert info["rule"] == "+ * C3 sin * C4 x + C0 * C2 * x x"
    assert game.real_state_len == 14
    assert r > 0.98  # measured 0.9874 for problem_seed=42


def test_no_dead_ends():
    # keep growing with S -> + S S until masked out; the mask must never be
    # all-zero (the len-1 production C0 is always applicable)
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    grow = g.proddict[g.symdict["S"]][0]  # + S S
    term = False
    for _ in range(50):
        mask = game.get_action_mask()
        assert mask.any(), "dead-end state reached"
        grow_actions = [a for a in np.nonzero(mask)[0] if a % g.nprods == grow]
        a = grow_actions[0] if grow_actions else int(np.nonzero(mask)[0][0])
        _, _, term, _, _ = game.step_wrapper(a)
        if term:
            break
    assert term


def test_invalid_action_fails_closed_without_fit(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("fit_expression must not be called on the guard path")
    monkeypatch.setattr(symreg_game, "fit_expression", boom)

    # invalid: position 1 holds the pad token at reset
    game = SymRegGame()
    g = game.grammar
    game.reset_wrapper()
    obs, r, term, trunc, info = game.step_wrapper(act(g, 1, 0))
    assert r == -1.0 and term and info["rule"] is None

    # overflow: fabricate a nearly-full state, then apply the 6-token production
    game.reset_wrapper()
    game.real_state_len = 13
    obs, r, term, trunc, info = game.step_wrapper(act(g, 0, 6))  # 13+5 > 15
    assert r == -1.0 and term and info["rule"] is None
