"""Action-space, transposition, terminal-handling, and structural-invariant
tests for ``sraz.core.mcts``.

Ten tests over the shared synthetic environments in ``tests/mcts_envs.py``:

1.  grid_bandit_finds_best_cell — MultiDiscrete 3x3 bandit: probs shape,
    masked cell stays at 0, argmax cell and best-Q action are the true best
    cell (1, 2), and every visited cell's Q exactly equals its reward.
2.  grid_ucbs_masked_neg_inf — calc_masked_ucbs multi-dim fallback: -inf at
    the illegal cell, finite everywhere legal.
3.  grid_counts_consistency — root.total_N == sum(root.action_N.values())
    == n_simulations (the pre-loop root expansion backs up no edge).
4.  transposition_shares_nodes — make_transposition(3) builds exactly the 10
    reachable (z, o) states, interior states shared across orderings.
5.  dead_end_raises — expanding an all-zero-mask state raises
    ValueError("Action mask says no valid moves").
6.  truncation_treated_as_terminal — truncated=True episodes end the search
    exactly like terminated=True: terminal node, exact Q at the root.
7.  total_N_invariant_across_envs — total_N == sum(action_N.values()) at
    every nonterminal expanded node, across five different environments.
8.  game_state_restored_after_search — hashable_obs and step_count are
    identical before/after perform_simulations.
9.  terminal_nodes_have_no_edges — terminal nodes carry direct_reward but
    have empty action_N / action_Q.
10. policy_original_untouched_without_noise — without Dirichlet noise,
    nn_policy stays bitwise-equal to nn_policy_original.

All searches here are deterministic (no noise, no rollouts), so no RNG
seeding is required.
"""

from __future__ import annotations

import numpy as np
import pytest

from sraz.core.mcts import MCTS
from tests.helpers.mcts_envs import (
    GridBanditGame,
    UniformNet,
    make_bandit,
    make_binary_tree,
    make_chain,
    make_dead_end,
    make_delayed_trap,
    make_transposition,
    make_truncation,
)


# ---------------------------------------------------------------------------
# Shared grid-bandit scenario
# ---------------------------------------------------------------------------

GRID_REWARDS = np.array([
    [0.0, 0.1, 0.0],
    [0.2, 0.0, 1.0],
    [0.0, 0.3, 0.0],
])
GRID_MASK = np.array([
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
])
GRID_BEST_CELL = (1, 2)  # argmax of GRID_REWARDS over legal cells


def _grid_mcts(n_simulations: int, backup_rule: str = "max"):
    """Fresh GridBanditGame + UniformNet + MCTS (backup_rule="max" so Q values
    are exact maxima of identical backups, immune to mean-rounding)."""
    game = GridBanditGame(GRID_REWARDS, GRID_MASK)
    net = UniformNet((3, 3))
    mcts = MCTS(game, net, n_simulations=n_simulations, temperature=1.0,
                c_exploration=1.0, backup_rule=backup_rule)
    return game, mcts


# ---------------------------------------------------------------------------
# 1-3: MultiDiscrete action space (multi-dimensional fallback path)
# ---------------------------------------------------------------------------


def test_grid_bandit_finds_best_cell():
    """3x3 one-step bandit with (0,0) illegal: after ~400 sims the visit
    distribution and the Q table both single out the true best cell (1,2),
    and each visited cell's Q equals its reward exactly (one-step game)."""
    game, mcts = _grid_mcts(n_simulations=400)
    root_key = game.hashable_obs
    probs = mcts.perform_simulations(None)

    assert probs.shape == (3, 3), f"probs shape {probs.shape} != (3, 3)"
    assert np.isclose(probs.sum(), 1.0), f"probs do not sum to 1: {probs}"
    assert probs[0, 0] == 0.0, f"illegal cell (0,0) got mass {probs[0, 0]}"

    argmax_cell = np.unravel_index(np.argmax(probs), probs.shape)
    assert tuple(int(x) for x in argmax_cell) == GRID_BEST_CELL, (
        f"argmax cell {argmax_cell} != {GRID_BEST_CELL}; probs\n{probs}"
    )

    root = mcts.nodes[root_key]
    best_q_action = max(root.action_Q, key=root.action_Q.get)
    assert tuple(int(x) for x in best_q_action) == GRID_BEST_CELL, (
        f"best-Q action {best_q_action} != {GRID_BEST_CELL}; "
        f"action_Q={root.action_Q}"
    )

    # One-step game + max backup rule: Q(cell) is exactly its reward.
    for action, q in root.action_Q.items():
        expected = float(GRID_REWARDS[action])
        assert q == expected, (
            f"Q{tuple(int(x) for x in action)}={q} != reward {expected}"
        )


def test_grid_ucbs_masked_neg_inf():
    """After a grid search, calc_masked_ucbs (2D mask ⇒ multi-dimensional
    fallback path) is -inf at the illegal cell (0,0) and finite at every
    legal cell."""
    game, mcts = _grid_mcts(n_simulations=100)
    mcts.perform_simulations(None)
    root = mcts.nodes[game.hashable_obs]

    ucbs = mcts.calc_masked_ucbs(root, None)
    assert ucbs.shape == (3, 3), f"ucbs shape {ucbs.shape} != (3, 3)"
    assert np.isneginf(ucbs[0, 0]), f"ucbs[0,0] should be -inf, got {ucbs[0, 0]}"
    for cell in zip(*np.nonzero(GRID_MASK)):
        assert np.isfinite(ucbs[cell]), (
            f"ucbs at legal cell {cell} should be finite, got {ucbs[cell]}"
        )


def test_grid_counts_consistency():
    """The pre-loop root expansion backs up no edge (total_N stays 0); each of
    the N simulations then backs up exactly one root edge. Hence
    root.total_N == sum(root.action_N.values()) == n_simulations."""
    n_sims = 200
    game, mcts = _grid_mcts(n_simulations=n_sims)
    mcts.perform_simulations(None)
    root = mcts.nodes[game.hashable_obs]

    sum_action_n = sum(root.action_N.values())
    assert root.total_N == sum_action_n, (
        f"root.total_N={root.total_N} != sum(action_N)={sum_action_n}"
    )
    assert root.total_N == n_sims, (
        f"root.total_N={root.total_N} != n_simulations={n_sims}"
    )


# ---------------------------------------------------------------------------
# 4: transpositions share nodes
# ---------------------------------------------------------------------------


def test_transposition_shares_nodes():
    """State = (n_zeros, n_ones): different action orderings collide on one
    node, so a depth-3 search builds exactly (3+1)(3+2)/2 == 10 nodes (an
    unshared binary tree would need 15)."""
    depth = 3
    game = make_transposition(depth)
    net = UniformNet(2)
    mcts = MCTS(game, net, n_simulations=300)
    mcts.perform_simulations(None)

    n_reachable = (depth + 1) * (depth + 2) // 2  # == 10
    assert len(mcts.nodes) == n_reachable, (
        f"expected {n_reachable} shared nodes, got {len(mcts.nodes)}: "
        f"{sorted(mcts.nodes)}"
    )

    # The canonical transposition state (1,1) — reachable via 01 and 10 —
    # is present and interior.
    assert (1, 1) in mcts.nodes, f"(1,1) missing from {sorted(mcts.nodes)}"
    assert not mcts.nodes[(1, 1)].is_terminal_state, (
        "(1,1) at depth 2 < 3 must be nonterminal"
    )

    # All four terminal states appear and are flagged terminal.
    for z in range(depth + 1):
        state = (z, depth - z)
        assert state in mcts.nodes, f"terminal {state} missing"
    for state, node in mcts.nodes.items():
        z, o = state
        if z + o == depth:
            assert node.is_terminal_state, (
                f"state {state} at full depth should be terminal"
            )
        else:
            assert not node.is_terminal_state, (
                f"state {state} at depth {z + o} < {depth} should be nonterminal"
            )


# ---------------------------------------------------------------------------
# 5-6: dead ends and truncation
# ---------------------------------------------------------------------------


def test_dead_end_raises():
    """The root expands fine, but the second simulation descends into the
    nonterminal all-zero-mask state 'pit' and query_net_masked raises."""
    game = make_dead_end()
    net = UniformNet(1)
    mcts = MCTS(game, net, n_simulations=5)
    with pytest.raises(ValueError, match="no valid moves"):
        mcts.perform_simulations(None)


def test_truncation_treated_as_terminal():
    """An episode ending via truncated=True is handled exactly like
    termination: the 'cutoff' node is terminal with direct_reward == 0.75,
    and Q(root, 0) == 0.75 exactly (max backup of identical values)."""
    reward = 0.75
    game = make_truncation(reward)
    net = UniformNet(1)
    mcts = MCTS(game, net, n_simulations=10, backup_rule="max")
    mcts.perform_simulations(None)

    root = mcts.nodes["root"]
    assert root.action_Q[(0,)] == reward, (
        f"Q(root, (0,))={root.action_Q[(0,)]} != {reward}"
    )

    cutoff = mcts.nodes["cutoff"]
    assert cutoff.is_terminal_state is True, (
        f"'cutoff' is_terminal_state={cutoff.is_terminal_state}, expected True"
    )
    assert cutoff.direct_reward == reward, (
        f"'cutoff' direct_reward={cutoff.direct_reward} != {reward}"
    )


# ---------------------------------------------------------------------------
# 7: total_N invariant across environments
# ---------------------------------------------------------------------------

ENV_NET_CASES = [
    pytest.param(lambda: make_bandit([0.2, 1.0, -0.5]),
                 lambda: UniformNet(3), id="bandit"),
    pytest.param(lambda: make_binary_tree(
                     3, [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]),
                 lambda: UniformNet(2), id="binary_tree"),
    pytest.param(make_delayed_trap, lambda: UniformNet(2), id="delayed_trap"),
    pytest.param(lambda: make_transposition(3),
                 lambda: UniformNet(2), id="transposition"),
    pytest.param(lambda: GridBanditGame(GRID_REWARDS, GRID_MASK),
                 lambda: UniformNet((3, 3)), id="grid_bandit"),
]


@pytest.mark.parametrize("game_factory, net_factory", ENV_NET_CASES)
def test_total_N_invariant_across_envs(game_factory, net_factory):
    """After ~100 sims in any environment, every nonterminal expanded node
    satisfies total_N == sum(action_N.values()).

    Note: 'expanded' (nn_policy populated) counts only nonterminal nodes —
    one-step games (bandit, grid_bandit) expand exactly one node (the root),
    so the breadth check is on nodes *created* (>= 2), with >= 1 expanded."""
    game = game_factory()
    net = net_factory()
    mcts = MCTS(game, net, n_simulations=100)
    mcts.perform_simulations(None)

    expanded = 0
    for state, node in mcts.nodes.items():
        if node.is_terminal_state:
            continue  # terminal nodes carry no edges (see test 9)
        if node.nn_policy is None:
            continue  # reached but unexpanded — tolerate
        expanded += 1
        sum_action_n = sum(node.action_N.values())
        assert node.total_N == sum_action_n, (
            f"node {state!r}: total_N={node.total_N} != "
            f"sum(action_N)={sum_action_n}"
        )
    assert expanded >= 1, "search should have expanded at least the root"
    assert len(mcts.nodes) >= 2, (
        f"search should have created >= 2 nodes, got {len(mcts.nodes)}"
    )


# ---------------------------------------------------------------------------
# 8: game state restoration
# ---------------------------------------------------------------------------


def test_game_state_restored_after_search():
    """perform_simulations must leave the game exactly where it found it:
    same hashable_obs, same step_count."""
    cases = [
        (make_chain([1.0, 2.0, 3.0]), UniformNet(1)),
        (GridBanditGame(GRID_REWARDS, GRID_MASK), UniformNet((3, 3))),
    ]
    for game, net in cases:
        mcts = MCTS(game, net, n_simulations=100)
        obs_before = game.hashable_obs
        steps_before = game.step_count
        mcts.perform_simulations(None)
        assert game.hashable_obs == obs_before, (
            f"hashable_obs changed: {obs_before!r} -> {game.hashable_obs!r}"
        )
        assert game.step_count == steps_before, (
            f"step_count changed: {steps_before} -> {game.step_count}"
        )


# ---------------------------------------------------------------------------
# 9: terminal nodes carry rewards but no edges
# ---------------------------------------------------------------------------


def test_terminal_nodes_have_no_edges():
    """In make_bandit each arm k has its own terminal state ('arm', k):
    each terminal node has empty action_N / action_Q and direct_reward equal
    to the arm reward that created it."""
    rewards = [0.2, 1.0, -0.5]
    game = make_bandit(rewards)
    net = UniformNet(3)
    mcts = MCTS(game, net, n_simulations=50)
    mcts.perform_simulations(None)

    terminal_nodes = {s: n for s, n in mcts.nodes.items()
                      if n.is_terminal_state}
    assert len(terminal_nodes) == len(rewards), (
        f"expected all {len(rewards)} arms visited, got terminal states "
        f"{sorted(terminal_nodes)}"
    )
    for state, node in terminal_nodes.items():
        _, k = state
        assert node.action_N == {}, (
            f"terminal {state!r} has edges: action_N={node.action_N}"
        )
        assert node.action_Q == {}, (
            f"terminal {state!r} has edges: action_Q={node.action_Q}"
        )
        assert node.direct_reward == rewards[k], (
            f"terminal {state!r}: direct_reward={node.direct_reward} != "
            f"{rewards[k]}"
        )


# ---------------------------------------------------------------------------
# 10: nn_policy untouched without noise
# ---------------------------------------------------------------------------


def test_policy_original_untouched_without_noise():
    """A plain search (add_noise=False) never modifies nn_policy after
    expansion, so it stays bitwise-equal to nn_policy_original."""
    game = make_delayed_trap()
    net = UniformNet(2)
    mcts = MCTS(game, net, n_simulations=100)
    mcts.perform_simulations(None)

    root = mcts.nodes["root"]
    assert root.nn_policy_original is not None, "root was never expanded"
    assert np.array_equal(root.nn_policy, root.nn_policy_original), (
        f"nn_policy {root.nn_policy} diverged from original "
        f"{root.nn_policy_original} without noise"
    )
