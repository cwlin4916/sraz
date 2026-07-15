"""Rollout-evaluation and direct-query tests for ``sraz.core.mcts``.

Covers ``MCTS._rollout_value`` (nonterminal leaf evaluation via uniform
random game completion) and the ``n_simulations < 0`` direct-policy-query
path in ``perform_simulations``.

Key technique: ``n_simulations=0`` expands only the root node (one net
query, zero simulations), so ``root.nn_value`` directly exposes the
leaf-evaluation result (rollout/net blend) with zero search noise.
``make_chain`` has exactly one legal action per state, so every random
rollout deterministically yields the full path total.

Ten tests:

 1. rollout_replaces_net_value          — blend=0.0: nn_value is the exact rollout return
 2. blend_extremes_and_midpoint         — blend=1.0 → pure net value; blend=0.5 → exact midpoint
 3. budget_zero_falls_back_to_net       — rollout_budget=0: _rollout_value returns None,
                                          nn_value falls back to the net value
 4. budget_truncates_incomplete_rollouts — a rollout cut off mid-episode is excluded from
                                          the aggregate; exact _search_rollout_budget bookkeeping
 5. mean_vs_max_modes                   — rollout_mode="mean" averages, "max" takes the best
 6. rollout_does_not_query_net          — rollouts never call net.predict (root expansion only)
 7. budget_resets_per_search            — _search_rollout_budget refills on every
                                          perform_simulations call
 8. direct_query_negative_sims          — n_simulations=-1 returns the net prior and builds
                                          no tree at all
 9. direct_query_respects_mask          — masked arms get zero mass, the rest renormalise
10. rollout_multidim_action_space       — MultiDiscrete rollouts step via np.unravel_index
"""

from __future__ import annotations

import numpy as np

from sraz.core.mcts import MCTS
from tests.mcts_envs import (
    FixedPolicyNet,
    GridBanditGame,
    TableGame,
    UniformNet,
    make_bandit,
    make_chain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expand_root(mcts: MCTS):
    """Run one ``perform_simulations`` call and return the root node.

    With ``n_simulations=0`` this expands exactly the root (one net query,
    optional rollouts) and runs zero simulations, so the returned node's
    ``nn_value`` is the raw leaf-evaluation result.
    """
    root_hash = mcts.game.hashable_obs
    mcts.perform_simulations(None)
    return mcts.nodes[root_hash]


# ---------------------------------------------------------------------------
# Rollout leaf evaluation (nn_value at the root, n_simulations=0)
# ---------------------------------------------------------------------------


def test_rollout_replaces_net_value():
    """Chain [1,2,3] has a single legal action per state, so every rollout
    returns exactly 6.0. With rollout_blend=0.0 the net value (7.0) is fully
    replaced: root.nn_value == 6.0."""
    np.random.seed(1234)
    game = make_chain([1.0, 2.0, 3.0])
    net = UniformNet(1, value=7.0)
    mcts = MCTS(game, net, n_simulations=0, rollout_n=3, rollout_blend=0.0)

    root = _expand_root(mcts)
    assert float(root.nn_value) == 6.0, (
        f"blend=0.0 should give the pure rollout return 6.0, "
        f"got nn_value={float(root.nn_value)}"
    )


def test_blend_extremes_and_midpoint():
    """leaf_value = (1-blend)*rollout + blend*net. On the chain the rollout
    return is exactly 6.0 and the net value 7.0, so blend=1.0 → 7.0 (pure
    net) and blend=0.5 → 6.5 exactly."""
    np.random.seed(1234)
    for blend, expected in ((1.0, 7.0), (0.5, 6.5)):
        game = make_chain([1.0, 2.0, 3.0])
        net = UniformNet(1, value=7.0)
        mcts = MCTS(game, net, n_simulations=0, rollout_n=3,
                    rollout_blend=blend)
        root = _expand_root(mcts)
        assert float(root.nn_value) == expected, (
            f"blend={blend}: expected nn_value={expected}, "
            f"got {float(root.nn_value)}"
        )


def test_budget_zero_falls_back_to_net():
    """rollout_budget=0: the for-loop breaks before any rollout starts,
    _rollout_value returns None, and nn_value keeps the net value 7.0."""
    np.random.seed(1234)
    game = make_chain([1.0, 2.0, 3.0])
    net = UniformNet(1, value=7.0)
    mcts = MCTS(game, net, n_simulations=0, rollout_n=3, rollout_budget=0,
                rollout_blend=0.0)

    # Direct check of the None signal (game is at the root, budget is 0).
    rv = mcts._rollout_value(None)
    assert rv is None, f"budget=0 should yield no completed rollout, got {rv}"

    root = _expand_root(mcts)
    assert float(root.nn_value) == 7.0, (
        f"with no completed rollout nn_value must fall back to the net "
        f"value 7.0, got {float(root.nn_value)}"
    )


def test_budget_truncates_incomplete_rollouts():
    """Each chain episode costs exactly 3 budget units (one per step).

    - budget=7, rollout_n=5: rollouts 1 and 2 complete (6 units), rollout 3
      is cut off after one step (budget hits 0 mid-episode) and is EXCLUDED
      from the aggregate; rollouts 4-5 never start. nn_value is still the
      exact mean of two complete returns, 6.0. Consumed = 7 → remaining 0.
    - budget=2 (< one episode): the single attempted rollout is cut off, no
      completions, fallback to net value 7.0. Consumed = 2 → remaining 0.
    - budget=8, rollout_n=2: both rollouts complete (6 units), rollout_n is
      then exhausted → remaining budget is a nonzero 8 - 6 = 2.
    """
    np.random.seed(1234)

    # Case 1: partial third rollout excluded, aggregate stays exact.
    game = make_chain([1.0, 2.0, 3.0])
    net = UniformNet(1, value=7.0)
    mcts = MCTS(game, net, n_simulations=0, rollout_n=5, rollout_budget=7,
                rollout_blend=0.0)
    root = _expand_root(mcts)
    assert float(root.nn_value) == 6.0, (
        f"two complete rollouts of 6.0 each; the truncated third must be "
        f"excluded, got nn_value={float(root.nn_value)}"
    )
    assert mcts._search_rollout_budget == 0, (
        f"budget 7 minus 7 consumed steps (3+3+1) should leave 0, "
        f"got {mcts._search_rollout_budget}"
    )

    # Case 2: budget smaller than one episode → no completions → net value.
    game2 = make_chain([1.0, 2.0, 3.0])
    net2 = UniformNet(1, value=7.0)
    mcts2 = MCTS(game2, net2, n_simulations=0, rollout_n=5, rollout_budget=2,
                 rollout_blend=0.0)
    root2 = _expand_root(mcts2)
    assert float(root2.nn_value) == 7.0, (
        f"budget=2 cannot complete a 3-step episode; nn_value must fall "
        f"back to the net value 7.0, got {float(root2.nn_value)}"
    )
    assert mcts2._search_rollout_budget == 0, (
        f"budget 2 minus 2 consumed steps should leave 0, "
        f"got {mcts2._search_rollout_budget}"
    )

    # Case 3: rollout_n exhausted before budget → nonzero remainder.
    game3 = make_chain([1.0, 2.0, 3.0])
    net3 = UniformNet(1, value=7.0)
    mcts3 = MCTS(game3, net3, n_simulations=0, rollout_n=2, rollout_budget=8,
                 rollout_blend=0.0)
    root3 = _expand_root(mcts3)
    assert float(root3.nn_value) == 6.0, (
        f"both rollouts complete within budget 8, "
        f"got nn_value={float(root3.nn_value)}"
    )
    assert mcts3._search_rollout_budget == 2, (
        f"budget 8 minus 2*3 consumed steps should leave 2, "
        f"got {mcts3._search_rollout_budget}"
    )


def test_mean_vs_max_modes():
    """Bandit [0.0, 1.0]: each rollout is one uniform-random arm pull.
    mode="mean" over 30 pulls lands strictly between the arm values;
    mode="max" is exactly 1.0 once any pull hits arm 1 (seeded, 30 pulls)."""
    np.random.seed(0)
    game = make_bandit([0.0, 1.0])
    net = UniformNet(2, value=-5.0)  # sentinel: rollout must replace this
    mcts = MCTS(game, net, n_simulations=0, rollout_n=30,
                rollout_mode="mean", rollout_blend=0.0)
    root = _expand_root(mcts)
    assert 0.2 < float(root.nn_value) < 0.8, (
        f"mean of 30 uniform pulls over rewards (0, 1) should lie in "
        f"(0.2, 0.8), got {float(root.nn_value)}"
    )

    np.random.seed(0)
    game_max = make_bandit([0.0, 1.0])
    net_max = UniformNet(2, value=-5.0)
    mcts_max = MCTS(game_max, net_max, n_simulations=0, rollout_n=30,
                    rollout_mode="max", rollout_blend=0.0)
    root_max = _expand_root(mcts_max)
    assert float(root_max.nn_value) == 1.0, (
        f"max over 30 seeded pulls must hit the 1.0 arm at least once, "
        f"got {float(root_max.nn_value)}"
    )


def test_rollout_does_not_query_net():
    """Rollouts pick uniform random actions via the game interface only;
    the sole net query is the root expansion itself."""
    np.random.seed(1234)
    game = make_chain([1.0, 2.0, 3.0])
    net = UniformNet(1, value=7.0)
    mcts = MCTS(game, net, n_simulations=0, rollout_n=5, rollout_blend=0.0)
    _expand_root(mcts)
    assert net.n_predict_calls == 1, (
        f"only the root expansion should query the net, "
        f"got n_predict_calls={net.n_predict_calls}"
    )


def test_budget_resets_per_search():
    """_search_rollout_budget is refilled to rollout_budget at the top of
    every perform_simulations call. First call: root expansion runs 2
    rollouts x 3 steps → 10 - 6 = 4 left. Second call: the root already
    exists, no expansion and no rollouts, so the budget stays at the fresh
    full value 10 — proving the reset happened."""
    np.random.seed(1234)
    game = make_chain([1.0, 2.0, 3.0])
    net = UniformNet(1, value=7.0)
    mcts = MCTS(game, net, n_simulations=0, rollout_n=2, rollout_budget=10,
                rollout_blend=0.0)

    mcts.perform_simulations(None)
    assert mcts._search_rollout_budget == 4, (
        f"first call: budget 10 minus 2 rollouts * 3 steps should leave 4, "
        f"got {mcts._search_rollout_budget}"
    )

    mcts.perform_simulations(None)
    assert mcts._search_rollout_budget == 10, (
        f"second call: root already expanded, budget must be reset to the "
        f"full 10, got {mcts._search_rollout_budget}"
    )


# ---------------------------------------------------------------------------
# Direct-query path (n_simulations < 0)
# ---------------------------------------------------------------------------


def test_direct_query_negative_sims():
    """n_simulations=-1 skips the tree entirely: the returned probs are the
    (masked, temperature-1) net prior and no node is ever created."""
    game = make_bandit([0.0, 1.0])
    net = FixedPolicyNet([0.7, 0.3])
    mcts = MCTS(game, net, n_simulations=-1, temperature=1.0)

    probs = mcts.perform_simulations(None)
    assert probs.shape == (2,), f"expected shape (2,), got {probs.shape}"
    assert np.allclose(probs, [0.7, 0.3]), (
        f"direct query should return the net prior [0.7, 0.3], got {probs}"
    )
    assert len(mcts.nodes) == 0, (
        f"no tree should be built on the direct-query path, "
        f"got {len(mcts.nodes)} nodes: {list(mcts.nodes)}"
    )


def test_direct_query_respects_mask():
    """Only arms {0, 2} exist; the prior [0.5, 0.3, 0.2] must be masked to
    [0.5, 0, 0.2] and renormalised to [5/7, 0, 2/7]."""
    transitions = {
        "root": {0: ("a", 0.0, True, False), 2: ("b", 0.0, True, False)},
    }
    game = TableGame(transitions, "root", n_actions=3)
    net = FixedPolicyNet([0.5, 0.3, 0.2])
    mcts = MCTS(game, net, n_simulations=-1, temperature=1.0)

    probs = mcts.perform_simulations(None)
    expected = np.array([0.5 / 0.7, 0.0, 0.2 / 0.7])
    assert np.allclose(probs, expected), (
        f"masked direct query should renormalise over legal arms: "
        f"expected {expected}, got {probs}"
    )
    assert probs[1] == 0.0, (
        f"masked arm 1 must get exactly zero mass, got {probs[1]}"
    )
    assert len(mcts.nodes) == 0, (
        f"no tree should be built on the direct-query path, "
        f"got {len(mcts.nodes)} nodes: {list(mcts.nodes)}"
    )


# ---------------------------------------------------------------------------
# Multi-dimensional action spaces in rollouts
# ---------------------------------------------------------------------------


def test_rollout_multidim_action_space():
    """GridBanditGame has a MultiDiscrete action space, so _rollout_value
    must step with np.unravel_index tuples. All cells reward 1.0, so both
    rollouts return exactly 1.0 regardless of the sampled cell."""
    np.random.seed(1234)
    game = GridBanditGame(np.ones((2, 2)))
    net = UniformNet((2, 2), value=0.0)
    mcts = MCTS(game, net, n_simulations=0, rollout_n=2, rollout_blend=0.0)

    root = _expand_root(mcts)
    assert float(root.nn_value) == 1.0, (
        f"every 2x2 cell rewards 1.0, so the rollout mean must be exactly "
        f"1.0, got {float(root.nn_value)}"
    )
