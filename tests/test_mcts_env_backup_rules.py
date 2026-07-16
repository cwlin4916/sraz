"""Backup-rule and Q-normalization tests for ``sraz.core.mcts``.

Covers ``MCTS.update_edge`` (mean/max/topk/softmax backup rules, edge
bookkeeping, global q_min/q_max tracking) and the min-max Q normalization
inside ``MCTS.calc_masked_ucbs`` (both the 1D fast path and the
multi-dimensional fallback), using the synthetic environments from
``tests/mcts_envs.py``.

Tests:

1.  test_update_edge_mean_exact — running mean after each update.
2.  test_update_edge_max_exact — running max after each update.
3.  test_update_edge_topk_exact — mean of top-3 values (k=3) after each update.
4.  test_update_edge_softmax_exact — tau-log-mean-exp formula after each update.
5.  test_softmax_between_mean_and_max — mean < softmaxQ < max for a
    non-constant value list; large-tau limit approaches the mean.
6.  test_topk_fewer_than_k_equals_mean — 2 values with k=3 gives their mean.
7.  test_update_edge_bookkeeping — action_N / action_values / total_N /
    q_min / q_max bookkeeping semantics of update_edge.
8.  test_constructor_rejects_unknown_rule — backup_rule="median" asserts.
9.  test_ucb_qnorm_regimes — the three q_norm regimes (infinite init bounds,
    q_min == q_max, spread bounds) on a hand-built node, asserted bit-exactly
    on the 1D fast path.
10. test_ucb_qnorm_regimes_2d_fallback — the same three q_norm regimes
    asserted bit-exactly on the multi-dimensional fallback, plus a
    masked-cell variant.
11. test_ucb_masked_entries_neg_inf_1d_and_2d — masked positions are -inf in
    the 1D fast path and the multi-dimensional fallback.
12. test_chain_exact_total_integration — backup_rule="max" wiring reaches
    search(): root Q equals the exact chain return.
"""

from __future__ import annotations

import numpy as np
import pytest

from sraz.core.mcts import EPS, MCTS, MCTSTreeNode
from tests.mcts_envs import (
    GridBanditGame,
    UniformNet,
    make_bandit,
    make_chain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_mcts(**kwargs) -> MCTS:
    """MCTS over a tiny bandit; the game/net are inert scaffolding for
    unit-level calls to ``update_edge`` / ``calc_masked_ucbs``."""
    game = make_bandit([0.0, 1.0])
    net = UniformNet(2)
    return MCTS(game, net, **kwargs)


def _softmax_q(values: list[float], tau: float) -> float:
    """Independent recomputation of the softmax backup formula."""
    v = np.array(values, dtype=np.float64)
    return float(v.max() + tau * np.log(np.mean(np.exp((v - v.max()) / tau))))


# The shared value sequence for the per-rule exactness tests.
VALUE_SEQUENCE = [0.0, 1.0, 0.5, 0.5]


def _feed_and_collect(mcts: MCTS, values: list[float]) -> tuple[MCTSTreeNode, list[float]]:
    """Feed ``values`` one at a time through ``update_edge`` on a fresh node,
    recording ``action_Q[(0,)]`` after each update."""
    node = MCTSTreeNode(0.0, False)
    qs = []
    for v in values:
        mcts.update_edge(node, (0,), v)
        qs.append(node.action_Q[(0,)])
    return node, qs


# ---------------------------------------------------------------------------
# update_edge: per-rule exactness
# ---------------------------------------------------------------------------


def test_update_edge_mean_exact():
    """mean rule: Q after each update equals the running mean."""
    mcts = _fresh_mcts(backup_rule="mean")
    _, qs = _feed_and_collect(mcts, VALUE_SEQUENCE)
    for i, q in enumerate(qs):
        prefix = VALUE_SEQUENCE[: i + 1]
        expected = sum(prefix) / len(prefix)
        assert q == expected, (
            f"mean rule after {i + 1} updates: Q={q}, expected {expected} "
            f"(values so far {prefix})"
        )


def test_update_edge_max_exact():
    """max rule: Q after each update equals the running max."""
    mcts = _fresh_mcts(backup_rule="max")
    _, qs = _feed_and_collect(mcts, VALUE_SEQUENCE)
    for i, q in enumerate(qs):
        prefix = VALUE_SEQUENCE[: i + 1]
        expected = max(prefix)
        assert q == expected, (
            f"max rule after {i + 1} updates: Q={q}, expected {expected} "
            f"(values so far {prefix})"
        )


def test_update_edge_topk_exact():
    """topk rule (k=3): Q equals the mean of the top-3 values so far
    (mean of all values while fewer than 3 have been backed up)."""
    mcts = _fresh_mcts(backup_rule="topk", backup_topk=3)
    _, qs = _feed_and_collect(mcts, VALUE_SEQUENCE)
    for i, q in enumerate(qs):
        prefix = VALUE_SEQUENCE[: i + 1]
        top = sorted(prefix, reverse=True)[:3]
        expected = sum(top) / len(top)
        assert q == expected, (
            f"topk rule after {i + 1} updates: Q={q}, expected {expected} "
            f"(top-3 of {prefix} is {top})"
        )


def test_update_edge_softmax_exact():
    """softmax rule (tau=0.1): Q equals
    v.max() + tau * log(mean(exp((v - v.max()) / tau))) after each update."""
    tau = 0.1
    mcts = _fresh_mcts(backup_rule="softmax", backup_tau=tau)
    _, qs = _feed_and_collect(mcts, VALUE_SEQUENCE)
    for i, q in enumerate(qs):
        prefix = VALUE_SEQUENCE[: i + 1]
        expected = _softmax_q(prefix, tau)
        assert np.isclose(q, expected, rtol=0.0, atol=1e-12), (
            f"softmax rule after {i + 1} updates: Q={q}, expected {expected} "
            f"(values so far {prefix})"
        )


# ---------------------------------------------------------------------------
# update_edge: softmax/topk edge properties
# ---------------------------------------------------------------------------


def test_softmax_between_mean_and_max():
    """For a non-constant value list, the softmax Q lies strictly between the
    mean and the max; with a huge tau it collapses to the mean."""
    values = [0.0, 1.0, 0.5, 0.5]
    v_mean = sum(values) / len(values)
    v_max = max(values)

    # Sharp tau: strictly between mean and max.
    mcts = _fresh_mcts(backup_rule="softmax", backup_tau=0.1)
    node, _ = _feed_and_collect(mcts, values)
    q = node.action_Q[(0,)]
    assert v_mean < q < v_max, (
        f"softmax Q={q} not strictly between mean={v_mean} and max={v_max} "
        f"for values {values}"
    )

    # Large-tau limit: within 1e-3 of the mean.
    mcts_flat = _fresh_mcts(backup_rule="softmax", backup_tau=100.0)
    node_flat, _ = _feed_and_collect(mcts_flat, values)
    q_flat = node_flat.action_Q[(0,)]
    assert abs(q_flat - v_mean) < 1e-3, (
        f"large-tau softmax Q={q_flat} should be within 1e-3 of mean={v_mean}, "
        f"diff={abs(q_flat - v_mean)}"
    )


def test_topk_fewer_than_k_equals_mean():
    """With only 2 backed-up values and k=3, the topk Q is the mean of the 2."""
    mcts = _fresh_mcts(backup_rule="topk", backup_topk=3)
    node, _ = _feed_and_collect(mcts, [0.25, 0.75])
    q = node.action_Q[(0,)]
    assert q == 0.5, f"topk with 2 values [0.25, 0.75] and k=3: Q={q}, expected 0.5"


# ---------------------------------------------------------------------------
# update_edge: bookkeeping and constructor validation
# ---------------------------------------------------------------------------


def test_update_edge_bookkeeping():
    """update_edge increments action_N per call, accumulates the raw value
    list in action_values, never touches total_N, and q_min/q_max track the
    running min/max over every edge-Q snapshot (across all edges)."""
    mcts = _fresh_mcts(backup_rule="mean")
    node = MCTSTreeNode(0.0, False)
    values = [0.5, -1.0, 2.0]

    for i, v in enumerate(values):
        mcts.update_edge(node, (0,), v)
        assert node.action_N[(0,)] == i + 1, (
            f"after update {i + 1}: action_N={node.action_N[(0,)]}, expected {i + 1}"
        )
        assert node.action_values[(0,)] == values[: i + 1], (
            f"after update {i + 1}: action_values={node.action_values[(0,)]}, "
            f"expected {values[: i + 1]}"
        )
        assert node.total_N == 0, (
            f"update_edge must not touch total_N, got {node.total_N} after "
            f"update {i + 1} (search() increments it separately)"
        )

    # Mean-rule Q snapshots along the way were 0.5, -0.25, 0.5; q_min/q_max
    # track the running min/max of those snapshots (q_min keeps -0.25 even
    # though no edge currently holds that Q).
    assert node.action_Q[(0,)] == 0.5, (
        f"final mean Q={node.action_Q[(0,)]}, expected 0.5"
    )
    assert mcts.q_min == -0.25, f"q_min={mcts.q_min}, expected -0.25"
    assert mcts.q_max == 0.5, f"q_max={mcts.q_max}, expected 0.5"

    # A second edge participates in the same global min/max tracking.
    mcts.update_edge(node, (1,), 3.0)
    assert mcts.q_max == 3.0, (
        f"q_max should track Q across edges: q_max={mcts.q_max}, expected 3.0"
    )
    assert mcts.q_min == -0.25, (
        f"q_min must be unaffected by the higher edge: q_min={mcts.q_min}"
    )
    assert node.total_N == 0, (
        f"total_N still untouched after second edge, got {node.total_N}"
    )


def test_constructor_rejects_unknown_rule():
    """backup_rule outside (mean, max, topk, softmax) trips the constructor
    assertion."""
    game = make_bandit([0.0, 1.0])
    net = UniformNet(2)
    with pytest.raises(AssertionError, match="Unknown backup_rule"):
        MCTS(game, net, backup_rule="median")


# ---------------------------------------------------------------------------
# calc_masked_ucbs: q_norm regimes and masking
# ---------------------------------------------------------------------------


def _hand_built_node(mask: np.ndarray) -> MCTSTreeNode:
    """Node with uniform prior over 4 actions, edge Qs {0, 1, 2} on actions
    0..2 (action 3 unvisited), N = [2, 3, 5, 0], total_N = 10."""
    node = MCTSTreeNode(0.0, False)
    node.nn_policy = np.full(4, 0.25, dtype=np.float64)
    node.action_mask = mask
    node.action_Q = {(0,): 0.0, (1,): 1.0, (2,): 2.0}
    node.action_N = {(0,): 2, (1,): 3, (2,): 5}
    node.total_N = 10
    return node


def test_ucb_qnorm_regimes():
    """The three q_norm regimes in calc_masked_ucbs (1D fast path), asserted
    bit-exactly (the expected values replicate the float64 operation order of
    the fast path, so any single-op perturbation — e.g. dropping EPS —
    must fail):
    (a) q_min/q_max at their +-inf init values -> Q term contributes 0;
    (b) q_min == q_max -> q_norm 0.5 everywhere;
    (c) spread bounds (0, 2) with edge Qs {0, 1, 2} -> q_norm {0, 0.5, 1}."""
    c = 1.5
    mcts = _fresh_mcts(c_exploration=c)
    node = _hand_built_node(np.ones(4, dtype=bool))

    n_arr = np.array([2.0, 3.0, 5.0, 0.0])
    q_arr = np.array([0.0, 1.0, 2.0, 0.0])  # unvisited action 3 defaults to 0
    u = c * 0.25 * np.sqrt(node.total_N + EPS) / (1.0 + n_arr)

    # (a) Fresh search stats: q_min=inf, q_max=-inf -> q_norm is all zeros,
    # so the UCB is exactly the exploration term.
    assert mcts.q_min == float("inf") and mcts.q_max == float("-inf"), (
        f"fresh MCTS should init q_min/q_max to +-inf, got "
        f"({mcts.q_min}, {mcts.q_max})"
    )
    ucbs_a = mcts.calc_masked_ucbs(node, msg=None)
    np.testing.assert_array_equal(
        ucbs_a, u,
        err_msg="with infinite q bounds the UCB must equal the exploration "
                "term bit-exactly (including the EPS inside sqrt)",
    )

    # (b) Degenerate spread: q_min == q_max -> q_norm is 0.5 everywhere.
    mcts.q_min = 1.0
    mcts.q_max = 1.0
    ucbs_b = mcts.calc_masked_ucbs(node, msg=None)
    np.testing.assert_array_equal(
        ucbs_b, 0.5 + u,
        err_msg="with q_min == q_max the UCB must be exactly "
                "0.5 + exploration term",
    )

    # (c) Spread bounds: q_norm = (Q - q_min) / (q_max - q_min) = {0, 0.5, 1}
    # at the visited actions and 0 at the unvisited one (Q defaults to 0);
    # all four quotients are exact in float64.
    mcts.q_min = 0.0
    mcts.q_max = 2.0
    q_norm = (q_arr - mcts.q_min) / (mcts.q_max - mcts.q_min)
    np.testing.assert_array_equal(
        q_norm, [0.0, 0.5, 1.0, 0.0],
        err_msg="self-check: expected q_norm values must be [0, 0.5, 1, 0]",
    )
    ucbs_c = mcts.calc_masked_ucbs(node, msg=None)
    np.testing.assert_array_equal(
        ucbs_c, q_norm + u,
        err_msg="spread-regime UCB must equal q_norm + exploration term "
                "bit-exactly",
    )


def _hand_built_node_2d(mask: np.ndarray) -> MCTSTreeNode:
    """2D analogue of :func:`_hand_built_node`: non-uniform prior
    [[0.4, 0.3], [0.2, 0.1]] over a 2x2 grid, edge Qs {0, 1, 2} on cells
    (0,0)/(0,1)/(1,0) (cell (1,1) unvisited), N = [[2, 3], [5, 0]],
    total_N = 10. The non-uniform prior makes the exploration term sensitive
    to dropping the prior factor."""
    node = MCTSTreeNode(0.0, False)
    node.nn_policy = np.array([[0.4, 0.3], [0.2, 0.1]], dtype=np.float64)
    node.action_mask = mask
    node.action_Q = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 2.0}
    node.action_N = {(0, 0): 2, (0, 1): 3, (1, 0): 5}
    node.total_N = 10
    return node


def test_ucb_qnorm_regimes_2d_fallback():
    """The three q_norm regimes on the multi-dimensional fallback of
    calc_masked_ucbs, asserted bit-exactly (the expected values replicate the
    per-cell float64 operation order of the fallback loop:
    q_norm + c * P[a] * sqrt(total_N + EPS) / (1 + N[a])):
    (a) infinite init bounds -> UCB is exactly the per-cell exploration term;
    (b) q_min == q_max -> exactly 0.5 + exploration term;
    (c) spread bounds (0, 2) -> q_norm [[0, 0.5], [1, 0]] + exploration term,
        the unvisited cell (1, 1) taking the Q = 0 default.
    A masked variant of (c) checks the masked cell is -inf while legal cells
    keep their exact values."""
    c = 1.5
    mcts = _fresh_mcts(c_exploration=c)
    node = _hand_built_node_2d(np.ones((2, 2), dtype=bool))

    p_grid = node.nn_policy
    n_grid = np.array([[2, 3], [5, 0]])
    # Per-cell exploration term with the fallback's op order:
    # (c * P[a]) * sqrt(total_N + EPS) / (1 + N[a]).
    u = c * p_grid * np.sqrt(node.total_N + EPS) / (1 + n_grid)

    # (a) Fresh search stats: q_norm contributes exactly 0 per cell.
    ucbs_a = mcts.calc_masked_ucbs(node, msg=None)
    assert ucbs_a.shape == (2, 2), (
        f"2D UCB shape {ucbs_a.shape}, expected (2, 2)"
    )
    np.testing.assert_array_equal(
        ucbs_a, u,
        err_msg="2D fallback with infinite q bounds: UCB must equal the "
                "per-cell exploration term bit-exactly",
    )

    # (b) Degenerate spread: q_norm is exactly 0.5 at every legal cell.
    mcts.q_min = 1.0
    mcts.q_max = 1.0
    ucbs_b = mcts.calc_masked_ucbs(node, msg=None)
    np.testing.assert_array_equal(
        ucbs_b, 0.5 + u,
        err_msg="2D fallback with q_min == q_max: UCB must be exactly "
                "0.5 + exploration term",
    )

    # (c) Spread bounds (0, 2): Qs {0, 1, 2, 0-default} -> q_norm
    # [[0, 0.5], [1, 0]], every quotient exact in float64.
    mcts.q_min = 0.0
    mcts.q_max = 2.0
    q_norm = np.array([[0.0, 0.5], [1.0, 0.0]])
    ucbs_c = mcts.calc_masked_ucbs(node, msg=None)
    np.testing.assert_array_equal(
        ucbs_c, q_norm + u,
        err_msg="2D fallback spread-regime UCB must equal "
                "q_norm + exploration term bit-exactly",
    )

    # Masked variant of (c): cell (0, 1) illegal -> exactly -inf there, and
    # bit-identical UCBs at the remaining legal cells.
    node_masked = _hand_built_node_2d(np.array([[1, 0], [1, 1]], dtype=bool))
    expected_masked = (q_norm + u).copy()
    expected_masked[0, 1] = -np.inf
    ucbs_m = mcts.calc_masked_ucbs(node_masked, msg=None)
    np.testing.assert_array_equal(
        ucbs_m, expected_masked,
        err_msg="2D fallback masked variant: masked cell must be -inf and "
                "legal cells bit-identical to the unmasked values",
    )


def test_ucb_masked_entries_neg_inf_1d_and_2d():
    """Masked action positions are exactly -inf in calc_masked_ucbs, on both
    the 1D fast path (hand-built node) and the multi-dimensional fallback
    (GridBanditGame root after a short search)."""
    # --- 1D fast path: mask [1, 0, 1, 0] -> positions 1 and 3 are -inf.
    mcts = _fresh_mcts(c_exploration=1.0)
    node = _hand_built_node(np.array([1, 0, 1, 0], dtype=bool))
    ucbs = mcts.calc_masked_ucbs(node, msg=None)
    assert ucbs.shape == (4,), f"1D UCB shape {ucbs.shape}, expected (4,)"
    assert np.isneginf(ucbs[1]) and np.isneginf(ucbs[3]), (
        f"masked 1D positions must be -inf, got ucbs={ucbs}"
    )
    assert np.all(np.isfinite(ucbs[[0, 2]])), (
        f"legal 1D positions must be finite, got ucbs={ucbs}"
    )

    # --- 2D fallback: GridBanditGame with cell (1, 1) masked out.
    # (Exact UCB values of the fallback are pinned by
    # test_ucb_qnorm_regimes_2d_fallback; this checks the mask wiring through
    # a real search.)
    reward_grid = np.array([[0.1, 0.9], [0.4, 5.0]])
    mask_grid = np.array([[1, 1], [1, 0]])
    game = GridBanditGame(reward_grid, mask_grid)
    net = UniformNet((2, 2))
    mcts2d = MCTS(game, net, n_simulations=12, c_exploration=1.5)
    mcts2d.perform_simulations(None)

    root = mcts2d.nodes[game.hashable_obs]
    ucbs2d = mcts2d.calc_masked_ucbs(root, msg=None)
    assert ucbs2d.shape == (2, 2), f"2D UCB shape {ucbs2d.shape}, expected (2, 2)"
    assert np.isneginf(ucbs2d[1, 1]), (
        f"masked cell (1, 1) must be -inf, got {ucbs2d[1, 1]} (ucbs={ucbs2d})"
    )
    np.testing.assert_array_equal(
        np.isneginf(ucbs2d),
        ~game.mask_grid,
        err_msg=f"-inf pattern must match the inverted mask; ucbs={ucbs2d}",
    )
    assert np.all(np.isfinite(ucbs2d[game.mask_grid])), (
        f"legal cells must be finite, got ucbs={ucbs2d}"
    )


# ---------------------------------------------------------------------------
# Integration: backup rule wiring reaches search()
# ---------------------------------------------------------------------------


def test_chain_exact_total_integration():
    """make_chain([0.5, -0.25, 2.0]) with backup_rule="max": once one
    simulation traverses the full chain, the root Q is exactly the chain
    return 0.5 + (-0.25) + 2.0 = 2.25 (max is immune to the pre-terminal
    partial backups that contaminate the mean)."""
    game = make_chain([0.5, -0.25, 2.0])
    net = UniformNet(1, value=0.0)
    mcts = MCTS(game, net, n_simulations=8, backup_rule="max")
    mcts.perform_simulations(None)

    root = mcts.nodes[0]  # make_chain root state is 0
    q = root.action_Q[(0,)]
    assert q == 2.25, f"max-rule root Q={q}, expected exact chain return 2.25"
    assert root.action_N[(0,)] == 8, (
        f"single-action chain: all 8 simulations should visit the edge, "
        f"got N={root.action_N[(0,)]}"
    )
