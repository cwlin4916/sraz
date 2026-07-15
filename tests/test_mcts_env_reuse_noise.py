"""Dirichlet-noise injection and tree-reuse tests for ``sraz.core.mcts``.

Seven tests covering ``perform_simulations`` noise handling,
``perform_simulations_reuse`` re-injection, and ``advance_to`` tree reuse:

1. fresh_root_noise_applied — ``n_simulations=0`` + ``add_noise=True`` mixes
   Dirichlet noise into a fresh root's ``nn_policy``; the noise residual is a
   distribution over legal actions and the masked entry stays exactly 0.
2. noise_skipped_on_visited_root — ``perform_simulations`` guards on
   ``total_N == 0``: a second ``add_noise`` call on a visited root leaves
   ``nn_policy`` bit-for-bit unchanged.
3. reuse_reinjects_fresh_noise — ``perform_simulations_reuse`` restores
   ``nn_policy_original`` before mixing, so repeated calls inject fresh noise
   each time (no noise-on-noise compounding) and the residual check holds
   after every call.
4. reuse_without_noise_matches_plain — without noise/rollouts the reuse entry
   point behaves identically to ``perform_simulations``: same probs array and
   same root visit counts on a depth-2 binary tree.
5. advance_to_steps_and_warm_starts — after a root search, ``advance_to``
   steps the game to the chosen child, whose node already carries visits; a
   follow-up reuse search warm-starts it (``total_N`` grows by exactly
   ``n_simulations``).
6. nodes_retained_across_advance — ``advance_to`` performs no pruning:
   ``len(mcts.nodes)`` is unchanged.
7. noise_masked_positions_stay_zero_after_search — a full noisy search on a
   masked bandit never leaks probability or visits onto the masked arm; its
   UCB stays ``-inf``.

Noise math (defaults ``alpha=0.3``, ``eps=0.25``): after injection,
``nn_policy == (1-eps)*nn_policy_original + eps*masked_noise`` where
``masked_noise`` is a distribution supported on the legal actions. So the
residual ``r = (nn_policy - (1-eps)*nn_policy_original) / eps`` must be
entrywise >= -1e-12, sum to ~1, and be exactly 0 at masked positions.

Every test seeds the global numpy RNG at the top — MCTS draws its Dirichlet
noise from ``np.random``.
"""

from __future__ import annotations

import numpy as np

from sraz.core.mcts import MCTS
from tests.mcts_envs import TableGame, UniformNet, make_binary_tree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MASKED_ARM = 2  # the one illegal arm of the masked bandit below


def _make_masked_bandit() -> TableGame:
    """4-action one-step bandit with arm 2 absent (masked out).

    Legal arms: 0 -> 0.1, 1 -> 0.2, 3 -> 0.3, all terminal. The masked
    prior after ``query_net_masked`` with a uniform net is
    ``[1/3, 1/3, 0, 1/3]``.
    """
    transitions = {
        "root": {
            0: ("a", 0.1, True, False),
            1: ("b", 0.2, True, False),
            3: ("c", 0.3, True, False),
        },
    }
    return TableGame(transitions, "root", n_actions=4)


def _assert_valid_noise_residual(node, eps: float, masked_idx: int, tag: str):
    """Check that ``node.nn_policy`` is a valid (1-eps)/eps mix of the
    original policy with a masked noise distribution."""
    policy = np.asarray(node.nn_policy, dtype=np.float64)
    original = np.asarray(node.nn_policy_original, dtype=np.float64)
    residual = (policy - (1.0 - eps) * original) / eps

    assert np.all(residual >= -1e-12), (
        f"[{tag}] noise residual has negative entries: {residual}"
    )
    assert np.isclose(residual.sum(), 1.0), (
        f"[{tag}] noise residual should sum to 1, got {residual.sum()} "
        f"(residual={residual})"
    )
    assert residual[masked_idx] == 0.0, (
        f"[{tag}] residual at masked arm {masked_idx} should be exactly 0, "
        f"got {residual[masked_idx]}"
    )


# ---------------------------------------------------------------------------
# Noise injection on fresh vs visited roots
# ---------------------------------------------------------------------------


def test_fresh_root_noise_applied():
    """``n_simulations=0`` expands the root (one net query, zero visits), so
    ``add_noise=True`` hits the fresh-root branch: ``nn_policy`` becomes a
    (1-eps)/eps mix of the masked prior with masked Dirichlet noise."""
    np.random.seed(11)
    game = _make_masked_bandit()
    net = UniformNet(4)
    mcts = MCTS(game, net, n_simulations=0, temperature=1.0, c_exploration=1.5)

    probs = mcts.perform_simulations(None, add_noise=True)
    root = mcts.nodes["root"]

    # Noise actually landed: mixed policy differs from the stored original.
    assert not np.array_equal(root.nn_policy, root.nn_policy_original), (
        f"nn_policy should differ from nn_policy_original after noise, "
        f"got identical arrays {root.nn_policy}"
    )
    # Masked entry stays exactly zero and the mix is still a distribution.
    assert root.nn_policy[MASKED_ARM] == 0.0, (
        f"masked arm leaked policy mass: nn_policy={root.nn_policy}"
    )
    assert np.isclose(root.nn_policy.sum(), 1.0), (
        f"noise-mixed nn_policy should sum to 1, got {root.nn_policy.sum()}"
    )
    _assert_valid_noise_residual(
        root, mcts.dirichlet_epsilon, MASKED_ARM, "fresh root")

    # Zero counts => masked-prior fallback returns the (noise-mixed) policy.
    assert np.all(np.isfinite(probs)), f"non-finite probs: {probs}"
    assert np.isclose(probs.sum(), 1.0), f"probs do not sum to 1: {probs}"
    assert probs[MASKED_ARM] == 0.0, f"masked arm got probability: {probs}"
    np.testing.assert_allclose(
        probs, root.nn_policy,
        err_msg="zero-count fallback should return the noise-mixed prior",
    )


def test_noise_skipped_on_visited_root():
    """``perform_simulations`` injects noise only when ``total_N == 0``: a
    second ``add_noise=True`` call on the now-visited root must leave
    ``nn_policy`` bit-for-bit unchanged."""
    np.random.seed(23)
    game = _make_masked_bandit()
    net = UniformNet(4)
    mcts = MCTS(game, net, n_simulations=25, temperature=1.0,
                c_exploration=1.5)

    mcts.perform_simulations(None, add_noise=True)
    root = mcts.nodes["root"]
    assert root.total_N == 25, (
        f"root should carry 25 visits after the first search, "
        f"got {root.total_N}"
    )
    snapshot = np.array(root.nn_policy, copy=True)

    mcts.perform_simulations(None, add_noise=True)
    assert np.array_equal(root.nn_policy, snapshot), (
        f"visited-root noise guard failed: nn_policy changed from "
        f"{snapshot} to {root.nn_policy}"
    )


def test_reuse_reinjects_fresh_noise():
    """``perform_simulations_reuse`` restores ``nn_policy_original`` before
    mixing, so each call injects *fresh* noise: the original is preserved
    across calls, the residual check holds after every call, and the two
    post-call policies differ (no noise-on-noise compounding)."""
    np.random.seed(37)
    game = _make_masked_bandit()
    net = UniformNet(4)
    mcts = MCTS(game, net, n_simulations=10, temperature=1.0,
                c_exploration=1.5)

    # Visit the root first (no noise) so both reuse calls hit a visited node.
    mcts.perform_simulations(None)
    root = mcts.nodes["root"]
    assert root.total_N == 10, (
        f"root should be visited before the reuse calls, "
        f"total_N={root.total_N}"
    )
    original_before = np.array(root.nn_policy_original, copy=True)

    mcts.perform_simulations_reuse(None, add_noise=True)
    policy_after_first = np.array(root.nn_policy, copy=True)
    original_after_first = np.array(root.nn_policy_original, copy=True)
    _assert_valid_noise_residual(
        root, mcts.dirichlet_epsilon, MASKED_ARM, "reuse call 1")

    mcts.perform_simulations_reuse(None, add_noise=True)
    policy_after_second = np.array(root.nn_policy, copy=True)
    original_after_second = np.array(root.nn_policy_original, copy=True)
    _assert_valid_noise_residual(
        root, mcts.dirichlet_epsilon, MASKED_ARM, "reuse call 2")

    # The clean original never changes.
    assert np.array_equal(original_before, original_after_first), (
        f"nn_policy_original mutated by first reuse call: "
        f"{original_before} -> {original_after_first}"
    )
    assert np.array_equal(original_before, original_after_second), (
        f"nn_policy_original mutated by second reuse call: "
        f"{original_before} -> {original_after_second}"
    )
    # Fresh Dirichlet draw each call => the mixed policies differ.
    assert not np.array_equal(policy_after_first, policy_after_second), (
        f"reuse calls should inject fresh noise each time, but both produced "
        f"{policy_after_first}"
    )


# ---------------------------------------------------------------------------
# Tree reuse: perform_simulations_reuse and advance_to
# ---------------------------------------------------------------------------


def test_reuse_without_noise_matches_plain():
    """Without noise or rollouts the search is fully deterministic, so
    ``perform_simulations_reuse`` on a fresh tree must reproduce
    ``perform_simulations`` exactly: identical probs and identical root
    visit counts."""
    np.random.seed(41)
    leaf_rewards = [0.1, 0.7, 0.3, 0.9]
    kwargs = dict(n_simulations=40, temperature=1.0, c_exploration=1.5,
                  backup_rule="max")

    game_plain = make_binary_tree(2, leaf_rewards)
    mcts_plain = MCTS(game_plain, UniformNet(2), **kwargs)
    probs_plain = mcts_plain.perform_simulations(None)

    game_reuse = make_binary_tree(2, leaf_rewards)
    mcts_reuse = MCTS(game_reuse, UniformNet(2), **kwargs)
    probs_reuse = mcts_reuse.perform_simulations_reuse(None)

    assert np.array_equal(probs_plain, probs_reuse), (
        f"probs diverge: plain={probs_plain}, reuse={probs_reuse}"
    )

    root_plain = mcts_plain.nodes[game_plain.hashable_obs]
    root_reuse = mcts_reuse.nodes[game_reuse.hashable_obs]
    assert root_plain.total_N == root_reuse.total_N == 40, (
        f"root visit totals diverge: plain={root_plain.total_N}, "
        f"reuse={root_reuse.total_N}"
    )
    assert root_plain.action_N == root_reuse.action_N, (
        f"root visit counts diverge: plain={root_plain.action_N}, "
        f"reuse={root_reuse.action_N}"
    )


def test_advance_to_steps_and_warm_starts():
    """After a 100-sim root search on a depth-3 tree, the best child's node
    already carries visits. ``advance_to`` steps the game there, and a 50-sim
    reuse search warm-starts that node: its ``total_N`` grows by exactly 50
    (every simulation backs up one edge at the new root)."""
    np.random.seed(53)
    # Best leaf is 7 (=0b111) with reward 1.0 => optimal first action is 1.
    leaf_rewards = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]
    game = make_binary_tree(3, leaf_rewards)
    mcts = MCTS(game, UniformNet(2), n_simulations=100, temperature=1.0,
                c_exploration=1.5, backup_rule="max")

    probs = mcts.perform_simulations(None)
    best = int(np.argmax(probs))
    # Deterministic search (no noise/rollouts) on this tree picks action 1.
    assert best == 1, (
        f"expected first action 1 toward leaf 7 (reward 1.0), got {best} "
        f"(probs={probs})"
    )

    child_state = (1, best)
    assert child_state in mcts.nodes, (
        f"depth-1 state {child_state} missing from tree "
        f"(nodes: {sorted(mcts.nodes)})"
    )
    child = mcts.nodes[child_state]
    n_before = child.total_N
    assert n_before > 0, (
        f"warm-start child {child_state} should already have visits, "
        f"total_N={n_before}"
    )

    mcts.advance_to(best)
    assert game.hashable_obs == child_state, (
        f"advance_to should step the game to {child_state}, "
        f"got {game.hashable_obs}"
    )

    mcts.n_simulations = 50
    probs_reuse = mcts.perform_simulations_reuse(None)
    assert np.all(np.isfinite(probs_reuse)), (
        f"non-finite reuse probs: {probs_reuse}"
    )
    assert np.isclose(probs_reuse.sum(), 1.0), (
        f"reuse probs do not sum to 1: {probs_reuse}"
    )
    # Warm start: the existing node keeps its counts and gains exactly one
    # backup per simulation.
    assert child.total_N == n_before + 50, (
        f"new root total_N should be {n_before} + 50, got {child.total_N}"
    )


def test_nodes_retained_across_advance():
    """``advance_to`` only steps the game — it never prunes ``mcts.nodes``,
    so the node count is unchanged (in particular it does not shrink)."""
    np.random.seed(61)
    game = make_binary_tree(3, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0])
    mcts = MCTS(game, UniformNet(2), n_simulations=60, temperature=1.0,
                c_exploration=1.5, backup_rule="max")
    mcts.perform_simulations(None)

    n_nodes_before = len(mcts.nodes)
    assert n_nodes_before > 1, (
        f"search should have grown a tree, got {n_nodes_before} node(s)"
    )
    mcts.advance_to(1)
    n_nodes_after = len(mcts.nodes)
    assert n_nodes_after == n_nodes_before, (
        f"advance_to changed the node count: {n_nodes_before} -> "
        f"{n_nodes_after}"
    )


# ---------------------------------------------------------------------------
# Mask integrity under a full noisy search
# ---------------------------------------------------------------------------


def test_noise_masked_positions_stay_zero_after_search():
    """A full 50-sim noisy search on the masked bandit must keep the masked
    arm at zero probability, zero visits, and -inf UCB."""
    np.random.seed(71)
    game = _make_masked_bandit()
    net = UniformNet(4)
    mcts = MCTS(game, net, n_simulations=50, temperature=1.0,
                c_exploration=1.5)

    probs = mcts.perform_simulations(None, add_noise=True)
    assert np.all(np.isfinite(probs)), f"non-finite probs: {probs}"
    assert np.isclose(probs.sum(), 1.0), f"probs do not sum to 1: {probs}"
    assert probs[MASKED_ARM] == 0.0, (
        f"masked arm leaked probability after noisy search: {probs}"
    )

    root = mcts.nodes["root"]
    assert (MASKED_ARM,) not in root.action_N, (
        f"masked arm was visited: action_N={root.action_N}"
    )
    ucbs = mcts.calc_masked_ucbs(root, msg=None)
    assert np.isinf(ucbs[MASKED_ARM]) and ucbs[MASKED_ARM] < 0, (
        f"UCB at masked arm should be -inf, got {ucbs[MASKED_ARM]} "
        f"(ucbs={ucbs})"
    )
