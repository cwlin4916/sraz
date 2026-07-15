"""Search-correctness tests for ``sraz.core.mcts`` on analytically-solved games.

Every environment (from ``tests/mcts_envs.py``) has a known optimal action and
known exact Q values, so MCTS behaviour is asserted exactly — not statistically.
No Dirichlet noise and no rollouts are used, so every search here is fully
deterministic (``np.argmax`` tie-breaks by first index).

Eight tests:

1. bandit_exact_Q_and_optimal_arm
2. bandit_negative_rewards
3. adversarial_prior_overcome
4. binary_tree_optimal_first_action
5. delayed_trap_bellman
6. chain_max_backup_exact_total
7. value_guided_search
8. temperature_sharpening_and_flattening

Conventions relied on throughout (verified against the implementation):

- Edge keys in ``node.action_Q`` / ``action_N`` are 1-tuples ``(k,)`` from
  ``np.unravel_index`` over the 1D action space.
- ``perform_simulations`` restores the game state after every simulation, so
  the root node is ``mcts.nodes[game.hashable_obs]`` after the search.
- One-step bandits back up the same deterministic value on every visit of an
  edge, so ``action_Q`` equals the arm reward exactly under every backup rule.
"""

from __future__ import annotations

import numpy as np

from sraz.core.mcts import MCTS
from tests.mcts_envs import (
    FixedPolicyNet,
    UniformNet,
    ValueTableNet,
    make_bandit,
    make_binary_tree,
    make_chain,
    make_delayed_trap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _root(mcts: MCTS):
    """Root node of the finished search (game state is restored to the root)."""
    return mcts.nodes[mcts.game.hashable_obs]


# ---------------------------------------------------------------------------
# 1-2. Bandits: exact Q values and optimal-arm identification
# ---------------------------------------------------------------------------


def test_bandit_exact_Q_and_optimal_arm():
    """3-armed bandit with rewards [0.2, 1.0, -0.5]: after 60 simulations the
    visit-count distribution peaks on arm 1, and every visited arm's Q equals
    its reward EXACTLY (each edge always backs up the same deterministic
    value, so even the mean rule reproduces the reward bit-for-bit here)."""
    rewards = [0.2, 1.0, -0.5]
    game = make_bandit(rewards)
    net = UniformNet(3)
    mcts = MCTS(game, net, n_simulations=60)
    probs = mcts.perform_simulations(None)

    assert int(np.argmax(probs)) == 1, (
        f"optimal arm is 1 (reward 1.0), got argmax(probs)={np.argmax(probs)} "
        f"with probs={probs}"
    )

    root = _root(mcts)
    visited = 0
    for k, r in enumerate(rewards):
        key = (k,)
        if root.action_N.get(key, 0) == 0:
            continue
        visited += 1
        assert root.action_Q[key] == r, (
            f"arm {k}: Q={root.action_Q[key]!r} should EXACTLY equal "
            f"reward {r!r} (N={root.action_N[key]})"
        )
    assert visited == 3, (
        f"all 3 arms should be visited at 60 sims, action_N={dict(root.action_N)}"
    )


def test_bandit_negative_rewards():
    """All-negative bandit [-1.0, -2.0]: min-max Q normalization must still
    rank arm 0 first (least bad), and both Qs are exact."""
    game = make_bandit([-1.0, -2.0])
    net = UniformNet(2)
    mcts = MCTS(game, net, n_simulations=40)
    probs = mcts.perform_simulations(None)

    assert int(np.argmax(probs)) == 0, (
        f"least-bad arm is 0 (reward -1.0), got argmax(probs)="
        f"{np.argmax(probs)} with probs={probs}"
    )

    root = _root(mcts)
    assert root.action_N.get((0,), 0) > 0 and root.action_N.get((1,), 0) > 0, (
        f"both arms should be visited, action_N={dict(root.action_N)}"
    )
    assert root.action_Q[(0,)] == -1.0, (
        f"Q(arm 0)={root.action_Q[(0,)]!r} should be exactly -1.0"
    )
    assert root.action_Q[(1,)] == -2.0, (
        f"Q(arm 1)={root.action_Q[(1,)]!r} should be exactly -2.0"
    )
    assert root.action_N[(0,)] > root.action_N[(1,)], (
        f"arm 0 should dominate visits, action_N={dict(root.action_N)}"
    )


# ---------------------------------------------------------------------------
# 3. Adversarial prior: Q signal overcomes a wrong prior
# ---------------------------------------------------------------------------


def test_adversarial_prior_overcome():
    """Prior mass 0.85 sits on a worthless arm; the single rewarding arm has
    prior 0.05. Once arm 3's reward 1.0 enters the min-max normalization its
    q_norm hits 1.0 and the search locks onto it — the final visit counts
    must peak on arm 3 despite the adversarial prior."""
    game = make_bandit([0.0, 0.0, 0.0, 1.0])
    net = FixedPolicyNet([0.85, 0.05, 0.05, 0.05])
    mcts = MCTS(game, net, n_simulations=300)
    probs = mcts.perform_simulations(None)

    assert int(np.argmax(probs)) == 3, (
        f"search should overcome the prior and pick arm 3, got argmax(probs)="
        f"{np.argmax(probs)} with probs={probs}"
    )

    root = _root(mcts)
    assert root.action_N[(3,)] > root.action_N[(0,)], (
        f"rewarding arm 3 should out-visit prior-favoured arm 0, "
        f"action_N={dict(root.action_N)}"
    )
    assert root.action_Q[(3,)] == 1.0, (
        f"Q(arm 3)={root.action_Q[(3,)]!r} should be exactly 1.0"
    )


# ---------------------------------------------------------------------------
# 4. Binary tree: deep credit assignment picks the optimal first action
# ---------------------------------------------------------------------------


def test_binary_tree_optimal_first_action():
    """Depth-3 tree, leaf rewards [0.1..0.7, 1.0]: the best leaf is 0b111, so
    the optimal first action is 1. After 200 sims both the visit counts and
    the root Q values must point at action 1."""
    game = make_binary_tree(3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
    net = UniformNet(2)
    mcts = MCTS(game, net, n_simulations=200)
    probs = mcts.perform_simulations(None)

    assert int(np.argmax(probs)) == 1, (
        f"optimal first action is 1 (toward leaf 0b111), got argmax(probs)="
        f"{np.argmax(probs)} with probs={probs}"
    )

    root = _root(mcts)
    assert root.action_N.get((0,), 0) > 0 and root.action_N.get((1,), 0) > 0, (
        f"both first actions should be visited, action_N={dict(root.action_N)}"
    )
    best_q_action = max(root.action_Q, key=root.action_Q.get)
    assert best_q_action == (1,), (
        f"best-Q first action should be (1,), got {best_q_action} "
        f"with action_Q={dict(root.action_Q)}"
    )


# ---------------------------------------------------------------------------
# 5. Delayed trap: Bellman composition rejects the greedy first step
# ---------------------------------------------------------------------------


def test_delayed_trap_bellman():
    """Action 0 pays +0.5 now (total 0.5); action 1 pays -0.1 now, +1.0 later
    (total 0.9). Only the Bellman composition Q = immediate_reward + V(s')
    ranks action 1 above the greedy action 0.

    - mean backup: Q(1) approaches 0.9 from below (the first backup of the
      edge lands before its subtree completes, contributing -0.1 + nn_value),
      but must already exceed Q(0), and visit counts must pick action 1.
    - max backup: both Qs are EXACT path totals, 0.9 and 0.5.
    """
    # -- mean backup: correct ranking, contaminated-but-ordered Q values --
    game = make_delayed_trap()
    net = UniformNet(2)
    mcts = MCTS(game, net, n_simulations=100)  # default backup_rule="mean"
    probs = mcts.perform_simulations(None)
    root = _root(mcts)

    assert int(np.argmax(probs)) == 1, (
        f"patient action 1 (total 0.9) must beat greedy action 0 (total 0.5), "
        f"got argmax(probs)={np.argmax(probs)} with probs={probs}"
    )
    q0, q1 = root.action_Q[(0,)], root.action_Q[(1,)]
    assert q1 > q0, f"mean backup: Q(1)={q1} should exceed Q(0)={q0}"
    # Pre-terminal partial backups drag the mean below the true 0.9.
    assert q1 < 0.9, (
        f"mean backup is contaminated by the partial first backup, so "
        f"Q(1)={q1} should sit strictly below the true total 0.9"
    )

    # -- max backup: exact path totals --
    game_max = make_delayed_trap()
    net_max = UniformNet(2)
    mcts_max = MCTS(game_max, net_max, n_simulations=50, backup_rule="max")
    mcts_max.perform_simulations(None)
    root_max = _root(mcts_max)

    assert root_max.action_Q[(1,)] == 0.9, (
        f"max backup: Q(1)={root_max.action_Q[(1,)]!r} should be exactly 0.9"
    )
    assert root_max.action_Q[(0,)] == 0.5, (
        f"max backup: Q(0)={root_max.action_Q[(0,)]!r} should be exactly 0.5"
    )


# ---------------------------------------------------------------------------
# 6. Chain: dense per-step rewards accumulate exactly through the recursion
# ---------------------------------------------------------------------------


def test_chain_max_backup_exact_total():
    """Forced chain with step rewards [1, 2, 3]: with max backup, the root Q
    must reach the exact path total 1+2+3 = 6.0 (three simulations suffice to
    traverse the whole chain; 20 leaves plenty of slack)."""
    game = make_chain([1.0, 2.0, 3.0])
    net = UniformNet(1)
    mcts = MCTS(game, net, n_simulations=20, backup_rule="max")
    mcts.perform_simulations(None)
    root = _root(mcts)

    assert root.action_Q[(0,)] == 6.0, (
        f"root Q={root.action_Q[(0,)]!r} should be exactly 6.0 "
        f"(= 1.0 + 2.0 + 3.0 accumulated through the recursion)"
    )


# ---------------------------------------------------------------------------
# 7. Value-guided search: the value head alone steers the tree
# ---------------------------------------------------------------------------


def test_value_guided_search():
    """Depth-2 tree with ALL leaf rewards zero — the environment gives no
    reward signal at all. The value net returns 1.0 exactly at the depth-1
    right child (1, 1) and 0.0 elsewhere, so only the value head can break
    the symmetry: after 50 sims the visit counts must favour first action 1."""
    game = make_binary_tree(2, [0.0, 0.0, 0.0, 0.0])
    net = ValueTableNet(2, lambda obs: 1.0 if obs == (1, 1) else 0.0)
    mcts = MCTS(game, net, n_simulations=50)
    probs = mcts.perform_simulations(None)
    root = _root(mcts)

    assert int(np.argmax(probs)) == 1, (
        f"value head at (1,1) should steer the search toward action 1, "
        f"got argmax(probs)={np.argmax(probs)} with probs={probs}, "
        f"action_N={dict(root.action_N)}"
    )
    assert root.action_N[(1,)] > root.action_N[(0,)], (
        f"action 1 should out-visit action 0, action_N={dict(root.action_N)}"
    )


# ---------------------------------------------------------------------------
# 8. Temperature: end-to-end sharpening vs flattening of the same search
# ---------------------------------------------------------------------------


def test_temperature_sharpening_and_flattening():
    """Two MCTS instances differing ONLY in temperature (0.1 vs 4.0), no
    noise, deterministic game — so both searches produce identical visit
    counts and only the final exponentiation differs. The low-temperature
    distribution must be strictly sharper (higher max prob), and both must
    be valid distributions."""
    rewards = [0.2, 1.0, -0.5]

    def run(temperature: float):
        game = make_bandit(rewards)
        net = UniformNet(3)
        mcts = MCTS(game, net, n_simulations=60, temperature=temperature)
        probs = mcts.perform_simulations(None)
        return probs, _root(mcts)

    probs_low, root_low = run(0.1)
    probs_high, root_high = run(4.0)

    # Identical searches: the visit counts must agree exactly.
    assert dict(root_low.action_N) == dict(root_high.action_N), (
        f"searches should be identical up to temperature: "
        f"{dict(root_low.action_N)} vs {dict(root_high.action_N)}"
    )
    # More than one arm visited, otherwise both distributions degenerate to
    # a point mass and sharpness is undefined.
    n_visited = sum(1 for n in root_low.action_N.values() if n > 0)
    assert n_visited >= 2, (
        f"need >= 2 visited arms for a meaningful comparison, "
        f"action_N={dict(root_low.action_N)}"
    )

    for name, probs in (("low", probs_low), ("high", probs_high)):
        assert np.all(np.isfinite(probs)), f"non-finite {name}-T probs: {probs}"
        assert np.isclose(probs.sum(), 1.0), (
            f"{name}-T probs do not sum to 1: {probs} (sum={probs.sum()!r})"
        )

    assert probs_low.max() > probs_high.max(), (
        f"T=0.1 must sharpen relative to T=4.0: max(low)={probs_low.max()} "
        f"vs max(high)={probs_high.max()}"
    )
    # Both must still agree on the winning arm.
    assert int(np.argmax(probs_low)) == int(np.argmax(probs_high)) == 1, (
        f"both temperatures must keep arm 1 on top: "
        f"low={probs_low}, high={probs_high}"
    )
