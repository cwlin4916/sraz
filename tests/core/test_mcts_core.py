"""Audit tests for ``sraz.core.mcts``.

Six tests covering the three defensive patches landed in
``docs/notes/stage4/03.md`` §6 (Core MCTS audit findings):

1. node_stats_invariant_total_N_equals_sum_action_N
2. n_simulations_one_does_not_return_nan
3. invalid_actions_never_selected
4. uniform_policy_is_not_mutated_in_place
5. terminal_reward_backpropagates_to_root_Q
6. temperature_transform_is_well_defined

All tests use small in-file toy ``Game`` / ``PolicyValueNet`` shims so that
``MCTS`` behaviour can be audited without dragging in derivation-game or
gripper-env state.
"""

from __future__ import annotations

import copy
from typing import Tuple

import gymnasium.spaces as spaces
import numpy as np
import pytest

from sraz.core.game import Game
from sraz.core.mcts import MCTS
from sraz.core.policy_value_net import PolicyValueNet


# ---------------------------------------------------------------------------
# Toy Game shims
# ---------------------------------------------------------------------------


class ToyTwoActionGame(Game[int, int]):
    """One-step deterministic 2-action game.

    Action 0 → state 1 with reward 1.0 (terminal).
    Action 1 → state 2 with reward 0.0 (terminal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(1,), dtype=np.int64)
        self.reset_wrapper()

    def reset(self, **_):
        self.state = 0
        return self.state, {}

    def step(self, action: int):
        assert action in (0, 1)
        self.state = 1 if action == 0 else 2
        reward = 1.0 if action == 0 else 0.0
        return self.state, reward, True, False, {}

    def get_action_mask(self) -> np.ndarray:
        return np.array([1, 1], dtype=bool)

    @property
    def hashable_obs(self):
        return int(self.state)


class ToyPaddedGame(Game[int, int]):
    """Padded action space (size 4) with only actions 0,1 legal.

    One-step deterministic: action 0 → reward 1, action 1 → reward 0.5.
    Actions 2 and 3 must NEVER be selected.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3, shape=(1,), dtype=np.int64)
        self.reset_wrapper()

    def reset(self, **_):
        self.state = 0
        return self.state, {}

    def step(self, action: int):
        assert action in (0, 1, 2, 3)
        self.state = 1
        reward = {0: 1.0, 1: 0.5, 2: -99.0, 3: -99.0}[int(action)]
        return self.state, reward, True, False, {}

    def get_action_mask(self) -> np.ndarray:
        return np.array([1, 1, 0, 0], dtype=bool)

    @property
    def hashable_obs(self):
        return int(self.state)


class ToyDepth2Game(Game[Tuple[int, int], int]):
    """Two-step deterministic game for the depth-2 invariant test.

    State = (depth, path). At depth 0, two actions {0,1}; at depth 1, two
    actions {0,1}. Terminal reward at depth 2 = product of action choices
    (0 → 0.0, 1 → 1.0), i.e. only path (1,1) gives reward 1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=3, shape=(2,), dtype=np.int64)
        self.reset_wrapper()

    def reset(self, **_):
        self.state: Tuple[int, int] = (0, 0)
        return self.state, {}

    def step(self, action: int):
        depth, acc = self.state
        new_acc = (acc << 1) | int(action)
        new_depth = depth + 1
        self.state = (new_depth, new_acc)
        terminated = new_depth >= 2
        reward = 1.0 if (terminated and new_acc == 0b11) else 0.0
        return self.state, reward, terminated, False, {}

    def get_action_mask(self) -> np.ndarray:
        return np.array([1, 1], dtype=bool)

    @property
    def hashable_obs(self):
        return tuple(self.state)


# ---------------------------------------------------------------------------
# Toy nets
# ---------------------------------------------------------------------------


class _StubNet(PolicyValueNet):
    """Shared stub of the abstract methods on :class:`PolicyValueNet` so
    test-only subclasses don't have to repeat them."""

    def train(self, examples):
        pass

    def save_checkpoint(self, save_dir):
        pass

    def load_checkpoint(self, save_dir):
        pass

    def push_multiprocessing(self):
        pass

    def pop_multiprocessing(self, *args):
        pass


class FreshUniformNet(_StubNet):
    """Returns a *fresh* uniform-prior array on every call (the safe baseline,
    matching :class:`UniformPolicyValueNet`)."""

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def predict(self, _state):
        return (
            np.ones(self.action_size, dtype=np.float64) / self.action_size,
            np.array(0.0, dtype=np.float64),
        )


class CachedUniformNet(_StubNet):
    """Returns the SAME pre-allocated policy array every call — surfaces any
    accidental in-place mutation inside ``query_net_masked``."""

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size
        self._policy = np.ones(action_size, dtype=np.float64) / action_size
        self._value = np.array(0.0, dtype=np.float64)

    def predict(self, _state):
        return self._policy, self._value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _root_hash(mcts: MCTS):
    return mcts.game.hashable_obs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_node_stats_invariant_total_N_equals_sum_action_N():
    """For every expanded node, ``node.total_N`` should equal
    ``sum(node.action_N.values())`` — a basic backup-counter consistency
    invariant. Run on the 2-deep toy game so multiple nodes get expanded."""
    game = ToyDepth2Game()
    net = FreshUniformNet(game.action_space.n)
    mcts = MCTS(game, net, n_simulations=64, temperature=1.0, c_exploration=1.5)
    mcts.perform_simulations(None)

    expanded = 0
    for state, node in mcts.nodes.items():
        if node.is_terminal_state:
            # Terminal nodes are not on the backup path; they have no edges.
            continue
        if node.nn_policy is None:
            # Reached but unexpanded — shouldn't happen after a finished search,
            # but tolerate.
            continue
        expanded += 1
        assert node.total_N == sum(node.action_N.values()), (
            f"node {state}: total_N={node.total_N}, "
            f"sum action_N={sum(node.action_N.values())}"
        )
    assert expanded >= 1, "search should have expanded at least the root"


def test_n_simulations_one_does_not_return_nan():
    """With ``n_simulations=1`` exactly one simulation runs after the root
    expansion, so the returned distribution must be finite and sum to 1
    (it will be concentrated on the visited action — that's fine).
    With ``n_simulations=0`` the visit counts are all zero, and the
    defensive fallback must return the masked prior, not NaN."""
    # n=1: finite and normalized, even if concentrated.
    game = ToyTwoActionGame()
    net = FreshUniformNet(2)
    mcts = MCTS(game, net, n_simulations=1, temperature=1.0, c_exploration=1.5)
    probs = mcts.perform_simulations(None)
    assert np.all(np.isfinite(probs)), f"non-finite probs at n=1: {probs}"
    assert np.isclose(probs.sum(), 1.0), f"probs do not sum to 1 at n=1: {probs}"

    # n=0: fallback to masked prior; both legal actions get mass.
    game0 = ToyTwoActionGame()
    net0 = FreshUniformNet(2)
    mcts0 = MCTS(game0, net0, n_simulations=0, temperature=1.0, c_exploration=1.5)
    probs0 = mcts0.perform_simulations(None)
    assert np.all(np.isfinite(probs0)), f"non-finite probs at n=0: {probs0}"
    assert np.isclose(probs0.sum(), 1.0), f"probs do not sum to 1 at n=0: {probs0}"
    assert probs0[0] > 0 and probs0[1] > 0, (
        f"fallback must give both legal actions positive mass, got {probs0}"
    )


def test_invalid_actions_never_selected():
    """Padded mask = [1,1,0,0]; under many sims, masked positions stay at
    zero probability and the UCB at those positions is ``-inf``."""
    game = ToyPaddedGame()
    net = FreshUniformNet(4)
    mcts = MCTS(game, net, n_simulations=64, temperature=1.0, c_exploration=1.5)
    probs = mcts.perform_simulations(None)

    assert probs.shape == (4,)
    assert probs[2] == 0.0 and probs[3] == 0.0, f"masked entries leaked mass: {probs}"
    assert np.isclose(probs.sum(), 1.0)

    # UCBs at masked positions are -inf.
    root = mcts.nodes[mcts.game.hashable_obs]
    ucbs = mcts.calc_masked_ucbs(root, msg=None)
    assert np.isinf(ucbs[2]) and ucbs[2] < 0, f"ucbs[2] should be -inf, got {ucbs[2]}"
    assert np.isinf(ucbs[3]) and ucbs[3] < 0, f"ucbs[3] should be -inf, got {ucbs[3]}"


def test_uniform_policy_is_not_mutated_in_place():
    """``query_net_masked`` used to do ``policy *= mask; policy /= sum``
    in place — safe for fresh-array nets but a latent trap for any cached
    net. The defensive copy ensures the cache stays untouched."""
    game = ToyPaddedGame()
    net = CachedUniformNet(4)
    snapshot = net._policy.copy()
    mcts = MCTS(game, net, n_simulations=16, temperature=1.0, c_exploration=1.5)
    mcts.perform_simulations(None)
    np.testing.assert_array_equal(
        net._policy,
        snapshot,
        err_msg="CachedUniformNet._policy was mutated by MCTS",
    )


def test_terminal_reward_backpropagates_to_root_Q():
    """Action 0 leads to reward 1, action 1 to reward 0. After enough
    simulations, root.action_Q[0] should exceed root.action_Q[1]."""
    game = ToyTwoActionGame()
    net = FreshUniformNet(2)
    mcts = MCTS(game, net, n_simulations=64, temperature=1.0, c_exploration=1.5)
    mcts.perform_simulations(None)

    root = mcts.nodes[_root_hash(mcts)]
    # Actions are stored as 1-tuples (np.unravel_index of a 1D action space).
    good, bad = (0,), (1,)
    assert root.action_N.get(good, 0) > 0, "good action was never visited"
    assert root.action_N.get(bad, 0) > 0, "bad action was never visited"
    assert root.action_Q[good] > root.action_Q[bad], (
        f"Q(good)={root.action_Q[good]} should exceed Q(bad)={root.action_Q[bad]}"
    )


def test_temperature_transform_is_well_defined():
    """Helper ``_temperature_probs_with_fallback`` must:
    - return finite probs summing to 1 for nonzero counts,
    - concentrate on argmax at low temperature,
    - return the masked-prior fallback (not NaN) when all counts are zero."""
    game = ToyPaddedGame()
    net = FreshUniformNet(4)
    mcts = MCTS(game, net, n_simulations=4, temperature=1.0, c_exploration=1.5)

    # Seed mynode by running a tiny search so nn_policy is populated.
    mcts.perform_simulations(None)
    root = mcts.nodes[mcts.game.hashable_obs]

    counts = np.array([10.0, 1.0, 0.0, 0.0])
    for T in (0.25, 1.0, 4.0):
        mcts.temperature = T
        probs = mcts._temperature_probs_with_fallback(counts, root)
        assert np.all(np.isfinite(probs)), f"non-finite probs at T={T}: {probs}"
        assert np.isclose(probs.sum(), 1.0)
    # At low temperature argmax dominates.
    mcts.temperature = 0.1
    probs = mcts._temperature_probs_with_fallback(counts, root)
    assert int(np.argmax(probs)) == 0

    # All-zero counts ⇒ fallback to masked prior, not NaN.
    mcts.temperature = 1.0
    probs0 = mcts._temperature_probs_with_fallback(np.zeros(4), root)
    assert np.all(np.isfinite(probs0))
    assert np.isclose(probs0.sum(), 1.0)
    assert probs0[2] == 0.0 and probs0[3] == 0.0
