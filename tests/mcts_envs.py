"""Synthetic environments and stub networks for testing ``sraz.core.mcts``.

Every environment here has an *analytically known* ground truth (optimal
action, exact Q values, exact path returns), so MCTS behaviour can be
asserted exactly rather than statistically wherever possible.

The workhorse is :class:`TableGame`, a fully table-driven deterministic MDP:
``transitions[state][action] = (next_state, reward, terminated, truncated)``.
States must be hashable (ints or tuples). The action mask at a state marks
exactly the actions present in its transition row, padded to ``n_actions``.
Because nodes are keyed by ``hashable_obs``, two paths that reach the same
state key share an MCTS node — which makes transposition tests trivial to
express.

Factory functions build the named scenarios used by the test modules:

- :func:`make_bandit` — K-armed one-step bandit with known arm rewards.
- :func:`make_chain` — linear chain, one legal action per state, known
  total return ``sum(step_rewards)``.
- :func:`make_binary_tree` — depth-D binary tree with rewards on the final
  step; optimal leaf/first-action known by construction.
- :func:`make_delayed_trap` — greedy first-step reward misleads; correct
  Bellman backup (immediate reward + future value) prefers the other arm.
- :func:`make_dead_end` — nonterminal state with an all-zero action mask
  (MCTS raises ``ValueError`` when it tries to expand it).
- :func:`make_truncation` — episode ends via ``truncated=True``; MCTS must
  treat it exactly like termination.
- :func:`make_transposition` — state is the multiset of actions taken, so
  different orderings collide on the same node.

:class:`GridBanditGame` is separate: it has a ``MultiDiscrete`` action space
(non-scalar ``action_space.shape``), exercising the multi-dimensional
fallback path in ``MCTS.calc_masked_ucbs`` and the tuple-action stepping in
``MCTS.search``.

Stub networks (all subclass :class:`CountingNet`, which counts ``predict``
calls in ``n_predict_calls``):

- :class:`UniformNet` — fresh uniform prior over a given shape, fixed value.
- :class:`FixedPolicyNet` — fixed prior array (returned as a fresh copy),
  fixed value; use for adversarial-prior tests.
- :class:`ValueTableNet` — uniform prior, value looked up per-observation
  via a callable; use for value-guided-search tests.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable

import gymnasium.spaces as spaces
import numpy as np

from sraz.core.game import Game
from sraz.core.policy_value_net import PolicyValueNet

# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

# transitions[state][action] = (next_state, reward, terminated, truncated)
TransitionTable = dict[Hashable, dict[int, tuple[Hashable, float, bool, bool]]]


class TableGame(Game[Hashable, int]):
    """Deterministic table-driven single-player game over hashable states.

    ``transitions[state]`` maps each *legal* action at ``state`` to a
    ``(next_state, reward, terminated, truncated)`` tuple. Actions absent
    from the row are masked out. A state with an empty row and no terminal
    flag is a dead end: its mask is all zeros and ``MCTS.query_net_masked``
    raises when asked to expand it.
    """

    def __init__(self, transitions: TransitionTable, initial_state: Hashable,
                 n_actions: int) -> None:
        super().__init__()
        self.transitions = transitions
        self.initial_state = initial_state
        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(max(1, len(transitions)))
        self.reset_wrapper()

    def reset(self, **_):
        self.state = self.initial_state
        return self.state, {}

    def step(self, action: int):
        action = int(action)
        row = self.transitions.get(self.state, {})
        assert action in row, (
            f"illegal action {action} at state {self.state!r} "
            f"(legal: {sorted(row)})"
        )
        next_state, reward, terminated, truncated = row[action]
        self.state = next_state
        return self.state, reward, terminated, truncated, {}

    def get_action_mask(self) -> np.ndarray:
        row = self.transitions.get(self.state, {})
        mask = np.zeros(self.n_actions, dtype=bool)
        for action in row:
            mask[action] = True
        return mask

    @property
    def hashable_obs(self):
        return self.state


class GridBanditGame(Game[int, tuple]):
    """One-step bandit over a 2D action grid with a ``MultiDiscrete`` space.

    ``action_space.shape == (2,)`` is non-scalar, so MCTS takes the
    multi-dimensional fallback in ``calc_masked_ucbs`` and steps with tuple
    actions. Reward for cell ``(r, c)`` is ``reward_grid[r, c]``; cells where
    ``mask_grid`` is 0 are illegal. The episode terminates after one step.
    """

    def __init__(self, reward_grid: np.ndarray,
                 mask_grid: np.ndarray | None = None) -> None:
        super().__init__()
        self.reward_grid = np.asarray(reward_grid, dtype=np.float64)
        if mask_grid is None:
            mask_grid = np.ones_like(self.reward_grid)
        self.mask_grid = np.asarray(mask_grid).astype(bool)
        assert self.mask_grid.shape == self.reward_grid.shape
        self.action_space = spaces.MultiDiscrete(list(self.reward_grid.shape))
        self.observation_space = spaces.Discrete(2)
        self.reset_wrapper()

    def reset(self, **_):
        self.state = 0  # 0 = fresh, 1 = done
        return self.state, {}

    def step(self, action: tuple):
        r, c = (int(x) for x in action)
        assert self.state == 0, "GridBanditGame is a one-step game"
        assert self.mask_grid[r, c], f"illegal cell ({r}, {c})"
        self.state = 1
        return self.state, float(self.reward_grid[r, c]), True, False, {}

    def get_action_mask(self) -> np.ndarray:
        if self.state == 0:
            return self.mask_grid.copy()
        return np.zeros_like(self.mask_grid)

    @property
    def hashable_obs(self):
        return self.state


# ---------------------------------------------------------------------------
# Factory functions for named scenarios
# ---------------------------------------------------------------------------


def make_bandit(rewards: list[float]) -> TableGame:
    """K-armed one-step bandit. Arm ``k`` yields ``rewards[k]``, terminal.

    Ground truth: the optimal arm is ``argmax(rewards)``. Every backup of
    edge ``k`` is the same deterministic value ``rewards[k]``, so
    Q(root, k) == rewards[k] bit-exactly under ``backup_rule="max"`` for
    any reward, and under ``"mean"`` only when the reward is dyadic
    (mean backup recomputes ``sum(values)/len(values)``, and e.g.
    ``sum([0.2]*3)/3 != 0.2`` in float64 — use rewards like 0.25/0.5/1.0
    for exact mean-backup assertions).
    """
    transitions: TransitionTable = {
        "root": {k: (("arm", k), float(r), True, False)
                 for k, r in enumerate(rewards)},
    }
    return TableGame(transitions, "root", n_actions=len(rewards))


def make_chain(step_rewards: list[float]) -> TableGame:
    """Linear chain with exactly one legal action (0) per state.

    Step ``i`` yields ``step_rewards[i]``; the last step terminates.
    Ground truth: the return of the unique trajectory is
    ``sum(step_rewards)``, and with ``backup_rule="max"`` the root Q reaches
    it exactly once one simulation has traversed the full chain.
    """
    n = len(step_rewards)
    assert n >= 1
    transitions: TransitionTable = {
        i: {0: (i + 1, float(step_rewards[i]), i + 1 == n, False)}
        for i in range(n)
    }
    return TableGame(transitions, 0, n_actions=1)


def make_binary_tree(depth: int, leaf_rewards: list[float]) -> TableGame:
    """Full binary tree of the given depth; rewards only on the final step.

    States are ``(d, path)`` with ``path`` the bit-string of actions taken.
    Reaching leaf ``p`` at depth ``depth`` yields ``leaf_rewards[p]``.
    Ground truth: the optimal leaf is ``argmax(leaf_rewards)`` and the
    optimal first action is its top bit.
    """
    assert len(leaf_rewards) == 2 ** depth
    transitions: TransitionTable = {}
    for d in range(depth):
        for path in range(2 ** d):
            row = {}
            for a in (0, 1):
                child = (path << 1) | a
                terminal = d + 1 == depth
                reward = float(leaf_rewards[child]) if terminal else 0.0
                row[a] = ((d + 1, child), reward, terminal, False)
            transitions[(d, path)] = row
    return TableGame(transitions, (0, 0), n_actions=2)


def make_delayed_trap() -> TableGame:
    """Two-step trap: the greedy first-step reward is a losing move.

    - Action 0: reward +0.5 now, then forced step with reward 0.0 → total 0.5.
    - Action 1: reward -0.1 now, then forced step with reward +1.0 → total 0.9.

    Ground truth: optimal first action is 1. With ``backup_rule="max"``,
    Q(root, 0) == 0.5 and Q(root, 1) == 0.9 exactly once both trajectories
    complete. With ``backup_rule="mean"``, the first backup of action 1
    lands before its subtree is expanded (it backs up ``-0.1 + nn_value``),
    so Q(root, 1) approaches 0.9 from below but still exceeds Q(root, 0).
    """
    transitions: TransitionTable = {
        "root": {
            0: ("greedy", 0.5, False, False),
            1: ("patient", -0.1, False, False),
        },
        "greedy": {0: ("end_g", 0.0, True, False)},
        "patient": {0: ("end_p", 1.0, True, False)},
    }
    return TableGame(transitions, "root", n_actions=2)


def make_dead_end() -> TableGame:
    """Root's only action leads to a nonterminal state with no legal actions.

    Ground truth: expanding the dead end calls ``query_net_masked`` with an
    all-zero mask, which raises ``ValueError("Action mask says no valid
    moves")``. The root expansion itself succeeds; the second simulation
    descends into the dead end and raises.
    """
    transitions: TransitionTable = {
        "root": {0: ("pit", 0.0, False, False)},
        "pit": {},
    }
    return TableGame(transitions, "root", n_actions=1)


def make_truncation(reward: float = 0.75) -> TableGame:
    """One-step episode ending via ``truncated=True`` (not terminated).

    Ground truth: MCTS must treat truncation exactly like termination —
    the node is terminal, and Q(root, 0) == reward exactly.
    """
    transitions: TransitionTable = {
        "root": {0: ("cutoff", float(reward), False, True)},
    }
    return TableGame(transitions, "root", n_actions=1)


def make_transposition(depth: int = 3,
                       reward_fn: Callable[[int, int], float] | None = None,
                       ) -> TableGame:
    """Order-independent game: state is ``(n_zeros, n_ones)`` taken so far.

    Both actions are always legal until ``n_zeros + n_ones == depth``, at
    which point the episode terminates with ``reward_fn(n_zeros, n_ones)``
    (default: ``n_ones / depth``). Different action orderings reach the same
    state key, so interior nodes are shared (transpositions).

    Ground truth: the reachable state set is ``{(z, o) : z + o <= depth}``,
    i.e. ``(depth+1)(depth+2)/2`` states — strictly fewer than the
    ``2^(depth+1) - 1`` nodes of the unshared tree.
    """
    if reward_fn is None:
        reward_fn = lambda z, o: o / depth
    transitions: TransitionTable = {}
    for z in range(depth):
        for o in range(depth - z):
            row = {}
            for a, (nz, no) in ((0, (z + 1, o)), (1, (z, o + 1))):
                terminal = nz + no == depth
                reward = float(reward_fn(nz, no)) if terminal else 0.0
                row[a] = ((nz, no), reward, terminal, False)
            transitions[(z, o)] = row
    return TableGame(transitions, (0, 0), n_actions=2)


# ---------------------------------------------------------------------------
# Stub networks
# ---------------------------------------------------------------------------


class CountingNet(PolicyValueNet):
    """Base stub: no-op abstract methods plus a ``predict`` call counter."""

    def __init__(self) -> None:
        self.n_predict_calls = 0

    def predict(self, state):  # pragma: no cover - overridden
        raise NotImplementedError

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


class UniformNet(CountingNet):
    """Fresh uniform prior over ``shape`` (int or tuple), fixed scalar value."""

    def __init__(self, shape: int | tuple, value: float = 0.0) -> None:
        super().__init__()
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.value = float(value)

    def predict(self, _state):
        self.n_predict_calls += 1
        size = int(np.prod(self.shape))
        policy = np.full(self.shape, 1.0 / size, dtype=np.float64)
        return policy, np.array(self.value, dtype=np.float64)


class FixedPolicyNet(CountingNet):
    """Fixed prior (returned as a fresh copy each call), fixed scalar value."""

    def __init__(self, policy, value: float = 0.0) -> None:
        super().__init__()
        self.policy = np.asarray(policy, dtype=np.float64)
        self.value = float(value)

    def predict(self, _state):
        self.n_predict_calls += 1
        return self.policy.copy(), np.array(self.value, dtype=np.float64)


class ValueTableNet(CountingNet):
    """Uniform prior; value computed per-observation by ``value_fn(obs)``."""

    def __init__(self, shape: int | tuple,
                 value_fn: Callable[[Any], float]) -> None:
        super().__init__()
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.value_fn = value_fn

    def predict(self, state):
        self.n_predict_calls += 1
        size = int(np.prod(self.shape))
        policy = np.full(self.shape, 1.0 / size, dtype=np.float64)
        return policy, np.array(float(self.value_fn(state)), dtype=np.float64)
