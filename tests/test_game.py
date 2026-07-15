"""Game ABC machinery: wrapper bookkeeping, hashable_obs, stash/unstash, clone, EnvGame."""

import copy

import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from sraz.core.game import Game, EnvGame


# --------------------------------------------------------------------------- #
# Tiny concrete Game subclass to exercise the ABC machinery
# --------------------------------------------------------------------------- #

class CountingGame(Game[int, int]):
    """Counter game: action 1 increments, anything else decrements.

    Terminates when counter reaches `target`. `history` is a mutable list so
    tests can detect shallow-vs-deep copying in stash/clone.
    """

    def __init__(self, target=3):
        super().__init__()
        self.target = target
        self.counter = 0
        self.history = []

    def reset(self, start=0):
        self.counter = start
        self.history = [start]
        return self.counter, {"start": start}

    def step(self, action):
        self.counter += 1 if action == 1 else -1
        self.history.append(self.counter)
        terminated = self.counter >= self.target
        reward = 1.0 if terminated else -0.1
        return self.counter, reward, terminated, False, {"count": self.counter}

    def get_action_mask(self):
        return [self.counter > -self.target, True]


# --------------------------------------------------------------------------- #
# Tiny gymnasium env for EnvGame
# --------------------------------------------------------------------------- #

class TinyCorridorEnv(gym.Env):
    """1-D corridor: action 1 moves right, 0 moves left (floored at 0)."""

    def __init__(self, length=3):
        self.length = length
        self.pos = 0
        self.last_reset_seed = "never-reset"
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(length + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_reset_seed = seed
        self.pos = 0
        return self.pos, {"seed": seed}

    def step(self, action):
        self.pos = max(self.pos + (1 if action == 1 else -1), 0)
        terminated = self.pos >= self.length
        reward = 1.0 if terminated else 0.0
        return self.pos, reward, terminated, False, {"pos": self.pos}

    def get_action_mask(self):
        return np.array([self.pos > 0, True])

    def render(self):
        return f"pos={self.pos}"


class NoMaskEnv(gym.Env):
    """Env without get_action_mask, to hit EnvGame's ValueError path."""

    def __init__(self):
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(1)

    def reset(self, seed=None, options=None):
        return 0, {}

    def step(self, action):
        return 0, 0.0, True, False, {}


class BuggyMaskEnv(NoMaskEnv):
    """Env whose get_action_mask exists but raises AttributeError internally."""

    def get_action_mask(self):
        return self.no_such_attribute  # AttributeError from INSIDE the impl


@pytest.fixture
def game():
    return CountingGame()


# --------------------------------------------------------------------------- #
# Construction / abstractness
# --------------------------------------------------------------------------- #

def test_game_abc_not_instantiable():
    with pytest.raises(TypeError):
        Game()


def test_new_game_fields_are_none(game):
    assert game.obs is None
    assert game.reward is None
    assert game.terminated is None
    assert game.truncated is None
    assert game.info is None
    assert game.step_count is None


# --------------------------------------------------------------------------- #
# reset_wrapper / step_wrapper bookkeeping
# --------------------------------------------------------------------------- #

def test_reset_wrapper_sets_bookkeeping(game):
    obs, info = game.reset_wrapper()
    assert (obs, info) == (0, {"start": 0})
    assert game.obs == 0
    assert game.info == {"start": 0}
    assert game.step_count == 0
    assert game.reward is None
    assert game.terminated is None
    assert game.truncated is None


def test_reset_wrapper_forwards_kwargs(game):
    obs, info = game.reset_wrapper(start=2)
    assert obs == 2
    assert info == {"start": 2}
    assert game.obs == 2


def test_step_wrapper_updates_fields_and_returns(game):
    game.reset_wrapper()
    obs, reward, term, trunc, info = game.step_wrapper(1)
    assert (obs, reward, term, trunc) == (1, -0.1, False, False)
    assert info == {"count": 1}
    # attributes mirror the returned tuple
    assert game.obs == obs
    assert game.reward == reward
    assert game.terminated is term
    assert game.truncated is trunc
    assert game.info == info
    assert game.step_count == 1
    game.step_wrapper(1)
    assert game.step_count == 2


def test_step_wrapper_to_termination(game):
    game.reset_wrapper()
    for _ in range(2):
        _, _, term, _, _ = game.step_wrapper(1)
        assert not term
    obs, reward, term, trunc, _ = game.step_wrapper(1)
    assert term and not trunc
    assert obs == 3
    assert reward == 1.0
    assert game.step_count == 3


def test_step_wrapper_before_reset_sets_step_count_zero(game):
    # Current behavior: stepping without a prior reset takes the `else 0`
    # branch, so the first step yields step_count == 0 (vs 1 after a reset).
    game.step_wrapper(1)
    assert game.step_count == 0
    game.step_wrapper(1)
    assert game.step_count == 1


def test_reset_wrapper_clears_previous_episode(game):
    game.reset_wrapper()
    for _ in range(3):
        game.step_wrapper(1)
    assert game.terminated is True
    game.reset_wrapper()
    assert game.reward is None
    assert game.terminated is None
    assert game.truncated is None
    assert game.step_count == 0
    assert game.obs == 0


# --------------------------------------------------------------------------- #
# hashable_obs
# --------------------------------------------------------------------------- #

def test_hashable_obs_passthrough_for_hashable(game):
    game.reset_wrapper(start=5)
    assert game.hashable_obs == 5
    assert {game.hashable_obs: "v"}[5] == "v"


def test_hashable_obs_nested_lists_become_tuples(game):
    game.obs = [[1, 2], [3, 4]]
    assert game.hashable_obs == ((1, 2), (3, 4))


def test_hashable_obs_numpy_array_uses_tobytes(game):
    a = np.array([1, 2, 3], dtype=np.int64)
    game.obs = a
    h1 = game.hashable_obs
    assert h1 == a.tobytes()
    # equal arrays hash equal, different arrays differ
    game.obs = np.array([1, 2, 3], dtype=np.int64)
    assert game.hashable_obs == h1
    game.obs = np.array([1, 2, 4], dtype=np.int64)
    assert game.hashable_obs != h1
    hash(h1)  # actually hashable


def test_hashable_obs_tuple_containing_arrays(game):
    a = np.array([1.0, 2.0])
    b = np.array([3.0])
    game.obs = (a, b)
    assert game.hashable_obs == (a.tobytes(), b.tobytes())


def test_hashable_obs_reshape_collision(game):
    # Caveat of the generic implementation: tobytes() drops shape, so two
    # arrays with the same dtype and flat contents but different shapes
    # collapse to the same hashable key.
    flat = np.arange(4, dtype=np.int64)
    game.obs = flat
    h_flat = game.hashable_obs
    game.obs = flat.reshape(2, 2)
    assert game.hashable_obs == h_flat


def test_hashable_obs_dict_returns_bytes_not_typeerror(game):
    # Current behavior: np.asarray wraps a dict into a 0-d object array whose
    # tobytes() is the object's pointer bytes, so the documented TypeError
    # path is not reached and equal dicts need not hash equal.
    d = {"a": 1}
    game.obs = d
    key = game.hashable_obs
    assert isinstance(key, bytes)
    hash(key)  # actually usable as a dict key
    # Same object -> same pointer -> stable key while the object is alive.
    assert game.hashable_obs == key
    # An equal-but-distinct dict lives at a different address, so it gets a
    # DIFFERENT key: equality of observations is not respected (the bug).
    game.obs = {"a": 1}
    assert game.hashable_obs != key
    del d  # keep the first dict alive until after the comparison above


# --------------------------------------------------------------------------- #
# stash_state / unstash_state
# --------------------------------------------------------------------------- #

def test_stash_unstash_round_trip(game):
    game.reset_wrapper()
    game.step_wrapper(1)
    stash = game.stash_state()
    snapshot = (game.obs, game.reward, game.terminated, game.truncated,
                game.info, game.step_count, game.counter, list(game.history))

    game.step_wrapper(1)
    game.step_wrapper(1)
    assert game.terminated is True
    assert game.step_count == 3

    returned = game.unstash_state(stash)
    assert returned is game
    assert (game.obs, game.reward, game.terminated, game.truncated,
            game.info, game.step_count, game.counter, game.history) == snapshot


def test_stash_reusable_after_unstash(game):
    # Fields that are REBOUND on step (ints, bools) survive repeated unstash
    # of the same stash object...
    game.reset_wrapper()
    stash = game.stash_state()
    game.step_wrapper(1)
    game.unstash_state(stash)
    game.step_wrapper(1)
    game.step_wrapper(1)
    game.unstash_state(stash)  # MCTS may unstash the same object again
    assert game.counter == 0
    assert game.step_count == 0
    assert game.terminated is None
    # ...but unstash_state copies REFERENCES (__dict__.update), so after the
    # first unstash game.history IS stash.history, and the in-place appends
    # from the two steps corrupted the stash. Current behavior: the second
    # unstash hands back the mutated list, not the snapshot [0].
    assert game.history is stash.history
    assert game.history == [0, 1, 2]


def test_stash_is_deep_snapshot(game):
    game.reset_wrapper()
    stash = game.stash_state()
    game.history.append(999)  # mutate in place; a shallow stash would leak
    game.unstash_state(stash)
    assert game.history == [0]


# --------------------------------------------------------------------------- #
# clone
# --------------------------------------------------------------------------- #

def test_clone_is_independent_playable_copy(game):
    game.reset_wrapper()
    game.step_wrapper(1)
    clone = game.clone()
    assert clone is not game
    assert clone.history is not game.history

    clone.step_wrapper(1)
    clone.step_wrapper(1)
    assert clone.terminated is True
    assert clone.counter == 3
    # original untouched
    assert game.counter == 1
    assert game.step_count == 1
    assert game.terminated is False
    assert game.history == [0, 1]


def test_mutating_original_does_not_affect_clone(game):
    game.reset_wrapper()
    clone = game.clone()
    game.step_wrapper(1)
    game.history.append(-42)
    assert clone.counter == 0
    assert clone.step_count == 0
    assert clone.history == [0]


# --------------------------------------------------------------------------- #
# EnvGame
# --------------------------------------------------------------------------- #

def test_envgame_aliases_spaces_and_delegates():
    env = TinyCorridorEnv()
    game = EnvGame(env)  # concrete: all abstract methods implemented
    # The constructor aliases (does not copy) the env's spaces.
    assert game.action_space is env.action_space
    assert game.observation_space is env.observation_space

    obs, info = game.reset_wrapper()
    assert obs == 0
    assert game.step_count == 0

    obs, reward, term, trunc, info = game.step_wrapper(1)
    assert obs == 1
    assert env.pos == 1  # the underlying env actually moved
    assert not term
    assert game.step_count == 1
    assert game.obs == 1
    assert game.info == {"pos": 1}


def test_envgame_reset_kwargs_passthrough():
    env = TinyCorridorEnv()
    game = EnvGame(env)
    _, info = game.reset_wrapper(seed=123)
    assert env.last_reset_seed == 123
    assert info == {"seed": 123}


def test_envgame_termination_bookkeeping():
    game = EnvGame(TinyCorridorEnv(length=2))
    game.reset_wrapper()
    game.step_wrapper(1)
    obs, reward, term, trunc, _ = game.step_wrapper(1)
    assert term and not trunc
    assert reward == 1.0
    assert game.terminated is True
    assert game.step_count == 2


def test_envgame_action_mask_delegates():
    env = TinyCorridorEnv()
    game = EnvGame(env)
    game.reset_wrapper()
    assert list(game.get_action_mask()) == [False, True]
    game.step_wrapper(1)
    assert list(game.get_action_mask()) == [True, True]


def test_envgame_action_mask_missing_raises_valueerror():
    game = EnvGame(NoMaskEnv())
    game.reset_wrapper()
    with pytest.raises(ValueError, match="get_action_mask"):
        game.get_action_mask()


def test_envgame_action_mask_internal_attributeerror_becomes_valueerror():
    # Current behavior: the `except AttributeError` in EnvGame.get_action_mask
    # is broad enough to also catch AttributeErrors raised INSIDE a real
    # get_action_mask implementation, converting a genuine bug into the
    # misleading "must implement `get_action_mask`" ValueError.
    game = EnvGame(BuggyMaskEnv())
    game.reset_wrapper()
    with pytest.raises(ValueError, match="get_action_mask"):
        game.get_action_mask()


def test_envgame_render_delegates():
    game = EnvGame(TinyCorridorEnv())
    game.reset_wrapper()
    game.step_wrapper(1)
    assert game.render() == "pos=1"


def test_envgame_clone_deepcopies_env():
    game = EnvGame(TinyCorridorEnv())
    game.reset_wrapper()
    clone = game.clone()
    assert clone.env is not game.env
    clone.step_wrapper(1)
    clone.step_wrapper(1)
    assert clone.env.pos == 2
    assert game.env.pos == 0
    assert game.step_count == 0


def test_envgame_stash_unstash_restores_env_state():
    game = EnvGame(TinyCorridorEnv())
    game.reset_wrapper()
    stash = game.stash_state()
    game.step_wrapper(1)
    game.step_wrapper(1)
    game.unstash_state(stash)
    assert game.env.pos == 0
    assert game.obs == 0
    assert game.step_count == 0
