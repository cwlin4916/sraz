"""Behavioral tests for ``sraz.core.agent.Agent``.

Coverage:

- ``Agent.policy``: flat normalized output, temperature handling
  (low temperature ~ argmax vs temperature=1 visit-count sampling),
  Dirichlet-noise path (add_noise=True), input-state immutability,
  external-policy path, flatness assertion.
- ``play_one_round``: termination, experience-tuple shape, obs copies,
  discounted-reward math, ``max_moves`` truncation, seed determinism.
- ``play_one_round_reuse_tree``: same contract as ``play_one_round``
  (documenting the one known divergence in the returned total reward).
- ``play_for_experience`` / ``play_for_experience_reuse_tree``: 4-tuple shape,
  reset-seed plumbing, same-seeds -> same-trajectory determinism.
- RNG plumbing: ``_construct_rngs`` / ``_randseed`` determinism,
  ``_extract_leaf_eval_data``.

All tests use a tiny deterministic ChainGame and a uniform-prior stub net, so
no torch training or GPU work is involved.
"""

from __future__ import annotations

import warnings

import gymnasium.spaces as spaces
import numpy as np
import pytest

from sraz.core.agent import Agent
from sraz.core.game import Game
from sraz.core.policy_value_net import PolicyValueNet


# ---------------------------------------------------------------------------
# Toy Game / net shims
# ---------------------------------------------------------------------------


class ChainGame(Game):
    """Deterministic chain of ``length`` steps with a 2-action space.

    Action 0 gives reward 1.0, action 1 gives reward 0.0; the episode
    terminates after ``length`` steps regardless of actions. ``reset(seed=s)``
    stores ``s % 5`` as a start offset in the observation so tests can verify
    that reset seeds are plumbed through. obs = np.array([start, t]).
    """

    def __init__(self, length: int = 3) -> None:
        super().__init__()
        self.length = length
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(2,), dtype=np.int64)
        self.reset_wrapper()

    def reset(self, seed=None, **_):
        self.start = 0 if seed is None else int(seed) % 5
        self.t = 0
        return np.array([self.start, self.t], dtype=np.int64), {}

    def step(self, action):
        assert action in (0, 1)
        self.t += 1
        reward = 1.0 if action == 0 else 0.0
        terminated = self.t >= self.length
        obs = np.array([self.start, self.t], dtype=np.int64)
        return obs, reward, terminated, False, {"action": int(action), "t": self.t}

    def get_action_mask(self):
        return np.array([1, 1], dtype=bool)


class UniformNet(PolicyValueNet):
    """Uniform-prior, zero-value stub net (fresh arrays every call)."""

    def __init__(self, action_size: int) -> None:
        self.action_size = action_size

    def predict(self, _state):
        return (np.ones(self.action_size, dtype=np.float64) / self.action_size,
                np.array(0.0, dtype=np.float64))

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


MCTS_PARAMS = dict(n_simulations=16, temperature=1.0, c_exploration=1.5)


@pytest.fixture
def game():
    return ChainGame(length=3)


@pytest.fixture
def agent(game):
    return Agent(game, UniformNet(2), mcts_params=dict(MCTS_PARAMS))


def _check_experience_contract(experience, step_infos, game_length):
    """Shared shape assertions for the experience list of one full episode."""
    assert len(experience) == game_length
    assert len(step_infos) == game_length
    for i, item in enumerate(experience):
        assert isinstance(item, tuple) and len(item) == 3
        obs, move_probs, disc_reward = item
        assert isinstance(obs, np.ndarray) and obs.shape == (2,)
        assert obs[1] == i  # obs recorded BEFORE the i-th move
        assert isinstance(move_probs, np.ndarray)
        assert move_probs.shape == (2,)
        assert np.all(np.isfinite(move_probs)) and np.all(move_probs >= 0)
        assert np.isclose(move_probs.sum(), 1.0)
        assert isinstance(disc_reward, float)


# ---------------------------------------------------------------------------
# Agent.policy
# ---------------------------------------------------------------------------


def test_policy_returns_flat_normalized_distribution(agent, game):
    np.random.seed(0)
    probs = agent.policy(game, add_noise=False)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2,)
    assert np.all(np.isfinite(probs)) and np.all(probs >= 0)
    assert np.isclose(probs.sum(), 1.0)
    # Action 0 has reward 1.0 vs 0.0, so visits (and thus mass) favor it.
    assert probs[0] > probs[1]


def test_policy_low_temperature_near_argmax_vs_temp_one_sampling(agent, game):
    """temperature=1 keeps mass on both actions (visit-count proportional
    sampling distribution); a low override concentrates near-greedily on the
    argmax action. Exact temperature 0 is unsupported (see the NaN test)."""
    np.random.seed(0)
    probs_t1 = agent.policy(game, add_noise=False)  # mcts_params temperature=1.0
    probs_low = agent.policy(game, add_noise=False, temperature_override=0.05)

    # At T=1 both actions retain positive probability (exploration visits).
    assert probs_t1[0] > 0 and probs_t1[1] > 0

    # At T=0.05 the best action takes essentially all the mass.
    assert int(np.argmax(probs_low)) == 0
    assert probs_low[0] > 0.99
    # And the low-temperature distribution is strictly more concentrated.
    assert probs_low[0] > probs_t1[0]


def test_policy_temperature_zero_is_a_hard_argmax(agent, game):
    """temperature_override=0.0 is the greedy limit of counts**(1/T): all mass
    on the most-visited action.

    This previously divided the log visit counts by zero and returned NaN, so
    callers had to fake greedy selection with a small positive temperature
    (e.g. 0.05). MCTS._temperature_probs_with_fallback now special-cases T = 0
    before the dividing path.
    """
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # no divide-by-zero
        probs = agent.policy(game, add_noise=False, temperature_override=0.0)
    assert probs.shape == (2,)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)
    assert (probs > 0).sum() == 1
    assert set(probs.tolist()) == {0.0, 1.0}


def test_policy_temperature_zero_agrees_with_a_tiny_temperature(agent, game):
    """The T -> 0 limit: a very small positive temperature already selects the
    same action, so the special case is continuous with the general path."""
    np.random.seed(0)
    greedy = agent.policy(game, add_noise=False, temperature_override=0.0)
    np.random.seed(0)
    nearly = agent.policy(game, add_noise=False, temperature_override=0.01)
    assert int(np.argmax(greedy)) == int(np.argmax(nearly))


def test_policy_add_noise_true_perturbs_and_is_reproducible_via_injected_rng(agent, game):
    """The self-play default add_noise=True injects Dirichlet noise at the MCTS
    root. That randomness is drawn from the injected ``rng`` (NOT the process-
    global ``np.random``), so passing a freshly seeded Generator makes the
    result reproducible, still a valid distribution, and (at seed 3) with visit
    counts differing from the noiseless search."""
    probs_clean = agent.policy(game, add_noise=False)

    probs_noisy = agent.policy(game, add_noise=True, rng=np.random.default_rng(3))
    probs_noisy_again = agent.policy(game, add_noise=True, rng=np.random.default_rng(3))

    assert probs_noisy.shape == (2,)
    assert np.all(np.isfinite(probs_noisy)) and np.all(probs_noisy >= 0)
    assert np.isclose(probs_noisy.sum(), 1.0)
    np.testing.assert_allclose(probs_noisy, probs_noisy_again)
    assert not np.allclose(probs_noisy, probs_clean)


def test_policy_does_not_mutate_input_state(agent, game):
    np.random.seed(0)
    obs_before = game.obs.copy()
    step_count_before = game.step_count
    agent.policy(game, add_noise=False)
    np.testing.assert_array_equal(game.obs, obs_before)
    assert game.step_count == step_count_before
    assert game.t == 0


def test_policy_uses_external_policy_on_a_clone(game):
    received = []

    def external(state):
        received.append(state)
        return np.array([0.25, 0.75])

    agent = Agent(game, UniformNet(2), mcts_params={}, external_policy=external)
    probs = agent.policy(game)
    np.testing.assert_allclose(probs, [0.25, 0.75])
    # The external policy is handed a clone, never the caller's object --
    # a distinct Game carrying the same observation.
    assert len(received) == 1
    assert received[0] is not game
    assert isinstance(received[0], ChainGame)
    np.testing.assert_array_equal(received[0].obs, game.obs)
    assert received[0].obs is not game.obs


def test_policy_rejects_non_flat_move_probs(game):
    agent = Agent(game, UniformNet(2), mcts_params={},
                  external_policy=lambda s: np.full((2, 2), 0.25))
    with pytest.raises(AssertionError, match="flat"):
        agent.policy(game)


# ---------------------------------------------------------------------------
# play_one_round
# ---------------------------------------------------------------------------


def test_play_one_round_terminates_with_correct_shapes(agent, game):
    np.random.seed(0)
    experience, total, step_infos = agent.play_one_round(
        game, random_seed=0, add_noise=False)
    _check_experience_contract(experience, step_infos, game_length=3)
    assert isinstance(total, float)
    # Rewards are 0/1 per step, three steps.
    assert 0.0 <= total <= 3.0
    # The caller's game object is untouched (Agent plays on a clone).
    assert game.t == 0


def test_play_one_round_obs_are_independent_copies(agent, game):
    np.random.seed(0)
    experience, _, _ = agent.play_one_round(game, random_seed=0, add_noise=False)
    obs_list = [obs for (obs, _, _) in experience]
    # Distinct array objects, and mutating one does not affect the others.
    assert obs_list[0] is not obs_list[1]
    saved = obs_list[1].copy()
    obs_list[0][:] = -1
    np.testing.assert_array_equal(obs_list[1], saved)


def test_play_one_round_discounted_rewards_exact():
    """With a deterministic always-action-0 external policy every step earns
    reward 1.0, so with discount 0.5 over 3 steps the per-step discounted
    returns are [1.75, 1.5, 1.0].

    NOTE (documented current behavior): the second return value of
    play_one_round is the *discounted* return from step 0 (the
    ``cumulative_reward`` variable is overwritten by the discounting loop at
    src/sraz/core/agent.py:143-152), not the plain reward sum — which would
    be 3.0 here. play_one_round_reuse_tree returns the plain sum instead
    (agent.py:207); the inconsistency is reported as a suspected bug."""
    game = ChainGame(length=3)
    agent = Agent(game, UniformNet(2), mcts_params={}, reward_discount=0.5,
                  external_policy=lambda s: np.array([1.0, 0.0]))
    experience, total, step_infos = agent.play_one_round(game, random_seed=0)
    assert [info["action"] for info in step_infos] == [0, 0, 0]
    np.testing.assert_allclose([e[2] for e in experience], [1.75, 1.5, 1.0])
    assert total == pytest.approx(1.75)  # discounted return, NOT the sum 3.0


def test_play_one_round_undiscounted_total_equals_reward_sum(agent, game):
    """With reward_discount=1.0 (the default) the returned total coincides
    with the plain reward sum and with experience[0][2]."""
    np.random.seed(0)
    experience, total, step_infos = agent.play_one_round(
        game, random_seed=0, add_noise=False)
    reward_sum = sum(1.0 if info["action"] == 0 else 0.0 for info in step_infos)
    assert total == pytest.approx(reward_sum)
    assert experience[0][2] == pytest.approx(total)


def test_play_one_round_max_moves_truncates():
    game = ChainGame(length=5)
    agent = Agent(game, UniformNet(2), mcts_params={},
                  external_policy=lambda s: np.array([1.0, 0.0]))
    experience, total, step_infos = agent.play_one_round(
        game, max_moves=2, random_seed=0)
    assert len(experience) == 2
    assert len(step_infos) == 2
    assert total == pytest.approx(2.0)


def test_play_one_round_random_seed_determinism_and_variation():
    """Same random_seed -> identical action sequence; two seeds verified (at
    test-authoring time) to differ produce different trajectories. Uses a
    stochastic uniform external policy so the interaction RNG matters."""
    game = ChainGame(length=6)
    agent = Agent(game, UniformNet(2), mcts_params={},
                  external_policy=lambda s: np.array([0.5, 0.5]))

    def actions(seed):
        _, _, infos = agent.play_one_round(game, random_seed=seed)
        return [info["action"] for info in infos]

    assert actions(0) == actions(0)
    assert actions(1) == actions(1)
    # np.random.default_rng(seed) is stable across runs; seeds 0 and 1 give
    # different length-6 action sequences.
    assert actions(0) != actions(1)


# ---------------------------------------------------------------------------
# play_one_round_reuse_tree
# ---------------------------------------------------------------------------


def test_play_one_round_reuse_tree_same_contract(agent, game):
    """Reuse-tree variant honors the same return contract: (experience list
    of (obs, move_probs, discounted_reward), total reward, step infos), one
    entry per move, terminating at the game's natural end."""
    np.random.seed(0)
    experience, total, step_infos = agent.play_one_round_reuse_tree(
        game, random_seed=0, add_noise=False)
    _check_experience_contract(experience, step_infos, game_length=3)
    assert isinstance(total, float)
    assert 0.0 <= total <= 3.0
    assert game.t == 0  # caller's game untouched


def test_play_one_round_reuse_tree_total_is_plain_reward_sum():
    """Documented divergence from play_one_round: the reuse-tree variant
    returns sum(collected_rewards) (agent.py:207), NOT the discounted return.

    Ground truth comes from the game semantics: ChainGame pays 1.0 for
    action 0 and 0.0 for action 1, and records the action in step info."""
    gamma = 0.5
    game = ChainGame(length=3)
    agent = Agent(game, UniformNet(2), mcts_params=dict(MCTS_PARAMS),
                  reward_discount=gamma)
    np.random.seed(0)
    experience, total, step_infos = agent.play_one_round_reuse_tree(
        game, random_seed=0, add_noise=False)
    rewards = [1.0 if info["action"] == 0 else 0.0 for info in step_infos]
    assert total == pytest.approx(sum(rewards))
    # Experience tuples carry the discounted returns of those same rewards.
    expected_d = []
    acc = 0.0
    for r in reversed(rewards):
        acc = r + gamma * acc
        expected_d.append(acc)
    expected_d.reverse()
    np.testing.assert_allclose([e[2] for e in experience], expected_d)
    # The two contracts genuinely diverge on this trajectory (all rewards 1.0
    # under seed 0: total == 3.0 while the step-0 discounted return is 1.75).
    assert total != pytest.approx(experience[0][2])


def test_play_one_round_reuse_tree_max_moves_truncates():
    game = ChainGame(length=5)
    agent = Agent(game, UniformNet(2), mcts_params=dict(MCTS_PARAMS))
    np.random.seed(0)
    experience, _, step_infos = agent.play_one_round_reuse_tree(
        game, max_moves=2, random_seed=0, add_noise=False)
    assert len(experience) == 2
    assert len(step_infos) == 2
    # The two recorded observations are the pre-move states t=0 and t=1.
    assert [int(obs[1]) for (obs, _, _) in experience] == [0, 1]
    # Caller's game object untouched (played on a clone inside the MCTS).
    assert game.t == 0


# ---------------------------------------------------------------------------
# play_for_experience / play_for_experience_reuse_tree
# ---------------------------------------------------------------------------


def test_play_for_experience_shape_and_reset_seed_plumbing(agent, game):
    np.random.seed(0)
    result = agent.play_for_experience(
        game, id=0, reset_seed=3, interaction_seed=7, add_noise=False)
    assert isinstance(result, tuple) and len(result) == 4
    experience, total, step_infos, leaf_data = result
    _check_experience_contract(experience, step_infos, game_length=3)
    assert isinstance(total, float)
    # reset(seed=3) sets start offset 3 % 5 == 3 in the observation.
    assert all(obs[0] == 3 for (obs, _, _) in experience)
    # ChainGame has no leaf_evaluator, so leaf data is None.
    assert leaf_data is None


def test_play_for_experience_matches_manual_reset_plus_play_one_round(agent, game):
    """play_for_experience(reset_seed, interaction_seed) is exactly
    clone + reset_wrapper(seed=reset_seed) + play_one_round(random_seed=
    interaction_seed) plus the leaf-data 4th element -- and rerunning with the
    same seeds reproduces the trajectory (no hidden agent-level RNG state)."""
    def run():
        np.random.seed(0)
        return agent.play_for_experience(
            game, id=0, reset_seed=11, interaction_seed=42, add_noise=False)

    exp_a, total_a, infos_a, leaf_a = run()
    exp_b, total_b, infos_b, leaf_b = run()
    assert total_a == total_b
    assert infos_a == infos_b
    assert leaf_a is None and leaf_b is None
    assert len(exp_a) == len(exp_b)
    for (obs_a, probs_a, r_a), (obs_b, probs_b, r_b) in zip(exp_a, exp_b):
        np.testing.assert_array_equal(obs_a, obs_b)
        np.testing.assert_allclose(probs_a, probs_b)
        assert r_a == pytest.approx(r_b)

    # Manual composition reproduces the same trajectory.
    np.random.seed(0)
    manual = game.clone()
    manual.reset_wrapper(seed=11)
    exp_m, total_m, infos_m = agent.play_one_round(
        manual, random_seed=42, add_noise=False)
    assert total_m == total_a
    assert infos_m == infos_a
    assert len(exp_m) == len(exp_a)
    for (obs_m, probs_m, r_m), (obs_a_, probs_a_, r_a_) in zip(exp_m, exp_a):
        np.testing.assert_array_equal(obs_m, obs_a_)
        np.testing.assert_allclose(probs_m, probs_a_)
        assert r_m == pytest.approx(r_a_)


def test_play_for_experience_reuse_tree_shape(agent, game):
    np.random.seed(0)
    result = agent.play_for_experience_reuse_tree(
        game, id=0, reset_seed=2, interaction_seed=5, add_noise=False)
    assert isinstance(result, tuple) and len(result) == 4
    experience, total, step_infos, leaf_data = result
    _check_experience_contract(experience, step_infos, game_length=3)
    assert all(obs[0] == 2 for (obs, _, _) in experience)
    assert leaf_data is None


# ---------------------------------------------------------------------------
# RNG plumbing / leaf-eval helper
# ---------------------------------------------------------------------------


def test_construct_rngs_and_randseed_determinism(game):
    seeds = {"mcts": 1, "train": 2, "eval": 3, "external_policy": 4}
    agent_a = Agent(game, UniformNet(2), mcts_params={}, random_seeds=seeds)
    agent_b = Agent(game, UniformNet(2), mcts_params={}, random_seeds=dict(seeds))

    assert set(agent_a.rngs.keys()) == set(Agent.RNG_NAMES)
    assert agent_a.random_seeds == seeds

    seq_a = [agent_a._randseed("mcts") for _ in range(5)]
    seq_b = [agent_b._randseed("mcts") for _ in range(5)]
    assert seq_a == seq_b
    assert all(0 <= s < 2**31 - 1 for s in seq_a)
    # Different named streams are independent generators.
    seq_train = [agent_b._randseed("train") for _ in range(5)]
    assert seq_train != seq_a


def test_construct_rngs_partial_seeds_still_builds_all_streams(game):
    agent = Agent(game, UniformNet(2), mcts_params={},
                  random_seeds={"mcts": 7})
    assert set(agent.rngs.keys()) == set(Agent.RNG_NAMES)
    for name in Agent.RNG_NAMES:
        s = agent._randseed(name)
        assert isinstance(s, int) and 0 <= s < 2**31 - 1
    # The stream that WAS seeded is reproducible even with the others absent.
    agent_b = Agent(game, UniformNet(2), mcts_params={},
                    random_seeds={"mcts": 7})
    agent_c = Agent(game, UniformNet(2), mcts_params={},
                    random_seeds={"mcts": 7})
    assert ([agent_b._randseed("mcts") for _ in range(5)]
            == [agent_c._randseed("mcts") for _ in range(5)])


def test_extract_leaf_eval_data(game):
    # No leaf_evaluator attribute -> None.
    assert Agent._extract_leaf_eval_data(game) is None

    # leaf_evaluator with export_caches -> its exported dict.
    class FakeLeafEvaluator:
        def export_caches(self):
            return {"cache": [1, 2, 3]}

    game2 = ChainGame(length=3)
    game2.leaf_evaluator = FakeLeafEvaluator()
    assert Agent._extract_leaf_eval_data(game2) == {"cache": [1, 2, 3]}

    # leaf_evaluator without export_caches -> None.
    game3 = ChainGame(length=3)
    game3.leaf_evaluator = object()
    assert Agent._extract_leaf_eval_data(game3) is None


# ---------------------------------------------------------------------------
# Regression: leaf-eval caches are extracted from the game actually stepped
# (fix #2). Previously play_for_experience read an unstepped outer clone, so
# export_caches() was always empty and Trainer's cross-worker merge was a no-op.
# ---------------------------------------------------------------------------


class _RecordingLeafEval:
    """Minimal leaf_evaluator that records a key on every game step."""

    def __init__(self):
        self.seen = []

    def record(self, key):
        self.seen.append(key)

    def export_caches(self):
        return {"seen": list(self.seen)}


class LeafEvalChainGame(ChainGame):
    """ChainGame whose leaf_evaluator records (t, action) on each real step."""

    def __init__(self, length=3):
        super().__init__(length)
        self.leaf_evaluator = _RecordingLeafEval()

    def step(self, action):
        out = super().step(action)
        self.leaf_evaluator.record((int(self.t), int(action)))
        return out


def test_play_for_experience_extracts_leaf_eval_from_stepped_game():
    game = LeafEvalChainGame(length=3)
    agent = Agent(game, UniformNet(2), mcts_params=dict(MCTS_PARAMS))
    *_, leaf_data = agent.play_for_experience(
        game, id=0, reset_seed=0, interaction_seed=0, add_noise=False)
    assert leaf_data is not None
    # Exactly one record per real step of the 3-step episode. The MCTS-internal
    # clones (created inside policy()) accumulate their own caches and are
    # discarded, so they must NOT be the source of the export.
    assert len(leaf_data["seen"]) == 3


def test_play_for_experience_reuse_tree_extracts_leaf_eval():
    game = LeafEvalChainGame(length=3)
    agent = Agent(game, UniformNet(2), mcts_params=dict(MCTS_PARAMS))
    *_, leaf_data = agent.play_for_experience_reuse_tree(
        game, id=0, reset_seed=0, interaction_seed=0, add_noise=False)
    assert leaf_data is not None
    # Non-empty: the reuse-tree MCTS steps this game in place via advance_to,
    # so its real-trajectory caches survive (pre-fix this was always empty).
    assert len(leaf_data["seen"]) >= 1
