"""Pure UCT: the uniform net, zero-temperature selection, and the regime in
which the backup rule cannot bind.

The load-bearing test here is
:func:`test_backup_rule_is_a_no_op_at_shallow_budgets`. It records that the
mean-vs-max comparison the study sets out to make is *empty* at the swept
simulation budgets, and -- via its high-budget counterpart -- that this is a
property of the regime rather than a plumbing bug.
"""

import numpy as np
import pytest

from sraz.core.mcts import MCTS
from sraz.core.policy_value_net import UniformPolicyValueNet
from sraz.instances.symreg.evaluate import greedy_eval, greedy_episode
from sraz.instances.symreg.game import SymRegGame
from sraz.instances.symreg.targets import get_target


N_ACTIONS = 15 * 7


def uct_mcts(target="quad_A", n_simulations=12, backup_rule="mean", rollout_n=0):
    game = SymRegGame(target=target, lmfit_max_nfev=50)
    game.reset_wrapper()
    net = UniformPolicyValueNet(n_actions=N_ACTIONS)
    return MCTS(game.clone(), net, n_simulations=n_simulations, temperature=1.0,
                c_exploration=1.0, backup_rule=backup_rule,
                rollout_n=rollout_n, rollout_budget=10 ** 6)


def edges_with_differing_values(mcts) -> int:
    """Edges visited more than once whose backed-up values are not all equal."""
    return sum(1 for node in mcts.nodes.values()
               for vals in node.action_values.values()
               if len({round(v, 12) for v in vals}) > 1)


# ---------------------------------------------------------------------------
# UniformPolicyValueNet
# ---------------------------------------------------------------------------

def test_uniform_net_predicts_a_normalised_uniform_prior_and_scalar_value():
    net = UniformPolicyValueNet(n_actions=N_ACTIONS)
    policy, value = net.predict(np.zeros(15, dtype=np.int64))
    assert policy.shape == (N_ACTIONS,)
    assert policy.sum() == pytest.approx(1.0)
    assert len(set(policy.tolist())) == 1          # genuinely uniform
    assert value.shape == ()                        # MCTS.query_net demands 0-d
    assert float(value) == 0.0


def test_uniform_net_value_is_configurable():
    net = UniformPolicyValueNet(n_actions=4, value=0.5)
    _, value = net.predict(np.zeros(15, dtype=np.int64))
    assert float(value) == 0.5


def test_uniform_net_rejects_an_empty_action_space():
    with pytest.raises(ValueError, match="must be positive"):
        UniformPolicyValueNet(n_actions=0)


def test_uniform_net_returns_a_fresh_array_each_call():
    """MCTS masks the prior in place, so a shared buffer would be corrupted
    after the first masked state."""
    net = UniformPolicyValueNet(n_actions=8)
    first, _ = net.predict(np.zeros(15, dtype=np.int64))
    second, _ = net.predict(np.zeros(15, dtype=np.int64))
    assert first is not second
    first *= 0.0
    assert second.sum() == pytest.approx(1.0)


def test_uniform_net_train_is_a_no_op():
    net = UniformPolicyValueNet(n_actions=8)
    before, _ = net.predict(np.zeros(15, dtype=np.int64))
    out = net.train([(np.zeros(15), np.ones(8) / 8, 1.0)] * 4)
    assert len(out) == 5           # the shape Trainer._train_network unpacks
    _, _, train_losses, policy_losses, value_losses = out
    assert train_losses == policy_losses == value_losses == [0.0]
    after, value = net.predict(np.zeros(15, dtype=np.int64))
    assert np.array_equal(before, after)   # training changed nothing
    assert float(value) == 0.0


def test_uniform_prior_survives_mcts_masking():
    """query_net_masked must see a uniform prior over the *legal* actions."""
    mcts = uct_mcts(n_simulations=8)
    mcts.perform_simulations("", add_noise=False)
    root = mcts.nodes[mcts.game.hashable_obs]
    legal = root.action_mask.astype(bool)
    assert legal.sum() == 7                      # every production applies at S
    assert root.nn_policy[legal] == pytest.approx(1.0 / 7)
    assert root.nn_policy[~legal] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Zero-temperature selection
# ---------------------------------------------------------------------------

def test_temperature_zero_is_one_hot_on_the_most_visited_action():
    mcts = uct_mcts(n_simulations=12)
    mcts.temperature = 0.0
    probs = mcts.perform_simulations("", add_noise=False)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)
    assert (probs > 0).sum() == 1                # all mass on one action

    root = mcts.nodes[mcts.game.hashable_obs]
    counts = np.zeros(N_ACTIONS)
    for a, n in root.action_N.items():
        counts[a] = n
    assert int(np.argmax(probs)) == int(np.argmax(counts))


def test_temperature_zero_breaks_ties_at_the_lowest_index():
    """Determinism under ties is what makes a greedy episode reproducible."""
    mcts = uct_mcts(n_simulations=4)
    mcts.temperature = 0.0
    counts = np.zeros(N_ACTIONS)
    counts[[3, 9, 20]] = 5      # a three-way tie
    probs = mcts._temperature_probs_with_fallback(counts, None)
    assert probs[3] == 1.0 and probs.sum() == pytest.approx(1.0)


def test_temperature_zero_does_not_divide_by_zero():
    """The general path is counts**(1/T); T = 0 must not reach it."""
    mcts = uct_mcts(n_simulations=6)
    mcts.temperature = 0.0
    with np.errstate(all="raise"):
        probs = mcts.perform_simulations("", add_noise=False)
    assert np.isfinite(probs).all()


def test_positive_temperature_still_spreads_mass():
    """Guard that the T = 0 special case did not capture the normal path."""
    mcts = uct_mcts(n_simulations=20)
    mcts.temperature = 1.0
    probs = mcts.perform_simulations("", add_noise=False)
    assert (probs > 0).sum() > 1
    assert probs.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The backup rule cannot bind at shallow budgets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_simulations", [4, 6, 8, 10, 12, 20])
def test_backup_rule_is_a_no_op_at_shallow_budgets(n_simulations):
    """mean and max agree exactly across the swept budgets.

    With the network disabled every leaf is worth 0 and the game is
    deterministic, so an edge backs up `immediate_reward(s,a) + V(s')` -- the
    same number on every visit. mean and max of a constant list coincide, so
    the two rules cannot produce different searches no matter how the visits
    fall. The comparison is empty here by construction, not by accident.
    """
    p_mean = uct_mcts(n_simulations=n_simulations, backup_rule="mean") \
        .perform_simulations("", add_noise=False)
    m_max = uct_mcts(n_simulations=n_simulations, backup_rule="max")
    p_max = m_max.perform_simulations("", add_noise=False)

    assert np.array_equal(p_mean, p_max)
    assert edges_with_differing_values(m_max) == 0


def test_backup_rule_binds_once_search_is_deep_enough():
    """The counterpart: at 200 simulations the rules do diverge.

    Without this, the no-op test above would be indistinguishable from the
    backup_rule argument never having been wired up at all.
    """
    p_mean = uct_mcts(n_simulations=200, backup_rule="mean") \
        .perform_simulations("", add_noise=False)
    m_max = uct_mcts(n_simulations=200, backup_rule="max")
    p_max = m_max.perform_simulations("", add_noise=False)

    assert edges_with_differing_values(m_max) > 0
    assert not np.array_equal(p_mean, p_max)


def test_shallow_search_only_ever_reaches_the_constant_terminal():
    """Why the greedy curve is flat at R^2 = 0: at these budgets the only
    terminal search sees is the single-token constant, worth exactly 0."""
    mcts = uct_mcts(n_simulations=12)
    mcts.debug_state = {}
    mcts.perform_simulations("", add_noise=False)
    assert mcts.debug_state["count_terminal_reward_returns"] > 0
    assert mcts.debug_state["best_terminal_reward_seen"] == pytest.approx(0.0, abs=1e-9)


def test_deep_search_reaches_a_structured_terminal():
    mcts = uct_mcts(n_simulations=200)
    mcts.debug_state = {}
    mcts.perform_simulations("", add_noise=False)
    assert mcts.debug_state["best_terminal_reward_seen"] > 0.5


# ---------------------------------------------------------------------------
# Greedy episode
# ---------------------------------------------------------------------------

class _StubAgent:
    """Minimal agent surface that greedy_episode needs."""

    def __init__(self, game, n_simulations=12, backup_rule="mean"):
        self.game = game
        self.net = UniformPolicyValueNet(n_actions=N_ACTIONS)
        self.mcts_params = {"n_simulations": n_simulations, "temperature": 1.0,
                            "c_exploration": 1.0, "backup_rule": backup_rule}
        self.calls = []

    def policy(self, state, msg=None, add_noise=True, temperature_override=None):
        self.calls.append({"add_noise": add_noise, "temperature": temperature_override})
        mcts = MCTS(state.clone(), self.net, **self.mcts_params)
        if temperature_override is not None:
            mcts.temperature = temperature_override
        return mcts.perform_simulations("", add_noise=add_noise)


def test_greedy_episode_disables_noise_and_selects_at_zero_temperature():
    agent = _StubAgent(SymRegGame(target="lin_D", lmfit_max_nfev=50))
    out = greedy_episode(agent, temperature=0.0)
    assert agent.calls, "policy was never consulted"
    assert all(c["add_noise"] is False for c in agent.calls)
    assert all(c["temperature"] == 0.0 for c in agent.calls)
    assert out["rule"] is not None and out["actions"]


def test_greedy_episode_is_reproducible():
    """No RNG is consulted, so repeated calls must agree exactly -- this is what
    licenses reading a flat greedy curve as 'frozen' rather than 'noisy'."""
    agent = _StubAgent(SymRegGame(target="quad_A", lmfit_max_nfev=50))
    first = greedy_episode(agent, temperature=0.0)
    second = greedy_episode(agent, temperature=0.0)
    assert first == second


def test_greedy_episode_finds_the_exact_structure_on_lin_D():
    """y = 2x is one production away, so even a shallow search must land it."""
    agent = _StubAgent(SymRegGame(target="lin_D", lmfit_max_nfev=50), n_simulations=10)
    out = greedy_episode(agent, temperature=0.0)
    assert out["rule"] == "* C1 x"
    assert out["reward"] == pytest.approx(1.0, abs=1e-9)


def test_greedy_eval_aggregates_episodes():
    agent = _StubAgent(SymRegGame(target="lin_D", lmfit_max_nfev=50), n_simulations=10)
    mean_r, best_r, best_rule = greedy_eval(agent, n_episodes=3, temperature=0.0)
    assert mean_r == pytest.approx(best_r)   # a fixed target makes episodes identical
    assert best_rule == "* C1 x"
