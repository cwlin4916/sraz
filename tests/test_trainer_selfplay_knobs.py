"""Trainer -> self-play plumbing for the Dirichlet-noise and temperature knobs.

`Trainer` used to call `play_for_experience` without either argument, so
self-play always ran with noise on at the agent's own temperature. Turning
noise off is a precondition for measuring structural search, so these tests
pin that the settings actually arrive rather than being silently defaulted.
"""

import json

import numpy as np
import pytest

from sraz.core.policy_value_net import UniformPolicyValueNet
from sraz.instances.symreg.config import SymRegConfig
from sraz.instances.symreg.targets import get_target
from sraz.training.trainer import Trainer


class _SpyAgent:
    """Records the keyword arguments self-play is invoked with."""

    def __init__(self, n_actions=8):
        self.net = UniformPolicyValueNet(n_actions=n_actions)
        self.n_actions = n_actions
        self.calls = []
        self._seeds = iter(range(1000, 2000))

    def _randseed(self, name):
        return next(self._seeds)

    def play_for_experience(self, game, id, reset_seed, interaction_seed,
                            add_noise=True, temperature_override=None):
        self.calls.append({"id": id, "add_noise": add_noise,
                           "temperature_override": temperature_override,
                           "reset_seed": reset_seed,
                           "interaction_seed": interaction_seed})
        experience = [(np.zeros(4), np.ones(self.n_actions) / self.n_actions, 1.0)]
        return experience, 1.0, [{"rule": "C0"}], None

    def play_for_experience_reuse_tree(self, *a, **kw):
        return self.play_for_experience(*a, **kw)


class _StubGame:
    pass


def make_trainer(**kwargs):
    agent = _SpyAgent()
    trainer = Trainer(agent=agent, net=agent.net, game=_StubGame(),
                      n_games_per_train=3, n_past_iterations_to_train=2,
                      n_procs=-1, checkpoint_dir="/tmp/_unused_ckpt", **kwargs)
    return agent, trainer


def test_noise_defaults_to_on():
    """The learned runs documented in the first note rely on this default."""
    agent, trainer = make_trainer()
    trainer.train_iteration()
    assert len(agent.calls) == 3
    assert all(c["add_noise"] is True for c in agent.calls)
    assert all(c["temperature_override"] is None for c in agent.calls)


def test_noise_off_reaches_every_self_play_game():
    agent, trainer = make_trainer(self_play_add_noise=False)
    trainer.train_iteration()
    assert agent.calls and all(c["add_noise"] is False for c in agent.calls)


def test_self_play_temperature_reaches_every_game():
    agent, trainer = make_trainer(self_play_add_noise=False, self_play_temperature=1.0)
    trainer.train_iteration()
    assert all(c["temperature_override"] == 1.0 for c in agent.calls)


def test_knobs_persist_across_iterations():
    agent, trainer = make_trainer(self_play_add_noise=False, self_play_temperature=1.0)
    trainer.train_iteration()
    trainer.train_iteration()
    assert len(agent.calls) == 6
    assert all(c["add_noise"] is False for c in agent.calls)


def test_each_game_still_gets_its_own_seeds():
    """Binding the new kwargs must not disturb the positional seed arguments."""
    agent, trainer = make_trainer(self_play_add_noise=False)
    trainer.train_iteration()
    assert [c["id"] for c in agent.calls] == [0, 1, 2]
    seeds = [(c["reset_seed"], c["interaction_seed"]) for c in agent.calls]
    assert len(set(seeds)) == 3          # distinct per game
    assert all(isinstance(s, int) for pair in seeds for s in pair)


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

def test_config_defaults_keep_the_learned_setup():
    cfg = SymRegConfig()
    assert cfg.trainer.self_play_add_noise is True
    assert cfg.trainer.self_play_temperature is None
    assert cfg.agent.mcts_params["backup_rule"] == "mean"
    game, net, agent, trainer = cfg.build()
    assert not isinstance(net, UniformPolicyValueNet)
    assert trainer.self_play_add_noise is True


def test_use_uniform_net_builds_a_pure_uct_run():
    cfg = SymRegConfig().use_uniform_net()
    cfg.trainer.self_play_add_noise = False
    cfg.trainer.self_play_temperature = 1.0
    cfg.agent.mcts_params["backup_rule"] = "max"
    game, net, agent, trainer = cfg.build()

    assert isinstance(net, UniformPolicyValueNet)
    assert net.n_actions == game.state_len * game.grammar.nprods == 105
    assert agent.net is net and trainer.net is net
    assert trainer.self_play_add_noise is False
    assert trainer.self_play_temperature == 1.0
    assert agent.mcts_params["backup_rule"] == "max"


def test_use_uniform_net_is_chainable_and_returns_the_config():
    cfg = SymRegConfig()
    assert cfg.use_uniform_net() is cfg


def test_config_carries_target_and_nfev_into_the_game():
    cfg = SymRegConfig().use_uniform_net()
    cfg.game.kwargs |= {"target": "quad_D", "lmfit_max_nfev": 50}
    game, net, agent, trainer = cfg.build()
    assert game.target.name == "quad_D"
    assert game.lmfit_max_nfev == 50
    assert (game.xs[0], game.xs[-1]) == (-1.0, 1.0)


@pytest.mark.parametrize("target, expected", [
    ("quad_A", "quad_A"),                                  # name: kept verbatim
    (get_target("quad_A"), {"name": "quad_A", "family": "quadratic"}),  # object: unpacked
])
def test_config_with_a_target_is_serialisable(tmp_path, target, expected):
    """cfg.save() runs on every sweep cell, so neither spelling of `target`
    may break JSON serialisation."""
    cfg = SymRegConfig().use_uniform_net()
    cfg.game.kwargs |= {"target": target, "lmfit_max_nfev": 50}
    out = tmp_path / "config.json"
    cfg.save(str(out))

    blob = json.loads(out.read_text())
    assert blob["trainer"]["self_play_add_noise"] is True
    assert blob["game"]["kwargs"]["lmfit_max_nfev"] == 50
    got = blob["game"]["kwargs"]["target"]
    if isinstance(expected, str):
        assert got == expected
    else:
        assert {k: got[k] for k in expected} == expected
        assert got["coeffs"] == [1.0, -1.0, 2.0]
