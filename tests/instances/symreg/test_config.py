"""instances/symreg/config.py: SymRegConfig defaults and build() wiring."""

import json
from dataclasses import is_dataclass

import numpy as np
import pytest
import torch

from sraz.core.agent import Agent
from sraz.instances.symreg.config import SymRegConfig
from sraz.instances.symreg.game import SymRegGame
from sraz.instances.symreg.network import SymRegPolicyValueNet
from sraz.training.trainer import Trainer


def test_symreg_config_defaults():
    cfg = SymRegConfig()
    assert is_dataclass(cfg)
    assert cfg.game.game_cls is SymRegGame
    assert cfg.game.kwargs == {"max_len": 15, "redraw_constants": False, "problem_seed": 0}
    assert cfg.net.net_cls is SymRegPolicyValueNet
    assert cfg.net.kwargs == {}
    assert cfg.agent.mcts_params == {
        "n_simulations": 25, "temperature": 1.0, "c_exploration": 1.0,
    }
    assert cfg.agent.random_seeds == {"mcts": 0, "train": 1, "eval": 2, "external_policy": 3}
    assert (cfg.trainer.n_games_per_train, cfg.trainer.n_past_iterations_to_train) == (20, 10)
    assert cfg.trainer.n_procs == -1 and cfg.evaluator.n_procs == -1
    assert cfg.evaluator.n_games == 1
    assert cfg.run.n_iterations == 10
    assert cfg.run.plot_path == "symreg_training_metrics.png"
    assert cfg.run.accept_threshold == 0.55  # inherited RunConfig default


@pytest.fixture
def built(tmp_path):
    torch.manual_seed(0)  # net weight init draws from the global torch RNG
    np.random.seed(0)
    cfg = SymRegConfig()
    cfg.trainer.checkpoint_dir = str(tmp_path / "ckpt")  # keep any I/O in tmp
    game, net, agent, trainer = cfg.build()
    return cfg, game, net, agent, trainer


def test_build_returns_wired_objects(built):
    cfg, game, net, agent, trainer = built
    assert isinstance(game, SymRegGame)
    assert isinstance(net, SymRegPolicyValueNet)
    assert isinstance(agent, Agent)
    assert isinstance(trainer, Trainer)
    # the same objects are shared across the stack, not copies
    assert agent.game is game and agent.net is net
    assert trainer.agent is agent and trainer.net is net and trainer.game is game
    # build() passes the config's own dicts through: the agent aliases them
    assert agent.mcts_params is cfg.agent.mcts_params
    assert agent.random_seeds is cfg.agent.random_seeds
    assert agent.reward_discount == 1.0
    assert trainer.n_games_per_train == 20
    assert trainer.n_past_iterations_to_train == 10
    assert trainer.n_procs == -1


def test_build_action_space_sizes_consistent(built):
    _, game, net, _, _ = built
    g = game.grammar
    assert game.state_len == 15
    assert (g.nsym, g.nprods) == (11, 7)
    assert net.state_len == game.state_len
    assert net.n_tokens == g.nsym + 1  # vocabulary includes the pad token
    assert net.n_actions == game.state_len * g.nprods == 105
    assert int(game.action_space.n) == net.n_actions
    # observation slots range over the same vocabulary the net one-hot encodes
    assert all(int(n) == net.n_tokens for n in game.observation_space.nvec)
    assert net.model.input_size == net.state_len * net.n_tokens


def test_built_net_predicts_over_game_action_space(built):
    _, game, net, _, _ = built
    obs, _ = game.reset_wrapper(seed=0)
    policy, value = net.predict(obs)
    assert policy.shape == (int(game.action_space.n),)
    assert (policy >= 0).all()
    assert np.isfinite(policy).all()
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert value.shape == ()
    assert np.isfinite(value)
    # the action mask lives in the same flat action space as the policy
    assert game.get_action_mask().shape == policy.shape


def test_build_respects_game_kwargs_override(tmp_path):
    torch.manual_seed(0)
    cfg = SymRegConfig()
    cfg.trainer.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.game.kwargs["max_len"] = 9
    game, net, _, _ = cfg.build()
    assert game.state_len == 9
    assert net.n_actions == 9 * game.grammar.nprods == int(game.action_space.n)


def test_build_net_kwargs_override_computed_defaults(tmp_path):
    # build() merges with `computed | self.net.kwargs`, so user kwargs win.
    torch.manual_seed(0)
    cfg = SymRegConfig()
    cfg.trainer.checkpoint_dir = str(tmp_path / "ckpt")
    cfg.net.kwargs = {"hidden_size": 8, "n_hidden_layers": 1, "n_actions": 42}
    _, net, _, _ = cfg.build()
    assert net.n_actions == 42
    assert net.model.output_size == 42


def test_symreg_config_save_serializes_classes_as_strings(tmp_path):
    cfg = SymRegConfig()
    path = tmp_path / "symreg.json"
    cfg.save(path)
    loaded = json.loads(path.read_text())
    assert loaded["game"]["game_cls"] == "<callable: sraz.instances.symreg.game.SymRegGame>"
    assert loaded["net"]["net_cls"] == "<callable: sraz.instances.symreg.network.SymRegPolicyValueNet>"
    assert loaded["agent"]["mcts_params"]["n_simulations"] == 25
    assert loaded["trainer"]["n_procs"] == -1
