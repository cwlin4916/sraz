"""Config dataclasses (core) and SymRegConfig.build() wiring (symreg instance)."""

import inspect
import json
from dataclasses import fields, is_dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from sraz.core.agent import Agent
from sraz.core.config import (
    AgentConfig,
    EvaluatorConfig,
    GameConfig,
    MetaConfig,
    NetConfig,
    RunConfig,
    TrainerConfig,
)
from sraz.core.mcts import MCTS
from sraz.instances.symreg.config import SymRegConfig
from sraz.instances.symreg.game import SymRegGame
from sraz.instances.symreg.network import SymRegPolicyValueNet
from sraz.training.trainer import Trainer


class ConcreteConfig(MetaConfig):
    """Minimal concrete MetaConfig used to exercise the ABC surface."""

    def build(self):
        return ("game", "net", "agent", "trainer", "evaluator")


# ---------------------------------------------------------------------------
# Leaf dataclass defaults
# ---------------------------------------------------------------------------

def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.mcts_params == {
        "n_simulations": 100,
        "temperature": 1.0,
        "c_exploration": 1.0,
    }
    assert cfg.reward_discount == 1.0
    assert cfg.external_policy is None
    assert cfg.random_seeds == {}


def test_agent_config_mutable_defaults_independent():
    a, b = AgentConfig(), AgentConfig()
    a.mcts_params["n_simulations"] = 7
    a.random_seeds["mcts"] = 0
    assert b.mcts_params["n_simulations"] == 100
    assert b.random_seeds == {}


def test_agent_config_default_mcts_params_are_valid_mcts_kwargs():
    # The default dict must stay constructible as MCTS(**mcts_params) kwargs.
    mcts_kwargs = set(inspect.signature(MCTS.__init__).parameters) - {"self", "game", "net"}
    assert set(AgentConfig().mcts_params) <= mcts_kwargs


def test_trainer_evaluator_run_config_defaults():
    tr = TrainerConfig()
    assert (tr.n_games_per_train, tr.n_past_iterations_to_train, tr.n_procs) == (100, 20, 8)
    assert tr.checkpoint_dir == "checkpoints"

    ev = EvaluatorConfig()
    assert (ev.n_games, ev.n_procs) == (20, 8)

    run = RunConfig()
    assert run.n_iterations == 100
    assert run.accept_threshold == 0.55
    assert run.plot_every == 5
    assert run.plot_path == "training_metrics.png"


def test_game_and_net_config_defaults_independent():
    g1, g2 = GameConfig(), GameConfig()
    n1, n2 = NetConfig(), NetConfig()
    assert g1.game_cls is None and n1.net_cls is None
    g1.kwargs["max_len"] = 9
    n1.kwargs["hidden_size"] = 4
    assert g2.kwargs == {} and n2.kwargs == {}


# ---------------------------------------------------------------------------
# MetaConfig ABC contract
# ---------------------------------------------------------------------------

def test_meta_config_is_abstract():
    with pytest.raises(TypeError, match="build"):
        MetaConfig()


def test_meta_config_subclass_gets_default_subconfigs():
    cfg = ConcreteConfig()
    assert isinstance(cfg.game, GameConfig)
    assert isinstance(cfg.net, NetConfig)
    assert isinstance(cfg.agent, AgentConfig)
    assert isinstance(cfg.trainer, TrainerConfig)
    assert isinstance(cfg.evaluator, EvaluatorConfig)
    assert isinstance(cfg.run, RunConfig)
    # sub-config default factories are per-instance, not shared
    other = ConcreteConfig()
    cfg.agent.mcts_params["n_simulations"] = 1
    assert other.agent.mcts_params["n_simulations"] == 100


def test_meta_config_field_names():
    assert [f.name for f in fields(MetaConfig)] == [
        "game", "net", "agent", "trainer", "evaluator", "run",
    ]


def test_to_serializable_dict_stringifies_callables():
    cfg = ConcreteConfig()
    cfg.game = GameConfig(game_cls=SymRegGame, kwargs={"max_len": 15})
    cfg.agent.external_policy = len
    d = cfg._to_serializable_dict()
    assert d["game"]["game_cls"] == "<callable: sraz.instances.symreg.game.SymRegGame>"
    assert d["game"]["kwargs"] == {"max_len": 15}
    assert d["agent"]["external_policy"] == "<callable: builtins.len>"
    # scalars pass through unchanged, and the whole dict must be JSON-safe
    assert d["run"]["accept_threshold"] == 0.55
    json.dumps(d)


def test_to_serializable_dict_converts_tuples_and_falls_back_to_str():
    cfg = ConcreteConfig()
    cfg.game = GameConfig(kwargs={"sizes": (1, 2), "where": Path("x")})
    d = cfg._to_serializable_dict()
    # tuples become JSON lists; non-callable non-primitives fall back to str()
    assert d["game"]["kwargs"]["sizes"] == [1, 2]
    assert d["game"]["kwargs"]["where"] == "x"
    json.dumps(d)


def test_save_round_trip_creates_parents(tmp_path):
    cfg = ConcreteConfig()
    path = tmp_path / "nested" / "dir" / "config.json"
    cfg.save(path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert set(loaded) == {"game", "net", "agent", "trainer", "evaluator", "run"}
    assert loaded["trainer"]["n_games_per_train"] == 100
    assert loaded["agent"]["mcts_params"]["temperature"] == 1.0
    assert loaded["game"]["game_cls"] is None


def test_plot_writes_figure(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    cfg = ConcreteConfig()
    out = tmp_path / "figs" / "config.png"
    cfg.plot(save_path=out)
    assert out.exists()
    # plot() saves via savefig(...png); the file must actually be a PNG
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# SymRegConfig defaults and build() wiring
# ---------------------------------------------------------------------------

def test_symreg_config_defaults():
    cfg = SymRegConfig()
    assert is_dataclass(cfg)
    assert cfg.game.game_cls is SymRegGame
    # target=None / lmfit_max_nfev=None keep the default sinusoid instance on
    # [1, 3] with an uncapped inner optimizer -- the setup the first note's
    # numbers were measured on.
    assert cfg.game.kwargs == {
        "max_len": 15, "redraw_constants": False, "problem_seed": 0,
        "target": None, "lmfit_max_nfev": None,
    }
    assert cfg.net.net_cls is SymRegPolicyValueNet
    assert cfg.net.kwargs == {}
    assert cfg.agent.mcts_params == {
        "n_simulations": 25, "temperature": 1.0, "c_exploration": 1.0,
        "backup_rule": "mean",
    }
    assert cfg.trainer.self_play_add_noise is True
    assert cfg.trainer.self_play_temperature is None
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
