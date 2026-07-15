"""core/config.py: config dataclasses, MetaConfig ABC, serialization/save/plot."""

import inspect
import json
from dataclasses import fields
from pathlib import Path

import pytest

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
from sraz.instances.symreg.game import SymRegGame  # used only as a sample callable


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
