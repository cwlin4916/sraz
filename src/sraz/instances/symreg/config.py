from __future__ import annotations

from dataclasses import dataclass

from sraz.instances.symreg.game import SymRegGame
from sraz.instances.symreg.network import SymRegPolicyValueNet
from sraz.core.agent import Agent
from sraz.training.trainer import Trainer

from sraz.core.config import (
    MetaConfig,
    GameConfig,
    NetConfig,
    AgentConfig,
    TrainerConfig,
    EvaluatorConfig,
    RunConfig,
)


@dataclass
class SymRegConfig(MetaConfig):
    """Configuration for symbolic-regression AlphaZero training."""

    def __init__(self):
        super().__init__()
        self.game = GameConfig(
            game_cls=SymRegGame,
            kwargs={
                "max_len": 15,
                "redraw_constants": False,
                "problem_seed": 0,
            },
        )
        self.net = NetConfig(
            net_cls=SymRegPolicyValueNet,
            kwargs={},
        )
        self.agent = AgentConfig(
            mcts_params={
                "n_simulations": 25,
                "temperature": 1.0,
                "c_exploration": 1.0,
            },
            reward_discount=1.0,
            random_seeds={
                "mcts": 0,
                "train": 1,
                "eval": 2,
                "external_policy": 3,
            },
        )
        self.trainer = TrainerConfig(
            n_games_per_train=20,
            n_past_iterations_to_train=10,
            n_procs=-1,
            checkpoint_dir="checkpoints",
        )
        self.evaluator = EvaluatorConfig(
            n_games=1,
            n_procs=-1,
        )
        self.run = RunConfig(
            n_iterations=10,
            plot_path="symreg_training_metrics.png",
        )

    def build(self):
        """Build symreg game, network, agent, and trainer (no evaluator in v1)."""
        game = SymRegGame(**self.game.kwargs)
        net_kwargs = {
            "state_len": game.state_len,
            "n_tokens": game.grammar.nsym + 1,
            "n_actions": game.state_len * game.grammar.nprods,
        } | self.net.kwargs
        net = SymRegPolicyValueNet(**net_kwargs)
        agent = Agent(
            game=game,
            net=net,
            mcts_params=self.agent.mcts_params,
            reward_discount=self.agent.reward_discount,
            external_policy=self.agent.external_policy,
            random_seeds=self.agent.random_seeds,
        )
        trainer = Trainer(
            agent=agent,
            net=net,
            game=game,
            n_games_per_train=self.trainer.n_games_per_train,
            n_past_iterations_to_train=self.trainer.n_past_iterations_to_train,
            n_procs=self.trainer.n_procs,
            checkpoint_dir=self.trainer.checkpoint_dir,
        )
        return game, net, agent, trainer
