from __future__ import annotations

from dataclasses import dataclass

from sraz.instances.symreg.game import SymRegGame
from sraz.instances.symreg.network import SymRegPolicyValueNet
from sraz.instances.symreg.problems import get_problem
from sraz.core.agent import Agent
from sraz.core.policy_value_net import UniformPolicyValueNet
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

    def __init__(self, problem: str = "sine", pure_mcts: bool = False):
        super().__init__()
        # Which named instance to build (see instances/symreg/problems.py). The
        # grammar + target are injected in build(); the config's game.kwargs
        # stays limited to max_len/redraw/problem_seed so the shipped default is
        # unchanged.
        self.problem = problem
        # pure_mcts=True builds a net-free classic MCTS (uniform prior + random
        # rollouts) instead of the learned AlphaZero net -- the "no network"
        # search baseline.
        self.pure_mcts = pure_mcts
        self.game = GameConfig(
            game_cls=SymRegGame,
            kwargs={
                "max_len": 15,
                "redraw_constants": False,
                "problem_seed": 0,
                "target": None,
                "lmfit_max_nfev": None,
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
                "backup_rule": "mean",
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

    def use_uniform_net(self, value: float = 0.0) -> "SymRegConfig":
        """Swap the learned net for a uniform prior with constant value.

        Turns the run into pure UCT: `build` still returns a Trainer, and its
        iterations still batch self-play games, but `train` is a no-op, so
        nothing carries from one iteration to the next.
        """
        self.net = NetConfig(net_cls=UniformPolicyValueNet, kwargs={"value": value})
        return self

    def build(self):
        """Build symreg game, network, agent, and trainer (no evaluator in v1)."""
        prob = get_problem(self.problem)
        prob_kwargs = prob.game_kwargs()
        if self.game.kwargs.get("target") is not None:
            # A named target supplies its own ys, grid and formula. The problem
            # still supplies the grammar its members are scored by.
            prob_kwargs.pop("target_ys_fn")
            prob_kwargs.pop("target_infix")
        # prob_kwargs supplies (grammar, target); self.game.kwargs supplies
        # max_len/redraw/problem_seed/target. No key overlap, so the merge is clean.
        game = SymRegGame(**{**prob_kwargs, **self.game.kwargs})
        n_actions = game.state_len * game.grammar.nprods
        mcts_params = self.agent.mcts_params

        # Two routes to the net-free search: `pure_mcts`, which also points leaf
        # evaluation at rollouts, and `use_uniform_net()`, which swaps the net
        # and leaves the MCTS params alone.
        uniform = self.pure_mcts or self.net.net_cls is UniformPolicyValueNet
        if uniform:
            net_kwargs = (self.net.kwargs
                          if self.net.net_cls is UniformPolicyValueNet else {})
            net = UniformPolicyValueNet(n_actions=n_actions, **net_kwargs)
            if self.pure_mcts:
                # Net-free classic MCTS: the value comes from random rollouts,
                # not from the net's constant.
                mcts_params = {**mcts_params}
                mcts_params.setdefault("rollout_n", 20)
                mcts_params["rollout_blend"] = 0.0
        else:
            net_kwargs = {
                "state_len": game.state_len,
                "n_tokens": game.grammar.nsym + 1,
                "n_actions": n_actions,
            } | self.net.kwargs
            net = SymRegPolicyValueNet(**net_kwargs)
        agent = Agent(
            game=game,
            net=net,
            mcts_params=mcts_params,
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
            self_play_add_noise=self.trainer.self_play_add_noise,
            self_play_temperature=self.trainer.self_play_temperature,
        )
        return game, net, agent, trainer
