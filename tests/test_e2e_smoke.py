"""One tiny end-to-end training iteration plus a greedy episode."""

import numpy as np

from sraz.instances.symreg.config import SymRegConfig


def test_one_training_iteration_and_greedy_episode():
    cfg = SymRegConfig()
    cfg.agent.mcts_params["n_simulations"] = 5
    cfg.trainer.n_games_per_train = 2
    cfg.net.kwargs = {"random_seed": 0}
    cfg.game.kwargs["problem_seed"] = 42

    game, net, agent, trainer = cfg.build()

    trainer.train_iteration()
    stats = trainer.statistics_manager.to_list()
    assert len(stats) == 1
    record = stats[0]
    assert np.isfinite(record["train_loss"])

    # one greedy (noise-free, near-argmax temperature) episode
    eval_game = game.clone()
    eval_game.reset_wrapper()
    reward, terminated = 0.0, False
    for _ in range(20):
        pi = agent.policy(eval_game, "", add_noise=False, temperature_override=0.01)
        _, reward, terminated, _, info = eval_game.step_wrapper(int(np.argmax(pi)))
        if terminated:
            break
    assert terminated
    assert -1.0 <= reward <= 1.0
    assert info["rule"] is not None
