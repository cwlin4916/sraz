"""Greedy (zero-temperature, noise-free) episode evaluation.

Lives in the package rather than in a run script so both drivers and the test
suite exercise the same code path.
"""

from __future__ import annotations

import numpy as np


def greedy_episode(agent, reset_seed: int | None = None,
                   temperature: float = 0.0) -> dict:
    """Play one episode with search exploration disabled at action selection.

    Dirichlet noise is off and the visit counts are collapsed at
    `temperature` (0.0 = argmax), so with a deterministic net the whole
    trajectory is a function of the game and the search parameters alone --
    no RNG is consulted. Two calls on an unchanged agent return identical
    results, which is what makes a flat greedy curve meaningful rather than
    lucky.

    Returns a dict with the terminal reward, the emitted prefix sentence
    (None if the episode never terminated), and the action sequence.
    """
    game = agent.game.clone()
    game.reset_wrapper(seed=reset_seed)

    reward, rule, actions = 0.0, None, []
    for _ in range(game.state_len + 5):
        probs = agent.policy(game, "", add_noise=False,
                             temperature_override=temperature)
        action = int(np.argmax(probs))
        actions.append(action)
        _, reward, terminated, truncated, info = game.step_wrapper(action)
        if terminated or truncated:
            rule = info.get("rule")
            break

    return {"reward": float(reward), "rule": rule, "actions": actions}


def greedy_eval(agent, n_episodes: int = 1, temperature: float = 0.0,
                base_seed: int = 1000) -> tuple[float, float, str | None]:
    """Aggregate `n_episodes` greedy episodes.

    Returns (mean reward, best reward, best rule). Episodes differ only by
    their reset seed, which for a fixed target changes nothing -- so
    n_episodes > 1 is only meaningful when constants are redrawn per episode.
    """
    rewards, best_r, best_rule = [], -np.inf, None
    for i in range(n_episodes):
        out = greedy_episode(agent, reset_seed=base_seed + i, temperature=temperature)
        rewards.append(out["reward"])
        if out["reward"] > best_r:
            best_r, best_rule = out["reward"], out["rule"]
    return float(np.mean(rewards)), float(best_r), best_rule
