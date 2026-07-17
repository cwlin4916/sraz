# sraz

AlphaZero on grammar games, starting with **symbolic regression**.

A minimal single-player AlphaZero engine (extracted from
[AlphaZero_PP](https://github.com/cwlin4916/AlphaZero_PP)) plus one game
instance: a grammar-derivation MDP whose terminal reward is the R² of an
lmfit constant fit of the derived expression against a hidden target,
`4·sin(4x) + C0 + C1·x + C2·x²` (ported from the original AlphaGrammar
`srgame.py`).

## Layout

```
src/sraz/core/          game/net ABCs, MCTS, self-play Agent, config dataclasses
src/sraz/training/      Trainer (self-play -> replay window -> net.train)
src/sraz/utils/         determinism, multiprocessing, checkpoints, stats
src/sraz/instances/symreg/   the SR grammar game, MLP policy/value net, config
scripts/run/run_symreg.py    training driver
Claude-research/        research notes, results, figures
```

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # if needed
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
pytest tests/ -q                            # ~30 s
python scripts/run/run_symreg.py --seed 42  # first experiment, a few minutes
```

Results land in `experiments/symreg/<timestamp>_.../` (config, jsonl logs,
reward curve, checkpoint). See [Claude-research/notes/01-game-spec.md](Claude-research/notes/01-game-spec.md) for the
first documented run.
