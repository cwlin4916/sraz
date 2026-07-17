"""Trainer orchestration: construction, one iteration, replay window,
train_multiple accounting, checkpoint round-trips, multiprocessing hooks."""

from pathlib import Path

import numpy as np
import pytest
import torch

from sraz.instances.symreg.config import SymRegConfig
from sraz.instances.symreg.network import SymRegPolicyValueNet
from sraz.training.trainer import Trainer
from sraz.utils.checkpoint import CheckpointManager
from sraz.utils.multiprocessing import MultiprocessingManager


# ---------------------------------------------------------------------------
# Hand-rolled stubs (Trainer only needs a thin slice of Agent/Net/Game)
# ---------------------------------------------------------------------------

class _StubNet:
    """Records train() calls; returns canned per-epoch losses."""

    def __init__(self):
        self.train_calls = []
        self.push_count = 0
        self.pop_count = 0

    def train(self, examples):
        self.train_calls.append(list(examples))
        # (model, batch_losses, train_losses, policy_losses, value_losses)
        return None, [0.5], [0.5, 0.3], [0.4, 0.2], [0.1, 0.1]

    def push_multiprocessing(self):
        self.push_count += 1

    def pop_multiprocessing(self, *args):
        self.pop_count += 1

    def save_checkpoint(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "stub_net.pt").write_text("weights")

    def load_checkpoint(self, save_dir):
        self.loaded_from = Path(save_dir)


class _StubAgent:
    """Plays canned episodes; mirrors the Agent surface Trainer touches."""

    def __init__(self, net, rewards=(1.0,), policies=None, with_leaf_data=False):
        self.net = net
        self.rewards = list(rewards)
        self.policies = policies or [np.full(4, 0.25)]
        self.with_leaf_data = with_leaf_data
        self.rngs = {"train": np.random.default_rng(0),
                     "mcts": np.random.default_rng(1)}
        # Attributes CheckpointManager expects (the real Agent lacks
        # all_training_examples -- see test_save_checkpoint_with_real_agent).
        self.all_training_examples = []
        self.game = {"tag": "stub-game"}
        self.play_calls = 0
        self.play_kwargs = []

    def _randseed(self, rng_name):
        return int(self.rngs[rng_name].integers(0, 2**31 - 1))

    def play_for_experience(self, game, i, reset_seed, interaction_seed,
                            add_noise=True, temperature_override=None):
        self.play_calls += 1
        self.play_kwargs.append({"add_noise": add_noise,
                                 "temperature_override": temperature_override})
        reward = float(self.rewards[i % len(self.rewards)])
        experience = [(np.zeros(3), p.copy(), reward) for p in self.policies]
        step_infos = [{"step": k} for k in range(len(self.policies))]
        leaf = {"game_index": i} if self.with_leaf_data else None
        return experience, reward, step_infos, leaf


class _ExplodingAgent(_StubAgent):
    def play_for_experience(self, game, i, reset_seed, interaction_seed,
                            add_noise=True, temperature_override=None):
        raise RuntimeError("boom")


class _RecordingLeafEval:
    def __init__(self):
        self.snapshots = 0
        self.merged = []

    def snapshot_baseline(self):
        self.snapshots += 1

    def merge_caches(self, data):
        self.merged.append(data)


class _LeafEvalGame:
    def __init__(self):
        self.leaf_evaluator = _RecordingLeafEval()


def _stub_trainer(tmp_path, n_games=2, rewards=(1.0, -0.5), policies=None,
                  game=None, agent_cls=_StubAgent, **trainer_kwargs):
    net = _StubNet()
    agent = agent_cls(net, rewards=rewards, policies=policies)
    trainer = Trainer(
        agent=agent, net=net, game=game if game is not None else object(),
        n_games_per_train=n_games, n_procs=-1,
        checkpoint_dir=tmp_path / "ckpts", **trainer_kwargs)
    return trainer, agent, net


def _cheap_symreg_cfg(tmp_path):
    cfg = SymRegConfig()
    cfg.agent.mcts_params["n_simulations"] = 2
    cfg.trainer.n_games_per_train = 1
    cfg.trainer.checkpoint_dir = str(tmp_path / "ckpts")
    cfg.net.kwargs = {"random_seed": 0, "training_params": {"epochs": 1}}
    cfg.game.kwargs["problem_seed"] = 42
    return cfg


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_config_build_wires_trainer(tmp_path):
    cfg = _cheap_symreg_cfg(tmp_path)
    game, net, agent, trainer = cfg.build()
    assert trainer.agent is agent
    assert trainer.net is net
    assert trainer.game is game
    assert agent.net is net
    assert trainer.n_games_per_train == 1
    assert trainer.n_procs == -1
    assert trainer.n_past_iterations_to_train == cfg.trainer.n_past_iterations_to_train
    assert trainer.checkpoint_manager.checkpoint_dir == Path(cfg.trainer.checkpoint_dir)
    assert trainer.all_training_examples == []
    assert trainer.statistics_manager.to_list() == []


def test_trainer_defaults(tmp_path, monkeypatch):
    # chdir first: the default checkpoint_dir is the relative "checkpoints",
    # so an eager mkdir would land in the cwd -- keep it inside tmp_path.
    monkeypatch.chdir(tmp_path)
    net = _StubNet()
    trainer = Trainer(agent=_StubAgent(net), net=net, game=object())
    assert trainer.n_games_per_train == 100
    assert trainer.n_past_iterations_to_train == 20
    assert trainer.n_procs is None
    assert trainer.use_tree_reuse is False
    assert trainer.checkpoint_manager.checkpoint_dir == Path("checkpoints")
    # Construction must not create the checkpoint directory eagerly.
    assert not (tmp_path / "checkpoints").exists()
    assert trainer._last_diagnostics == []


# ---------------------------------------------------------------------------
# Collecting experience (sequential mode, n_procs=-1)
# ---------------------------------------------------------------------------

def test_collect_training_examples_one_result_per_game(tmp_path):
    trainer, agent, net = _stub_trainer(tmp_path, n_games=3,
                                        rewards=(0.5, -0.25, 1.0))
    results = trainer._collect_training_examples()
    assert len(results) == 3
    assert agent.play_calls == 3
    for i, (experience, reward, step_infos, leaf) in enumerate(results):
        assert reward == pytest.approx((0.5, -0.25, 1.0)[i])
        assert len(experience) == 1
        assert leaf is None
    # push/pop happen exactly once around collection (on agent.net).
    assert net.push_count == 1
    assert net.pop_count == 1


def test_collect_pops_multiprocessing_state_on_failure(tmp_path):
    trainer, _, net = _stub_trainer(tmp_path, agent_cls=_ExplodingAgent)
    with pytest.raises(RuntimeError, match="boom"):
        trainer._collect_training_examples()
    assert net.push_count == 1
    assert net.pop_count == 1  # finally-clause restored state


def test_collect_snapshots_leaf_evaluator_baseline(tmp_path):
    game = _LeafEvalGame()
    trainer, _, _ = _stub_trainer(tmp_path, game=game)
    trainer._collect_training_examples()
    assert game.leaf_evaluator.snapshots == 1


def test_collect_uses_tree_reuse_play_fn_when_enabled(tmp_path):
    # use_tree_reuse=True must select play_for_experience_reuse_tree
    # (trainer.py:88-91); the plain method must not run.
    class _ReuseAgent(_StubAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.reuse_calls = 0

        def play_for_experience_reuse_tree(self, game, i, reset_seed,
                                           interaction_seed, add_noise=True,
                                           temperature_override=None):
            self.reuse_calls += 1
            return _StubAgent.play_for_experience(
                self, game, i, reset_seed, interaction_seed,
                add_noise=add_noise, temperature_override=temperature_override)

        def play_for_experience(self, game, i, reset_seed, interaction_seed,
                                add_noise=True, temperature_override=None):
            raise AssertionError("plain play_for_experience must not be used")

    trainer, agent, _ = _stub_trainer(tmp_path, n_games=2,
                                      agent_cls=_ReuseAgent,
                                      use_tree_reuse=True)
    results = trainer._collect_training_examples()
    assert agent.reuse_calls == 2
    assert len(results) == 2


# ---------------------------------------------------------------------------
# train_iteration
# ---------------------------------------------------------------------------

def test_train_iteration_statistics_from_stub(tmp_path):
    trainer, _, net = _stub_trainer(tmp_path, n_games=2, rewards=(1.0, 0.0),
                                    policies=[np.full(4, 0.25)])
    trainer.train_iteration()

    assert len(net.train_calls) == 1
    assert len(net.train_calls[0]) == 2  # 2 games x 1 step

    records = trainer.statistics_manager.to_list()
    assert len(records) == 1
    rec = records[0]
    assert rec["avg_reward"] == pytest.approx(0.5)
    assert rec["num_examples"] == 2
    assert rec["train_loss"] == pytest.approx(np.mean([0.5, 0.3]))
    assert rec["train_loss_policy"] == pytest.approx(np.mean([0.4, 0.2]))
    assert rec["train_loss_value"] == pytest.approx(np.mean([0.1, 0.1]))
    # Uniform-over-4 MCTS targets have entropy ln(4).
    assert rec["mcts_target_entropy"] == pytest.approx(np.log(4))
    assert rec["policy_kl_gap"] == pytest.approx(0.3 - np.log(4))
    assert "timestamp" in rec


def test_train_iteration_entropy_averages_over_examples(tmp_path):
    uniform = np.full(4, 0.25)
    onehot = np.array([1.0, 0.0, 0.0, 0.0])
    trainer, _, _ = _stub_trainer(tmp_path, n_games=1, rewards=(1.0,),
                                  policies=[uniform, onehot])
    trainer.train_iteration()
    rec = trainer.statistics_manager.to_list()[0]
    # Entropy is averaged over examples; zero entries are excluded from the log.
    assert rec["mcts_target_entropy"] == pytest.approx(np.log(4) / 2)


def test_train_iteration_populates_diagnostics_and_merges_leaf_caches(tmp_path):
    game = _LeafEvalGame()
    net = _StubNet()
    agent = _StubAgent(net, rewards=(0.5, 0.5), with_leaf_data=True)
    trainer = Trainer(agent=agent, net=net, game=game, n_games_per_train=2,
                      n_procs=-1, checkpoint_dir=tmp_path / "ckpts")
    trainer.train_iteration()
    assert trainer._last_diagnostics == [[{"step": 0}], [{"step": 0}]]
    assert game.leaf_evaluator.merged == [{"game_index": 0}, {"game_index": 1}]


def test_train_iteration_symreg_end_to_end(tmp_path):
    # MCTS exploration RNG now flows from the Agent's seeded "mcts" stream
    # (SymRegConfig sets it), so the episode is reproducible without touching
    # the process-global np.random. torch.manual_seed pins net weight init.
    torch.manual_seed(0)
    cfg = _cheap_symreg_cfg(tmp_path)
    game, net, agent, trainer = cfg.build()

    trainer.train_iteration()

    # Replay buffer: 1 iteration x 1 game x episode-length examples.
    assert len(trainer.all_training_examples) == 1
    assert len(trainer.all_training_examples[0]) == 1
    episode = trainer.all_training_examples[0][0]
    assert len(episode) >= 1
    obs, policy, value = episode[0]
    assert policy.shape == (game.action_space.n,)
    assert policy.sum() == pytest.approx(1.0)

    rec = trainer.statistics_manager.to_list()[0]
    assert rec["num_examples"] == len(episode)
    for key in ("train_loss", "train_loss_policy", "train_loss_value",
                "mcts_target_entropy", "policy_kl_gap", "avg_reward"):
        assert np.isfinite(rec[key])

    # One step-info list per game, one info dict per episode step.
    assert len(trainer._last_diagnostics) == 1
    assert len(trainer._last_diagnostics[0]) == len(episode)


def test_collect_training_examples_deterministic_across_builds(tmp_path):
    # Determinism now flows entirely from the Agent's seeded RNG streams
    # (SymRegConfig fixes "mcts"/"train"), which feed per-game reset and
    # interaction seeds; the interaction seed drives MCTS exploration. No
    # global np.random seeding needed.
    rewards = []
    lengths = []
    for _ in range(2):
        cfg = _cheap_symreg_cfg(tmp_path)
        _, _, _, trainer = cfg.build()
        results = trainer._collect_training_examples()
        rewards.append([r[1] for r in results])
        lengths.append([len(r[0]) for r in results])
    assert rewards[0] == rewards[1]
    assert lengths[0] == lengths[1]


# ---------------------------------------------------------------------------
# Replay window (_process_training_examples)
# ---------------------------------------------------------------------------

def _one_game_iteration(tag):
    return [[(f"obs-{tag}", np.array([1.0]), 0.1)]]


def test_replay_window_trims_old_iterations(tmp_path):
    trainer, _, _ = _stub_trainer(tmp_path, n_past_iterations_to_train=2)
    trainer._process_training_examples(_one_game_iteration("a"))
    trainer._process_training_examples(_one_game_iteration("b"))
    flat = trainer._process_training_examples(_one_game_iteration("c"))
    assert len(trainer.all_training_examples) == 2
    assert [ex[0] for ex in flat] == ["obs-b", "obs-c"]  # oldest dropped


def test_replay_window_none_keeps_everything(tmp_path):
    trainer, _, _ = _stub_trainer(tmp_path, n_past_iterations_to_train=None)
    for tag in "abc":
        flat = trainer._process_training_examples(_one_game_iteration(tag))
    assert len(trainer.all_training_examples) == 3
    assert [ex[0] for ex in flat] == ["obs-a", "obs-b", "obs-c"]


# ---------------------------------------------------------------------------
# train_multiple
# ---------------------------------------------------------------------------

def test_train_multiple_runs_n_iterations(tmp_path):
    trainer, _, net = _stub_trainer(tmp_path)
    trainer.train_multiple(n_iterations=3)
    assert len(net.train_calls) == 3
    assert len(trainer.statistics_manager.to_list()) == 3
    assert len(trainer.all_training_examples) == 3


def test_train_multiple_start_at_skips_completed_iterations(tmp_path):
    trainer, _, net = _stub_trainer(tmp_path)
    trainer.train_multiple(n_iterations=3, start_at=1)
    assert len(net.train_calls) == 2  # iterations 1 and 2 only


def test_train_multiple_start_at_past_end_is_noop(tmp_path):
    trainer, _, net = _stub_trainer(tmp_path)
    trainer.train_multiple(n_iterations=2, start_at=5)
    assert net.train_calls == []
    assert trainer.statistics_manager.to_list() == []


def test_train_multiple_checkpoint_every_writes_hardcoded_dir(tmp_path, monkeypatch):
    # train_multiple writes to the literal Path("checkpoints"), NOT to the
    # configured checkpoint_dir (trainer.py:244) -- documented current
    # behavior; chdir into tmp_path so nothing lands outside it.
    monkeypatch.chdir(tmp_path)
    trainer, _, _ = _stub_trainer(tmp_path)
    trainer.train_multiple(n_iterations=2, checkpoint_every=2)
    expected = tmp_path / "checkpoints" / f"{trainer.run_start_time}_iter_2"
    assert (expected / CheckpointManager.AGENT_CHECKPOINT_FILE).exists()
    assert (expected / "stub_net.pt").exists()
    # No checkpoint after iteration 1: (i+1) % checkpoint_every == 1 at i=0.
    assert not (tmp_path / "checkpoints" / f"{trainer.run_start_time}_iter_1").exists()
    # The configured directory was never used.
    assert not (tmp_path / "ckpts").exists()


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

def test_save_load_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    net = SymRegPolicyValueNet(state_len=2, n_tokens=3, n_actions=4,
                               hidden_size=8, n_hidden_layers=1, random_seed=0)
    agent = _StubAgent(net)
    agent.all_training_examples.append(("obs", "policy", 0.5))
    trainer = Trainer(agent=agent, net=net, game=object(),
                      checkpoint_dir=tmp_path)
    original_weights = {k: v.clone() for k, v in net.model.state_dict().items()}
    original_rng_state = agent.rngs["train"].bit_generator.state
    original_game = dict(agent.game)

    ckpt = tmp_path / "roundtrip"
    trainer.save_checkpoint(ckpt)
    assert trainer.checkpoint_manager.validate_checkpoint(ckpt)

    # Mutate everything the checkpoint covers.
    with torch.no_grad():
        for p in net.model.parameters():
            p.add_(1.0)
    agent.all_training_examples.append("junk")
    agent.rngs["train"].integers(0, 100, size=10)  # advance the RNG
    agent.game["tag"] = "mutated"

    trainer.load_checkpoint(ckpt)

    for k, v in net.model.state_dict().items():
        assert torch.equal(v, original_weights[k])
    assert agent.all_training_examples == [("obs", "policy", 0.5)]
    assert agent.rngs["train"].bit_generator.state == original_rng_state
    assert agent.game == original_game  # restored from the pickled copy


def test_save_checkpoint_default_path_uses_run_start_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer, _, _ = _stub_trainer(tmp_path)
    trainer.save_checkpoint()  # default: checkpoints/<run_start_time>_final
    expected = tmp_path / "checkpoints" / f"{trainer.run_start_time}_final"
    assert (expected / CheckpointManager.AGENT_CHECKPOINT_FILE).exists()


def test_load_checkpoint_missing_raises(tmp_path):
    trainer, _, _ = _stub_trainer(tmp_path)
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint(tmp_path / "does-not-exist")


def test_save_checkpoint_with_real_agent_raises_attribute_error(tmp_path):
    # Suspected bug: CheckpointManager reads agent.all_training_examples, but
    # that attribute lives on Trainer, not Agent (trainer.py:256 /
    # checkpoint.py:46); scripts/run/run_symreg.py bypasses this method for
    # the same reason. Documenting current behavior.
    cfg = _cheap_symreg_cfg(tmp_path)
    _, _, _, trainer = cfg.build()
    with pytest.raises(AttributeError, match="all_training_examples"):
        trainer.save_checkpoint(tmp_path / "real-agent-ckpt")


# ---------------------------------------------------------------------------
# Multiprocessing hooks
# ---------------------------------------------------------------------------

def test_push_pop_multiprocessing_are_symmetric_noops(tmp_path):
    trainer, _, _ = _stub_trainer(tmp_path)
    stash = trainer.push_multiprocessing()
    assert stash is None
    assert trainer.pop_multiprocessing(stash) is None
    assert trainer.pop_multiprocessing() is None  # tolerates no stash at all


def test_trainer_works_with_multiprocessing_manager_protocol(tmp_path):
    # Double-push/double-pop errors are MultiprocessingManager behavior,
    # already covered in tests/test_utils.py; here we only check that a real
    # Trainer satisfies the push/pop protocol the manager requires.
    trainer, _, _ = _stub_trainer(tmp_path)
    manager = MultiprocessingManager(trainer)
    manager.push()
    manager.pop()
    manager.push()  # reusable after a full push/pop cycle
    manager.pop()
