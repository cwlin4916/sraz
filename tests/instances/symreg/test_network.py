"""instances/symreg/network.py: SymRegPolicyValueNet one-hot MLP."""

import numpy as np
import pytest
import torch

from sraz.instances.symreg.network import SymRegPolicyValueNet
from sraz.utils import get_device


@pytest.fixture(autouse=True)
def _restore_torch_determinism():
    """SymRegPolicyValueNet(random_seed=...) flips the process-global
    torch.use_deterministic_algorithms switch as a side effect and never
    restores it (src/sraz/instances/symreg/network.py:32).  Snapshot and
    restore it around every test so the leak cannot bleed into the rest of
    the suite (or between tests in this file)."""
    mode = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(mode, warn_only=warn_only)


def _tiny_symreg_net(seed, **overrides):
    kwargs = dict(state_len=4, n_tokens=5, n_actions=6, random_seed=seed,
                  n_hidden_layers=1, hidden_size=16, device="cpu")
    kwargs.update(overrides)
    return SymRegPolicyValueNet(**kwargs)


def _tiny_examples(net, n=8, seed=0):
    """(state, policy, value) triples in the format SymRegPolicyValueNet.train eats."""
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n):
        state = rng.integers(0, net.n_tokens, size=net.state_len)
        policy = rng.random(net.n_actions).astype(np.float32)
        policy /= policy.sum()
        value = float(rng.uniform(-1, 1))
        examples.append((state, policy, value))
    return examples


def test_symreg_encode_one_hot():
    net = _tiny_symreg_net(seed=0)
    obs = [1, 0, 3, 2]
    vec = net._encode(obs)
    assert vec.shape == (net.state_len * net.n_tokens,)
    assert vec.dtype == np.float32
    grid = vec.reshape(net.state_len, net.n_tokens)
    assert (grid.sum(axis=1) == 1).all()
    for slot, tok in enumerate(obs):
        assert grid[slot, tok] == 1.0


def test_symreg_device_parameter():
    # explicit device string is stored verbatim as an instance attribute,
    # shadowing the class-level torch.device DEVICE
    net = _tiny_symreg_net(seed=0)  # helper passes device="cpu"
    assert net.DEVICE == "cpu"
    assert "DEVICE" in net.__dict__
    # device=None falls back to get_device()
    net_default = SymRegPolicyValueNet(state_len=3, n_tokens=4, n_actions=5,
                                       n_hidden_layers=0, hidden_size=8)
    assert net_default.DEVICE == get_device()


def test_symreg_predict_shapes_and_normalization():
    net = _tiny_symreg_net(seed=0)
    state = np.array([0, 2, 4, 4])
    policy, value = net.predict(state)
    assert policy.shape == (net.n_actions,)
    assert (policy >= 0).all()
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert value.shape == ()
    assert np.isfinite(value)


def test_symreg_predict_deterministic_given_seed():
    state = np.array([1, 1, 0, 3])
    p1, v1 = _tiny_symreg_net(seed=123).predict(state)
    p2, v2 = _tiny_symreg_net(seed=123).predict(state)
    assert np.array_equal(p1, p2)
    assert v1 == v2
    p3, _ = _tiny_symreg_net(seed=124).predict(state)
    assert not np.array_equal(p1, p3)


def test_symreg_training_params_merge():
    net = _tiny_symreg_net(seed=0, training_params={"epochs": 3, "batch_size": 4})
    assert net.training_params["epochs"] == 3
    assert net.training_params["batch_size"] == 4
    # untouched keys keep their defaults
    assert net.training_params["learning_rate"] == \
        SymRegPolicyValueNet.default_training_params["learning_rate"]
    # class-level defaults must not be mutated by the merge
    assert SymRegPolicyValueNet.default_training_params["epochs"] == 10


def test_symreg_train_returns_finite_and_decreases():
    torch.manual_seed(0)
    np.random.seed(0)
    net = _tiny_symreg_net(
        seed=0, training_params={"epochs": 6, "batch_size": 4,
                                 "learning_rate": 0.01})
    examples = _tiny_examples(net, n=8, seed=0)
    model, batch_losses, train_losses, policy_losses, value_losses = \
        net.train(examples)

    assert model is net.model
    assert len(train_losses) == 6
    assert len(policy_losses) == 6
    assert len(value_losses) == 6
    assert len(batch_losses) == 6 * 2  # 8 examples / batch_size 4 = 2 batches
    for seq in (batch_losses, train_losses, policy_losses, value_losses):
        assert np.isfinite(seq).all()
    # total loss = value + policy_weight * policy, and it goes down on a
    # tiny fixed dataset
    assert train_losses[0] == pytest.approx(
        value_losses[0] + net.training_params["policy_weight"] * policy_losses[0])
    assert train_losses[-1] < train_losses[0]


def test_symreg_train_accepts_prebuilt_dataset():
    torch.manual_seed(0)
    net = _tiny_symreg_net(
        seed=0, training_params={"epochs": 2, "batch_size": 4})
    examples = _tiny_examples(net, n=4, seed=1)
    states = torch.from_numpy(np.stack([net._encode(s) for s, _, _ in examples]))
    policies = torch.from_numpy(np.stack([p for _, p, _ in examples]))
    values = torch.tensor([v for _, _, v in examples], dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(states, policies, values)

    _, _, train_losses, _, _ = net.train(dataset, needs_reshape=False)
    assert len(train_losses) == 2
    assert np.isfinite(train_losses).all()


def test_symreg_checkpoint_round_trip_predictions_match(tmp_path):
    state = np.array([2, 0, 1, 4])
    net1 = _tiny_symreg_net(seed=7)
    p1, v1 = net1.predict(state)
    net1.save_checkpoint(tmp_path)
    # subclass overrides the checkpoint file name
    assert (tmp_path / "symreg_checkpoint.pt").is_file()

    net2 = _tiny_symreg_net(seed=8)
    p2, _ = net2.predict(state)
    assert not np.allclose(p1, p2)

    net2.load_checkpoint(tmp_path)
    p3, v3 = net2.predict(state)
    assert np.allclose(p1, p3)
    assert v3 == pytest.approx(float(v1))


def test_symreg_train_then_predict_still_normalized():
    torch.manual_seed(0)
    net = _tiny_symreg_net(
        seed=0, training_params={"epochs": 2, "batch_size": 4})
    net.train(_tiny_examples(net, n=4, seed=2))
    policy, value = net.predict(np.array([0, 1, 2, 3]))
    assert policy.shape == (net.n_actions,)
    assert policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.isfinite(value)
