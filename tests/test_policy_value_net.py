"""PolicyValueNet ABC contract, Torch checkpointing, and the SymReg MLP net."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from sraz.core.policy_value_net import (
    PolicyValueNet,
    TorchPolicyValueNet,
    PolicyValueNetModel,
)
from sraz.instances.symreg.network import SymRegPolicyValueNet
from sraz.utils import get_device


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


class _TinyNet(TorchPolicyValueNet):
    """Minimal concrete TorchPolicyValueNet: only fills in the abstract slots."""

    def train(self, examples):
        return None

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.as_tensor(state, dtype=torch.float32)), None


class _ToRecordingModel(nn.Module):
    """Tiny model that records every target passed to .to().

    Device moves are otherwise unobservable on CPU-only hardware (DEVICE is
    already cpu, so a no-op push/pop would look identical to the real one).
    """

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)
        self.to_targets = []

    def to(self, target, *args, **kwargs):
        self.to_targets.append(target)
        return super().to(target, *args, **kwargs)


def _state_dicts_equal(sd1, sd2):
    if sd1.keys() != sd2.keys():
        return False
    return all(torch.equal(sd1[k], sd2[k]) for k in sd1)


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


# ---------------------------------------------------------------------------
# PolicyValueNet ABC contract
# ---------------------------------------------------------------------------

def test_abc_not_instantiable():
    with pytest.raises(TypeError):
        PolicyValueNet()


def test_abc_declares_full_interface():
    assert PolicyValueNet.__abstractmethods__ == {
        "__init__", "train", "predict", "save_checkpoint", "load_checkpoint",
        "push_multiprocessing", "pop_multiprocessing",
    }


def test_torch_pvn_still_abstract():
    # TorchPolicyValueNet fills in checkpointing/multiprocessing but not
    # train/predict, so it must remain non-instantiable.
    assert TorchPolicyValueNet.__abstractmethods__ == {"train", "predict"}
    with pytest.raises(TypeError):
        TorchPolicyValueNet(nn.Linear(2, 2))


def test_incomplete_subclass_not_instantiable():
    class Incomplete(TorchPolicyValueNet):
        def train(self, examples):
            pass
        # predict left abstract

    with pytest.raises(TypeError):
        Incomplete(nn.Linear(2, 2))


# ---------------------------------------------------------------------------
# TorchPolicyValueNet: checkpointing + multiprocessing handoff
# ---------------------------------------------------------------------------

def test_save_checkpoint_creates_named_file_with_state_dict(tmp_path):
    torch.manual_seed(0)
    net = _TinyNet(nn.Linear(3, 2))
    net.save_checkpoint(tmp_path)
    path = tmp_path / TorchPolicyValueNet.save_file_name
    assert path.is_file()
    # the file must contain exactly the model's current state dict
    saved = torch.load(path)
    assert _state_dicts_equal(saved, net.model.state_dict())


def test_save_checkpoint_creates_nested_dirs(tmp_path):
    net = _TinyNet(nn.Linear(3, 2))
    deep = tmp_path / "a" / "b" / "c"
    net.save_checkpoint(deep)  # mkdir(parents=True) must handle this
    assert (deep / "model_checkpoint.pt").is_file()


def test_checkpoint_round_trip_weights_identical(tmp_path):
    torch.manual_seed(0)
    net = _TinyNet(nn.Linear(3, 2))
    original = {k: v.clone() for k, v in net.model.state_dict().items()}
    net.save_checkpoint(tmp_path)

    # perturb every parameter, confirm we actually diverged
    with torch.no_grad():
        for p in net.model.parameters():
            p.add_(1.0)
    assert not _state_dicts_equal(net.model.state_dict(), original)

    net.load_checkpoint(tmp_path)
    assert _state_dicts_equal(net.model.state_dict(), original)


def test_load_checkpoint_into_fresh_instance(tmp_path):
    torch.manual_seed(1)
    net1 = _TinyNet(nn.Linear(3, 2))
    net1.save_checkpoint(tmp_path)

    torch.manual_seed(2)
    net2 = _TinyNet(nn.Linear(3, 2))
    assert not _state_dicts_equal(net2.model.state_dict(), net1.model.state_dict())
    net2.load_checkpoint(tmp_path)
    assert _state_dicts_equal(net2.model.state_dict(), net1.model.state_dict())


def test_load_checkpoint_missing_file_raises(tmp_path):
    net = _TinyNet(nn.Linear(3, 2))
    with pytest.raises(FileNotFoundError):
        net.load_checkpoint(tmp_path / "nowhere")


def test_push_pop_multiprocessing_round_trip():
    torch.manual_seed(0)
    rec = _ToRecordingModel()
    net = _TinyNet(rec)
    assert net.model is rec  # nn.Module.to is in-place
    # the constructor already moved the model onto DEVICE
    assert rec.to_targets == [TorchPolicyValueNet.DEVICE]
    before = {k: v.clone() for k, v in net.model.state_dict().items()}

    pushed = net.push_multiprocessing()
    # default implementation carries no extra state and moves to CPU;
    # the recorded .to target makes this observable even when DEVICE == cpu
    assert pushed is None
    assert rec.to_targets[-1] == "cpu"
    assert all(p.device.type == "cpu" for p in net.model.parameters())

    net.pop_multiprocessing(pushed)
    assert rec.to_targets[-1] == TorchPolicyValueNet.DEVICE
    assert len(rec.to_targets) == 3  # init, push, pop — no extra moves
    assert all(p.device == TorchPolicyValueNet.DEVICE
               for p in net.model.parameters())
    assert _state_dicts_equal(
        {k: v.cpu() for k, v in net.model.state_dict().items()}, before)


# ---------------------------------------------------------------------------
# PolicyValueNetModel forward
# ---------------------------------------------------------------------------

def test_model_forward_batched_shapes():
    torch.manual_seed(0)
    model = PolicyValueNetModel(input_size=7, output_size=5,
                                n_hidden_layers=1, hidden_size=8)
    x = torch.randn(4, 7)
    policy, value = model(x)
    assert policy.shape == (4, 5)
    assert value.shape == (4,)  # value head squeezes its trailing dim
    assert torch.isfinite(policy).all()
    # value head is a bare Linear (no tanh): only finiteness is guaranteed
    assert torch.isfinite(value).all()
    # each batch row agrees with an unbatched forward of the same row
    p0, v0 = model(x[0])
    assert torch.allclose(policy[0], p0, atol=1e-6)
    assert torch.allclose(value[0], v0, atol=1e-6)


def test_model_forward_unbatched_shapes():
    torch.manual_seed(0)
    model = PolicyValueNetModel(input_size=7, output_size=5,
                                n_hidden_layers=0, hidden_size=8)
    policy, value = model(torch.randn(7))
    assert policy.shape == (5,)
    assert value.shape == ()  # scalar after squeeze(-1)


def test_model_body_depth_and_head_shapes():
    model = PolicyValueNetModel(input_size=3, output_size=4,
                                n_hidden_layers=2, hidden_size=6)
    # body = input block + n_hidden_layers hidden blocks
    assert len(model.body) == 3
    assert model.policy_head.out_features == 4
    assert model.value_head.out_features == 1
    assert model.input_size == 3 and model.output_size == 4


def test_model_policy_logits_unnormalized():
    torch.manual_seed(3)
    model = PolicyValueNetModel(input_size=6, output_size=9,
                                n_hidden_layers=0, hidden_size=8)
    policy, _ = model(torch.randn(2, 6))
    # forward returns raw logits, not a distribution
    sums = policy.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums))


# ---------------------------------------------------------------------------
# SymRegPolicyValueNet
# ---------------------------------------------------------------------------

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
