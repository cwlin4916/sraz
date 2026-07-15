"""core/policy_value_net.py: PolicyValueNet ABC, TorchPolicyValueNet checkpoint/
multiprocessing handoff, and the shared PolicyValueNetModel MLP."""

import pytest
import torch
import torch.nn as nn

from sraz.core.policy_value_net import (
    PolicyValueNet,
    TorchPolicyValueNet,
    PolicyValueNetModel,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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
