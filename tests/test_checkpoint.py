"""CheckpointManager save/load round-trip, validation, listing, deletion.

All file I/O goes through pytest's tmp_path. Stub agent/net objects stand in
for the real trainer; only the agent's attribute *values* are pickled, so
plain dicts/lists/tuples suffice.
"""

import pickle
from pathlib import Path

import pytest

from sraz.utils.checkpoint import CheckpointManager


class StubAgent:
    def __init__(self, game=None, examples=None, rngs=None):
        self.game = game
        self.all_training_examples = examples if examples is not None else []
        self.rngs = rngs


class RecordingNet:
    """Records the paths passed to save_checkpoint / load_checkpoint."""

    def __init__(self):
        self.saved_to = None
        self.loaded_from = None

    def save_checkpoint(self, path):
        self.saved_to = Path(path)

    def load_checkpoint(self, path):
        self.loaded_from = Path(path)


@pytest.fixture
def manager(tmp_path):
    return CheckpointManager(checkpoint_dir=tmp_path)


@pytest.fixture
def agent():
    return StubAgent(
        game={"name": "toy_game", "size": 3},
        examples=[([1, 2], 0.5), ([3, 4], -1.0)],
        rngs=("rng_state", 42),
    )


# ---------------------------------------------------------------- __init__

def test_init_coerces_checkpoint_dir_to_path():
    mgr = CheckpointManager("some/dir")
    assert mgr.checkpoint_dir == Path("some/dir")
    assert isinstance(mgr.checkpoint_dir, Path)


def test_init_default_checkpoint_dir():
    assert CheckpointManager().checkpoint_dir == Path("checkpoints")


# ------------------------------------------------------------------- save

def test_save_creates_dir_and_agent_file(manager, agent, tmp_path):
    save_path = tmp_path / "ckpt" / "iter_0001"  # nested, does not exist yet
    returned = manager.save_checkpoint(save_path, agent)
    assert returned == save_path
    agent_file = save_path / CheckpointManager.AGENT_CHECKPOINT_FILE
    assert agent_file.is_file()
    with agent_file.open("rb") as f:
        data = pickle.load(f)
    assert data == {
        "game": agent.game,
        "all_training_examples": agent.all_training_examples,
        "rngs": agent.rngs,
    }


def test_save_with_net_delegates_to_net(manager, agent, tmp_path):
    net = RecordingNet()
    save_path = tmp_path / "ckpt_net"
    manager.save_checkpoint(save_path, agent, net=net)
    assert net.saved_to == save_path
    # Providing a net must not skip the agent-side save.
    assert (save_path / CheckpointManager.AGENT_CHECKPOINT_FILE).is_file()


def test_save_without_net_writes_no_network_file(manager, agent, tmp_path):
    save_path = tmp_path / "ckpt_nonet"
    manager.save_checkpoint(save_path, agent)
    # The save really happened (agent file written) ...
    assert (save_path / CheckpointManager.AGENT_CHECKPOINT_FILE).is_file()
    # ... but no network file appears without a net.
    assert not (save_path / CheckpointManager.NETWORK_CHECKPOINT_FILE).exists()


def test_save_with_bad_agent_raises(manager, tmp_path):
    # An object lacking the expected attributes: the AttributeError propagates.
    with pytest.raises(AttributeError):
        manager.save_checkpoint(tmp_path / "bad", object())


def test_save_and_load_accept_str_paths(manager, agent, tmp_path):
    # Every public method takes str | Path; str inputs are coerced to Path.
    save_str = str(tmp_path / "str_ckpt")
    returned = manager.save_checkpoint(save_str, agent)
    assert isinstance(returned, Path)
    assert returned == tmp_path / "str_ckpt"

    fresh = StubAgent(examples=["stale"])
    loaded = manager.load_checkpoint(save_str, fresh)
    assert isinstance(loaded, Path)
    assert fresh.game == agent.game

    assert manager.validate_checkpoint(save_str) is True
    assert manager.list_checkpoints(base_dir=str(tmp_path)) == [tmp_path / "str_ckpt"]
    assert manager.delete_checkpoint(save_str) is True
    assert not (tmp_path / "str_ckpt").exists()


# ------------------------------------------------------------------- load

def test_save_load_round_trip(manager, agent, tmp_path):
    save_path = tmp_path / "rt"
    manager.save_checkpoint(save_path, agent)

    fresh = StubAgent(game=None, examples=["stale"], rngs=None)
    returned = manager.load_checkpoint(save_path, fresh)
    assert returned == save_path
    assert fresh.game == agent.game
    assert fresh.rngs == agent.rngs
    assert fresh.all_training_examples == agent.all_training_examples


def test_load_mutates_examples_list_in_place(manager, agent, tmp_path):
    # load clears + extends rather than reassigning, so the original list
    # object must be preserved (documented multiprocessing behavior).
    save_path = tmp_path / "inplace"
    manager.save_checkpoint(save_path, agent)

    fresh = StubAgent(examples=["stale"])
    original_list = fresh.all_training_examples
    manager.load_checkpoint(save_path, fresh)
    assert fresh.all_training_examples is original_list
    assert original_list == agent.all_training_examples


def test_load_exclude_keys(manager, agent, tmp_path):
    save_path = tmp_path / "excl"
    manager.save_checkpoint(save_path, agent)

    fresh = StubAgent(game="keep_game", examples=["stale"], rngs="keep_rngs")
    manager.load_checkpoint(save_path, fresh, exclude_keys=["rngs", "game"])
    assert fresh.game == "keep_game"
    assert fresh.rngs == "keep_rngs"
    # examples were not excluded, so they load normally
    assert fresh.all_training_examples == agent.all_training_examples


def test_load_exclude_training_examples_leaves_list_untouched(manager, agent, tmp_path):
    save_path = tmp_path / "excl_ex"
    manager.save_checkpoint(save_path, agent)

    fresh = StubAgent(examples=["stale"])
    manager.load_checkpoint(save_path, fresh, exclude_keys=["all_training_examples"])
    assert fresh.all_training_examples == ["stale"]
    assert fresh.game == agent.game  # non-excluded keys still load


def test_load_missing_checkpoint_raises_file_not_found(manager, agent, tmp_path):
    with pytest.raises(FileNotFoundError):
        manager.load_checkpoint(tmp_path / "nope", agent)


def test_load_with_net_delegates_to_net(manager, agent, tmp_path):
    save_path = tmp_path / "load_net"
    manager.save_checkpoint(save_path, agent)
    net = RecordingNet()
    manager.load_checkpoint(save_path, StubAgent(), net=net)
    assert net.loaded_from == save_path


# --------------------------------------------------------------- validate

def test_validate_good_checkpoint(manager, agent, tmp_path):
    save_path = tmp_path / "good"
    manager.save_checkpoint(save_path, agent)
    assert manager.validate_checkpoint(save_path) is True


def test_validate_missing_directory(manager, tmp_path):
    assert manager.validate_checkpoint(tmp_path / "absent") is False


def test_validate_directory_without_agent_file(manager, tmp_path):
    empty = tmp_path / "emptydir"
    empty.mkdir()
    assert manager.validate_checkpoint(empty) is False


def test_validate_corrupt_pickle(manager, tmp_path):
    ckpt = tmp_path / "corrupt"
    ckpt.mkdir()
    (ckpt / CheckpointManager.AGENT_CHECKPOINT_FILE).write_bytes(b"not a pickle")
    assert manager.validate_checkpoint(ckpt) is False


def test_validate_missing_required_keys(manager, tmp_path):
    ckpt = tmp_path / "missing_keys"
    ckpt.mkdir()
    with (ckpt / CheckpointManager.AGENT_CHECKPOINT_FILE).open("wb") as f:
        pickle.dump({"game": None, "rngs": None}, f)  # no all_training_examples
    assert manager.validate_checkpoint(ckpt) is False


def test_validate_non_dict_payload(manager, tmp_path):
    ckpt = tmp_path / "non_dict"
    ckpt.mkdir()
    with (ckpt / CheckpointManager.AGENT_CHECKPOINT_FILE).open("wb") as f:
        pickle.dump([1, 2, 3], f)  # .keys() fails -> caught -> False
    assert manager.validate_checkpoint(ckpt) is False


# ------------------------------------------------------------------- list

def test_list_checkpoints_missing_dir_returns_empty(tmp_path):
    mgr = CheckpointManager(tmp_path / "does_not_exist")
    assert mgr.list_checkpoints() == []


def test_list_checkpoints_no_checkpoints_returns_empty(manager, tmp_path):
    (tmp_path / "just_a_dir").mkdir()
    assert manager.list_checkpoints() == []


def test_list_checkpoints_finds_nested_and_sorts_reverse(manager, agent, tmp_path):
    # NOTE: "most recent first" is implemented as reverse-lexicographic sort
    # of the full paths, which matches chronology only for zero-padded names.
    a = tmp_path / "iter_0001"
    b = tmp_path / "iter_0002"
    nested = tmp_path / "runs" / "deep" / "iter_0003"
    for p in (a, b, nested):
        manager.save_checkpoint(p, agent)
    (tmp_path / "not_a_checkpoint").mkdir()  # dir without agent file: excluded

    found = manager.list_checkpoints()
    # Hard-coded expected order (all share the tmp_path prefix, so the
    # comparison is decided by the suffixes: "runs/..." > "iter_0002" >
    # "iter_0001"). Note iter_0003 comes first only because "runs" sorts
    # after "iter", not because it is newest.
    assert found == [nested, b, a]


def test_list_checkpoints_explicit_base_dir_overrides_default(agent, tmp_path):
    mgr = CheckpointManager(tmp_path / "default_dir")  # never created
    other = tmp_path / "other"
    ckpt = other / "run1"
    mgr.save_checkpoint(ckpt, agent)
    assert mgr.list_checkpoints(base_dir=other) == [ckpt]
    assert mgr.list_checkpoints() == []  # default dir still missing


def test_list_checkpoints_includes_base_dir_itself(manager, agent, tmp_path):
    # A checkpoint saved directly into the search dir is itself listed.
    manager.save_checkpoint(tmp_path, agent)
    assert manager.list_checkpoints() == [tmp_path]


# ----------------------------------------------------------------- delete

def test_delete_checkpoint_removes_directory(manager, agent, tmp_path):
    ckpt = tmp_path / "to_delete"
    manager.save_checkpoint(ckpt, agent)
    assert manager.delete_checkpoint(ckpt) is True
    assert not ckpt.exists()


def test_delete_missing_checkpoint_returns_false(manager, tmp_path):
    assert manager.delete_checkpoint(tmp_path / "ghost") is False


def test_delete_on_plain_file_returns_false(manager, tmp_path):
    # shutil.rmtree on a regular file raises; caught -> False, file survives.
    f = tmp_path / "plain_file.txt"
    f.write_text("hello")
    assert manager.delete_checkpoint(f) is False
    assert f.exists()
