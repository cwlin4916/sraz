"""Utils: env-var setup, StatisticsManager, MultiprocessingManager."""

import json
import os
import platform
import time
import warnings

import pytest
import torch

from sraz.utils.common import (
    THREAD_VARS,
    disable_numpy_multithreading,
    get_device,
    use_deterministic_cuda,
)
from sraz.utils.multiprocessing import (
    MultiprocessingManager,
    validate_multiprocessing_setup,
)
from sraz.utils.statistics import StatisticsManager


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------

def test_disable_numpy_multithreading_sets_all_thread_vars(monkeypatch):
    # Start from wrong / missing values; the call must force all to "1".
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    if platform.system() == "Darwin":
        # Register the OBJC var with monkeypatch so the direct os.environ
        # write inside disable_numpy_multithreading() is undone on teardown.
        monkeypatch.setenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "NO")
    disable_numpy_multithreading()
    for var in THREAD_VARS:
        assert os.environ[var] == "1", var


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS-only env var")
def test_disable_numpy_multithreading_sets_objc_fork_safety(monkeypatch):
    # setenv first registers a restore-to-original with monkeypatch (the
    # function under test writes os.environ directly); delenv then makes
    # the var missing for the actual check.
    monkeypatch.setenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "NO")
    monkeypatch.delenv("OBJC_DISABLE_INITIALIZE_FORK_SAFETY")
    disable_numpy_multithreading()
    assert os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] == "YES"


def test_use_deterministic_cuda_sets_cublas_workspace(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    use_deterministic_cuda()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_get_device_returns_valid_torch_device():
    dev = get_device()
    assert dev in ("cuda", "cpu")
    assert dev == ("cuda" if torch.cuda.is_available() else "cpu")
    # the string must be a usable torch device
    t = torch.zeros(2, device=torch.device(dev))
    assert t.device.type == dev


# ---------------------------------------------------------------------------
# statistics.py
# ---------------------------------------------------------------------------

def test_statistics_record_adds_timestamp_and_data():
    sm = StatisticsManager(run_start_time=12345)
    assert sm.run_start_time == 12345
    before = int(time.time())
    sm.record({"loss": 0.5, "iter": 3})
    after = int(time.time())
    (rec,) = sm.to_list()
    assert rec["loss"] == 0.5 and rec["iter"] == 3
    assert before <= rec["timestamp"] <= after
    # a caller-supplied timestamp overrides the auto one (dict-merge order)
    sm.record({"timestamp": 99})
    assert sm.to_list()[1]["timestamp"] == 99


def test_statistics_run_start_time_default_and_falsy_zero():
    now = int(time.time())
    assert abs(StatisticsManager().run_start_time - now) <= 2
    # `run_start_time or int(time.time())` means an explicit 0 is silently
    # replaced by the current time -- documenting current behavior.
    assert StatisticsManager(run_start_time=0).run_start_time != 0


def test_statistics_record_rejects_non_dict():
    sm = StatisticsManager()
    with pytest.raises(TypeError):
        sm.record([("a", 1)])
    with pytest.raises(TypeError):
        sm.record_many([{"ok": 1}, "not a dict"])
    # the valid record before the bad one was still appended
    assert len(sm.to_list()) == 1


def test_statistics_record_many_and_to_list_returns_copy():
    sm = StatisticsManager()
    sm.record_many([{"a": 1}, {"a": 2}, {"a": 3}])
    out = sm.to_list()
    assert [r["a"] for r in out] == [1, 2, 3]
    out.append({"a": 4})  # mutating the copy must not affect the manager
    assert len(sm.to_list()) == 3
    # The copy is shallow (list(self._records)): the dicts are shared, so
    # mutating one through the copy is visible in the manager -- documenting
    # current behavior.
    out[0]["a"] = 99
    assert sm.to_list()[0]["a"] == 99


def test_statistics_clear_empties_records(tmp_path):
    sm = StatisticsManager()
    sm.record({"x": 1})
    sm.record({"x": 2})
    path = tmp_path / "stats.jsonl"
    sm.save_jsonl(path)
    sm.clear()
    assert sm.to_list() == []
    # clear() also resets the written counter: the next append-save starts
    # from index 0 and appends after the existing lines.
    sm.record({"x": 3})
    sm.save_jsonl(path)
    lines = path.read_text().splitlines()
    assert [json.loads(l)["x"] for l in lines] == [1, 2, 3]


def test_statistics_save_jsonl_append_writes_only_new_records(tmp_path):
    sm = StatisticsManager()
    path = tmp_path / "stats.jsonl"
    sm.record({"i": 0})
    sm.record({"i": 1})
    sm.save_jsonl(path, append=True)
    sm.record({"i": 2})
    sm.save_jsonl(path, append=True)
    sm.save_jsonl(path, append=True)  # nothing new -> no extra lines
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["i"] for r in lines] == [0, 1, 2]


def test_statistics_save_jsonl_overwrite_rewrites_everything(tmp_path):
    sm = StatisticsManager()
    path = tmp_path / "stats.jsonl"
    path.write_text("stale garbage\n")
    sm.record({"i": 0})
    sm.record({"i": 1})
    sm.save_jsonl(path, append=False)
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["i"] for r in lines] == [0, 1]


def test_statistics_save_jsonl_empty_manager_creates_no_file(tmp_path):
    sm = StatisticsManager()
    path = tmp_path / "sub" / "dir" / "stats.jsonl"
    sm.save_jsonl(path)
    # mkdir(parents=True) runs BEFORE the empty-records early return, so the
    # parent directories exist even though no file was written.
    assert not path.exists()
    assert path.parent.is_dir()
    sm.record({"a": 1})
    sm.save_jsonl(path)
    assert path.exists() and len(path.read_text().splitlines()) == 1


def test_statistics_overwrite_does_not_reset_append_counter(tmp_path):
    # Current behavior: append=False rewrites the file but leaves
    # _records_written untouched, so a later append=True save re-appends
    # records that the overwrite already persisted (duplicate line for i=2).
    sm = StatisticsManager()
    path = tmp_path / "stats.jsonl"
    sm.record({"i": 0})
    sm.record({"i": 1})
    sm.save_jsonl(path, append=True)   # written counter -> 2
    sm.record({"i": 2})
    sm.save_jsonl(path, append=False)  # file now [0, 1, 2]; counter still 2
    sm.save_jsonl(path, append=True)   # appends [2] again
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert [r["i"] for r in lines] == [0, 1, 2, 2]


# ---------------------------------------------------------------------------
# multiprocessing.py: MultiprocessingManager push/pop
# ---------------------------------------------------------------------------

class _MPStub:
    """Minimal object implementing the push/pop multiprocessing protocol."""

    def __init__(self, name):
        self.name = name
        self.push_calls = 0
        self.popped_with = None

    def push_multiprocessing(self):
        self.push_calls += 1
        return f"stash-{self.name}"

    def pop_multiprocessing(self, stash):
        self.popped_with = stash


def test_mp_manager_push_pop_round_trip():
    a, b = _MPStub("a"), _MPStub("b")
    mgr = MultiprocessingManager(a, b)
    assert mgr.push() is mgr  # chainable
    assert a.push_calls == 1 and b.push_calls == 1
    assert mgr.stashes == ["stash-a", "stash-b"]
    assert mgr.pop() is mgr
    # each object gets back exactly its own stash
    assert a.popped_with == "stash-a"
    assert b.popped_with == "stash-b"
    assert mgr.stashes == []


def test_mp_manager_double_push_and_premature_pop_raise():
    stub = _MPStub("s")
    mgr = MultiprocessingManager(stub)
    with pytest.raises(RuntimeError, match="not pushed"):
        mgr.pop()
    mgr.push()
    with pytest.raises(RuntimeError, match="already pushed"):
        mgr.push()
    mgr.pop()
    with pytest.raises(RuntimeError, match="not pushed"):
        mgr.pop()


def test_mp_manager_reusable_after_pop():
    stub = _MPStub("s")
    mgr = MultiprocessingManager(stub)
    mgr.push().pop()
    mgr.push().pop()
    assert stub.push_calls == 2
    assert stub.popped_with == "stash-s"


# ---------------------------------------------------------------------------
# multiprocessing.py: starmap
# ---------------------------------------------------------------------------

def _affine(x, y):
    """Top-level picklable worker for spawn-based pools."""
    return 3 * x + y


def _affine_with_pid(x, y):
    """Top-level picklable worker that also reports the executing PID."""
    return 3 * x + y, os.getpid()


def test_starmap_negative_procs_runs_sequentially():
    # An unpicklable closure with an in-parent side effect can only succeed
    # on the sequential (in-process) branch: a spawn/forkserver pool would
    # fail to pickle it, and side effects in a child would be invisible here.
    seen = []

    def worker(x, y):
        seen.append((x, y))
        return 3 * x + y

    args = [(i, i * i) for i in range(8)]
    expected = [3 * x + y for x, y in args]
    result = MultiprocessingManager.starmap(worker, args, n_procs=-1)
    assert result == expected
    assert isinstance(result, list)
    assert seen == args  # executed in this process, in order


def test_starmap_negative_procs_empty_input():
    assert MultiprocessingManager.starmap(_affine, [], n_procs=-1) == []


def test_starmap_two_procs_matches_serial():
    # starmap silently falls back to sequential execution on pool failure,
    # so also assert every task ran in a child process (PID != parent PID):
    # the fallback would report the parent PID and fail this test.
    args = [(i, 10 - i) for i in range(6)]
    expected = [3 * x + y for x, y in args]
    result = MultiprocessingManager.starmap(_affine_with_pid, args, n_procs=2)
    values = [v for v, _pid in result]
    pids = [pid for _v, pid in result]
    assert values == expected  # order preserved across workers
    assert all(pid != os.getpid() for pid in pids)


def test_starmap_zero_procs_raises():
    # 0 is outside the documented contract (None or <0); the underlying
    # Pool rejects it and starmap does not catch ValueError.
    with pytest.raises(ValueError):
        MultiprocessingManager.starmap(_affine, [(1, 2)], n_procs=0)


# ---------------------------------------------------------------------------
# multiprocessing.py: validate_multiprocessing_setup
# ---------------------------------------------------------------------------

def test_validate_setup_noop_when_multiprocessing_disabled(monkeypatch):
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_multiprocessing_setup(False)  # must not warn


def test_validate_setup_accepts_fully_pinned_env(monkeypatch):
    for var in THREAD_VARS:
        monkeypatch.setenv(var, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_multiprocessing_setup(True)  # must not warn


def test_validate_setup_warns_on_missing_or_wrong_var(monkeypatch):
    for var in THREAD_VARS:
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    with pytest.warns(UserWarning, match="multithreading not disabled"):
        validate_multiprocessing_setup(True)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    with pytest.warns(UserWarning, match="multithreading not disabled"):
        validate_multiprocessing_setup(True)


def test_validate_setup_custom_thread_vars(monkeypatch):
    monkeypatch.delenv("SRAZ_TEST_THREADS", raising=False)
    with pytest.warns(UserWarning, match="SRAZ_TEST_THREADS"):
        validate_multiprocessing_setup(True, thread_vars=["SRAZ_TEST_THREADS"])
    monkeypatch.setenv("SRAZ_TEST_THREADS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_multiprocessing_setup(True, thread_vars=["SRAZ_TEST_THREADS"])
