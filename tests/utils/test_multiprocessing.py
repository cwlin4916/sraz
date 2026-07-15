"""utils/multiprocessing.py: MultiprocessingManager push/pop, starmap, validate."""

import os
import warnings

import pytest

from sraz.utils.common import THREAD_VARS
from sraz.utils.multiprocessing import (
    MultiprocessingManager,
    validate_multiprocessing_setup,
)


# ---------------------------------------------------------------------------
# MultiprocessingManager push/pop
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
# starmap
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
# validate_multiprocessing_setup
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
