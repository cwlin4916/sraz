"""utils/common.py: thread-var setup, deterministic CUDA, device selection."""

import os
import platform

import pytest
import torch

from sraz.utils.common import (
    THREAD_VARS,
    disable_numpy_multithreading,
    get_device,
    use_deterministic_cuda,
)


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
