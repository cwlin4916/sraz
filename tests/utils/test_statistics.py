"""utils/statistics.py: StatisticsManager record/persist semantics."""

import json
import time

import pytest

from sraz.utils.statistics import StatisticsManager


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
