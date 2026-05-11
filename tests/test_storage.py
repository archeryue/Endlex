from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from endlex.server.storage import (
    InvalidName,
    RunLocked,
    RunNotFound,
    Storage,
)


@pytest.fixture
def store(tmp_path: Path) -> Storage:
    return Storage(tmp_path)


def test_init_run_creates_files(store: Storage):
    store.init_run("run1", {"lr": 1e-4})
    run_dir = store.runs_dir / "run1"
    assert (run_dir / "config.json").exists()
    assert (run_dir / ".lock").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert json.loads((run_dir / "config.json").read_text()) == {"lr": 1e-4}


def test_init_run_idempotent_with_same_config(store: Storage):
    store.init_run("run1", {"lr": 1e-4})
    store.init_run("run1", {"lr": 1e-4})  # no error


def test_init_run_locked_on_different_config(store: Storage):
    store.init_run("run1", {"lr": 1e-4})
    with pytest.raises(RunLocked):
        store.init_run("run1", {"lr": 5e-5})


def test_init_run_force_overwrites(store: Storage):
    store.init_run("run1", {"lr": 1e-4})
    store.init_run("run1", {"lr": 5e-5}, force=True)
    assert store.get_config("run1") == {"lr": 5e-5}


@pytest.mark.parametrize("bad", ["../escape", ".hidden", "with/slash", "", "x" * 200])
def test_init_run_rejects_bad_names(store: Storage, bad: str):
    with pytest.raises(InvalidName):
        store.init_run(bad, {})


def test_append_and_read_metrics(store: Storage):
    store.init_run("r", {})
    n = store.append_metrics("r", [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.5}])
    assert n == 2
    events, off = store.read_metrics("r")
    assert events == [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.5}]
    assert off > 0


def test_read_metrics_resumes_from_offset(store: Storage):
    store.init_run("r", {})
    store.append_metrics("r", [{"step": 1}])
    events1, off1 = store.read_metrics("r")
    assert len(events1) == 1
    store.append_metrics("r", [{"step": 2}, {"step": 3}])
    events2, off2 = store.read_metrics("r", since_offset=off1)
    assert events2 == [{"step": 2}, {"step": 3}]
    assert off2 > off1


def test_append_metrics_empty_batch_is_noop(store: Storage):
    store.init_run("r", {})
    assert store.append_metrics("r", []) == 0


def test_append_metrics_missing_run_raises(store: Storage):
    with pytest.raises(RunNotFound):
        store.append_metrics("ghost", [{"step": 1}])


def test_list_runs(store: Storage):
    store.init_run("alpha", {"x": 1})
    store.init_run("beta", {"x": 2})
    store.append_metrics("alpha", [{"step": 5, "loss": 0.1}])
    summaries = {s.name: s for s in store.list_runs()}
    assert set(summaries) == {"alpha", "beta"}
    assert summaries["alpha"].num_events == 1
    assert summaries["alpha"].latest == {"step": 5, "loss": 0.1}
    assert summaries["beta"].num_events == 0


def test_delete_run(store: Storage):
    store.init_run("r", {})
    store.append_metrics("r", [{"step": 1}])
    store.delete_run("r")
    assert not store.run_exists("r")
    with pytest.raises(RunNotFound):
        store.delete_run("r")


def test_checkpoint_round_trip(store: Storage):
    store.init_run("r", {})
    payload = b"\x00\x01\x02" * 1024
    n = store.write_checkpoint_file("r", 1000, "model.pt", io.BytesIO(payload))
    assert n == len(payload)
    p = store.checkpoint_file_path("r", 1000, "model.pt")
    assert p.read_bytes() == payload
    # also reachable via the zero-padded form
    assert store.checkpoint_file_path("r", "1000", "model.pt") == p


def test_checkpoint_list(store: Storage):
    store.init_run("r", {})
    store.write_checkpoint_file("r", 1000, "model.pt", io.BytesIO(b"a"))
    store.write_checkpoint_file("r", 1000, "meta.json", io.BytesIO(b"{}"))
    store.write_checkpoint_file("r", 2000, "model.pt", io.BytesIO(b"b"))
    cks = store.list_checkpoints("r")
    assert [c["step"] for c in cks] == ["step_001000", "step_002000"]
    assert cks[0]["files"] == ["meta.json", "model.pt"]


def test_checkpoint_rejects_path_traversal(store: Storage):
    store.init_run("r", {})
    with pytest.raises(InvalidName):
        store.write_checkpoint_file("r", 1000, "../escape", io.BytesIO(b"x"))
    with pytest.raises(InvalidName):
        store.checkpoint_file_path("r", 1000, "../escape")
