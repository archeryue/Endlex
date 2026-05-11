from __future__ import annotations

import io
import json
import os
import time
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


def test_checkpoint_requires_existing_run(store: Storage):
    with pytest.raises(RunNotFound):
        store.write_checkpoint_file("ghost", 1000, "model.pt", io.BytesIO(b"x"))


# ---------- state (tags + archived) ----------

def test_get_state_defaults(store: Storage):
    store.init_run("r", {})
    assert store.get_state("r") == {
        "tags": [],
        "archived": False,
        "retention": {},
    }


def test_update_state_persists_tags(store: Storage):
    store.init_run("r", {})
    state = store.update_state("r", {"tags": ["foo", "bar"]})
    assert state["tags"] == ["foo", "bar"]
    assert store.get_state("r")["tags"] == ["foo", "bar"]


def test_update_state_dedupes_and_strips_tags(store: Storage):
    store.init_run("r", {})
    state = store.update_state("r", {"tags": [" foo ", "foo", "bar", "", "  "]})
    assert state["tags"] == ["foo", "bar"]


def test_update_state_archive(store: Storage):
    store.init_run("r", {})
    assert store.update_state("r", {"archived": True})["archived"] is True
    assert store.get_state("r")["archived"] is True
    assert store.update_state("r", {"archived": False})["archived"] is False


def test_update_state_rejects_non_string_tags(store: Storage):
    store.init_run("r", {})
    from endlex.server.storage import InvalidName
    with pytest.raises(InvalidName):
        store.update_state("r", {"tags": ["ok", 123]})


def test_state_appears_in_summary(store: Storage):
    store.init_run("r", {})
    store.update_state("r", {"tags": ["good"], "archived": True})
    s = store.summarize_run("r")
    assert s.tags == ["good"]
    assert s.archived is True


def test_state_missing_run_raises(store: Storage):
    with pytest.raises(RunNotFound):
        store.get_state("ghost")
    with pytest.raises(RunNotFound):
        store.update_state("ghost", {"archived": True})


# ---------- retention ----------

def _seed_ckpts(store: Storage, run: str, steps: list[int]) -> None:
    store.init_run(run, {})
    for s in steps:
        store.write_checkpoint_file(run, s, "model.pt", io.BytesIO(b"x"))


def test_prune_no_rules_is_noop(store: Storage):
    _seed_ckpts(store, "r", [100, 200, 300])
    deleted = store.prune_checkpoints("r", keep_last=0, max_age_seconds=0)
    assert deleted == []
    assert len(store.list_checkpoints("r")) == 3


def test_prune_keep_last_keeps_most_recent_steps(store: Storage):
    _seed_ckpts(store, "r", [100, 200, 300, 400, 500])
    deleted = store.prune_checkpoints("r", keep_last=2)
    assert sorted(deleted) == ["step_000100", "step_000200", "step_000300"]
    remaining = [c["step"] for c in store.list_checkpoints("r")]
    assert remaining == ["step_000400", "step_000500"]


def test_prune_keep_last_larger_than_count_is_noop(store: Storage):
    _seed_ckpts(store, "r", [100, 200])
    deleted = store.prune_checkpoints("r", keep_last=10)
    assert deleted == []


def test_prune_max_age_deletes_old(store: Storage, tmp_path: Path):
    _seed_ckpts(store, "r", [100, 200, 300])
    # Backdate step_000100 + 200 by 10 days; leave 300 fresh.
    old = time.time() - 10 * 86400
    for s in (100, 200):
        d = store.ckpt_dir / "r" / f"step_{s:06d}"
        os.utime(d, (old, old))
    deleted = store.prune_checkpoints("r", max_age_seconds=5 * 86400)
    assert sorted(deleted) == ["step_000100", "step_000200"]
    remaining = [c["step"] for c in store.list_checkpoints("r")]
    assert remaining == ["step_000300"]


def test_prune_union_of_keep_last_and_max_age(store: Storage):
    _seed_ckpts(store, "r", [100, 200, 300, 400])
    # Make 100 + 200 old; 300 + 400 fresh.
    old = time.time() - 10 * 86400
    for s in (100, 200):
        d = store.ckpt_dir / "r" / f"step_{s:06d}"
        os.utime(d, (old, old))
    # keep_last=1 wants to drop {100,200,300}, max_age wants to drop {100,200}.
    # Union of keepers = {400 (recent step)} ∪ {300,400 (within age)} = {300,400}.
    deleted = store.prune_checkpoints(
        "r", keep_last=1, max_age_seconds=5 * 86400
    )
    assert sorted(deleted) == ["step_000100", "step_000200"]
    remaining = [c["step"] for c in store.list_checkpoints("r")]
    assert remaining == ["step_000300", "step_000400"]


def test_resolved_retention_falls_back_to_server_default(tmp_path: Path):
    s = Storage(tmp_path, default_keep_last=3, default_max_age_days=7.0)
    s.init_run("r", {})
    keep, max_age = s.resolved_retention("r")
    assert keep == 3
    assert max_age == 7 * 86400


def test_resolved_retention_per_run_overrides_default(tmp_path: Path):
    s = Storage(tmp_path, default_keep_last=3, default_max_age_days=7.0)
    s.init_run("r", {})
    s.update_state("r", {"retention": {"keep_last": 1, "max_age_days": 0}})
    keep, max_age = s.resolved_retention("r")
    assert keep == 1
    assert max_age == 0


def test_update_state_validates_retention_shape(store: Storage):
    store.init_run("r", {})
    with pytest.raises(InvalidName):
        store.update_state("r", {"retention": [1, 2]})
