from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from endlex.server.app import create_app
from endlex.tracker import Tracker


@pytest.fixture
def server_data(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("ENDLEX_TOKEN", "tok")
    monkeypatch.setenv("ENDLEX_PUBLIC_READS", "1")
    return tmp_path / "server-data"


def _make_tracker(
    tmp_path: Path, *, online: bool, server_data: Path | None = None, **kwargs
) -> Tracker:
    local_dir = tmp_path / "local"
    if online:
        assert server_data is not None
        app = create_app(server_data)
        tc = TestClient(app)
        tc.headers["Authorization"] = "Bearer tok"
        return Tracker(
            project="p",
            name="r",
            config={"lr": 1e-4},
            local_dir=local_dir,
            batch_interval=0.05,  # fast for tests
            _client=tc,
            **kwargs,
        )
    return Tracker(
        project="p",
        name="r",
        config={"lr": 1e-4},
        local_dir=local_dir,
        url=None,
        **kwargs,
    )


# ---------- offline mode ----------

def test_offline_log_writes_local_jsonl(tmp_path: Path):
    t = _make_tracker(tmp_path, online=False)
    t.log({"step": 1, "loss": 2.0})
    t.log({"step": 2, "loss": 1.5})
    t.finish()
    path = tmp_path / "local" / "p" / "r" / "metrics.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines == [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.5}]


def test_offline_writes_config_json(tmp_path: Path):
    t = _make_tracker(tmp_path, online=False)
    t.finish()
    cfg = json.loads((tmp_path / "local" / "p" / "r" / "config.json").read_text())
    assert cfg == {"lr": 1e-4}


def test_offline_no_thread(tmp_path: Path):
    t = _make_tracker(tmp_path, online=False)
    assert t._thread is None
    t.finish()


# ---------- online (round-trip via ASGI) ----------

def test_online_init_uploads_config(tmp_path: Path, server_data: Path):
    t = _make_tracker(tmp_path, online=True, server_data=server_data)
    t.finish(timeout=5)
    server_cfg = server_data / "runs" / "r" / "config.json"
    assert json.loads(server_cfg.read_text()) == {"lr": 1e-4}


def test_online_log_flushes_to_server(tmp_path: Path, server_data: Path):
    t = _make_tracker(tmp_path, online=True, server_data=server_data, batch_size=2)
    t.log({"step": 1, "loss": 2.0})
    t.log({"step": 2, "loss": 1.5})
    t.log({"step": 3, "loss": 1.0})
    t.finish(timeout=5)
    # All three events should be on the server.
    server_metrics = server_data / "runs" / "r" / "metrics.jsonl"
    lines = [json.loads(l) for l in server_metrics.read_text().splitlines()]
    assert lines == [
        {"step": 1, "loss": 2.0},
        {"step": 2, "loss": 1.5},
        {"step": 3, "loss": 1.0},
    ]


def test_online_time_trigger_flushes_below_batch_size(
    tmp_path: Path, server_data: Path
):
    t = _make_tracker(
        tmp_path, online=True, server_data=server_data, batch_size=1000
    )
    t.log({"step": 1})
    # batch_size is huge so count trigger won't fire; rely on time trigger.
    time.sleep(0.2)
    t.finish(timeout=5)
    server_metrics = server_data / "runs" / "r" / "metrics.jsonl"
    assert "step" in server_metrics.read_text()


# ---------- backpressure ----------

def test_drop_oldest_on_full_queue(tmp_path: Path):
    # No URL → no daemon → queue just fills.
    t = _make_tracker(tmp_path, online=False, queue_max=3)
    for i in range(10):
        t.log({"step": i})
    assert t.dropped == 7
    # Local file has all 10 — drop only affects remote queue.
    lines = (
        (tmp_path / "local" / "p" / "r" / "metrics.jsonl")
        .read_text()
        .splitlines()
    )
    assert len(lines) == 10
    t.finish()


# ---------- performance contract ----------

def test_log_hot_path_under_100us(tmp_path: Path):
    """`.log()` budget per TECH_PLAN: <100 µs.

    Measured offline (no URL) so we exercise *only* the hot path:
    json.dumps + 2 file writes + deque append. Use median over 5000
    iterations to ignore one-off GC pauses.
    """
    t = _make_tracker(tmp_path, online=False, queue_max=100_000)
    payload = {
        "step": 1,
        "train/loss": 2.345,
        "train/mfu": 0.42,
        "train/dt": 0.123,
        "train/tok_per_sec": 12345.6,
        "train/lrm": 0.5,
    }
    # warm up
    for _ in range(500):
        t.log(payload)

    n = 5000
    samples = []
    for _ in range(n):
        start = time.perf_counter_ns()
        t.log(payload)
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    median_us = samples[n // 2] / 1000
    p99_us = samples[int(n * 0.99)] / 1000
    t.finish()
    # Generous on p99 because GC + scheduling can spike. Median must stay tight.
    assert median_us < 100, f"median {median_us:.1f}µs exceeds 100µs budget"
    assert p99_us < 500, f"p99 {p99_us:.1f}µs blew past 500µs ceiling"


# ---------- context manager ----------

def test_context_manager_calls_finish(tmp_path: Path):
    with _make_tracker(tmp_path, online=False) as t:
        t.log({"step": 1})
    assert t._finished is True


# ---------- bad args ----------

def test_missing_project_or_name_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        Tracker(project="", name="x", local_dir=tmp_path)
    with pytest.raises(ValueError):
        Tracker(project="x", name="", local_dir=tmp_path)
