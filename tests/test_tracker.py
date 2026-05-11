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


# ---------- retry-with-backoff ----------

def _mock_handler_factory(behaviors: list, *, init_ok: bool = True):
    """`behaviors` is a list of HTTP status codes (or callables) for /metrics POSTs.
    Each call to /metrics POPs the next behavior. /init always returns 200 unless
    init_ok=False (returns 503).
    """
    state = {"metrics_calls": 0, "init_calls": 0}

    def handler(request):
        if "/init" in request.url.path:
            state["init_calls"] += 1
            return httpx.Response(200 if init_ok else 503)
        if "/metrics" in request.url.path:
            i = state["metrics_calls"]
            state["metrics_calls"] += 1
            spec = behaviors[i] if i < len(behaviors) else behaviors[-1]
            if callable(spec):
                return spec(request)
            return httpx.Response(spec, json={"appended": 1})
        return httpx.Response(404)

    return handler, state


import httpx  # noqa: E402


def _make_tracker_with_handler(tmp_path: Path, handler, **kw):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    client.headers["Authorization"] = "Bearer t"
    kw.setdefault("retry_delays", (0.01, 0.02))  # tiny delays for fast tests
    return Tracker(
        project="p",
        name="n",
        config={},
        local_dir=tmp_path / "local",
        batch_size=1,
        batch_interval=0.05,
        _client=client,
        **kw,
    )


def test_post_retries_on_5xx_then_succeeds(tmp_path: Path):
    handler, state = _mock_handler_factory([503, 503, 200])
    t = _make_tracker_with_handler(tmp_path, handler)
    t.log({"step": 1})
    t.finish(timeout=5)
    # Init (1) + 3 metrics attempts (2 failures + 1 success).
    assert state["metrics_calls"] == 3
    assert t.failed_requests == 2
    assert "HTTP 503" in (t.last_error or "")


def test_post_retries_exhausted_returns_failure(tmp_path: Path):
    handler, state = _mock_handler_factory([503, 503, 503, 503])
    t = _make_tracker_with_handler(tmp_path, handler)
    t.log({"step": 1})
    t.finish(timeout=5)
    # retry_delays=(0.01, 0.02) → 3 total attempts.
    assert state["metrics_calls"] == 3
    assert t.failed_requests == 3


def test_post_4xx_does_not_retry(tmp_path: Path):
    handler, state = _mock_handler_factory([403])
    t = _make_tracker_with_handler(tmp_path, handler)
    t.log({"step": 1})
    t.finish(timeout=5)
    # 4xx → single attempt, no retry.
    assert state["metrics_calls"] == 1
    assert t.failed_requests == 1
    assert "HTTP 403" in (t.last_error or "")


def test_post_retries_on_transport_error(tmp_path: Path):
    """Transport-level errors (connection refused, etc.) also trigger retry."""
    calls = {"n": 0}

    def handler(request):
        if "/init" in request.url.path:
            return httpx.Response(200)
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("simulated network blip")
        return httpx.Response(200, json={"appended": 1})

    t = _make_tracker_with_handler(tmp_path, handler)
    t.log({"step": 1})
    t.finish(timeout=5)
    assert calls["n"] == 2
    assert t.failed_requests == 1
    assert "ConnectError" in (t.last_error or "")


def test_retry_does_not_run_on_trainer_thread(tmp_path: Path):
    """Hot path stays fast even when the daemon is in the middle of backoff."""
    import time as _t

    handler, _ = _mock_handler_factory([503, 503, 503])  # always 5xx; daemon sleeps
    t = _make_tracker_with_handler(
        tmp_path, handler, retry_delays=(0.2, 0.4)  # noticeable sleeps
    )
    # While the daemon is retrying, log() on the trainer thread must stay tight.
    samples = []
    for _ in range(200):
        start = _t.perf_counter_ns()
        t.log({"step": 1, "v": 1.0})
        samples.append(_t.perf_counter_ns() - start)
    samples.sort()
    median_us = samples[len(samples) // 2] / 1000
    p99_us = samples[int(len(samples) * 0.99)] / 1000
    t.finish(timeout=5)
    assert median_us < 100, f"median log() {median_us:.1f}µs blew past budget while retrying"
    assert p99_us < 500, f"p99 log() {p99_us:.1f}µs blew past 500µs ceiling"
