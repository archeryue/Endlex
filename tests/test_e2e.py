"""End-to-end smoke: real uvicorn on a real port + real Tracker + checkpoint upload."""

from __future__ import annotations

from pathlib import Path

import httpx

from endlex import Tracker, upload_checkpoint


def test_metrics_and_checkpoint_round_trip(live_server, tmp_path: Path):
    url, server_data = live_server
    local_dir = tmp_path / "local"

    tracker = Tracker(
        project="proj",
        name="e2e",
        config={"lr": 1e-4, "seed": 42},
        local_dir=local_dir,
        url=url,
        token="e2e-tok",
        batch_size=10,
        batch_interval=0.1,
    )
    try:
        for step in range(25):
            tracker.log(
                {
                    "step": step,
                    "train/loss": 2.0 - step * 0.01,
                    "train/mfu": 0.4,
                }
            )

        # write & upload a fake checkpoint while the tracker is still streaming
        model = tmp_path / "model.pt"
        model.write_bytes(b"\xab\xcd" * 4096)
        meta = tmp_path / "meta.json"
        meta.write_text('{"step": 1000, "lr": 1e-4}')
        ok = upload_checkpoint(
            "e2e",
            1000,
            {"model.pt": model, "meta.json": meta},
            url=url,
            token="e2e-tok",
        )
        assert ok is True
    finally:
        tracker.finish(timeout=10)

    # ---- verify via real HTTP ----
    with httpx.Client(base_url=url) as c:
        runs = c.get("/api/runs").json()
        assert any(r["name"] == "e2e" for r in runs)

        info = c.get("/api/runs/e2e").json()
        assert info["config"] == {"lr": 1e-4, "seed": 42}
        assert info["summary"]["num_events"] == 25
        assert info["checkpoints"][0]["step"] == "step_001000"

        body = c.get("/api/runs/e2e/metrics").json()
        steps = [e["step"] for e in body["events"]]
        assert steps == list(range(25))

        # checkpoint bytes round-trip
        r = c.get("/api/runs/e2e/ckpt/1000/model.pt")
        assert r.status_code == 200
        assert r.content == model.read_bytes()

        # dashboard HTML renders the live run
        r = c.get("/")
        assert r.status_code == 200
        assert "e2e" in r.text


def test_sse_stream_emits_sse_framing(live_server):
    """Open a real SSE stream and read at least one full event frame.

    Chunked-encoding boundaries make multi-chunk httpx reads timing-sensitive,
    so we only assert on the first frame's framing here. End-to-end metric
    delivery via SSE is covered by the run-page browser test.
    """
    import json as _j

    url, _ = live_server
    auth = {"Authorization": "Bearer e2e-tok"}

    with httpx.Client(base_url=url) as c:
        assert c.post("/api/runs/sse/init", json={}, headers=auth).status_code == 200
        assert (
            c.post(
                "/api/runs/sse/metrics",
                json=[{"step": 0, "train/loss": 5.0}],
                headers=auth,
            ).status_code
            == 200
        )

    with httpx.Client(base_url=url, timeout=5.0) as c:
        with c.stream(
            "GET", "/api/runs/sse/metrics/stream?poll_interval=0.05"
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            assert r.headers.get("cache-control") == "no-cache"
            # iter_raw yields bytes as they arrive (vs iter_bytes(chunk_size=N)
            # which blocks until N bytes accumulate — wrong for infinite streams).
            buf = b""
            for raw in r.iter_raw():
                buf += raw
                if b"event: metric" in buf and b'"step":0' in buf:
                    break
            txt = buf.decode("utf-8")
            assert ": stream open" in txt
            assert "event: metric" in txt
            assert '"step":0' in txt


def test_sse_stream_404_for_missing_run(live_server):
    url, _ = live_server
    with httpx.Client(base_url=url, timeout=5.0) as c:
        r = c.get("/api/runs/ghost/metrics/stream")
        assert r.status_code == 404


def test_sse_stream_hard_lifetime_closes_gen(live_server):
    """A short max_lifetime should let the server close the gen on its own,
    proving the lifetime cap actually triggers (defense-in-depth against
    wedged clients)."""
    import time as _t

    url, _ = live_server
    auth = {"Authorization": "Bearer e2e-tok"}
    with httpx.Client(base_url=url) as c:
        assert c.post("/api/runs/lt/init", json={}, headers=auth).status_code == 200

    t0 = _t.time()
    with httpx.Client(base_url=url, timeout=10.0) as c:
        # 0.5s lifetime cap → server closes the chunked stream after ~0.5s.
        with c.stream(
            "GET",
            "/api/runs/lt/metrics/stream?poll_interval=0.1&max_lifetime=1.0",
        ) as r:
            assert r.status_code == 200
            # iter_raw exits cleanly when the server closes the chunked stream.
            for _ in r.iter_raw():
                pass
    elapsed = _t.time() - t0
    # Should be ~1s; allow 4s headroom for slow CI.
    assert 0.5 < elapsed < 4.0, f"stream lifetime {elapsed:.2f}s outside expected window"


def test_sse_stream_exits_on_client_disconnect(live_server):
    """Open a stream, close it from the client, then confirm the server is
    still responsive (would deadlock or 500 if the disconnect leaked)."""
    url, _ = live_server
    auth = {"Authorization": "Bearer e2e-tok"}
    with httpx.Client(base_url=url) as c:
        assert c.post("/api/runs/dc/init", json={}, headers=auth).status_code == 200

    with httpx.Client(base_url=url, timeout=5.0) as c:
        with c.stream(
            "GET", "/api/runs/dc/metrics/stream?poll_interval=0.05"
        ) as r:
            assert r.status_code == 200
            # Read one byte then close.
            for chunk in r.iter_raw():
                if chunk:
                    break

    # The server should still answer a follow-up request promptly.
    with httpx.Client(base_url=url, timeout=2.0) as c:
        r = c.get("/api/runs")
        assert r.status_code == 200


def test_offline_tracker_still_logs_locally(tmp_path: Path, monkeypatch):
    """If ENDLEX_URL is unset the tracker must still produce a full local JSONL."""
    monkeypatch.delenv("ENDLEX_URL", raising=False)
    local = tmp_path / "local"
    t = Tracker(project="p", name="offline", config={}, local_dir=local, url=None)
    for i in range(50):
        t.log({"step": i})
    t.finish()
    lines = (local / "p" / "offline" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 50
