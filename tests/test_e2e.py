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
