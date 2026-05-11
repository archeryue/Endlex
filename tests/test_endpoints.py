from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from endlex.server.app import create_app


TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv("ENDLEX_TOKEN", TOKEN)
    monkeypatch.setenv("ENDLEX_PUBLIC_READS", "1")
    app = create_app(tmp_path)
    return TestClient(app)


def _init(client: TestClient, name: str, config: dict | None = None):
    r = client.post(f"/api/runs/{name}/init", json=config or {"lr": 1e-4}, headers=AUTH)
    assert r.status_code == 200, r.text
    return r


# ---------- writes ----------

def test_init_requires_auth(client):
    r = client.post("/api/runs/r/init", json={})
    assert r.status_code == 401


def test_init_creates_run(client):
    _init(client, "r1")
    r = client.get("/api/runs/r1")
    assert r.status_code == 200
    assert r.json()["config"] == {"lr": 1e-4}


def test_init_idempotent_same_config(client):
    _init(client, "r")
    _init(client, "r")  # no 409


def test_init_conflict_on_different_config(client):
    _init(client, "r", {"lr": 1e-4})
    r = client.post("/api/runs/r/init", json={"lr": 5e-5}, headers=AUTH)
    assert r.status_code == 409


def test_init_force_overwrites(client):
    _init(client, "r", {"lr": 1e-4})
    r = client.post(
        "/api/runs/r/init?force=true", json={"lr": 5e-5}, headers=AUTH
    )
    assert r.status_code == 200
    assert client.get("/api/runs/r").json()["config"] == {"lr": 5e-5}


def test_init_rejects_bad_name(client):
    r = client.post("/api/runs/..bad/init", json={}, headers=AUTH)
    assert r.status_code == 400


def test_metrics_append_and_read(client):
    _init(client, "r")
    r = client.post(
        "/api/runs/r/metrics",
        json=[{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.5}],
        headers=AUTH,
    )
    assert r.status_code == 200
    assert r.json() == {"appended": 2}

    r = client.get("/api/runs/r/metrics")
    body = r.json()
    assert body["events"] == [
        {"step": 1, "loss": 2.0},
        {"step": 2, "loss": 1.5},
    ]
    assert body["offset"] > 0


def test_metrics_cursor_resume(client):
    _init(client, "r")
    client.post("/api/runs/r/metrics", json=[{"step": 1}], headers=AUTH)
    first = client.get("/api/runs/r/metrics").json()
    client.post("/api/runs/r/metrics", json=[{"step": 2}, {"step": 3}], headers=AUTH)
    second = client.get(f"/api/runs/r/metrics?since={first['offset']}").json()
    assert second["events"] == [{"step": 2}, {"step": 3}]
    assert second["offset"] > first["offset"]


def test_metrics_missing_run_is_404(client):
    r = client.get("/api/runs/ghost/metrics")
    assert r.status_code == 404


def test_checkpoint_upload_and_download(client):
    _init(client, "r")
    model_bytes = b"\x10\x20\x30" * 1024
    meta_bytes = json.dumps({"step": 1000}).encode()
    r = client.post(
        "/api/runs/r/ckpt/1000",
        files=[
            ("files", ("model.pt", io.BytesIO(model_bytes), "application/octet-stream")),
            ("files", ("meta.json", io.BytesIO(meta_bytes), "application/json")),
        ],
        headers=AUTH,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["written"]["model.pt"] == len(model_bytes)

    # show up in summary
    info = client.get("/api/runs/r").json()
    assert info["checkpoints"][0]["step"] == "step_001000"
    assert set(info["checkpoints"][0]["files"]) == {"model.pt", "meta.json"}

    # download round-trips
    r = client.get("/api/runs/r/ckpt/1000/model.pt")
    assert r.status_code == 200
    assert r.content == model_bytes


def test_delete_run(client):
    _init(client, "r")
    r = client.delete("/api/runs/r", headers=AUTH)
    assert r.status_code == 204
    assert client.get("/api/runs/r").status_code == 404


# ---------- reads ----------

def test_list_runs(client):
    _init(client, "alpha", {"x": 1})
    _init(client, "beta", {"x": 2})
    r = client.get("/api/runs")
    assert r.status_code == 200
    names = {row["name"] for row in r.json()}
    assert names == {"alpha", "beta"}


def test_html_dashboard_renders(client):
    _init(client, "r1")
    client.post(
        "/api/runs/r1/metrics",
        json=[{"step": 5, "train/loss": 1.234, "val/bpb": 0.987}],
        headers=AUTH,
    )
    r = client.get("/")
    assert r.status_code == 200
    assert "r1" in r.text
    assert "train/loss" in r.text
    assert "1.2340" in r.text  # latest train/loss formatted


def test_html_run_page_renders(client):
    _init(client, "r", {"lr": 1e-4})
    r = client.get("/run/r")
    assert r.status_code == 200
    # config visible
    assert "0.0001" in r.text
    # chart.js loaded + panel scaffolding
    assert "chart.umd.min.js" in r.text
    assert 'id="charts"' in r.text
    assert "train/loss vs step" in r.text


def test_html_run_page_missing_run_is_404(client):
    r = client.get("/run/ghost")
    assert r.status_code == 404


# ---------- state ----------

def test_patch_state_updates_tags(client):
    _init(client, "r")
    r = client.patch(
        "/api/runs/r/state",
        json={"tags": ["best", "exp-1"]},
        headers=AUTH,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["tags"] == ["best", "exp-1"]
    assert client.get("/api/runs/r").json()["summary"]["tags"] == ["best", "exp-1"]


def test_patch_state_archives_and_filters(client):
    _init(client, "keep")
    _init(client, "hidden")
    client.patch("/api/runs/hidden/state", json={"archived": True}, headers=AUTH)
    names = {r["name"] for r in client.get("/api/runs").json()}
    assert names == {"keep"}
    names = {r["name"] for r in client.get("/api/runs?include_archived=true").json()}
    assert names == {"keep", "hidden"}


def test_patch_state_requires_auth(client):
    _init(client, "r")
    r = client.patch("/api/runs/r/state", json={"archived": True})
    assert r.status_code == 401


def test_patch_state_missing_run(client):
    r = client.patch("/api/runs/ghost/state", json={"archived": True}, headers=AUTH)
    assert r.status_code == 404


# ---------- compare ----------

def test_compare_renders_with_runs(client):
    _init(client, "alpha")
    _init(client, "beta")
    r = client.get("/compare?runs=alpha,beta")
    assert r.status_code == 200
    assert "alpha" in r.text
    assert "beta" in r.text
    # Chart.js loaded; charts div present
    assert "chart.umd.min.js" in r.text
    assert 'id="charts"' in r.text


def test_compare_filters_missing_runs(client):
    _init(client, "alpha")
    r = client.get("/compare?runs=alpha,ghost")
    assert r.status_code == 200
    assert "alpha" in r.text
    # ghost shows up in "missing" notice
    assert "ghost" in r.text
    assert "missing" in r.text


def test_compare_empty_param(client):
    r = client.get("/compare")
    assert r.status_code == 200
    assert "No runs selected" in r.text


# ---------- retention ----------

def test_ckpt_upload_prunes_per_run_retention(client):
    _init(client, "r")
    # Per-run policy: keep only the last 2 checkpoints.
    client.patch(
        "/api/runs/r/state",
        json={"retention": {"keep_last": 2}},
        headers=AUTH,
    )
    for step in (100, 200, 300, 400):
        r = client.post(
            f"/api/runs/r/ckpt/{step}",
            files=[("files", ("model.pt", io.BytesIO(b"x"), "application/octet-stream"))],
            headers=AUTH,
        )
        assert r.status_code == 200
    # The final upload's prune list should include the now-oldest dirs.
    assert r.json()["pruned"], "expected the last upload to prune old ckpts"
    steps = [c["step"] for c in client.get("/api/runs/r").json()["checkpoints"]]
    assert steps == ["step_000300", "step_000400"]


def test_admin_prune_all_endpoint(client):
    _init(client, "a")
    _init(client, "b")
    for step in (100, 200, 300):
        client.post(
            f"/api/runs/a/ckpt/{step}",
            files=[("files", ("model.pt", io.BytesIO(b"x"), "application/octet-stream"))],
            headers=AUTH,
        )
    # No retention set yet on `a` → prune is no-op.
    r = client.post("/api/admin/prune", headers=AUTH)
    assert r.status_code == 200
    assert r.json() == {"pruned": {}}
    # Set retention then prune.
    client.patch(
        "/api/runs/a/state",
        json={"retention": {"keep_last": 1}},
        headers=AUTH,
    )
    r = client.post("/api/admin/prune", headers=AUTH)
    body = r.json()
    assert "a" in body["pruned"]
    assert sorted(body["pruned"]["a"]) == ["step_000100", "step_000200"]
    assert "b" not in body["pruned"]


def test_admin_prune_requires_auth(client):
    r = client.post("/api/admin/prune")
    assert r.status_code == 401


# SSE stream tests live in test_e2e.py — TestClient buffers ASGI
# streaming responses too eagerly to assert on event timing.
