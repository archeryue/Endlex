from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from endlex.checkpoint_sync import upload_checkpoint, upload_checkpoint_async
from endlex.server.app import create_app
from endlex.server.storage import Storage


@pytest.fixture
def server_data(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("ENDLEX_TOKEN", "tok")
    monkeypatch.setenv("ENDLEX_PUBLIC_READS", "1")
    root = tmp_path / "server-data"
    Storage(root).init_run("r", {})
    return root


@pytest.fixture
def client(server_data: Path) -> TestClient:
    app = create_app(server_data)
    tc = TestClient(app)
    tc.headers["Authorization"] = "Bearer tok"
    return tc


def _write_files(tmp: Path) -> dict[str, Path]:
    model = tmp / "model.pt"
    model.write_bytes(b"\x10\x20\x30" * 1024)
    meta = tmp / "meta.json"
    meta.write_text('{"step": 1000}')
    return {"model.pt": model, "meta.json": meta}


def test_upload_checkpoint_roundtrip(tmp_path: Path, server_data: Path, client):
    files = _write_files(tmp_path)
    ok = upload_checkpoint("r", 1000, files, _client=client)
    assert ok is True
    server_model = server_data / "checkpoints" / "r" / "step_001000" / "model.pt"
    server_meta = server_data / "checkpoints" / "r" / "step_001000" / "meta.json"
    assert server_model.read_bytes() == files["model.pt"].read_bytes()
    assert server_meta.read_text() == files["meta.json"].read_text()


def test_upload_returns_false_on_missing_file(tmp_path: Path, client):
    ok = upload_checkpoint(
        "r", 1000, {"ghost.pt": tmp_path / "does-not-exist.pt"}, _client=client
    )
    assert ok is False


def test_upload_returns_false_on_404_run(tmp_path: Path, client):
    files = _write_files(tmp_path)
    ok = upload_checkpoint("nonexistent-run", 1000, files, _client=client)
    assert ok is False


def test_upload_returns_false_when_url_unset(
    tmp_path: Path, monkeypatch
):
    monkeypatch.delenv("ENDLEX_URL", raising=False)
    files = _write_files(tmp_path)
    ok = upload_checkpoint("r", 1000, files)  # no _client, no url
    assert ok is False


def test_upload_async_returns_thread(tmp_path: Path, server_data: Path, client):
    files = _write_files(tmp_path)
    t = upload_checkpoint_async("r", 2000, files, _client=client)
    t.join(timeout=10)
    assert not t.is_alive()
    assert (
        server_data / "checkpoints" / "r" / "step_002000" / "model.pt"
    ).read_bytes() == files["model.pt"].read_bytes()
