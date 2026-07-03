from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from endlex.checkpoint_sync import (
    download_checkpoint,
    upload_checkpoint,
    upload_checkpoint_async,
)
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


# ---------- integrity + chunked path ----------

def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_upload_records_verified_checksum(tmp_path: Path, server_data: Path, client):
    files = _write_files(tmp_path)
    assert upload_checkpoint("r", 1000, files, _client=client) is True
    r = client.get("/api/runs/r")
    meta = r.json()["checkpoints"][0]["files_meta"]
    assert meta["model.pt"]["sha256"] == _sha(files["model.pt"])


def test_upload_large_file_goes_chunked(tmp_path: Path, server_data: Path, client):
    big = tmp_path / "model.pt"
    big.write_bytes(os.urandom(10_000))
    ok = upload_checkpoint(
        "r",
        3000,
        {"model.pt": big},
        chunk_threshold=1_000,  # force the chunked path
        chunk_size=3_000,       # 4 chunks
        _client=client,
    )
    assert ok is True
    stored = server_data / "checkpoints" / "r" / "step_003000" / "model.pt"
    assert stored.read_bytes() == big.read_bytes()
    # no staging residue
    assert not (stored.parent / "model.pt.part").exists()


def test_chunked_upload_resumes_from_partial(tmp_path: Path, server_data: Path, client):
    """A .part left by an interrupted upload is continued, not restarted."""
    big = tmp_path / "model.pt"
    big.write_bytes(os.urandom(9_000))
    sha = _sha(big)
    # Simulate a prior attempt that got the first 4000 bytes through.
    r = client.put(
        "/api/runs/r/ckpt/4000/files/model.pt",
        params={"offset": 0, "total": 9_000, "sha256": sha},
        content=big.read_bytes()[:4_000],
    )
    assert r.status_code == 200
    ok = upload_checkpoint(
        "r",
        4000,
        {"model.pt": big},
        chunk_threshold=1_000,
        chunk_size=3_000,
        _client=client,
    )
    assert ok is True
    stored = server_data / "checkpoints" / "r" / "step_004000" / "model.pt"
    assert stored.read_bytes() == big.read_bytes()


def test_chunked_upload_skips_identical_complete_file(
    tmp_path: Path, server_data: Path, client
):
    big = tmp_path / "model.pt"
    big.write_bytes(os.urandom(5_000))
    kw = dict(chunk_threshold=1_000, chunk_size=2_000, _client=client)
    assert upload_checkpoint("r", 5000, {"model.pt": big}, **kw) is True
    # Re-upload of the identical file short-circuits via the status probe.
    assert upload_checkpoint("r", 5000, {"model.pt": big}, **kw) is True


# ---------- download (the pull half of weights sync) ----------

def test_download_checkpoint_roundtrip(tmp_path: Path, server_data: Path, client):
    files = _write_files(tmp_path)
    assert upload_checkpoint("r", 1000, files, _client=client) is True
    dest = tmp_path / "pulled"
    out = download_checkpoint("r", step=1000, dest=dest, _client=client)
    assert sorted(p.name for p in out) == ["meta.json", "model.pt"]
    assert (dest / "model.pt").read_bytes() == files["model.pt"].read_bytes()
    assert not (dest / "model.pt.part").exists()


def test_download_latest_and_subset(tmp_path: Path, server_data: Path, client):
    files = _write_files(tmp_path)
    upload_checkpoint("r", 1000, files, _client=client)
    upload_checkpoint("r", 2000, files, _client=client)
    dest = tmp_path / "pulled"
    out = download_checkpoint("r", dest=dest, files=["meta.json"], _client=client)
    assert [p.name for p in out] == ["meta.json"]


def test_download_verifies_sha256(tmp_path: Path, server_data: Path, client):
    files = _write_files(tmp_path)
    upload_checkpoint("r", 1000, files, _client=client)
    # Corrupt the stored file *after* upload; manifest sha no longer matches.
    stored = server_data / "checkpoints" / "r" / "step_001000" / "model.pt"
    stored.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        download_checkpoint(
            "r", step=1000, dest=tmp_path / "pulled", files=["model.pt"], _client=client
        )
    assert not (tmp_path / "pulled" / "model.pt").exists()


def test_download_missing_run_raises(tmp_path: Path, client):
    with pytest.raises(RuntimeError, match="run lookup failed"):
        download_checkpoint("ghost", dest=tmp_path, _client=client)
