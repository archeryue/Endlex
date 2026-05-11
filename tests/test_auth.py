from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from endlex.server.auth import require_read_auth, require_write_auth


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ENDLEX_TOKEN", "s3cret")
    monkeypatch.setenv("ENDLEX_PUBLIC_READS", "1")
    app = FastAPI()

    @app.post("/w", dependencies=[Depends(require_write_auth)])
    def w():
        return {"ok": True}

    @app.get("/r", dependencies=[Depends(require_read_auth)])
    def r():
        return {"ok": True}

    return TestClient(app)


def test_write_requires_auth(client):
    assert client.post("/w").status_code == 401


def test_write_rejects_wrong_token(client):
    r = client.post("/w", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 403


def test_write_accepts_good_token(client):
    r = client.post("/w", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_read_open_by_default(client):
    assert client.get("/r").status_code == 200


def test_read_protected_when_configured(monkeypatch):
    monkeypatch.setenv("ENDLEX_TOKEN", "s3cret")
    monkeypatch.setenv("ENDLEX_PUBLIC_READS", "0")
    app = FastAPI()

    @app.get("/r", dependencies=[Depends(require_read_auth)])
    def r():
        return {"ok": True}

    c = TestClient(app)
    assert c.get("/r").status_code == 401
    assert c.get("/r", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_missing_server_token_is_500(monkeypatch):
    monkeypatch.delenv("ENDLEX_TOKEN", raising=False)
    app = FastAPI()

    @app.post("/w", dependencies=[Depends(require_write_auth)])
    def w():
        return {"ok": True}

    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/w", headers={"Authorization": "Bearer anything"})
    assert r.status_code == 500
