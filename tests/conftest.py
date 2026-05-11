"""Shared pytest fixtures."""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from endlex.server.app import create_app


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"server on :{port} never came up")


@pytest.fixture
def live_server(tmp_path: Path, monkeypatch):
    """Boot a real uvicorn on a random port. Yields (url, data_root)."""
    monkeypatch.setenv("ENDLEX_TOKEN", "e2e-tok")
    monkeypatch.setenv("ENDLEX_PUBLIC_READS", "1")
    data_root = tmp_path / "server-data"
    port = _find_free_port()
    app = create_app(data_root)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_for_port(port)
        yield f"http://127.0.0.1:{port}", data_root
    finally:
        server.should_exit = True
        thread.join(timeout=5)
