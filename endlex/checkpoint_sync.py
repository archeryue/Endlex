"""Best-effort checkpoint upload helper for the trainer side.

Call :func:`upload_checkpoint_async` from ``save_checkpoint``; the trainer
returns immediately while a daemon thread streams the multipart POST. Local
save remains the source of truth — failures are logged to stderr and
discarded.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Mapping

import httpx


def upload_checkpoint(
    run_name: str,
    step: int,
    files: Mapping[str, str | os.PathLike[str]],
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 600.0,
    _client: httpx.Client | None = None,
) -> bool:
    """Stream a step's checkpoint files to the Endlex server.

    ``files`` maps the filename the server stores under (e.g. ``"model.pt"``)
    to the local path to read. Returns ``True`` on a 200 response. Any
    failure (missing file, network error, non-200, bad auth) is logged to
    stderr and yields ``False``.
    """
    url = url or os.environ.get("ENDLEX_URL")
    token = token or os.environ.get("ENDLEX_TOKEN")
    if _client is None and not url:
        print(
            "[endlex] ENDLEX_URL unset; skipping checkpoint upload",
            file=sys.stderr,
        )
        return False

    handles: list = []
    try:
        upload_files = []
        for filename, path in files.items():
            p = Path(path)
            if not p.exists():
                print(f"[endlex] missing checkpoint file: {p}", file=sys.stderr)
                return False
            fh = open(p, "rb")
            handles.append(fh)
            upload_files.append(
                ("files", (filename, fh, "application/octet-stream"))
            )

        if _client is not None:
            client = _client
            owns_client = False
        else:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            client = httpx.Client(base_url=url, headers=headers, timeout=timeout)
            owns_client = True
        try:
            r = client.post(
                f"/api/runs/{run_name}/ckpt/{int(step)}",
                files=upload_files,
            )
        finally:
            if owns_client:
                client.close()

        if r.status_code != 200:
            print(
                f"[endlex] checkpoint upload failed: {r.status_code} {r.text}",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as e:  # noqa: BLE001 — best-effort
        print(f"[endlex] checkpoint upload error: {e}", file=sys.stderr)
        return False
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass


def upload_checkpoint_async(
    run_name: str,
    step: int,
    files: Mapping[str, str | os.PathLike[str]],
    **kwargs,
) -> threading.Thread:
    """Spawn a daemon thread that uploads the checkpoint; return immediately."""
    t = threading.Thread(
        target=upload_checkpoint,
        args=(run_name, step, files),
        kwargs=kwargs,
        name=f"endlex-ckpt-{run_name}-{step}",
        daemon=True,
    )
    t.start()
    return t
