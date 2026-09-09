"""Checkpoint sync helpers for the trainer side (and pullers).

Upload: call :func:`upload_checkpoint_async` from ``save_checkpoint``; the
trainer returns immediately while a daemon thread streams the POST. Local
save remains the source of truth — failures are logged to stderr and
discarded.

Integrity: every file's sha256 is computed client-side and verified by the
server before the file lands (the server stages to ``.part`` and renames
atomically) — a torn transfer can never be mistaken for a real checkpoint.

Large files: proxies commonly cap request bodies (Cloudflare: ~100 MB), so
files above ``chunk_threshold`` are shipped as sequential chunks to
``PUT /api/runs/<run>/ckpt/<step>/files/<name>`` and reassembled server-side.
Interrupted transfers resume from the server-reported offset instead of
restarting a multi-GB upload. Servers predating the chunk API get the legacy
single multipart POST as a fallback.

Download: :func:`download_checkpoint` pulls a step's files to any box and
verifies each against the server-recorded sha256 — the reverse half of
"weights sync" (home box → dev machine).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import httpx

# Cap concurrent async uploads so a burst of checkpoints on a slow link
# doesn't open unbounded parallel TCP streams and file handles.
_UPLOAD_SEM = threading.Semaphore(4)

# Files above the threshold go through the chunked endpoint. Both sized to
# stay comfortably under Cloudflare's ~100 MB request-body cap.
_DEFAULT_CHUNK_THRESHOLD = 64 * 1024 * 1024
_DEFAULT_CHUNK_SIZE = 48 * 1024 * 1024

_RETRY_DELAYS: tuple[float, ...] = (1.0, 3.0, 10.0)

# Disable keep-alive on the transfer client. A checkpoint ships as many large
# sequential chunk requests; pooling the TLS connection between them means a
# connection that the tunnel (Cloudflare) has rotated/half-closed gets reused,
# and the next chunk desyncs the TLS keystream -> SSLV3_ALERT_BAD_RECORD_MAC.
# Retries then reuse the same poisoned pooled connection and give up. Forcing a
# fresh connection per request (one extra handshake per ~48 MB chunk, negligible)
# makes chunk uploads and their retries robust over tunnels. Metric streaming uses
# a separate client and is unaffected.
_NO_KEEPALIVE = httpx.Limits(max_keepalive_connections=0)


def _log(msg: str) -> None:
    print(f"[endlex] {msg}", file=sys.stderr)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _ChunkApiUnsupported(Exception):
    """Server predates the chunked upload API (404/405 on the PUT route)."""


def _request_with_retry(fn, *, retry_delays=_RETRY_DELAYS) -> httpx.Response | None:
    """Retry on 5xx + transport errors; return final response or None."""
    r: httpx.Response | None = None
    for i in range(len(retry_delays) + 1):
        try:
            r = fn()
            if r.status_code < 500:
                return r
        except httpx.HTTPError as e:
            _log(f"transfer error: {type(e).__name__}: {e}")
            r = None
        if i < len(retry_delays):
            time.sleep(retry_delays[i])
    return r


def _upload_multipart(
    client: httpx.Client,
    run_name: str,
    step: int,
    files: Mapping[str, Path],
    checksums: Mapping[str, str],
    retry_delays: tuple[float, ...],
) -> bool:
    handles: list = []
    try:
        upload_files = []
        for filename, p in files.items():
            fh = open(p, "rb")
            handles.append(fh)
            upload_files.append(
                ("files", (filename, fh, "application/octet-stream"))
            )

        def _do() -> httpx.Response:
            for fh in handles:
                fh.seek(0)  # a retried request must resend from the top
            return client.post(
                f"/api/runs/{run_name}/ckpt/{int(step)}",
                files=upload_files,
                headers={
                    "X-Endlex-Checksums": json.dumps(
                        {k: checksums[k] for k in files}
                    )
                },
            )

        r = _request_with_retry(_do, retry_delays=retry_delays)
        if r is None or r.status_code != 200:
            detail = f"{r.status_code} {r.text}" if r is not None else "gave up"
            _log(f"checkpoint upload failed: {detail}")
            return False
        return True
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass


def _upload_chunked(
    client: httpx.Client,
    run_name: str,
    step: int,
    filename: str,
    path: Path,
    sha256: str,
    chunk_size: int,
    retry_delays: tuple[float, ...],
) -> bool:
    """Ship one file in sequential chunks; resume from the server's offset."""
    total = path.stat().st_size
    base = f"/api/runs/{run_name}/ckpt/{int(step)}/files/{filename}"

    # Where did a previous (interrupted) attempt get to?
    r = _request_with_retry(lambda: client.get(f"{base}/status"),
                            retry_delays=retry_delays)
    if r is not None and r.status_code in (404, 405):
        raise _ChunkApiUnsupported()
    offset = 0
    if r is not None and r.status_code == 200:
        st = r.json()
        if st.get("complete") and st.get("sha256") == sha256:
            return True  # identical file already fully uploaded
        if not st.get("complete"):
            offset = int(st.get("received", 0))

    restarted = False
    with path.open("rb") as f:
        while offset < total:
            f.seek(offset)
            data = f.read(chunk_size)
            sent_offset = offset

            def _do() -> httpx.Response:
                return client.put(
                    base,
                    params={
                        "offset": sent_offset,
                        "total": total,
                        "sha256": sha256,
                    },
                    content=data,
                )

            r = _request_with_retry(_do, retry_delays=retry_delays)
            if r is None:
                _log(f"chunk upload gave up: {filename} @ {offset}")
                return False
            if r.status_code in (404, 405):
                raise _ChunkApiUnsupported()
            if r.status_code == 409:
                # Offset drifted (e.g. a retried chunk actually landed).
                # Trust the server's resume point and continue from there.
                try:
                    offset = int(r.json().get("received", 0))
                except (json.JSONDecodeError, ValueError, TypeError):
                    _log(f"chunk upload failed: {filename}: {r.text}")
                    return False
                continue
            if r.status_code == 422:
                # Assembled file failed the hash — staging was discarded.
                # One clean restart from zero, then give up.
                if restarted:
                    _log(f"checksum mismatch persists: {filename}")
                    return False
                restarted = True
                offset = 0
                continue
            if r.status_code != 200:
                _log(f"chunk upload failed: {r.status_code} {r.text}")
                return False
            body = r.json()
            offset = int(body.get("received", offset + len(data)))
            if body.get("complete"):
                return True
    return offset >= total


def upload_checkpoint(
    run_name: str,
    step: int,
    files: Mapping[str, str | os.PathLike[str]],
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 600.0,
    chunk_threshold: int | None = None,
    chunk_size: int | None = None,
    retry_delays: tuple[float, ...] = _RETRY_DELAYS,
    _client: httpx.Client | None = None,
) -> bool:
    """Ship a step's checkpoint files to the Endlex server.

    ``files`` maps the filename the server stores under (e.g. ``"model.pt"``)
    to the local path to read. Returns ``True`` when every file landed and
    verified. Any failure (missing file, network error, non-200, checksum
    mismatch after retries) is logged to stderr and yields ``False``.
    """
    url = url or os.environ.get("ENDLEX_URL")
    token = token or os.environ.get("ENDLEX_TOKEN")
    if _client is None and not url:
        _log("ENDLEX_URL unset; skipping checkpoint upload")
        return False

    threshold = (
        chunk_threshold
        if chunk_threshold is not None
        else int(os.environ.get("ENDLEX_CHUNK_THRESHOLD", _DEFAULT_CHUNK_THRESHOLD))
    )
    csize = (
        chunk_size
        if chunk_size is not None
        else int(os.environ.get("ENDLEX_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE))
    )

    paths: dict[str, Path] = {}
    for filename, path in files.items():
        p = Path(path)
        if not p.exists():
            _log(f"missing checkpoint file: {p}")
            return False
        paths[filename] = p

    if _client is not None:
        client = _client
        owns_client = False
    else:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        client = httpx.Client(base_url=url, headers=headers, timeout=timeout, limits=_NO_KEEPALIVE)
        owns_client = True

    try:
        checksums = {name: _sha256_of_file(p) for name, p in paths.items()}
        small = {n: p for n, p in paths.items() if p.stat().st_size <= threshold}
        large = {n: p for n, p in paths.items() if n not in small}

        ok = True
        for name, p in large.items():
            try:
                ok &= _upload_chunked(
                    client, run_name, step, name, p,
                    checksums[name], csize, retry_delays,
                )
            except _ChunkApiUnsupported:
                # Old server — fall back to one multipart POST and hope the
                # path between here and there has no body-size cap.
                _log(
                    "server lacks chunked upload API; falling back to "
                    f"single-request upload for {name} "
                    f"({p.stat().st_size / 1e6:.0f} MB)"
                )
                small[name] = p
        if small:
            ok &= _upload_multipart(
                client, run_name, step, small, checksums, retry_delays
            )
        return ok
    except Exception as e:  # noqa: BLE001 — best-effort, never crash the trainer
        _log(f"checkpoint upload error: {e}")
        return False
    finally:
        if owns_client:
            client.close()


def upload_checkpoint_async(
    run_name: str,
    step: int,
    files: Mapping[str, str | os.PathLike[str]],
    **kwargs,
) -> threading.Thread:
    """Spawn a daemon thread that uploads the checkpoint; return immediately."""
    def _upload_with_sem(*args, **kw):
        with _UPLOAD_SEM:
            upload_checkpoint(*args, **kw)

    t = threading.Thread(
        target=_upload_with_sem,
        args=(run_name, step, files),
        kwargs=kwargs,
        name=f"endlex-ckpt-{run_name}-{step}",
        daemon=True,
    )
    t.start()
    return t


def download_checkpoint(
    run_name: str,
    step: int | None = None,
    dest: str | os.PathLike[str] = ".",
    files: list[str] | None = None,
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: float = 600.0,
    _client: httpx.Client | None = None,
) -> list[Path]:
    """Pull a checkpoint's files from the server, verifying sha256 where known.

    ``step=None`` means the latest step. ``files=None`` means every file in
    the step. Files stream to ``dest/<filename>`` via a ``.part`` staging file
    so an interrupted download never leaves a plausible-looking partial file.
    Raises on failure — this is a deliberate, user-facing fetch, unlike the
    best-effort trainer-side upload.
    """
    url = url or os.environ.get("ENDLEX_URL")
    token = token or os.environ.get("ENDLEX_TOKEN")
    if _client is None and not url:
        raise RuntimeError("ENDLEX_URL unset and no url= given")

    if _client is not None:
        client = _client
        owns_client = False
    else:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        client = httpx.Client(base_url=url, headers=headers, timeout=timeout, limits=_NO_KEEPALIVE)
        owns_client = True

    try:
        r = client.get(f"/api/runs/{run_name}")
        if r.status_code != 200:
            raise RuntimeError(f"run lookup failed: {r.status_code} {r.text}")
        checkpoints: list[dict[str, Any]] = r.json().get("checkpoints", [])
        if not checkpoints:
            raise RuntimeError(f"run {run_name!r} has no checkpoints")

        if step is None:
            entry = checkpoints[-1]  # server returns numeric step order
        else:
            wanted_dir = f"step_{int(step):06d}"
            matches = [c for c in checkpoints if c["step"] == wanted_dir]
            if not matches:
                have = ", ".join(c["step"] for c in checkpoints)
                raise RuntimeError(f"no checkpoint {wanted_dir} (have: {have})")
            entry = matches[0]

        step_int = int(entry["step"].split("_", 1)[1])
        wanted = files if files is not None else list(entry["files"])
        meta = entry.get("files_meta") or {}

        dest_dir = Path(dest)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        for filename in wanted:
            if filename not in entry["files"]:
                raise RuntimeError(f"no file {filename!r} in {entry['step']}")
            target = dest_dir / filename
            part = dest_dir / (filename + ".part")
            h = hashlib.sha256()
            with client.stream(
                "GET", f"/api/runs/{run_name}/ckpt/{step_int}/{filename}"
            ) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"download failed: {filename}: {resp.status_code}"
                    )
                expected = (
                    resp.headers.get("x-endlex-sha256")
                    or (meta.get(filename) or {}).get("sha256")
                )
                with part.open("wb") as f:
                    for chunk in resp.iter_bytes(1 << 20):
                        f.write(chunk)
                        h.update(chunk)
            if expected and h.hexdigest() != expected:
                part.unlink(missing_ok=True)
                raise RuntimeError(
                    f"sha256 mismatch for {filename}: "
                    f"expected {expected}, got {h.hexdigest()}"
                )
            os.replace(part, target)
            out.append(target)
        return out
    finally:
        if owns_client:
            client.close()
