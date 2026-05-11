"""FastAPI app for Endlex.

Build the app via :func:`create_app`. The ``endlex-server`` console script
launches uvicorn with ``--factory`` against this function. Tests construct
their own app with a tmp data root.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import asyncio
import json as _json

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates

from endlex.server.auth import require_read_auth, require_write_auth
from endlex.server.storage import (
    InvalidName,
    RunLocked,
    RunNotFound,
    Storage,
)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app(data_root: str | os.PathLike[str] | None = None) -> FastAPI:
    root = (
        data_root if data_root is not None else os.environ.get("ENDLEX_DATA", "./data")
    )
    storage = Storage(
        root,
        default_keep_last=int(os.environ.get("ENDLEX_CKPT_KEEP_LAST", "0")),
        default_max_age_days=float(os.environ.get("ENDLEX_CKPT_MAX_AGE_DAYS", "0")),
    )
    templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))
    app = FastAPI(title="Endlex", version="0.1.0")
    app.state.storage = storage
    app.state.templates = templates
    _register_routes(app)
    return app


def _storage_of(request: Request) -> Storage:
    return request.app.state.storage


def _templates_of(request: Request) -> Jinja2Templates:
    return request.app.state.templates


StorageDep = Annotated[Storage, Depends(_storage_of)]


def _register_routes(app: FastAPI) -> None:  # noqa: C901 — long but flat
    @app.get("/health")
    async def health(storage: StorageDep):
        """Unauthenticated liveness/readiness probe.

        Cheap on purpose — just counts run directories. Don't add expensive
        work here; monitoring agents poll this every few seconds.
        """
        try:
            n_runs = sum(
                1 for p in storage.runs_dir.iterdir() if p.is_dir()
            )
        except OSError:
            n_runs = -1
        return {
            "status": "ok",
            "version": app.version,
            "runs": n_runs,
        }

    @app.exception_handler(InvalidName)
    async def _bad_name(_: Request, exc: InvalidName):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(RunNotFound)
    async def _not_found(_: Request, exc: RunNotFound):
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(RunLocked)
    async def _locked(_: Request, exc: RunLocked):
        return JSONResponse(
            {
                "error": (
                    f"run {exc} already exists with a different config; "
                    "pass ?force=1 to overwrite"
                )
            },
            status_code=409,
        )

    # ---------- writes ----------

    @app.post(
        "/api/runs/{name}/init",
        dependencies=[Depends(require_write_auth)],
    )
    async def init_run(
        name: str,
        config: dict[str, Any],
        storage: StorageDep,
        force: bool = Query(default=False),
    ):
        storage.init_run(name, config, force=force)
        return {"name": name, "config": config}

    @app.post(
        "/api/runs/{name}/metrics",
        dependencies=[Depends(require_write_auth)],
    )
    async def append_metrics(
        name: str,
        events: list[dict[str, Any]],
        storage: StorageDep,
    ):
        appended = storage.append_metrics(name, events)
        return {"appended": appended}

    @app.post(
        "/api/runs/{name}/ckpt/{step}",
        dependencies=[Depends(require_write_auth)],
    )
    async def upload_ckpt(
        name: str,
        step: str,
        storage: StorageDep,
        files: list[UploadFile],
    ):
        if not files:
            raise HTTPException(400, "no files in upload")
        written: dict[str, int] = {}
        for f in files:
            if not f.filename:
                raise HTTPException(400, "file missing filename")
            written[f.filename] = storage.write_checkpoint_file(
                name, step, f.filename, f.file
            )
        # Apply retention immediately after each successful upload so the
        # disk doesn't blow past the cap between sweeps.
        keep_last, max_age = storage.resolved_retention(name)
        pruned = storage.prune_checkpoints(
            name, keep_last=keep_last, max_age_seconds=max_age
        )
        return {"name": name, "step": step, "written": written, "pruned": pruned}

    @app.post(
        "/api/admin/prune",
        dependencies=[Depends(require_write_auth)],
    )
    async def prune_all(storage: StorageDep):
        """Apply retention to every run. For cron use."""
        result: dict[str, list[str]] = {}
        for s in storage.list_runs():
            keep_last, max_age = storage.resolved_retention(s.name)
            pruned = storage.prune_checkpoints(
                s.name, keep_last=keep_last, max_age_seconds=max_age
            )
            if pruned:
                result[s.name] = pruned
        return {"pruned": result}

    @app.delete(
        "/api/runs/{name}",
        dependencies=[Depends(require_write_auth)],
        status_code=204,
    )
    async def delete_run(name: str, storage: StorageDep):
        storage.delete_run(name)
        return Response(status_code=204)

    @app.patch(
        "/api/runs/{name}/state",
        dependencies=[Depends(require_write_auth)],
    )
    async def patch_state(
        name: str,
        patch: dict[str, Any],
        storage: StorageDep,
    ):
        return storage.update_state(name, patch)

    # ---------- reads (JSON) ----------

    @app.get("/api/runs", dependencies=[Depends(require_read_auth)])
    async def list_runs(
        storage: StorageDep,
        include_archived: bool = Query(default=False),
    ):
        rows = storage.list_runs()
        if not include_archived:
            rows = [r for r in rows if not r.archived]
        return [asdict(r) for r in rows]

    @app.get("/api/runs/{name}", dependencies=[Depends(require_read_auth)])
    async def get_run(name: str, storage: StorageDep):
        if not storage.run_exists(name):
            raise RunNotFound(name)
        return {
            "name": name,
            "config": storage.get_config(name),
            "summary": asdict(storage.summarize_run(name)),
            "checkpoints": storage.list_checkpoints(name),
        }

    @app.get(
        "/api/runs/{name}/metrics",
        dependencies=[Depends(require_read_auth)],
    )
    async def get_metrics(
        name: str,
        storage: StorageDep,
        since: int = Query(default=0, ge=0),
    ):
        events, new_offset = storage.read_metrics(name, since_offset=since)
        return {"events": events, "offset": new_offset}

    @app.get(
        "/api/runs/{name}/metrics/stream",
        dependencies=[Depends(require_read_auth)],
    )
    async def stream_metrics(
        request: Request,
        name: str,
        storage: StorageDep,
        since: int = Query(default=0, ge=0),
        poll_interval: float = Query(default=0.5, ge=0.05, le=5.0),
        max_lifetime: float = Query(
            default=float(os.environ.get("ENDLEX_SSE_MAX_LIFETIME_SEC", "3600")),
            ge=1.0,
            le=86400.0,
        ),
    ):
        # Validate up front so a typo doesn't return a 200 SSE that errors mid-stream.
        if not storage.run_exists(name):
            raise RunNotFound(name)

        async def gen():
            cursor = since
            yield b": stream open\n\n"
            start = asyncio.get_event_loop().time()
            last_keepalive = start
            while True:
                # Cap the lifetime so a wedged client can't pin a thread forever.
                if asyncio.get_event_loop().time() - start > max_lifetime:
                    return
                # Race the disconnect signal with the poll interval. On
                # http.disconnect, receive() returns immediately; otherwise
                # wait_for times out and we proceed to read the file. This
                # replaces the brittle is_disconnected() polling pattern that
                # leaks the gen when the client closes silently.
                try:
                    msg = await asyncio.wait_for(
                        request.receive(), timeout=poll_interval
                    )
                    if msg.get("type") == "http.disconnect":
                        return
                except asyncio.TimeoutError:
                    pass  # no signal in this window — normal path
                # Run sync file IO off the event loop to avoid blocking peers.
                events, new_cursor = await asyncio.to_thread(
                    storage.read_metrics, name, since_offset=cursor
                )
                for e in events:
                    yield (
                        b"event: metric\ndata: "
                        + _json.dumps(e, separators=(",", ":")).encode()
                        + b"\n\n"
                    )
                if new_cursor != cursor:
                    cursor = new_cursor
                    yield f"event: cursor\ndata: {cursor}\n\n".encode()
                now = asyncio.get_event_loop().time()
                if now - last_keepalive > 30:
                    yield b": keep-alive\n\n"
                    last_keepalive = now

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
            },
        )

    @app.get(
        "/api/runs/{name}/ckpt/{step}/{filename}",
        dependencies=[Depends(require_read_auth)],
    )
    async def download_ckpt(
        name: str,
        step: str,
        filename: str,
        storage: StorageDep,
    ):
        path = storage.checkpoint_file_path(name, step, filename)
        return FileResponse(path, filename=filename)

    @app.get(
        "/api/runs/{name}/export.html",
        response_class=HTMLResponse,
        dependencies=[Depends(require_read_auth)],
    )
    async def export_run(
        request: Request,
        name: str,
        storage: StorageDep,
        download: bool = Query(default=False),
    ):
        if not storage.run_exists(name):
            raise RunNotFound(name)
        events, _ = storage.read_metrics(name)
        import datetime as _dt

        response = _templates_of(request).TemplateResponse(
            request,
            "export.html",
            {
                "name": name,
                "config": storage.get_config(name),
                "summary": asdict(storage.summarize_run(name)),
                "checkpoints": storage.list_checkpoints(name),
                "events": events,
                "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
        )
        if download:
            response.headers["Content-Disposition"] = (
                f'attachment; filename="endlex-{name}.html"'
            )
        return response

    # ---------- reads (HTML) ----------

    @app.get(
        "/",
        response_class=HTMLResponse,
        dependencies=[Depends(require_read_auth)],
    )
    async def dashboard(request: Request, storage: StorageDep):
        # Always return all runs; the page JS hides archived rows by
        # default with a toggle so users can re-show them without a
        # server round-trip.
        runs = [asdict(s) for s in storage.list_runs()]
        return _templates_of(request).TemplateResponse(
            request, "dashboard.html", {"runs": runs}
        )

    @app.get(
        "/run/{name}",
        response_class=HTMLResponse,
        dependencies=[Depends(require_read_auth)],
    )
    async def run_page(request: Request, name: str, storage: StorageDep):
        if not storage.run_exists(name):
            raise RunNotFound(name)
        return _templates_of(request).TemplateResponse(
            request,
            "run.html",
            {
                "name": name,
                "config": storage.get_config(name),
                "summary": asdict(storage.summarize_run(name)),
                "checkpoints": storage.list_checkpoints(name),
            },
        )

    @app.get(
        "/compare",
        response_class=HTMLResponse,
        dependencies=[Depends(require_read_auth)],
    )
    async def compare(
        request: Request,
        storage: StorageDep,
        runs: str = Query(default=""),
    ):
        wanted = [n.strip() for n in runs.split(",") if n.strip()]
        valid: list[str] = []
        for n in wanted:
            try:
                if storage.run_exists(n):
                    valid.append(n)
            except InvalidName:
                continue
        return _templates_of(request).TemplateResponse(
            request,
            "compare.html",
            {"runs": valid, "missing": [n for n in wanted if n not in valid]},
        )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "endlex.server.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.environ.get("ENDLEX_PORT", "8000")),
        log_level="info",
    )
