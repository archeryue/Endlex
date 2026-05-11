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

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
    storage = Storage(root)
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
        return {"name": name, "step": step, "written": written}

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


def main() -> None:
    import uvicorn

    uvicorn.run(
        "endlex.server.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=int(os.environ.get("ENDLEX_PORT", "8000")),
        log_level="info",
    )
