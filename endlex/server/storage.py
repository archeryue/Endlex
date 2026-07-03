"""Storage layer for Endlex.

Layout under ``data_root``::

    runs/<name>/config.json     # training config + env snapshot
    runs/<name>/metrics.jsonl   # append-only, one JSON dict per .log()
    runs/<name>/.lock           # single-writer sentinel checked at init
    checkpoints/<name>/step_<NNNNNN>/<file>
    checkpoints/<name>/step_<NNNNNN>/.manifest.json  # {file: {size, sha256, uploaded_at}}

Checkpoint files are streamed to ``<file>.part`` and atomically renamed into
place only after the full payload (and its sha256, when provided) checks out —
a partially-uploaded checkpoint can never be mistaken for a complete one.

No DB. JSONL is tail-friendly; rm -rf cleans up.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_STEP_RE = re.compile(r"^[0-9]{1,12}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_MANIFEST_NAME = ".manifest.json"
_PART_SUFFIX = ".part"


class StorageError(Exception):
    pass


class InvalidName(StorageError):
    pass


class RunNotFound(StorageError):
    pass


class RunLocked(StorageError):
    """init called on an already-active run (concurrent writer) or a config conflict."""


class ChecksumMismatch(StorageError):
    """Uploaded bytes don't hash to the client-declared sha256."""


class ChunkOffsetMismatch(StorageError):
    """Chunk arrived at the wrong offset. ``received`` is the resume point."""

    def __init__(self, received: int):
        super().__init__(f"expected chunk at offset {received}")
        self.received = received


@dataclass(frozen=True)
class RunSummary:
    name: str
    last_updated: float | None
    num_events: int
    latest: dict[str, Any] | None
    tags: list[str]
    archived: bool
    project: str


_DEFAULT_STATE: dict[str, Any] = {"tags": [], "archived": False, "retention": {}}


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise InvalidName(f"invalid run name: {name!r}")


def _validate_step(step: str) -> None:
    if not _STEP_RE.match(str(step)):
        raise InvalidName(f"invalid step: {step!r}")


def _validate_filename(filename: str) -> None:
    if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
        raise InvalidName(f"invalid filename: {filename!r}")
    # Reserved names: the manifest sidecar, dotfiles, and .part staging files
    # would collide with internal bookkeeping.
    if filename.startswith(".") or filename.endswith(_PART_SUFFIX):
        raise InvalidName(f"invalid filename: {filename!r}")


def _validate_sha256(sha256: str) -> str:
    if not _SHA256_RE.match(sha256):
        raise InvalidName(f"invalid sha256: {sha256!r}")
    return sha256.lower()


def _step_dirname(step: str | int) -> str:
    return f"step_{int(step):06d}"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via tmp + rename so a crash can't leave a torn file behind."""
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Storage:
    def __init__(
        self,
        data_root: str | os.PathLike[str],
        *,
        default_keep_last: int = 0,
        default_max_age_days: float = 0.0,
    ):
        self.data_root = Path(data_root)
        self.runs_dir = self.data_root / "runs"
        self.ckpt_dir = self.data_root / "checkpoints"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.default_keep_last = max(0, int(default_keep_last))
        self.default_max_age_days = max(0.0, float(default_max_age_days))
        # Tracks open fds holding an exclusive flock on each run's .lock file.
        # flock within one OS process converts rather than blocks, so we pair
        # it with this in-memory dict to catch concurrent in-process callers.
        self._lock_fds: dict[str, int] = {}
        self._lock_fds_mu = threading.Lock()
        # Serializes manifest read-modify-write and chunk appends. Uploads are
        # disk-bound and single-user, so one coarse lock is plenty.
        self._ckpt_mu = threading.Lock()

    # ----- runs -----

    def init_run(
        self, name: str, config: dict[str, Any], *, force: bool = False
    ) -> None:
        _validate_name(name)
        run_dir = self.runs_dir / name
        cfg_path = run_dir / "config.json"
        lock_path = run_dir / ".lock"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Compare against the config already on disk. A re-init with an
        # *identical* config is treated as a resume (crashed trainer restarting
        # with the same run) and is always allowed — the crashed process never
        # got to call /finish, so blocking on the stale lock would strand the
        # whole resumed session. A *different* config still requires force,
        # whether or not the lock is held.
        same_config = False
        config_conflict = False
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text())
            except (OSError, json.JSONDecodeError):
                existing = None  # corrupt/unreadable — don't block re-init on it
            if existing is not None:
                same_config = existing == config
                config_conflict = not same_config

        with self._lock_fds_mu:
            in_process_locked = name in self._lock_fds
            if in_process_locked and not force and not same_config:
                raise RunLocked(name)

            # Linux flock treats each open file description as an independent
            # locker, even within the same process. Release the old fd *before*
            # opening the new one, otherwise flock(LOCK_EX|LOCK_NB) on the new
            # fd fails with EWOULDBLOCK even on a force re-init.
            if in_process_locked:
                old_fd = self._lock_fds.pop(name)
                try:
                    os.close(old_fd)
                except OSError:
                    pass

            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (BlockingIOError, OSError):
                os.close(fd)
                raise RunLocked(name)
            self._lock_fds[name] = fd

        # Config conflict: reject if the run already has a different config on
        # disk. This catches the "server restarted, trainer re-inits with wrong
        # config" case where _lock_fds was empty but the run files exist.
        if config_conflict and not force:
            with self._lock_fds_mu:
                if self._lock_fds.get(name) == fd:
                    del self._lock_fds[name]
            try:
                os.close(fd)
            except OSError:
                pass
            raise RunLocked(name)

        _atomic_write_text(cfg_path, json.dumps(config, indent=2, sort_keys=True))
        lock_path.write_text(json.dumps({"pid": os.getpid(), "created_at": time.time()}))
        (run_dir / "metrics.jsonl").touch(exist_ok=True)

    def delete_run(self, name: str) -> None:
        _validate_name(name)
        run_dir = self.runs_dir / name
        if not run_dir.exists():
            raise RunNotFound(name)
        shutil.rmtree(run_dir)
        ckpt = self.ckpt_dir / name
        if ckpt.exists():
            shutil.rmtree(ckpt)
        self._release_lock(name)

    def finish_run(self, name: str) -> None:
        """Release the active-writer lock so the run can be re-initialised."""
        _validate_name(name)
        self._release_lock(name)

    def _release_lock(self, name: str) -> None:
        with self._lock_fds_mu:
            fd = self._lock_fds.pop(name, None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    def run_exists(self, name: str) -> bool:
        _validate_name(name)
        return (self.runs_dir / name).is_dir()

    def get_config(self, name: str) -> dict[str, Any]:
        _validate_name(name)
        cfg_path = self.runs_dir / name / "config.json"
        if not cfg_path.exists():
            raise RunNotFound(name)
        return json.loads(cfg_path.read_text())

    # ----- metrics -----

    def append_metrics(self, name: str, events: Iterable[dict[str, Any]]) -> int:
        _validate_name(name)
        run_dir = self.runs_dir / name
        if not run_dir.is_dir():
            raise RunNotFound(name)
        path = run_dir / "metrics.jsonl"
        events_list = list(events)
        lines = [
            json.dumps(e, separators=(",", ":"), sort_keys=True) + "\n"
            for e in events_list
        ]
        if not lines:
            return 0
        # Single O_APPEND write for the whole batch — POSIX guarantees the
        # write is atomic relative to other O_APPEND writers for sizes below
        # PIPE_BUF. Even for larger batches, a single writer (enforced at
        # init) means no interleaving.
        data = "".join(lines).encode("utf-8")
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            # os.write may report a short write for large batches — loop.
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
        finally:
            os.close(fd)
        # Bump the summary cache so the dashboard's _summarize stays O(1)
        # per run instead of re-reading the whole JSONL every page load.
        self._bump_summary_cache(run_dir, events_list, path)
        return len(lines)

    def read_metrics(
        self, name: str, *, since_offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        """Return (events, new_offset). Pass new_offset back as since_offset to resume."""
        _validate_name(name)
        run_dir = self.runs_dir / name
        if not run_dir.is_dir():
            raise RunNotFound(name)
        path = run_dir / "metrics.jsonl"
        events: list[dict[str, Any]] = []
        if not path.exists():
            return events, since_offset
        with path.open("rb") as f:
            f.seek(since_offset)
            data = f.read()
            new_offset = since_offset + len(data)
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
        return events, new_offset

    # ----- listing / summary -----

    def list_runs(self) -> list[RunSummary]:
        if not self.runs_dir.exists():
            return []
        return [
            self._summarize(p)
            for p in sorted(self.runs_dir.iterdir())
            if p.is_dir()
        ]

    def summarize_run(self, name: str) -> RunSummary:
        _validate_name(name)
        run_dir = self.runs_dir / name
        if not run_dir.is_dir():
            raise RunNotFound(name)
        return self._summarize(run_dir)

    def _summarize(self, run_dir: Path) -> RunSummary:
        name = run_dir.name
        metrics_path = run_dir / "metrics.jsonl"
        cfg_path = run_dir / "config.json"
        state = self._read_state(run_dir)

        # Fast path: a .summary.json sidecar whose recorded metrics_size matches
        # the file on disk is authoritative. Stays in sync with append_metrics;
        # any external mutation flips the size and triggers a full rescan.
        if metrics_path.exists():
            try:
                actual_size = metrics_path.stat().st_size
            except OSError:
                actual_size = -1
            cache = self._read_summary_cache(run_dir)
            if cache and cache.get("metrics_size") == actual_size:
                return RunSummary(
                    name=name,
                    last_updated=cache.get("last_updated"),
                    num_events=int(cache.get("num_events", 0)),
                    latest=cache.get("latest"),
                    tags=list(state["tags"]),
                    archived=bool(state["archived"]),
                    project=str(state.get("project") or ""),
                )

        # Slow path: scan the JSONL and repopulate the cache for next time.
        last_updated: float | None = None
        num_events = 0
        latest: dict[str, Any] | None = None
        if metrics_path.exists() and metrics_path.stat().st_size > 0:
            stat = metrics_path.stat()
            last_updated = stat.st_mtime
            with metrics_path.open("rb") as f:
                last_line: bytes | None = None
                for raw in f:
                    raw = raw.strip()
                    if not raw:
                        continue
                    num_events += 1
                    last_line = raw
                if last_line is not None:
                    try:
                        latest = json.loads(last_line)
                    except json.JSONDecodeError:
                        latest = None
            self._write_summary_cache(
                run_dir,
                num_events=num_events,
                latest=latest,
                last_updated=last_updated,
                metrics_size=stat.st_size,
            )
        elif cfg_path.exists():
            last_updated = cfg_path.stat().st_mtime
        return RunSummary(
            name=name,
            last_updated=last_updated,
            num_events=num_events,
            latest=latest,
            tags=list(state["tags"]),
            archived=bool(state["archived"]),
            project=str(state.get("project") or ""),
        )

    # ----- summary cache -----

    @staticmethod
    def _read_summary_cache(run_dir: Path) -> dict[str, Any] | None:
        cache_path = run_dir / ".summary.json"
        if not cache_path.exists():
            return None
        try:
            return json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_summary_cache(
        run_dir: Path,
        *,
        num_events: int,
        latest: Any,
        last_updated: float | None,
        metrics_size: int,
    ) -> None:
        cache_path = run_dir / ".summary.json"
        try:
            _atomic_write_text(
                cache_path,
                json.dumps(
                    {
                        "num_events": num_events,
                        "latest": latest,
                        "last_updated": last_updated,
                        "metrics_size": metrics_size,
                    }
                ),
            )
        except OSError:
            pass  # cache is best-effort

    def _bump_summary_cache(
        self,
        run_dir: Path,
        events_just_appended: list[dict[str, Any]],
        metrics_path: Path,
    ) -> None:
        current = self._read_summary_cache(run_dir) or {}
        new_count = int(current.get("num_events", 0)) + len(events_just_appended)
        new_latest = (
            events_just_appended[-1]
            if events_just_appended
            else current.get("latest")
        )
        try:
            stat = metrics_path.stat()
        except OSError:
            return
        self._write_summary_cache(
            run_dir,
            num_events=new_count,
            latest=new_latest,
            last_updated=stat.st_mtime,
            metrics_size=stat.st_size,
        )

    @staticmethod
    def _read_state(run_dir: Path) -> dict[str, Any]:
        state_path = run_dir / "state.json"
        empty: dict[str, Any] = {
            "tags": [],
            "archived": False,
            "retention": {},
            "notes": "",
            "project": "",
            "panels": [],
        }
        if not state_path.exists():
            return dict(empty)
        try:
            raw = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            return dict(empty)
        return {
            "tags": list(raw.get("tags") or []),
            "archived": bool(raw.get("archived", False)),
            "retention": dict(raw.get("retention") or {}),
            "notes": str(raw.get("notes") or ""),
            "project": str(raw.get("project") or ""),
            "panels": list(raw.get("panels") or []),
        }

    # ----- state (tags / archived) -----

    def get_state(self, name: str) -> dict[str, Any]:
        _validate_name(name)
        run_dir = self.runs_dir / name
        if not run_dir.is_dir():
            raise RunNotFound(name)
        return self._read_state(run_dir)

    def update_state(
        self, name: str, patch: dict[str, Any]
    ) -> dict[str, Any]:
        _validate_name(name)
        run_dir = self.runs_dir / name
        if not run_dir.is_dir():
            raise RunNotFound(name)
        state = self._read_state(run_dir)
        if "tags" in patch:
            tags = patch["tags"]
            if not isinstance(tags, list) or not all(
                isinstance(t, str) for t in tags
            ):
                raise InvalidName("tags must be a list of strings")
            # Strip + dedupe while preserving order.
            seen: dict[str, None] = {}
            for t in tags:
                t = t.strip()
                if t and t not in seen:
                    seen[t] = None
            state["tags"] = list(seen)
        if "archived" in patch:
            state["archived"] = bool(patch["archived"])
        if "retention" in patch:
            r = patch["retention"]
            if not isinstance(r, dict):
                raise InvalidName("retention must be an object")
            normalized: dict[str, Any] = {}
            if "keep_last" in r:
                normalized["keep_last"] = max(0, int(r["keep_last"]))
            if "max_age_days" in r:
                normalized["max_age_days"] = max(0.0, float(r["max_age_days"]))
            state["retention"] = normalized
        if "notes" in patch:
            notes = patch["notes"]
            if not isinstance(notes, str):
                raise InvalidName("notes must be a string")
            # Cap to avoid unbounded growth.
            if len(notes) > 100_000:
                raise InvalidName("notes too long (max 100 000 chars)")
            state["notes"] = notes
        if "project" in patch:
            project = patch["project"]
            if not isinstance(project, str):
                raise InvalidName("project must be a string")
            if len(project) > 128:
                raise InvalidName("project name too long (max 128 chars)")
            state["project"] = project.strip()
        if "panels" in patch:
            panels = patch["panels"]
            if not isinstance(panels, list):
                raise InvalidName("panels must be a list")
            normalized_panels: list[dict[str, Any]] = []
            for p in panels:
                if not isinstance(p, dict):
                    raise InvalidName("each panel must be an object")
                for key in ("title", "x", "y"):
                    if key not in p or not isinstance(p[key], str) or not p[key].strip():
                        raise InvalidName(
                            f"panel missing non-empty string field: {key}"
                        )
                norm: dict[str, Any] = {
                    "title": p["title"],
                    "x": p["x"],
                    "y": p["y"],
                }
                # Optional axis-range overrides. None / missing => auto-scale.
                for key in ("xmin", "xmax", "ymin", "ymax"):
                    if key in p and p[key] is not None and p[key] != "":
                        try:
                            norm[key] = float(p[key])
                        except (TypeError, ValueError) as e:
                            raise InvalidName(
                                f"panel.{key} must be numeric, got {p[key]!r}"
                            ) from e
                normalized_panels.append(norm)
            state["panels"] = normalized_panels
        _atomic_write_text(
            run_dir / "state.json", json.dumps(state, indent=2, sort_keys=True)
        )
        return state

    def resolved_retention(self, name: str) -> tuple[int, float]:
        """Return (keep_last, max_age_seconds) for `name`.

        Per-run state.json overrides the server defaults. Either or both
        rules can be 0/unset (meaning "no limit on that dimension").
        """
        _validate_name(name)
        run_dir = self.runs_dir / name
        ret = self._read_state(run_dir).get("retention") or {}
        keep_last = int(ret.get("keep_last", self.default_keep_last))
        max_age_days = float(ret.get("max_age_days", self.default_max_age_days))
        return keep_last, max_age_days * 86400.0

    # ----- checkpoints -----

    def open_checkpoint_for_write(
        self, name: str, step: str | int, filename: str
    ) -> Path:
        _validate_name(name)
        _validate_step(str(step))
        _validate_filename(filename)
        if not (self.runs_dir / name).is_dir():
            raise RunNotFound(name)
        d = self.ckpt_dir / name / _step_dirname(step)
        d.mkdir(parents=True, exist_ok=True)
        return d / filename

    def write_checkpoint_file(
        self,
        name: str,
        step: str | int,
        filename: str,
        source: BinaryIO,
        *,
        expected_sha256: str | None = None,
    ) -> int:
        """Stream `source` to a .part file, verify, atomically rename. Returns bytes written.

        The sha256 is computed while streaming; if ``expected_sha256`` is given
        and doesn't match, the .part file is removed and :class:`ChecksumMismatch`
        raised — a corrupted transfer can never land as a real checkpoint file.
        """
        if expected_sha256 is not None:
            expected_sha256 = _validate_sha256(expected_sha256)
        dest = self.open_checkpoint_for_write(name, step, filename)
        part = dest.parent / (dest.name + _PART_SUFFIX)
        written = 0
        h = hashlib.sha256()
        try:
            with part.open("wb") as out:
                while True:
                    chunk = source.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    out.write(chunk)
                    h.update(chunk)
                    written += len(chunk)
            digest = h.hexdigest()
            if expected_sha256 is not None and digest != expected_sha256:
                raise ChecksumMismatch(
                    f"{filename}: expected {expected_sha256}, got {digest}"
                )
            os.replace(part, dest)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        self._record_in_manifest(dest.parent, filename, written, digest)
        return written

    # -- chunked upload (large files through proxies with request-size caps) --

    def checkpoint_upload_status(
        self, name: str, step: str | int, filename: str
    ) -> dict[str, Any]:
        """Where a (possibly interrupted) upload stands. Drives client resume."""
        _validate_name(name)
        _validate_step(str(step))
        _validate_filename(filename)
        if not (self.runs_dir / name).is_dir():
            raise RunNotFound(name)
        dest = self.ckpt_dir / name / _step_dirname(step) / filename
        part = dest.parent / (dest.name + _PART_SUFFIX)
        if dest.exists():
            meta = self.checkpoint_manifest(name, step).get(filename, {})
            return {
                "received": dest.stat().st_size,
                "complete": True,
                "sha256": meta.get("sha256"),
            }
        if part.exists():
            return {"received": part.stat().st_size, "complete": False, "sha256": None}
        return {"received": 0, "complete": False, "sha256": None}

    def append_checkpoint_chunk(
        self,
        name: str,
        step: str | int,
        filename: str,
        data: bytes,
        *,
        offset: int,
        total: int,
        sha256: str,
    ) -> dict[str, Any]:
        """Append one sequential chunk; finalize (verify + rename) on the last.

        Chunks must arrive in order: ``offset`` has to equal the bytes already
        staged, otherwise :class:`ChunkOffsetMismatch` reports the resume
        point. When the staged size reaches ``total`` the whole .part file is
        hashed against ``sha256``; mismatch discards the staging file.
        """
        sha256 = _validate_sha256(sha256)
        if total <= 0:
            raise InvalidName(f"invalid total size: {total}")
        dest = self.open_checkpoint_for_write(name, step, filename)
        part = dest.parent / (dest.name + _PART_SUFFIX)
        with self._ckpt_mu:
            # Re-upload of a completed file: start over (trainer may be
            # overwriting a checkpoint after a force re-init).
            current = part.stat().st_size if part.exists() else 0
            if offset != current:
                raise ChunkOffsetMismatch(current)
            if current + len(data) > total:
                raise InvalidName(
                    f"chunk overruns declared total ({current}+{len(data)} > {total})"
                )
            with part.open("ab") as out:
                out.write(data)
            new_size = current + len(data)
            if new_size < total:
                return {"received": new_size, "complete": False}
            digest = _sha256_of_file(part)
            if digest != sha256:
                part.unlink(missing_ok=True)
                raise ChecksumMismatch(
                    f"{filename}: expected {sha256}, got {digest}"
                )
            os.replace(part, dest)
            self._record_in_manifest(
                dest.parent, filename, new_size, digest, locked=True
            )
            return {"received": new_size, "complete": True, "sha256": digest}

    # -- manifest --

    def checkpoint_manifest(self, name: str, step: str | int) -> dict[str, Any]:
        _validate_name(name)
        _validate_step(str(step))
        p = self.ckpt_dir / name / _step_dirname(step) / _MANIFEST_NAME
        try:
            raw = json.loads(p.read_text())
            return raw if isinstance(raw, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _record_in_manifest(
        self,
        step_dir: Path,
        filename: str,
        size: int,
        sha256: str,
        *,
        locked: bool = False,
    ) -> None:
        def _do() -> None:
            p = step_dir / _MANIFEST_NAME
            try:
                manifest = json.loads(p.read_text())
                if not isinstance(manifest, dict):
                    manifest = {}
            except (OSError, json.JSONDecodeError):
                manifest = {}
            manifest[filename] = {
                "size": size,
                "sha256": sha256,
                "uploaded_at": time.time(),
            }
            try:
                _atomic_write_text(p, json.dumps(manifest, indent=2, sort_keys=True))
            except OSError:
                pass  # manifest is best-effort metadata

        if locked:  # caller already holds _ckpt_mu
            _do()
        else:
            with self._ckpt_mu:
                _do()

    def checkpoint_file_path(self, name: str, step: str | int, filename: str) -> Path:
        _validate_name(name)
        _validate_step(str(step))
        _validate_filename(filename)
        p = self.ckpt_dir / name / _step_dirname(step) / filename
        if not p.exists():
            raise RunNotFound(f"{name}/{_step_dirname(step)}/{filename}")
        return p

    def list_checkpoints(self, name: str) -> list[dict[str, Any]]:
        """Steps in numeric order. ``files`` keeps the legacy name-list shape;
        ``files_meta`` adds per-file size + sha256 from the manifest."""
        _validate_name(name)
        d = self.ckpt_dir / name
        if not d.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for _, step_dir in self._checkpoint_step_dirs(name):
            files: list[str] = []
            files_meta: dict[str, Any] = {}
            manifest = None
            for p in sorted(step_dir.iterdir()):
                if not p.is_file():
                    continue
                # Hide bookkeeping: manifest, staging .part files, dotfiles.
                if p.name.startswith(".") or p.name.endswith(_PART_SUFFIX):
                    continue
                if manifest is None:
                    try:
                        manifest = json.loads(
                            (step_dir / _MANIFEST_NAME).read_text()
                        )
                        if not isinstance(manifest, dict):
                            manifest = {}
                    except (OSError, json.JSONDecodeError):
                        manifest = {}
                files.append(p.name)
                entry = manifest.get(p.name) or {}
                files_meta[p.name] = {
                    "size": p.stat().st_size,
                    "sha256": entry.get("sha256"),
                }
            out.append(
                {"step": step_dir.name, "files": files, "files_meta": files_meta}
            )
        return out

    def _checkpoint_step_dirs(self, name: str) -> list[tuple[int, Path]]:
        """Return [(step_int, dir_path), ...] sorted by step ascending."""
        _validate_name(name)
        d = self.ckpt_dir / name
        if not d.is_dir():
            return []
        out: list[tuple[int, Path]] = []
        for p in d.iterdir():
            if not p.is_dir() or not p.name.startswith("step_"):
                continue
            try:
                step = int(p.name.split("_", 1)[1])
            except ValueError:
                continue
            out.append((step, p))
        out.sort()
        return out

    def prune_checkpoints(
        self,
        name: str,
        *,
        keep_last: int = 0,
        max_age_seconds: float | None = None,
    ) -> list[str]:
        """Apply retention rules to one run's checkpoints.

        Semantics (union-of-keepers): a step dir is kept if it falls within
        the ``keep_last`` most-recent set OR its mtime is within
        ``max_age_seconds``. With both rules unset (the defaults), this is a
        no-op. Returns the list of deleted step dir names.
        """
        if keep_last <= 0 and (max_age_seconds is None or max_age_seconds <= 0):
            return []
        steps = self._checkpoint_step_dirs(name)
        if not steps:
            return []
        keep: set[Path] = set()
        if keep_last > 0:
            keep.update(p for _, p in steps[-keep_last:])
        if max_age_seconds is not None and max_age_seconds > 0:
            cutoff = time.time() - max_age_seconds
            for _, p in steps:
                try:
                    if p.stat().st_mtime >= cutoff:
                        keep.add(p)
                except FileNotFoundError:
                    keep.add(p)  # vanished concurrently — nothing to prune
        deleted: list[str] = []
        for _, p in steps:
            if p in keep:
                continue
            shutil.rmtree(p, ignore_errors=True)
            deleted.append(p.name)
        return deleted
