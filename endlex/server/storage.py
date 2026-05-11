"""Storage layer for Endlex.

Layout under ``data_root``::

    runs/<name>/config.json     # training config + env snapshot
    runs/<name>/metrics.jsonl   # append-only, one JSON dict per .log()
    runs/<name>/.lock           # single-writer sentinel checked at init
    checkpoints/<name>/step_<NNNNNN>/<file>

No DB. JSONL is tail-friendly; rm -rf cleans up.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_STEP_RE = re.compile(r"^[0-9]{1,12}$")


class StorageError(Exception):
    pass


class InvalidName(StorageError):
    pass


class RunNotFound(StorageError):
    pass


class RunLocked(StorageError):
    """init called on an existing run whose config differs from what's on disk."""


@dataclass(frozen=True)
class RunSummary:
    name: str
    last_updated: float | None
    num_events: int
    latest: dict[str, Any] | None
    tags: list[str]
    archived: bool


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


def _step_dirname(step: str | int) -> str:
    return f"step_{int(step):06d}"


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

    # ----- runs -----

    def init_run(
        self, name: str, config: dict[str, Any], *, force: bool = False
    ) -> None:
        _validate_name(name)
        run_dir = self.runs_dir / name
        cfg_path = run_dir / "config.json"
        lock_path = run_dir / ".lock"
        if run_dir.exists() and lock_path.exists() and not force:
            try:
                existing = json.loads(cfg_path.read_text())
            except FileNotFoundError:
                existing = None
            if existing != config:
                raise RunLocked(name)
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True))
        lock_path.write_text(json.dumps({"created_at": time.time()}))
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
            os.write(fd, data)
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
            cache_path.write_text(
                json.dumps(
                    {
                        "num_events": num_events,
                        "latest": latest,
                        "last_updated": last_updated,
                        "metrics_size": metrics_size,
                    }
                )
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
        if not state_path.exists():
            return {
                "tags": [],
                "archived": False,
                "retention": {},
                "notes": "",
            }
        try:
            raw = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            return {
                "tags": [],
                "archived": False,
                "retention": {},
                "notes": "",
            }
        return {
            "tags": list(raw.get("tags") or []),
            "archived": bool(raw.get("archived", False)),
            "retention": dict(raw.get("retention") or {}),
            "notes": str(raw.get("notes") or ""),
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
        (run_dir / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True)
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
        self, name: str, step: str | int, filename: str, source: BinaryIO
    ) -> int:
        """Stream `source` to disk, return bytes written."""
        dest = self.open_checkpoint_for_write(name, step, filename)
        written = 0
        with dest.open("wb") as out:
            while True:
                chunk = source.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        return written

    def checkpoint_file_path(self, name: str, step: str | int, filename: str) -> Path:
        _validate_name(name)
        _validate_step(str(step))
        _validate_filename(filename)
        p = self.ckpt_dir / name / _step_dirname(step) / filename
        if not p.exists():
            raise RunNotFound(f"{name}/{_step_dirname(step)}/{filename}")
        return p

    def list_checkpoints(self, name: str) -> list[dict[str, Any]]:
        _validate_name(name)
        d = self.ckpt_dir / name
        if not d.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for step_dir in sorted(d.iterdir()):
            if not step_dir.is_dir():
                continue
            files = sorted(p.name for p in step_dir.iterdir() if p.is_file())
            out.append({"step": step_dir.name, "files": files})
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
                if p.stat().st_mtime >= cutoff:
                    keep.add(p)
        deleted: list[str] = []
        for _, p in steps:
            if p in keep:
                continue
            shutil.rmtree(p)
            deleted.append(p.name)
        return deleted
