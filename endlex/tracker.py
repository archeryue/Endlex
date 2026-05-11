"""Endlex client. wandb-shaped API for solo LLM training.

Compliance with TECH_PLAN performance rules:

* ``log()`` returns in <100 µs: ``json.dumps`` + line-buffered write + deque
  append. No locks on the hot path.
* No network I/O on the trainer thread. A daemon thread drains the queue
  and POSTs batches. If ``ENDLEX_URL`` is unset, no thread runs at all.
* Bounded ``deque(maxlen=...)`` drops oldest events under sustained
  backpressure; local JSONL still receives every event.
* Count- OR time-triggered batching (defaults: 100 events / 5 s).
* Local JSONL is the source of truth; remote is best-effort.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import httpx

_DEFAULT_BATCH_SIZE = 100
_DEFAULT_BATCH_INTERVAL = 5.0
_DEFAULT_QUEUE_MAX = 10_000
_DEFAULT_LOCAL_ROOT = "./endlex_runs"
_DEFAULT_RETRY_DELAYS: tuple[float, ...] = (0.5, 1.0, 2.0)


class Tracker:
    """wandb-shaped tracker. See module docstring for the performance contract."""

    def __init__(
        self,
        project: str,
        name: str,
        config: dict[str, Any] | None = None,
        *,
        local_dir: str | os.PathLike[str] | None = None,
        url: str | None = None,
        token: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        batch_interval: float = _DEFAULT_BATCH_INTERVAL,
        queue_max: int = _DEFAULT_QUEUE_MAX,
        force: bool = False,
        retry_delays: tuple[float, ...] = _DEFAULT_RETRY_DELAYS,
        _client: httpx.Client | None = None,  # test seam
    ) -> None:
        if not project or not name:
            raise ValueError("project and name are required")
        self.project = project
        self.name = name
        self.config = dict(config or {})
        self._url = (url if url is not None else os.environ.get("ENDLEX_URL")) or None
        self._token = (
            token if token is not None else os.environ.get("ENDLEX_TOKEN")
        )
        self._force = force
        self.batch_size = max(1, int(batch_size))
        self.batch_interval = float(batch_interval)
        self.retry_delays = tuple(float(d) for d in retry_delays)
        self._queue: deque[dict[str, Any]] = deque(maxlen=int(queue_max))
        self._dropped = 0
        self._failed_requests = 0
        self._last_error: str | None = None
        self._resynced = 0
        self._in_flight = 0  # events popped from queue but not yet POST'd

        root = Path(
            local_dir
            if local_dir is not None
            else os.environ.get("ENDLEX_LOCAL_DIR", _DEFAULT_LOCAL_ROOT)
        )
        self._dir = root / project / name
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "config.json").write_text(
            json.dumps(self.config, indent=2, sort_keys=True)
        )

        # Snapshot the local JSONL line count BEFORE we open it for append.
        # Anything beyond this index is "this session" and flows through the
        # queue; the resync path ships only these pre-existing lines so it
        # can't accidentally duplicate the events we're about to log.
        metrics_path = self._dir / "metrics.jsonl"
        self._initial_local_count = 0
        if metrics_path.exists():
            try:
                with metrics_path.open("rb") as f:
                    self._initial_local_count = sum(1 for ln in f if ln.strip())
            except OSError:
                pass

        # Line-buffered: each newline-terminated write hits the OS, so `kill -9`
        # of the trainer loses zero data (only an in-flight kernel buffer
        # crash on the host could).
        self._local = open(metrics_path, "a", buffering=1, encoding="utf-8")

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._finished = False
        self._init_ok = False  # set by daemon after successful POST /init

        if self._url or _client is not None:
            if _client is not None:
                self._client = _client
            else:
                headers = (
                    {"Authorization": f"Bearer {self._token}"} if self._token else {}
                )
                self._client = httpx.Client(
                    base_url=self._url,
                    headers=headers,
                    timeout=httpx.Timeout(30.0, connect=10.0),
                )
            self._thread = threading.Thread(
                target=self._loop,
                name=f"endlex-tracker-{name}",
                daemon=True,
            )
            self._thread.start()
        else:
            self._client = None
            self._thread = None

    # ---------- hot path ----------

    def log(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, separators=(",", ":"), sort_keys=True)
        # write+newline as two calls is fine: line buffering flushes on \n.
        self._local.write(line)
        self._local.write("\n")
        # Detect overflow: append on a full bounded deque silently drops the
        # leftmost element. Compare length before/after — same length means
        # we just dropped one to make room.
        if self._queue.maxlen is not None and len(self._queue) == self._queue.maxlen:
            self._dropped += 1
        self._queue.append(event)
        if len(self._queue) >= self.batch_size:
            self._wake.set()

    # ---------- lifecycle ----------

    def flush(self, *, timeout: float = 10.0) -> bool:
        """Block until the outbound queue is empty and no batch is in flight.

        Useful between epochs or before a checkpoint upload so the dashboard
        is current. Offline mode (no URL) is a no-op that returns True.
        Returns True if drained, False if the timeout elapsed first.
        """
        if self._thread is None:
            return True
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self._queue and self._in_flight == 0:
                return True
            self._wake.set()
            time.sleep(0.01)
        return not self._queue and self._in_flight == 0

    def finish(self, *, timeout: float = 30.0) -> None:
        if self._finished:
            return
        self._finished = True
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._client is not None:
            self._client.close()
        try:
            self._local.close()
        except Exception:
            pass
        # Surface any way the remote diverged from local — easy to miss
        # otherwise, since the trainer's hot path swallows all of this.
        warnings: list[str] = []
        if self._dropped > 0:
            warnings.append(
                f"{self._dropped} events dropped from remote queue "
                "(local JSONL still has them)"
            )
        if self._failed_requests > 0:
            warnings.append(
                f"{self._failed_requests} failed HTTP attempts "
                f"(last error: {self._last_error})"
            )
        if warnings:
            import sys

            print(
                f"[endlex] Tracker '{self.name}' finished with notable conditions:",
                file=sys.stderr,
            )
            for w in warnings:
                print(f"  - {w}", file=sys.stderr)

    def __enter__(self) -> "Tracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()

    @property
    def dropped(self) -> int:
        """Count of events dropped from the *remote* queue (local has them)."""
        return self._dropped

    @property
    def failed_requests(self) -> int:
        """Cumulative count of failed HTTP attempts (5xx, 4xx, or transport)."""
        return self._failed_requests

    @property
    def last_error(self) -> str | None:
        """Last observed error string, or None if no requests have failed."""
        return self._last_error

    @property
    def resynced(self) -> int:
        """Number of events shipped from local JSONL on startup resync."""
        return self._resynced

    # ---------- daemon ----------

    def _loop(self) -> None:
        if not self._init_remote():
            return  # remote is dead; local file keeps recording
        self._init_ok = True
        self._resynced = self._resync_local_to_remote()
        while not self._stop.is_set():
            self._wake.wait(timeout=self.batch_interval)
            self._wake.clear()
            self._drain_one_batch()
        self._drain_all()

    def _resync_local_to_remote(self) -> int:
        """Ship pre-existing local events that the server doesn't have yet.

        Use case: a cloud trainer that lost its network mid-run, kept logging
        locally, and is now resuming. The local file has events past the
        server's last-known offset.

        Scope is restricted to events that were already in the local file
        when the Tracker was constructed (``self._initial_local_count``).
        Events logged during *this* session flow through the queue only —
        otherwise resync and queue-drain would both ship them and produce
        duplicates.
        """
        n_initial = self._initial_local_count
        if n_initial == 0:
            return 0
        local_path = self._dir / "metrics.jsonl"
        try:
            lines: list[bytes] = []
            with local_path.open("rb") as f:
                for ln in f:
                    if ln.strip():
                        lines.append(ln)
                    if len(lines) >= n_initial:
                        break
        except OSError:
            return 0
        if not lines:
            return 0
        try:
            r = self._client.get(f"/api/runs/{self.name}")
            if r.status_code != 200:
                return 0
            server_count = int(r.json().get("summary", {}).get("num_events", 0))
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return 0
        if len(lines) <= server_count:
            return 0
        to_ship: list[dict[str, Any]] = []
        for raw in lines[server_count:]:
            try:
                to_ship.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        shipped = 0
        for i in range(0, len(to_ship), self.batch_size):
            chunk = to_ship[i : i + self.batch_size]
            if not self._post_batch(chunk):
                break
            shipped += len(chunk)
        return shipped

    def _request_with_retry(
        self, fn: Callable[[], httpx.Response]
    ) -> httpx.Response | None:
        """Exponential-backoff retry on 5xx and transport errors.

        Returns the final response (which may itself be a non-200 4xx) or
        ``None`` if every attempt raised. 4xx is *not* retried — it's the
        caller's bug, not the network's. Failures increment
        ``self._failed_requests`` so the metric is observable.
        """
        delays = self.retry_delays
        for i in range(len(delays) + 1):
            try:
                r = fn()
                if r.status_code == 200:
                    return r
                self._last_error = f"HTTP {r.status_code}"
                self._failed_requests += 1
                if r.status_code < 500:
                    return r  # non-retryable
            except httpx.HTTPError as e:
                self._last_error = f"{type(e).__name__}: {e}"
                self._failed_requests += 1
            if i < len(delays):
                time.sleep(delays[i])
        return None

    def _init_remote(self) -> bool:
        params = {"force": "true"} if self._force else None
        r = self._request_with_retry(
            lambda: self._client.post(
                f"/api/runs/{self.name}/init",
                json=self.config,
                params=params,
            )
        )
        return r is not None and r.status_code == 200

    def _take_batch(self) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for _ in range(self.batch_size):
            try:
                batch.append(self._queue.popleft())
            except IndexError:
                break
        return batch

    def _post_batch(self, batch: list[dict[str, Any]]) -> bool:
        r = self._request_with_retry(
            lambda: self._client.post(
                f"/api/runs/{self.name}/metrics", json=batch
            )
        )
        return r is not None and r.status_code == 200

    def _drain_one_batch(self) -> None:
        batch = self._take_batch()
        if not batch:
            return
        self._in_flight = len(batch)
        try:
            self._post_batch(batch)
        finally:
            self._in_flight = 0

    def _drain_all(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                return
            self._in_flight = len(batch)
            try:
                ok = self._post_batch(batch)
            finally:
                self._in_flight = 0
            if not ok:
                return
