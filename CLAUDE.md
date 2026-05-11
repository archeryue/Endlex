# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo status

v1 implemented per `TECH_PLAN.md`. **Read TECH_PLAN.md before changing behavior** — it is the authoritative spec for storage layout, HTTP API, client contract, and the performance rules below. If a request conflicts with it, surface the conflict rather than silently diverging.

## Layout

```
endlex/
  tracker.py            # client Tracker (hot-path log, daemon batching)
  checkpoint_sync.py    # upload_checkpoint{,_async} helpers
  server/
    app.py              # FastAPI factory `create_app(data_root)`
    storage.py          # disk layout: runs/, checkpoints/, JSONL append
    auth.py             # bearer-token dependencies
    templates/          # dashboard + per-run Chart.js page
tests/                  # pytest, includes a real-uvicorn e2e
deploy/                 # systemd unit, nginx snippet, env.example
```

## Commands

```bash
uv sync --extra server          # install deps incl. fastapi/uvicorn
uv run pytest -q                # full suite (~8s; includes uvicorn e2e + Chromium UI)
uv run pytest tests/test_tracker.py::test_log_hot_path_under_100us  # perf gate
uv run playwright install chromium  # one-time: required for tests/test_browser.py
uv run endlex-server            # run the server (port 8000 by default)
```

The cloud trainer should `pip install endlex` (no `--extra server`) — the client deps are just `httpx`.

## Features beyond v1

These are landed and tested; treat them as the authoritative behavior rather than re-reading TECH_PLAN for them:

- **Run state** in `runs/<name>/state.json`: `{tags, archived, retention, notes}`. Write via `PATCH /api/runs/<name>/state`. Dashboard surfaces tag chips + archived chip + show-archived toggle. Run page has an inline chip editor (Enter/comma to add, × or Backspace to remove) + a notes textarea.
- **Compare overlay**: `/compare?runs=a,b,c` overlays selected runs on the standard chart panels. Dashboard has row checkboxes + "Compare selected" button.
- **Checkpoint retention**: env defaults `ENDLEX_CKPT_KEEP_LAST` / `ENDLEX_CKPT_MAX_AGE_DAYS`, per-run override via state.json `retention`. Prune happens after each upload + on `POST /api/admin/prune` (for cron).
- **Search/filter**: dashboard search box grammar — substring | `tag:foo` | `key<op>num` (key on the latest metric event; ops `<`, `<=`, `>`, `>=`, `=`). Space-separated terms AND together.
- **Live updates via SSE**: `GET /api/runs/<name>/metrics/stream` emits `event: metric` per new event. Hard `max_lifetime` cap (env `ENDLEX_SSE_MAX_LIFETIME_SEC`, default 3600). Run + compare pages upgrade to EventSource after the initial poll and fall back to 5s polling on error.
- **Static HTML export**: `GET /api/runs/<name>/export.html` returns a self-contained report with all metrics embedded as JSON and the same chart panels (Chart.js from CDN). `?download=1` sets attachment header.
- **Summary cache**: `runs/<name>/.summary.json` sidecar bumped by `append_metrics`. Cache validity by `metrics_size`; mismatch triggers a rescan + repopulate. Keeps `list_runs()` O(1) per run even for very large JSONLs. Perf-gated test verifies cached `list_runs()` < 100 ms over 30 runs × 5k events and ≥5× faster than uncached.
- **/health**: unauthenticated probe returning `{status, version, runs}`.
- **Write auth in the browser**: `_base.html` exposes `endlex.authedFetch(url, opts)` which lazy-prompts for `ENDLEX_TOKEN`, caches in `localStorage`, clears on 401/403. Server uses `secrets.compare_digest` for constant-time token check.

## Tracker hardening beyond the original spec

- **Retry-with-backoff** on 5xx + transport errors. Daemon-thread only; hot path unchanged. Configurable via `retry_delays` kwarg (default `(0.5, 1.0, 2.0)` → 4 attempts). 4xx never retried.
- **Resync on init**: if the local JSONL has events past the server's count (cloud trainer restart), ship the gap. Scoped to `_initial_local_count` snapshotted at construction to avoid double-shipping events logged in the current session.
- **`flush(timeout)`**: synchronously drains the queue + waits for in-flight batch. Use between epochs or before checkpoint upload. Offline-mode no-op.
- **Warn-at-finish**: stderr warning if `dropped` or `failed_requests` > 0 at finish time. Easy to miss otherwise — hot path swallows everything.

## CI

`.github/workflows/test.yml` runs `pytest -q` on every push to main + on PRs. Caches uv resolution + Playwright Chromium. Local equivalent: `uv run pytest -q`.

## What Endlex is

Self-hosted, single-user replacement for the two wandb features that matter for solo LLM training: **metrics tracking + dashboard** and **model checkpoint sync** from cloud GPU boxes to a home box. Built primarily as the telemetry layer for ArcherChat (a from-scratch nanochat rewrite), but the design is generic for any single-user training workflow.

One server process, one storage tree under `$ENDLEX_DATA/`, one bearer token (`$ENDLEX_TOKEN`) on writes. Sits behind the user's existing nginx + Let's Encrypt cert.

## Architecture in one breath

- **Server**: FastAPI app exposing `/api/runs/...` (writes: init / metrics batch append / checkpoint upload) and read endpoints backing an HTML dashboard. Storage is plain files: `runs/<name>/{config.json, metrics.jsonl}` and `checkpoints/<name>/step_<N>/{model.pt, meta.json}`. No database in v1 — JSONL is append-only and tail-friendly.
- **Client (`endlex.Tracker`)**: wandb-shaped surface (`init` / `log` / `finish`). Writes locally first, then a daemon thread batch-POSTs to the server. Designed as a one-import drop-in for `wandb.log()` call sites.
- **Checkpoint upload**: separate daemon thread invoked from the trainer's `save_checkpoint`. Local save is the source of truth; remote upload is best-effort.

## Non-negotiable constraints

These are the rules that distinguish Endlex from wandb — code review should reject anything that violates them:

1. `Tracker.log()` must return in <100 µs. Local JSONL append + queue push only. No serialization beyond `json.dumps`, no network I/O on the trainer thread, ever.
2. Never force a CPU-GPU sync from the client. The trainer pre-extracts scalars via `.item()`; the client never touches tensors.
3. Bounded queues with drop-oldest under sustained backpressure. Local JSONL still gets every event — only remote is degraded.
4. Metrics endpoint takes a **batch** (`POST` body is an array of dicts). Time- and count-aware batching on the client (defaults: 100 events or 5 s). One HTTP round-trip per batch.
5. Metrics and checkpoint uploads run on **separate** daemon threads — slow checkpoint transfers must not starve low-latency metric flushes.
6. Local file is the source of truth. `kill -9` on the network process must lose zero data.
7. One writer per run. Enforce with a lock file at `init` — concurrent writers to the same `metrics.jsonl` would interleave.
8. No subprocess for system metrics. If/when GPU stats land, use `pynvml` inside the trainer process (which already has the CUDA context).

## Scope boundaries (from TECH_PLAN non-goals)

Single-user only. No multi-user/team features, no HA/clustering, no sweep orchestration, no hyperparameter-search UI. Auth is one shared bearer token — that is intentional and sufficient. Don't add user models, RBAC, or per-project tokens.

## Conventions worth knowing before the first PR

- Env vars are the configuration surface: `ENDLEX_URL`, `ENDLEX_TOKEN`, `ENDLEX_DATA`. If `ENDLEX_URL` is unset, the client runs fully offline (local logging only) — preserve that fallback.
- Read endpoints may be token-gated *or* open depending on deploy preference; write endpoints are always token-gated. Don't collapse the two.
- The roadmap in `TECH_PLAN.md` is staged v1 → v2 → v3. Don't pull v2/v3 items (run comparison, SSE, system metrics, static-HTML export) into v1 work without an explicit ask.
