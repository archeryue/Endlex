# Endlex Technical Plan

## Goals

- Replace wandb's two functions that matter for solo LLM training: **metrics tracking + dashboard** and **model checkpoint sync** between cloud GPU instances and a home box.
- Self-hosted on a home box that already has a domain + nginx + cert. Zero recurring cost, zero quotas, full control of the schema.
- Drop-in client API: swapping `wandb.log()` calls in nanochat-style trainers should be a one-import change.

## Non-goals

- Multi-user / team features (auth = one bearer token).
- High availability / clustering (single-node is fine).
- Sweep orchestration, hyperparameter search UI.
- General-purpose experiment tracking for projects beyond ArcherChat.

## Architecture

```
  Cloud H100 box                       Home box (domain + nginx + Endlex)
  ─────────────────                    ────────────────────────────────────
   trainer            ──── HTTPS ───→  Endlex server (FastAPI)
    tracker.log()                       POST /api/runs/<name>/metrics
    save_checkpoint()                   POST /api/runs/<name>/ckpt/<step>

                                        GET  /              (dashboard)
   Browser     ──────────────────────→  GET  /run/<name>    (chart page)
                                        GET  /api/...       (data)
                                        GET  /ckpt/...      (download)
```

One process, one storage tree, one auth boundary. Reuses the existing nginx + cert that already serves the user's domain.

## Storage layout (just files on disk)

```
$ENDLEX_DATA/
├── runs/
│   └── <run_name>/
│       ├── config.json            # training CLI args + env snapshot
│       └── metrics.jsonl          # append-only, one JSON dict per .log()
└── checkpoints/
    └── <run_name>/
        ├── step_001000/{model.pt, meta.json}
        ├── step_002000/{model.pt, meta.json}
        └── step_003000/{model.pt, meta.json}
```

No database in v1. JSONL is tail-friendly (the dashboard streams new lines as they land), `du -sh` answers "how much space is this using", `rm -rf` cleans up. Add SQLite later if multi-run cross-queries become painful.

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/runs/<name>/init` | Create run dir, store `config.json`. Idempotent. |
| `POST` | `/api/runs/<name>/metrics` | Append a **batch** of JSON dicts to `metrics.jsonl`. Body: `[{"step": ..., "train/loss": ...}, ...]` |
| `POST` | `/api/runs/<name>/ckpt/<step>` | Streamed multipart upload of `model.pt` + `meta.json` |
| `GET` | `/` | List runs with last-updated timestamp + summary metrics |
| `GET` | `/run/<name>` | HTML page: config + auto-refreshing charts |
| `GET` | `/api/runs` | JSON list of runs |
| `GET` | `/api/runs/<name>` | JSON: config + summary |
| `GET` | `/api/runs/<name>/metrics` | JSON: full metrics array (or NDJSON stream) |
| `GET` | `/api/runs/<name>/ckpt/<step>/<file>` | Download a checkpoint file |
| `DELETE` | `/api/runs/<name>` | Remove a run + all its checkpoints |

All write endpoints require `Authorization: Bearer $ENDLEX_TOKEN`. Read endpoints can be open or token-gated depending on how publicly the user wants the dashboard to be visible.

The metrics endpoint takes a **batch** (array), not a single event. This is the contract that lets the client coalesce multiple `.log()` calls into one HTTP round-trip.

## Client API

A small Python module mirrors the wandb surface that nanochat actually uses (`init`, `log`, `finish`). One-line drop-in for the trainer:

```python
# Before:
import wandb
wandb_run = wandb.init(project="nanochat", name=run_name, config=cfg)
wandb_run.log({"step": step, "train/loss": loss})

# After:
from endlex import Tracker
tracker = Tracker(project="archerchat", name=run_name, config=cfg)
tracker.log({"step": step, "train/loss": loss})
```

`Tracker` always writes to a local JSONL first (so a flaky network never costs metrics), then async-batches POSTs to the configured server. Server URL + token come from env (`ENDLEX_URL`, `ENDLEX_TOKEN`). If `ENDLEX_URL` is unset, runs fully offline (local logging only).

### Checkpoint upload patch

Roughly 10 lines added to `save_checkpoint`:

```python
if rank == 0 and os.environ.get("ENDLEX_URL"):
    # spawn a background thread so training never blocks on upload
    threading.Thread(
        target=_upload_checkpoint,
        args=(run_name, step, model_path, meta_path),
        daemon=True,
    ).start()
```

Local save remains the source of truth; remote upload is best-effort.

## Dashboard

Single-page-ish: server-side rendered HTML for the run list, client-side Chart.js (or Plotly) for the per-run charts.

Default chart panels (configurable per project later):
- `train/loss` vs `step`
- `val/bpb` vs `step`, vs `total_training_time`, vs `total_training_flops`
- `train/mfu` vs `step`
- `train/dt`, `train/tok_per_sec` vs `step` (stability check)
- `train/lrm` vs `step` (schedule sanity)

Auto-refresh every 5s by polling `/api/runs/<name>/metrics?since=<last_t>` for new lines. SSE/websockets are nicer but polling is fine at single-user scale.

## Performance contract

The client must contribute **< 0.1% of training step wall-clock**, verified by toggling the client off and comparing. wandb-style overheads (sidecar IPC, forced CPU-GPU syncs, gradient hooks, sync HTTP calls) are explicitly out.

Hard rules:

1. **`.log()` returns in < 100 µs.** Local JSONL append + queue push only. No serialization beyond `json.dumps`, no network I/O.
2. **No network I/O on the trainer thread, ever.** All HTTP calls happen in a daemon thread that drains the queue.
3. **No CPU-GPU syncs forced by the client.** The trainer pre-extracts scalars (`.item()`) before logging. Client never touches tensors.
4. **Bounded queues, drop-oldest under sustained backpressure.** Memory leak prevention. Local JSONL still has every event — only remote is degraded.
5. **Time- and count-aware batching.** Send when *either* N events queued OR T seconds elapsed (defaults: 100 / 5s). One HTTP call ships a batch.
6. **Checkpoint upload uses a separate daemon thread from metrics.** Big slow uploads must not starve low-latency metric flushes.
7. **No subprocess for system metrics.** Use `pynvml` directly inside the trainer (already has CUDA context). No `nvidia-smi` fork+exec polling.
8. **Local file is the source of truth.** `kill -9` on the network must lose zero data.

## Operational concerns

1. **Bandwidth direction.** Cloud → home is the home box's *download* speed (usually the faster half of asymmetric connections). 2 GB d24 model at 100 Mbps down ≈ 3 min. Negligible.
2. **Auth.** Single shared bearer token in env is plenty for one user. Do **not** leave write endpoints open — random scanners find them within hours.
3. **Cloud-side resilience.** Always save locally on cloud first (`save_checkpoint` already does this). Upload is best-effort; if it fails, user can `scp` from cloud before tearing the instance down.
4. **HTTPS.** Reuse the existing let's encrypt cert + nginx reverse-proxy in front of FastAPI. Never ship checkpoints (or bearer tokens) over plain HTTP.
5. **WSL2 inbound.** If the existing webpage hosting already works, do nothing extra. FastAPI must `bind 0.0.0.0`. WSL2 mirrored networking (Windows 11) or Windows host port-forwarding both work.
6. **Disk usage.** `$ENDLEX_DATA/checkpoints/` grows fast at 1–2 GB per d24 save. Either be selective in the trainer (only upload best-val_bpb checkpoints) or add a TTL/retention policy on the server (`keep latest N`, `delete older than N days`).
7. **Concurrency.** Multiple cloud runs writing to one server is fine because each writes to its own `<run_name>` dir. Two writers to the same `metrics.jsonl` would interleave — enforce one-writer-per-run by checking a lock file at `init`.

## Roadmap

### v1 — minimum viable, single file each side
- [ ] `server.py` — FastAPI app with the endpoints above
- [ ] `endlex/tracker.py` — `Tracker` class with the wandb-shaped API + performance contract
- [ ] `endlex/checkpoint_sync.py` — upload helper called from `save_checkpoint`
- [ ] `templates/dashboard.html` — list of runs
- [ ] `templates/run.html` — config + charts for one run
- [ ] Bearer-token auth middleware
- [ ] Systemd unit + nginx config snippet

### v2 — quality of life
- [ ] Run comparison view (overlay multiple runs on one chart)
- [ ] Tag / archive / delete runs from UI
- [ ] Configurable retention policy for checkpoints
- [ ] Metrics search ("show me all runs where val_bpb < 0.8")

### v3 — nice-to-haves
- [ ] SSE / websocket for live updates instead of polling
- [ ] System metrics (GPU util, VRAM, power) sourced via `pynvml` inside the trainer
- [ ] Sample text panel (model generations during training)
- [ ] Export run as a static HTML report (for sharing without exposing the live server)

## Use with ArcherChat

ArcherChat (the from-scratch nanochat rewrite) will adopt Endlex as its only telemetry from day one. The cloud speedrun stage relies on Endlex specifically so that final d24 weights end up on the home box automatically — the home box is then the launchpad for `chat_cli`, `chat_web`, and any post-hoc evaluation. The cloud H100 instance can be torn down the moment training finishes; nothing of value lives on it.
