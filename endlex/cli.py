"""``endlex`` command-line interface.

Weights sync: list runs, and pull checkpoint files down from the server or push
them up to it, with sha256 verification against the server's manifest.

    endlex runs                          # list runs
    endlex ckpts <run>                   # list a run's checkpoints
    endlex pull <run>                    # latest checkpoint -> ./<run>/<step>/
    endlex pull <run> --step 2000 --dest weights/ --files model.pt
    endlex push <run> --dir CKPT_DIR     # upload the latest step's files in CKPT_DIR
    endlex push <run> --dir CKPT_DIR --step 1920 --files model_001920.pt

Server + auth come from ``ENDLEX_URL`` / ``ENDLEX_TOKEN`` (or --url/--token).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import sys
from pathlib import Path

import httpx

from endlex import __version__
from endlex.checkpoint_sync import download_checkpoint, upload_checkpoint


def _client(args: argparse.Namespace) -> httpx.Client:
    url = args.url or os.environ.get("ENDLEX_URL")
    token = args.token or os.environ.get("ENDLEX_TOKEN")
    if not url:
        sys.exit("error: set ENDLEX_URL or pass --url")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=url, headers=headers, timeout=60.0)


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _cmd_runs(args: argparse.Namespace) -> None:
    with _client(args) as c:
        r = c.get("/api/runs", params={"include_archived": args.all})
        r.raise_for_status()
        rows = r.json()
    if not rows:
        print("no runs")
        return
    fmt = "{:<32} {:<16} {:>8}  {:<16} {}"
    print(fmt.format("NAME", "PROJECT", "EVENTS", "LAST UPDATE", "TAGS"))
    for row in rows:
        print(
            fmt.format(
                row["name"],
                row.get("project") or "-",
                row.get("num_events", 0),
                _fmt_ts(row.get("last_updated")),
                ",".join(row.get("tags") or []),
            )
        )


def _cmd_ckpts(args: argparse.Namespace) -> None:
    with _client(args) as c:
        r = c.get(f"/api/runs/{args.run}")
        if r.status_code == 404:
            sys.exit(f"error: run {args.run!r} not found")
        r.raise_for_status()
        cks = r.json().get("checkpoints", [])
    if not cks:
        print("no checkpoints")
        return
    for ck in cks:
        meta = ck.get("files_meta") or {}
        total = sum((meta.get(f) or {}).get("size") or 0 for f in ck["files"])
        print(f"{ck['step']}  ({total / 1e6:.1f} MB)")
        for f in ck["files"]:
            m = meta.get(f) or {}
            size = m.get("size")
            sha = m.get("sha256")
            size_s = f"{size / 1e6:.1f} MB" if size is not None else "?"
            sha_s = (sha[:12] + "…") if sha else "no checksum"
            print(f"    {f:<32} {size_s:>10}  {sha_s}")


def _cmd_pull(args: argparse.Namespace) -> None:
    dest = args.dest
    if dest is None:
        step_part = f"step_{args.step:06d}" if args.step is not None else "latest"
        dest = os.path.join(args.run, step_part)
    files = args.files.split(",") if args.files else None
    with _client(args) as c:
        try:
            paths = download_checkpoint(
                args.run, step=args.step, dest=dest, files=files, _client=c
            )
        except RuntimeError as e:
            sys.exit(f"error: {e}")
    for p in paths:
        print(f"pulled {p} ({p.stat().st_size / 1e6:.1f} MB, sha256 verified)")


def _cmd_push(args: argparse.Namespace) -> None:
    d = Path(args.dir)
    if not d.is_dir():
        sys.exit(f"error: --dir {d} is not a directory")

    # Resolve the step: explicit, else the highest one found among the dir's
    # step-numbered files (e.g. model_001920.pt / optim_001920_rank0.pt).
    step = args.step
    if step is None:
        steps = {
            int(m.group(1))
            for p in d.iterdir()
            if p.is_file() and (m := re.search(r"_(\d{6})(?=[._])", p.name))
        }
        if not steps:
            sys.exit(f"error: no step-numbered checkpoint files in {d} "
                     "(expected e.g. model_001920.pt); pass --step")
        step = max(steps)

    tag = f"{step:06d}"
    if args.files:
        names = [n.strip() for n in args.files.split(",") if n.strip()]
    else:
        names = sorted(p.name for p in d.iterdir() if p.is_file() and tag in p.name)
    if not names:
        sys.exit(f"error: no files for step {step} in {d}")

    files: dict[str, str] = {}
    for name in names:
        p = d / name
        if not p.exists():
            sys.exit(f"error: file not found: {p}")
        files[name] = str(p)

    total_mb = sum(os.path.getsize(v) for v in files.values()) / 1e6
    print(f"pushing step {step} to run {args.run!r} ({len(files)} files, {total_mb:.0f} MB):")
    for name, path in files.items():
        print(f"    {name}  ({os.path.getsize(path) / 1e6:.1f} MB)")

    # NOTE: do NOT hand upload_checkpoint the CLI's _client — let it build its own
    # (fresh-connection-per-chunk + long timeout). Reusing a keep-alive client is
    # exactly what triggers tunnel bad_record_mac on large chunked uploads.
    ok = upload_checkpoint(args.run, step, files, url=args.url, token=args.token)
    if not ok:
        sys.exit("error: upload failed (see [endlex] messages above)")
    print(f"pushed step {step} to {args.run} ({total_mb:.0f} MB, sha256-verified server-side)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="endlex", description="Endlex client CLI (runs / checkpoints / pull / push)"
    )
    parser.add_argument("--version", action="version", version=f"endlex {__version__}")
    parser.add_argument("--url", help="server URL (default: $ENDLEX_URL)")
    parser.add_argument("--token", help="bearer token (default: $ENDLEX_TOKEN)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_runs = sub.add_parser("runs", help="list runs")
    p_runs.add_argument("--all", action="store_true", help="include archived")
    p_runs.set_defaults(fn=_cmd_runs)

    p_ckpts = sub.add_parser("ckpts", help="list a run's checkpoints")
    p_ckpts.add_argument("run")
    p_ckpts.set_defaults(fn=_cmd_ckpts)

    p_pull = sub.add_parser("pull", help="download a checkpoint (sha256-verified)")
    p_pull.add_argument("run")
    p_pull.add_argument("--step", type=int, default=None, help="default: latest")
    p_pull.add_argument(
        "--dest", default=None, help="default: ./<run>/<step or latest>/"
    )
    p_pull.add_argument("--files", help="comma-separated subset (default: all)")
    p_pull.set_defaults(fn=_cmd_pull)

    p_push = sub.add_parser("push", help="upload a checkpoint's files to the server")
    p_push.add_argument("run")
    p_push.add_argument("--dir", default=".",
                        help="local dir holding the checkpoint files (default: .)")
    p_push.add_argument("--step", type=int, default=None,
                        help="default: highest step found in --dir")
    p_push.add_argument("--files",
                        help="comma-separated basenames in --dir (default: all files for the step)")
    p_push.set_defaults(fn=_cmd_push)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
