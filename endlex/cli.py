"""``endlex`` command-line interface.

The pull side of weights sync: list runs and fetch checkpoint files from the
server to whatever box you're on (dev laptop, eval machine), with sha256
verification against the server's manifest.

    endlex runs                          # list runs
    endlex ckpts <run>                   # list a run's checkpoints
    endlex pull <run>                    # latest checkpoint -> ./<run>/<step>/
    endlex pull <run> --step 2000 --dest weights/ --files model.pt

Server + auth come from ``ENDLEX_URL`` / ``ENDLEX_TOKEN`` (or --url/--token).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys

import httpx

from endlex import __version__
from endlex.checkpoint_sync import download_checkpoint


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="endlex", description="Endlex client CLI (runs / checkpoints / pull)"
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

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
