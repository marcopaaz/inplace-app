from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Never

from inplace.errors import PipelineError
from inplace.jobs import JobStore
from inplace.meta_client import MetaClient
from inplace.pipeline import check, format_render, format_report, render
from inplace.stages.creative import resolve_brand
from inplace.webapp import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inplace")
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser("check", help="Score a video and write a placeability report")
    check_parser.add_argument("video", type=Path)
    check_parser.add_argument("--jobs", type=Path, default=Path("jobs"))
    check_parser.add_argument("--force", action="append", default=[])

    render_parser = sub.add_parser("render", help="Track a window, place a creative, and write the visibility report")
    render_parser.add_argument("job")
    render_parser.add_argument("--window", type=int, required=True)
    render_parser.add_argument("--brand", required=True)
    render_parser.add_argument("--description", default="")
    render_parser.add_argument("--style", default="")
    render_parser.add_argument("--creative", type=int, default=0)
    render_parser.add_argument("--jobs", type=Path, default=Path("jobs"))
    render_parser.add_argument("--force", action="append", default=[])

    ui_parser = sub.add_parser("ui", help="Open a local page for upload, scoring, and placement")
    ui_parser.add_argument("--jobs", type=Path, default=Path("jobs"))
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    try:
        command = args.command
        if command == "check":
            client = MetaClient()
            job, report = check(args.video, args.jobs, client, set(args.force))
            print(format_report(job, report))
        elif command == "render":
            client = MetaClient()
            job = JobStore.open(args.jobs, args.job)
            visibility = render(
                job,
                window_index=args.window,
                brand=resolve_brand(args.brand, args.description, args.style),
                creative_index=args.creative,
                client=client,
                force=set(args.force),
            )
            print(format_render(job, visibility))
        elif command == "ui":
            serve(args.jobs, host=args.host, port=args.port)
        else:
            never: Never = command
            raise AssertionError(never)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
