#!/usr/bin/env python3
"""Parallel front-end for download_harbor_tasks.py.

The Harbor Spider2-DBT payload is ~18.3k mostly-small files (4.35 GB after the
oracle-solution filter), so the sequential downloader is request-latency bound
and takes hours. This reuses that module's filter verbatim -- Harbor's oracle
solution is still excluded -- and only parallelises the fetch loop.

Hugging Face rate-limits above roughly 8 concurrent requests; --workers 24 gets
429s on a cold run. Run once at 24 for speed, then again at the default 6 to
fill the gaps (download_file skips files whose size already matches).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import download_harbor_tasks as dl

REQUIRED = (
    "instruction.md",
    "workspace/dbt_project.yml",
    "workspace/profiles.yml",
    "tests/config.json",
    "tests/gold.duckdb",
)


def collect_jobs(tasks: list[str], output_dir: Path) -> list[tuple[str, Path, int | None]]:
    jobs: list[tuple[str, Path, int | None]] = []
    for task in tasks:
        root = f"{dl.DATASET_ROOT}/{task}"
        for item in dl.list_tree(root, recursive=True):
            if item.get("type") != "file":
                continue
            destination = dl.destination_for(item["path"].removeprefix(root + "/"))
            if destination is None:
                continue
            jobs.append((item["path"], output_dir / task / destination, item.get("size")))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--tasks", default="", help="Comma-separated task IDs; empty means all 64.")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    available = dl.list_tasks()
    tasks = [item for item in args.tasks.split(",") if item] or available
    unknown = sorted(set(tasks) - set(available))
    if unknown:
        raise SystemExit(f"Unknown task IDs: {', '.join(unknown)}")

    print(f"listing {len(tasks)} tasks...", flush=True)
    jobs = collect_jobs(tasks, args.output_dir.resolve())
    print(f"{len(jobs)} files to fetch with {args.workers} workers", flush=True)

    started = time.monotonic()
    completed = 0
    errors: list[tuple[str, str]] = []

    def fetch(job: tuple[str, Path, int | None]) -> None:
        dl.download_file(*job)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in as_completed(futures):
            completed += 1
            try:
                future.result()
            except Exception as exc:
                errors.append((futures[future][0], f"{type(exc).__name__}: {exc}"))
            if completed % 500 == 0:
                print(f"  {completed}/{len(jobs)}  {time.monotonic()-started:.0f}s", flush=True)

    print(f"finished in {time.monotonic()-started:.0f}s, errors={len(errors)}", flush=True)
    for path, error in errors[:20]:
        print(f"  ERR {path} {error}", flush=True)

    incomplete = [
        task for task in tasks
        if any(not (args.output_dir / task / name).exists() for name in REQUIRED)
    ]
    print(f"incomplete tasks: {incomplete}", flush=True)
    if errors or incomplete:
        raise SystemExit(f"re-run to fill gaps (errors={len(errors)}, incomplete={len(incomplete)})")


if __name__ == "__main__":
    main()
