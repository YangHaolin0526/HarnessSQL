#!/usr/bin/env python3
"""Download the clean Harbor Spider2-DBT task payloads without Docker."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


REPO_ID = "harborframework/harbor-datasets"
DATASET_ROOT = "datasets/spider2-dbt"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main"
RESOLVE_ROOT = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
KEEP_PREFIX = "environment/dbt_project/"
KEEP_EXACT = {
    "instruction.md",
    "tests/config.json",
    "tests/gold.duckdb",
    "tests/test_dbt.py",
}


def fetch_json(url: str, retries: int = 5) -> object:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def list_tree(path: str, recursive: bool) -> list[dict]:
    encoded = urllib.parse.quote(path, safe="/")
    query = urllib.parse.urlencode(
        {"recursive": str(recursive).lower(), "expand": "false", "limit": 1000}
    )
    result = fetch_json(f"{API_ROOT}/{encoded}?{query}")
    if not isinstance(result, list):
        raise RuntimeError(f"Unexpected Hugging Face tree response for {path}")
    return result


def list_tasks() -> list[str]:
    return sorted(
        item["path"].rsplit("/", 1)[-1]
        for item in list_tree(DATASET_ROOT, recursive=False)
        if item.get("type") == "directory"
    )


def destination_for(relative_path: str) -> Path | None:
    if relative_path.startswith(KEEP_PREFIX):
        return Path("workspace") / relative_path.removeprefix(KEEP_PREFIX)
    if relative_path in KEEP_EXACT:
        return Path(relative_path)
    return None


def download_file(remote_path: str, destination: Path, expected_size: int | None) -> None:
    if destination.exists() and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    encoded = urllib.parse.quote(remote_path, safe="/")
    url = f"{RESOLVE_ROOT}/{encoded}?download=true"

    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "spider2-dbt-local/1"})
            with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if expected_size is not None and part.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for {remote_path}: "
                    f"{part.stat().st_size} != {expected_size}"
                )
            part.replace(destination)
            return
        except Exception:
            part.unlink(missing_ok=True)
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def download_task(task_id: str, output_dir: Path) -> None:
    remote_root = f"{DATASET_ROOT}/{task_id}"
    task_dir = output_dir / task_id
    selected = 0
    for item in list_tree(remote_root, recursive=True):
        if item.get("type") != "file":
            continue
        remote_path = item["path"]
        relative = remote_path.removeprefix(remote_root + "/")
        destination = destination_for(relative)
        if destination is None:
            continue
        download_file(remote_path, task_dir / destination, item.get("size"))
        selected += 1

    required = [
        task_dir / "instruction.md",
        task_dir / "workspace/dbt_project.yml",
        task_dir / "workspace/profiles.yml",
        task_dir / "tests/config.json",
        task_dir / "tests/gold.duckdb",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"{task_id}: incomplete download; missing {missing}")
    print(f"[{task_id}] ready ({selected} files)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task IDs. Empty downloads all 64 clean Harbor tasks.",
    )
    parser.add_argument("--list", action="store_true", help="List task IDs and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    available = list_tasks()
    if args.list:
        print("\n".join(available))
        return

    requested = [item for item in args.tasks.split(",") if item] or available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise SystemExit(f"Unknown task IDs: {', '.join(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task_id in requested:
        download_task(task_id, args.output_dir)


if __name__ == "__main__":
    main()
