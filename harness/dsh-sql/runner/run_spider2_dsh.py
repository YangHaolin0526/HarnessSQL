#!/usr/bin/env python3
"""Run the Spider 2.0-Lite SQLite subset through the dsh SQL harness.

One `dsh --profile spider2sql` process per task, N in flight at once. Each
process gets its own database path, answer file, and workspace through the
environment; the harness composition (bundle-sql/cordis.patch.yml) turns those
into the agent's only capabilities.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DSH_ROOT = HERE.parent
DEFAULT_DSH_BIN = DSH_ROOT / "node_modules" / ".bin" / "dsh"

PRINT_LOCK = threading.Lock()


@dataclass
class Task:
    instance_id: str
    db: str
    question: str
    external_knowledge: str | None


@dataclass
class Result:
    instance_id: str
    outcome: str          # ok | recovered_stdout | no_sql | timeout | error
    seconds: float
    returncode: int | None
    sql_chars: int


def load_tasks(task_file: Path, only: set[str] | None, limit: int) -> list[Task]:
    tasks = []
    for line in task_file.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row["instance_id"].startswith("local"):
            continue          # the sqlite subset; bigquery/snowflake live elsewhere
        if only and row["instance_id"] not in only:
            continue
        tasks.append(Task(row["instance_id"], row["db"], row["question"], row.get("external_knowledge")))
    tasks.sort(key=lambda t: t.instance_id)
    return tasks[:limit] if limit else tasks


def build_prompt(task: Task, doc_dir: Path, max_doc_chars: int) -> str:
    parts = [
        f"Spider 2.0-Lite task {task.instance_id}. Dialect: SQLite.",
        "",
        f"Question: {task.question}",
    ]
    if task.external_knowledge:
        doc = doc_dir / task.external_knowledge
        if doc.is_file():
            text = doc.read_text(errors="replace")
            if len(text) > max_doc_chars:
                text = text[:max_doc_chars] + "\n...[truncated by harness]"
            parts += ["", "External documentation (authoritative for this task):", text]
    parts += [
        "",
        "Explore the database with sql_list_tables and sql_schema, verify your query with sql_exec, "
        "then call sql_submit exactly once with the final query.",
    ]
    return "\n".join(parts)


FENCE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def sql_from_stdout(text: str) -> str | None:
    """Last resort: the model answered in prose without calling sql_submit."""
    blocks = FENCE.findall(text or "")
    for block in reversed(blocks):
        candidate = block.strip()
        if re.match(r"^(SELECT|WITH)\b", candidate, re.IGNORECASE):
            return candidate.rstrip().rstrip(";") + ";"
    return None


def run_one(task: Task, args, out: Path) -> Result:
    logs = out / "logs"
    answers = out / "answers"
    submission = out / "submission"
    workspace = out / "workspace" / task.instance_id
    for d in (logs, answers, submission, workspace):
        d.mkdir(parents=True, exist_ok=True)

    answer_path = answers / f"{task.instance_id}.sql"
    answer_path.unlink(missing_ok=True)

    home = out / "homes" / task.instance_id
    if not (home / "profiles").is_dir():
        home.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(args.dsh_home) / "profiles", home / "profiles", symlinks=True)

    env = dict(os.environ)
    if args.node_bin:
        env["PATH"] = f"{Path(args.node_bin).resolve().parent}{os.pathsep}{env.get('PATH','')}"
    env["DSH_HOME"] = str(home)
    env["DSH_SQL_DB"] = str(Path(args.db_dir) / f"{task.db}.sqlite")
    env["DSH_SQL_ANSWER"] = str(answer_path)
    env["DSH_SQL_PROVIDER"] = args.provider
    env["DSH_SQL_MODEL"] = args.model
    env["DSH_SQL_VLLM_MODEL"] = args.model
    if args.base_url:
        route_env = "DSH_SQL_VLLM_BASE_URL" if args.provider == "vllm" else "DSH_SQL_GATEWAY_BASE_URL"
        env[route_env] = args.base_url
    env.setdefault("DSH_SQL_VLLM_CONTEXT", "32768")
    env.setdefault("DSH_SQL_MAX_TOKENS", "8192")
    env.setdefault("CHOKIDAR_USEPOLLING", "1")
    env.setdefault("CHOKIDAR_INTERVAL", "1000")
    env["DSH_SQL_QUERY_TIMEOUT_MS"] = str(args.query_timeout * 1000)
    env["DSH_TELEMETRY_DISABLED"] = "1"
    env.setdefault("DSH_PERMISSION_MODE", "read-only")

    prompt = build_prompt(task, Path(args.doc_dir), args.max_doc_chars)
    started = time.monotonic()
    returncode: int | None = None
    stdout = stderr = ""
    timed_out = False
    transient_markers = (
        "PI_AI_ERROR: Already borrowed",
        "TRANSPORT: Connection error",
        "System limit for number of file watchers reached",
        "ECONNREFUSED",
    )
    for attempt in range(1, 4):
        try:
            proc = subprocess.run(
                [str(Path(args.dsh_bin)), "--profile", args.profile, prompt],
                cwd=workspace, env=env, capture_output=True, text=True,
                timeout=args.task_timeout,
            )
            stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            timed_out = True
        is_transient = (
            not timed_out
            and returncode not in (0, None)
            and any(marker in (stderr or "") for marker in transient_markers)
        )
        if not is_transient or attempt == 3:
            break
        time.sleep(2)
    seconds = time.monotonic() - started

    (logs / f"{task.instance_id}.stdout.txt").write_text(stdout or "")
    (logs / f"{task.instance_id}.stderr.txt").write_text(stderr or "")

    sql = answer_path.read_text().strip() if answer_path.is_file() else ""
    outcome = "ok"
    if not sql:
        recovered = sql_from_stdout(stdout)
        if recovered:
            sql, outcome = recovered, "recovered_stdout"
        else:
            sql = "SELECT NULL;"
            outcome = "timeout" if timed_out else ("error" if returncode not in (0, None) else "no_sql")
    elif timed_out:
        outcome = "ok"

    (submission / f"{task.instance_id}.sql").write_text(sql if sql.endswith("\n") else sql + "\n")
    result = Result(task.instance_id, outcome, round(seconds, 1), returncode, len(sql))
    (out / "results" / f"{task.instance_id}.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "results" / f"{task.instance_id}.json").write_text(json.dumps(asdict(result), indent=1))
    with PRINT_LOCK:
        print(f"[{result.outcome:>17}] {task.instance_id}  {result.seconds:7.1f}s  {result.sql_chars:5d} chars",
              flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--task-file", type=Path, required=True,
                    help="Spider 2.0-Lite JSONL task file.")
    ap.add_argument("--db-dir", type=Path, required=True,
                    help="Directory containing <db>.sqlite files.")
    ap.add_argument("--doc-dir", type=Path, required=True,
                    help="Directory containing referenced external-knowledge files.")
    ap.add_argument("--provider", default="vllm")
    ap.add_argument("--model", default="Qwen3.5-9B")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--profile", default="spider2sql")
    ap.add_argument("--dsh-home", default=str(DSH_ROOT / "home"),
                    help="Template Harness home; each task gets its own copy of its profiles/ tree.")
    ap.add_argument("--dsh-bin", default=str(DEFAULT_DSH_BIN))
    ap.add_argument("--node-bin", default=shutil.which("node") or "",
                    help="Node executable; its parent is prepended to PATH.")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--task-timeout", type=int, default=1200)
    ap.add_argument("--query-timeout", type=int, default=60)
    ap.add_argument("--max-doc-chars", type=int, default=40000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    for label, path in (("task file", args.task_file), ("database directory", args.db_dir),
                        ("document directory", args.doc_dir), ("dsh executable", Path(args.dsh_bin))):
        if not path.exists():
            ap.error(f"{label} does not exist: {path}")
    only = {t.strip() for t in args.tasks.split(",") if t.strip()} or None
    tasks = load_tasks(args.task_file, only, args.limit)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.resume:
        done = {p.stem for p in (out / "results").glob("*.json")}
        skipped = [t for t in tasks if t.instance_id in done]
        tasks = [t for t in tasks if t.instance_id not in done]
        print(f"resume: {len(skipped)} already done, {len(tasks)} to run", flush=True)

    print(f"{len(tasks)} tasks | provider={args.provider} model={args.model} workers={args.workers}", flush=True)
    started = time.monotonic()
    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, t, args, out): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                with PRINT_LOCK:
                    print(f"[            crash] {task.instance_id}: {exc}", flush=True)
                results.append(Result(task.instance_id, "error", 0.0, None, 0))
    wall = time.monotonic() - started

    counts: dict[str, int] = {}
    for r in results:
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    summary = {
        "provider": args.provider,
        "model": args.model,
        "workers": args.workers,
        "tasks_run": len(results),
        "wall_seconds": round(wall, 1),
        "outcomes": counts,
        "mean_task_seconds": round(sum(r.seconds for r in results) / max(len(results), 1), 1),
        "max_task_seconds": max((r.seconds for r in results), default=0.0),
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
