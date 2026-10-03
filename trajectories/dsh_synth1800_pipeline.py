#!/usr/bin/env python3
"""Multi-seed Trajectory Synthesizer for 1800 Synthetic SQL Tasks using DSH-SQL.

Features:
1. Loads 1800 tasks from omnisql_synth1800_reforce.json.
2. Runs 3 rollouts (seeds 0, 1, 2) per question at temperature 0.7.
3. Distributes requests across hosted API or local vLLM endpoints (load balanced).
4. Executes and grades generated SQL against SQLite databases.
5. Selects the verified/consensus trajectory per task.
6. Exports high-quality training datasets (train.split.jsonl and train.packed.jsonl).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

PRINT_LOCK = threading.Lock()
ENDPOINT_LOCK = threading.Lock()
ENDPOINT_CYCLE = 0


@dataclass
class SynthTask:
    instance_id: str
    db_id: str
    question: str


def load_synth_tasks(task_json_path: Path, limit: int = 0) -> list[SynthTask]:
    with open(task_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = []
    for item in data:
        tasks.append(
            SynthTask(
                instance_id=item["instance_id"],
                db_id=item["db_id"],
                question=item["question"],
            )
        )
    tasks.sort(key=lambda t: t.instance_id)
    return tasks[:limit] if limit > 0 else tasks


def get_next_endpoint(endpoints: list[str]) -> tuple[int, str]:
    global ENDPOINT_CYCLE
    with ENDPOINT_LOCK:
        index = ENDPOINT_CYCLE % len(endpoints)
        ENDPOINT_CYCLE += 1
        return index, endpoints[index]


def build_prompt(task: SynthTask) -> str:
    return (
        f"Spider 2.0 Task {task.instance_id}. Dialect: SQLite.\n\n"
        f"Question: {task.question}\n\n"
        f"Explore the database with sql_list_tables and sql_schema, verify your query with sql_exec, "
        f"then call sql_submit exactly once with the final query."
    )


FENCE = re.compile(r"```sql\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def sql_from_stdout(text: str) -> str | None:
    blocks = FENCE.findall(text or "")
    for block in reversed(blocks):
        candidate = block.strip()
        if re.match(r"^(SELECT|WITH)\b", candidate, re.IGNORECASE):
            return candidate.rstrip().rstrip(";") + ";"
    return None


def execute_sqlite_query(db_path: Path, sql: str, timeout: int = 15) -> tuple[bool, str, list | None]:
    if not db_path.is_file() or not sql.strip():
        return False, "Database not found or empty SQL", None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        # Hash rows for fast equivalence comparison
        rows_str = json.dumps(rows, default=str, sort_keys=True)
        h = hashlib.sha256(rows_str.encode()).hexdigest()
        return True, h, rows
    except Exception as e:
        return False, str(e), None


def run_single_seed(
    task: SynthTask,
    seed: int,
    args,
    endpoints: list[str],
    out_dir: Path,
    dsh_bin: Path,
    node_bin_dir: Path,
    db_dir: Path,
) -> dict:
    endpoint_index, endpoint = get_next_endpoint(endpoints)
    task_seed_id = f"{task.instance_id}_seed{seed}"

    seed_dir = out_dir / "seeds" / str(seed)
    logs_dir = seed_dir / "logs"
    answers_dir = seed_dir / "answers"
    workspace = seed_dir / "workspaces" / task.instance_id
    home = seed_dir / "homes" / task.instance_id

    for d in (logs_dir, answers_dir, workspace, home):
        d.mkdir(parents=True, exist_ok=True)

    answer_path = answers_dir / f"{task.instance_id}.sql"
    db_file = db_dir / f"{task.db_id}.sqlite"
    if not db_file.is_file():
        # Check case-insensitive or direct match
        for f in db_dir.glob("*.sqlite"):
            if f.stem.lower() == task.db_id.lower():
                db_file = f
                break

    if answer_path.is_file():
        # Resume if already computed
        sql = answer_path.read_text(encoding="utf-8").strip()
        ok, res_hash, _ = execute_sqlite_query(db_file, sql)
        return {
            "instance_id": task.instance_id,
            "seed": seed,
            "endpoint_index": endpoint_index,
            "sql": sql,
            "exec_ok": ok,
            "res_hash": res_hash,
            "seconds": 0.0,
            "outcome": "cached",
        }

    # Setup isolated cordis profile from dsh_home / profiles
    profile_src = Path(args.dsh_home) / "profiles"
    if not profile_src.is_dir():
        profile_src = Path(args.dsh_root) / "home" / "profiles"

    if not (home / "profiles").is_dir() and profile_src.is_dir():
        shutil.copytree(profile_src, home / "profiles", symlinks=True)
        # Ensure node_modules inside the copied profile point to absolute dsh_root
        prof_mods = home / "profiles" / args.profile / "node_modules"
        if prof_mods.is_dir():
            (prof_mods / "dsh-bundle-sql").unlink(missing_ok=True)
            (prof_mods / "dsh-plugin-sql").unlink(missing_ok=True)
            (prof_mods / "dsh-bundle-sql").symlink_to(Path(args.dsh_root) / "bundle-sql")
            (prof_mods / "dsh-plugin-sql").symlink_to(Path(args.dsh_root) / "plugin-sql")

    env = dict(os.environ)
    env["PATH"] = f"{node_bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["DSH_HOME"] = str(home)
    env["DSH_SQL_DB"] = str(db_file)
    env["DSH_SQL_ANSWER"] = str(answer_path)
    env["DSH_SQL_PROVIDER"] = args.provider
    env["DSH_SQL_MODEL"] = args.model
    if args.provider == "vllm":
        env["DSH_SQL_VLLM_MODEL"] = args.model
        env["DSH_SQL_VLLM_BASE_URL"] = endpoint
        env.setdefault("VLLM_API_KEY", "local-vllm")
    else:
        env["DSH_SQL_GATEWAY_MODEL"] = args.model
        env["DSH_SQL_GATEWAY_BASE_URL"] = endpoint
        if not env.get("DSH_SQL_GATEWAY_KEY"):
            raise RuntimeError("DSH_SQL_GATEWAY_KEY is required for --provider gateway")
    env["DSH_SQL_QUERY_TIMEOUT_MS"] = str(args.query_timeout * 1000)
    env["DSH_TELEMETRY_DISABLED"] = "1"
    env["DSH_PERMISSION_MODE"] = "read-only"
    env["DSH_TEMPERATURE"] = str(args.temperature)
    env["DSH_SEED"] = str(seed * 1000 + 42)
    # Shared GPU servers can exhaust the host-wide inotify watcher limit.
    # Chokidar polling avoids a fast ENOSPC failure in that environment.
    env["CHOKIDAR_USEPOLLING"] = "1"
    env["CHOKIDAR_INTERVAL"] = "1000"

    prompt = build_prompt(task)
    started = time.monotonic()
    returncode = None
    try:
        proc = subprocess.run(
            [str(dsh_bin), "--profile", args.profile, prompt],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            timeout=args.task_timeout,
        )
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timed_out = True
    seconds = time.monotonic() - started

    (logs_dir / f"{task.instance_id}.stdout.txt").write_text(stdout or "", encoding="utf-8")
    (logs_dir / f"{task.instance_id}.stderr.txt").write_text(stderr or "", encoding="utf-8")

    sql = answer_path.read_text(encoding="utf-8").strip() if answer_path.is_file() else ""
    outcome = "ok"
    if not sql:
        recovered = sql_from_stdout(stdout)
        if recovered:
            sql, outcome = recovered, "recovered_stdout"
            answer_path.write_text(sql + "\n", encoding="utf-8")
        else:
            sql = "SELECT NULL;"
            outcome = "timeout" if timed_out else ("error" if returncode not in (0, None) else "no_sql")

    exec_ok, res_hash, _ = execute_sqlite_query(db_file, sql)

    # Capture session trajectory if exists
    session_files = list(home.glob("sessions/**/session.jsonl*"))
    session_path = str(session_files[0]) if session_files else ""

    res_item = {
        "instance_id": task.instance_id,
        "seed": seed,
        "endpoint_index": endpoint_index,
        "sql": sql,
        "exec_ok": exec_ok,
        "res_hash": res_hash,
        "seconds": round(seconds, 1),
        "outcome": outcome,
        "session_path": session_path,
    }

    with PRINT_LOCK:
        status_tag = "EXEC_OK" if exec_ok else "EXEC_FAIL"
        print(
            f"[{status_tag:>9}] Task {task.instance_id} Seed {seed} | {res_item['seconds']:5.1f}s | Hash: {res_hash[:8]}",
            flush=True,
        )

    return res_item


def process_task_3seeds(
    task: SynthTask,
    args,
    endpoints: list[str],
    out_dir: Path,
    dsh_bin: Path,
    node_bin_dir: Path,
    db_dir: Path,
) -> dict:
    seeds_results = []
    for s in range(args.num_votes):
        res = run_single_seed(task, s, args, endpoints, out_dir, dsh_bin, node_bin_dir, db_dir)
        seeds_results.append(res)

    # Selection Strategy:
    # 1. Look for valid execution queries
    valid_seeds = [r for r in seeds_results if r["exec_ok"] and r["outcome"] in ("ok", "recovered_stdout")]

    selected = None
    if valid_seeds:
        # Majority voting by result hash
        hash_counts = {}
        for r in valid_seeds:
            h = r["res_hash"]
            hash_counts[h] = hash_counts.get(h, 0) + 1

        # Pick hash with maximum votes
        best_hash = max(hash_counts, key=hash_counts.get)
        candidates = [r for r in valid_seeds if r["res_hash"] == best_hash]
        selected = candidates[0]
        selection_method = f"majority_vote_{hash_counts[best_hash]}/{len(seeds_results)}"
    else:
        # Fallback to first non-empty SQL
        fallback = [r for r in seeds_results if r["sql"] and r["sql"] != "SELECT NULL;"]
        if fallback:
            selected = fallback[0]
            selection_method = "fallback_non_empty"
        else:
            selected = seeds_results[0]
            selection_method = "fallback_default"

    summary_item = {
        "instance_id": task.instance_id,
        "question": task.question,
        "db_id": task.db_id,
        "selection_method": selection_method,
        "selected_seed": selected["seed"],
        "selected_sql": selected["sql"],
        "selected_exec_ok": selected["exec_ok"],
        "selected_res_hash": selected["res_hash"],
        "selected_session": selected.get("session_path", ""),
        "seeds": seeds_results,
    }

    # Write selected answer to unified output
    selected_dir = out_dir / "selected_answers"
    selected_dir.mkdir(parents=True, exist_ok=True)
    (selected_dir / f"{task.instance_id}.sql").write_text(selected["sql"] + "\n", encoding="utf-8")

    (out_dir / "selected_summaries" / f"{task.instance_id}.json").parent.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "selected_summaries" / f"{task.instance_id}.json", "w", encoding="utf-8") as f:
        json.dump(summary_item, f, indent=2, ensure_ascii=False)

    return summary_item


def export_training_datasets(out_dir: Path, summaries: list[dict]):
    sft_dir = out_dir / "sft_dataset"
    sft_dir.mkdir(parents=True, exist_ok=True)

    packed_path = sft_dir / "train.packed.jsonl"
    split_path = sft_dir / "train.split.jsonl"

    valid_count = 0
    with open(packed_path, "w", encoding="utf-8") as f_packed, open(split_path, "w", encoding="utf-8") as f_split:
        for s in summaries:
            if not s["selected_exec_ok"]:
                continue
            valid_count += 1
            entry = {
                "instance_id": s["instance_id"],
                "db_id": s["db_id"],
                "question": s["question"],
                "final_sql": s["selected_sql"],
                "selection_method": s["selection_method"],
                "session_path": s.get("selected_session", ""),
            }
            f_packed.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f_split.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"\n[SFT Export] Exported {valid_count} / {len(summaries)} verified positive trajectories to:\n  - {packed_path}\n  - {split_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--dsh-root", type=str, required=True)
    parser.add_argument("--dsh-bin", type=str, default="",
                        help="Override the dsh executable under --dsh-root.")
    parser.add_argument("--node-bin", type=str, default=shutil.which("node") or "",
                        help="Node.js executable. Node 22+ is recommended.")
    parser.add_argument("--db-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen3.6-27B")
    parser.add_argument("--provider", choices=("vllm", "gateway"), default="vllm")
    parser.add_argument("--ports", type=str, default="8000",
                        help="Comma-separated loopback vLLM ports.")
    parser.add_argument("--base-urls", default="",
                        help="Comma-separated OpenAI-compatible base URLs; overrides --ports.")
    parser.add_argument("--num-votes", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--task-timeout", type=int, default=600)
    parser.add_argument("--query-timeout", type=int, default=30)
    parser.add_argument("--profile", type=str, default="spider2sql")
    parser.add_argument("--dsh-home", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dsh_root = Path(args.dsh_root).resolve()
    dsh_bin = Path(args.dsh_bin).resolve() if args.dsh_bin else dsh_root / "node_modules/.bin/dsh"
    if not args.node_bin:
        parser.error("Node.js was not found; pass --node-bin")
    node_bin_dir = Path(args.node_bin).resolve().parent
    db_dir = Path(args.db_dir).resolve()
    if not args.dsh_home:
        args.dsh_home = str(dsh_root / "home")

    if args.base_urls:
        endpoints = [url.strip().rstrip("/") for url in args.base_urls.split(",") if url.strip()]
    elif args.provider == "vllm":
        endpoints = [f"http://127.0.0.1:{int(p.strip())}/v1"
                     for p in args.ports.split(",") if p.strip()]
    else:
        configured = os.environ.get("DSH_SQL_GATEWAY_BASE_URL", "").strip()
        endpoints = [configured] if configured else []
    for label, path in (("task file", Path(args.task_file)), ("database directory", db_dir),
                        ("dsh executable", dsh_bin), ("dsh home", Path(args.dsh_home))):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    if not endpoints:
        parser.error("configure --base-urls (or --ports for local vLLM)")

    print("======================================================================")
    print("DSH-SQL 1800 Synthetic Tasks Trajectory Generator (3-Seed Rollout)")
    print(f"Task File:    {args.task_file}")
    print(f"Model:        {args.model}")
    print(f"Provider:      {args.provider}")
    print(f"Endpoints:     {len(endpoints)} configured")
    print(f"Rollouts/Q:   {args.num_votes}")
    print(f"Workers:      {args.concurrency}")
    print(f"Output Dir:   {out_dir}")
    print("======================================================================")

    tasks = load_synth_tasks(Path(args.task_file), args.limit)
    print(f"Loaded {len(tasks)} tasks. Starting 3-seed rollout synthesis...")

    summaries = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(process_task_3seeds, t, args, endpoints, out_dir, dsh_bin, node_bin_dir, db_dir): t for t in tasks
        }
        for future in as_completed(futures):
            try:
                res = future.result()
                summaries.append(res)
            except Exception as e:
                print(f"Error processing task: {e}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nCompleted all {len(summaries)} tasks in {elapsed:.1f}s ({elapsed/3600:.2f}h)!")

    # Export verified SFT datasets
    export_training_datasets(out_dir, summaries)

    # Save master summary
    with open(out_dir / "synth1800_master_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_tasks": len(tasks),
                "completed_tasks": len(summaries),
                "num_votes": args.num_votes,
                "elapsed_seconds": elapsed,
                "model": args.model,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
