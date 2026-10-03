#!/usr/bin/env python3
"""Run one resumable DSH seed pass over the 735 dbt SQLite tasks."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dsh_synth1800_pipeline import SynthTask, execute_sqlite_query, run_single_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dsh-root", required=True)
    parser.add_argument("--dsh-bin", default="")
    parser.add_argument("--node-bin", default=shutil.which("node") or "")
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--provider", choices=("vllm", "gateway"), default="vllm")
    parser.add_argument("--base-url", default="",
                        help="OpenAI-compatible endpoint; defaults to the loopback --port.")
    parser.add_argument("--seed-id", type=int, required=True)
    parser.add_argument("--model", default="Qwen3.6-27B")
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--task-timeout", type=int, default=1200)
    parser.add_argument("--query-timeout", type=int, default=60)
    parser.add_argument("--profile", default="spider2sql")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.dsh_home = str(Path(args.dsh_root) / "home")

    payload = json.loads(Path(args.task_file).read_text())
    if args.limit:
        payload = payload[: args.limit]
    gold = {item["instance_id"]: item["gold_sql"] for item in payload}
    tasks = [SynthTask(item["instance_id"], item["db_id"], item["question"]) for item in payload]
    out = Path(args.out_dir).resolve()
    results_dir = out / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    dsh_root = Path(args.dsh_root).resolve()
    dsh_bin = Path(args.dsh_bin).resolve() if args.dsh_bin else dsh_root / "node_modules/.bin/dsh"
    if not args.node_bin:
        parser.error("Node.js was not found; pass --node-bin")
    node_bin = Path(args.node_bin).resolve().parent
    db_dir = Path(args.db_dir).resolve()
    endpoint = args.base_url.rstrip("/") or f"http://127.0.0.1:{args.port}/v1"

    def one(task: SynthTask) -> dict:
        result_path = results_dir / f"{task.instance_id}.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        result = run_single_seed(
            task, args.seed_id, args, [endpoint], out, dsh_bin, node_bin, db_dir
        )
        if not result.get("session_path"):
            home = out / "seeds" / str(args.seed_id) / "homes" / task.instance_id
            sessions = list(home.glob("sessions/**/session.jsonl*"))
            result["session_path"] = str(sessions[0]) if sessions else ""
        db_path = db_dir / f"{task.db_id}.sqlite"
        gold_ok, gold_hash, _ = execute_sqlite_query(db_path, gold[task.instance_id], args.query_timeout)
        result["gold_exec_ok"] = gold_ok
        result["gold_res_hash"] = gold_hash
        result["correct"] = bool(result["exec_ok"] and gold_ok and result["res_hash"] == gold_hash)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result

    started = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[seed={args.seed_id}] {result['instance_id']} correct={result['correct']} "
                f"outcome={result['outcome']} sec={result['seconds']}",
                flush=True,
            )
    summary = {
        "seed_id": args.seed_id,
        "total": len(tasks),
        "completed": len(results),
        "correct": sum(item["correct"] for item in results),
        "elapsed_seconds": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if len(results) == len(tasks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
