#!/usr/bin/env python3
"""Select exactly one correct DSH trajectory per task across three seed passes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--seeds", default="1000,2000,3000")
    args = parser.parse_args()
    tasks = {item["instance_id"]: item for item in json.loads(Path(args.task_file).read_text())}
    root = Path(args.run_root)
    seeds = [int(value) for value in args.seeds.split(",")]
    selected_dir = root / "training_set/selected_sessions"
    selected_dir.mkdir(parents=True, exist_ok=True)
    train_path = root / "training_set/train.correct_trajectory_index.jsonl"
    reject_path = root / "training_set/rejected_all_3_wrong.jsonl"
    agreement_path = root / "training_set/three_seed_outcomes.jsonl"
    kept = rejected = 0
    with train_path.open("w") as train, reject_path.open("w") as reject, agreement_path.open("w") as agreement:
        for task_id, task in sorted(tasks.items()):
            outcomes = []
            for seed in seeds:
                path = root / f"seed{seed}/results/{task_id}.json"
                if path.exists():
                    outcomes.append(json.loads(path.read_text()))
            correct = [item for item in outcomes if item.get("correct")]
            agreement.write(json.dumps({
                "sample_id": task_id,
                "correct_seeds": [item["seed"] for item in correct],
                "n_correct": len(correct),
                "completed_seeds": [item["seed"] for item in outcomes],
            }, ensure_ascii=False) + "\n")
            if not correct:
                rejected += 1
                reject.write(json.dumps({"sample_id": task_id, "outcomes": outcomes}, ensure_ascii=False) + "\n")
                continue
            chosen = correct[0]
            source = Path(chosen.get("session_path") or "")
            selected_session = ""
            if source.is_file():
                destination = selected_dir / f"{task_id}.jsonl"
                shutil.copy2(source, destination)
                selected_session = str(destination.resolve())
            train.write(json.dumps({
                "sample_id": task_id,
                "database_id": task["db_id"],
                "question": task["question"],
                "selected_seed": chosen["seed"],
                "final_sql": chosen["sql"],
                "trajectory_path": selected_session,
                "correct_seed_count": len(correct),
            }, ensure_ascii=False) + "\n")
            kept += 1
    summary = {"total": len(tasks), "kept": kept, "rejected_all_3_wrong": rejected, "seeds": seeds}
    (root / "training_set/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return 0 if kept + rejected == len(tasks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
