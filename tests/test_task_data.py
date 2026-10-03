from __future__ import annotations

import json
import sqlite3

from text2sql import task_data


def test_extract_sql_prefers_last_final_answer() -> None:
    text = """```sql
SELECT 1;
```
FINAL ANSWER:
```sql
SELECT 2;
```"""
    assert task_data.extract_sql(text) == "SELECT 2;"


def test_load_and_reward_one_task(tmp_path, monkeypatch) -> None:
    task_root = tmp_path / "tasks"
    db_root = tmp_path / "dbs"
    validated = task_root / "batch" / "validated"
    validated.mkdir(parents=True)
    db_root.mkdir()
    db_path = db_root / "demo.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE items(value INTEGER)")
        connection.executemany("INSERT INTO items VALUES (?)", [(1,), (2,)])

    expected_rows = [[3]]
    oracle_sql = "SELECT SUM(value) AS total FROM items"
    actual = task_data.EvaluationSQLite(db_path).execute(oracle_sql)
    row = {
        "sample_id": "demo-1",
        "instruction_environment_verifier": {
            "instruction": "Return the sum as total.",
            "verifier": {
                "order_sensitive": True,
                "ordered_result_sha256": actual["ordered_result_sha256"],
                "unordered_result_sha256": actual["unordered_result_sha256"],
                "expected_columns": ["total"],
            },
        },
        "oracle": {"sql": oracle_sql, "result": {"rows": expected_rows}},
        "provenance": {"database_id": "demo"},
        "task_ir": {"difficulty": {"level": "foundation", "measured_sql_features": {}}},
    }
    (validated / "pilot_tasks.jsonl").write_text(json.dumps(row) + "\n")

    monkeypatch.setitem(task_data.SOURCES, "spider2", {
        "root": task_root,
        "db_dir": db_root,
        "conn_prefix": "spider2-sqlite-",
        "batches": [],
    })
    tasks = task_data.load_tasks(sources=["spider2"])
    assert len(tasks) == 1
    reward = task_data.sql_reward(tasks[0], row["oracle"]["sql"])
    assert reward["reward"] == 1.0
