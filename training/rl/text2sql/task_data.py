"""Reference data for the RL SQLite task pool (Spider2-lite synth + dbt).

Two independently synthesised sources, both in the same
``harness-aware-sqlite-task-v2`` row format:

* **spider2 / sqlite** -- 1,800 tasks over the 30 Spider2-lite SQLite
  databases, in ``artifacts/data_synthesis/sqlite/<batch>/validated/``:
    candidate_200 (easy gate), candidate_harder_200 (hard gate),
    candidate_target40_200, candidate_extra_hard_200,
    candidate_target40_gpt56_500, candidate_frontier27b_gpt56_batch2
* **dbt** -- 735 tasks over 49 dbt-project databases (15 per database), in
  ``rl_run/data/dbt_sqlite/score20_30_per_db15/validated/``. These were used in
  SFT and were missing from the first RL run.

The two differ in exactly two respects that matter here, and both are handled
per-source rather than globally:

1. **Where the .sqlite file lives.** Spider2 databases sit in the benchmark's
   ``resource/databases``; dbt databases sit in the synthesis artifacts under
   ``dbt_sqlite/databases``. ``provenance.database_path`` in a dbt row may
   record a path from an earlier runtime and must never be trusted -- the path
   is always re-derived from the source's own database directory.
2. **The KTX connection id.** Spider2 databases are registered in the KTX
   project as ``spider2-sqlite-<slug>``; the dbt ones are registered as
   ``dbt-sqlite-<slug>``. A task whose connection id is not in the project's
   ``ktx.yaml`` cannot be queried at all, so ``sql_execution`` fails and the
   reward is a silent zero -- see ``scripts/build_ktx_dbt_project.py``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else None


SYNTH = _env_path("HARNESS_SQL_SYNTH_ROOT")
DB_DIR = _env_path("HARNESS_SQL_SPIDER_DB_DIR")
DBT_SYNTH = _env_path("HARNESS_SQL_DBT_SYNTH_ROOT")
DBT_DB_DIR = _env_path("HARNESS_SQL_DBT_DB_DIR")

try:
    from data_synthesis.codex_ktx_eval import EvaluationSQLite  # noqa: E402
    from data_synthesis.common import canonical_json  # noqa: E402
except ImportError:
    EvaluationSQLite = None
    canonical_json = None

BATCHES = [
    "candidate_200",
    "candidate_harder_200",
    "candidate_target40_200",
    "candidate_extra_hard_200",
    "candidate_target40_gpt56_500",
    "candidate_frontier27b_gpt56_batch2",
]

DBT_BATCHES = [
    "score20_30_per_db15",
]

# source -> (root holding <batch>/validated/pilot_tasks.jsonl, database dir,
#            KTX connection-id prefix, default batch list)
SOURCES: dict[str, dict[str, Any]] = {
    "spider2": {
        "root": SYNTH,
        "db_dir": DB_DIR,
        "conn_prefix": "spider2-sqlite-",
        "batches": BATCHES,
    },
    "dbt": {
        "root": DBT_SYNTH,
        "db_dir": DBT_DB_DIR,
        "conn_prefix": "dbt-sqlite-",
        "batches": DBT_BATCHES,
    },
}


def _slug(database_id: str) -> str:
    return database_id.lower().replace("_", "-")


def connection_id_for(database_id: str, source: str = "spider2") -> str:
    """KTX connection id. Must match a key in the project's ktx.yaml."""
    return SOURCES[source]["conn_prefix"] + _slug(database_id)


def db_path_for(database_id: str, source: str = "spider2") -> Path:
    """Always derived from the source's db dir, never from provenance."""
    db_dir = SOURCES[source]["db_dir"]
    if db_dir is None:
        raise RuntimeError(f"database root for {source!r} is not configured; see .env.example")
    return db_dir / f"{database_id}.sqlite"


def load_tasks(batches: list[str] | None = None,
               limit: int = 0,
               shuffle_seed: int | None = None,
               sources: list[str] | None = None) -> list[dict[str, Any]]:
    """Flatten the pilot_tasks.jsonl files into RL-ready task dicts.

    ``sources`` defaults to both pools. Passing ``batches`` restricts every
    selected source to those batch names, which is only meaningful for a
    single source; the old single-source call signature is unchanged.
    """
    out: list[dict[str, Any]] = []
    for source in (sources or list(SOURCES)):
        spec = SOURCES[source]
        root = spec["root"]
        if root is None:
            raise RuntimeError(f"task root for {source!r} is not configured; see .env.example")
        for batch in (batches or spec["batches"]):
            p = Path(root) / batch / "validated" / "pilot_tasks.jsonl"
            if not p.exists():
                raise FileNotFoundError(p)
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                iev = row["instruction_environment_verifier"]
                ver = iev["verifier"]
                oracle = row["oracle"]
                db = row["provenance"]["database_id"]
                out.append({
                    "sample_id": row["sample_id"],
                    "source": source,
                    "batch": batch,
                    "database_id": db,
                    "connection_id": connection_id_for(db, source),
                    "db_path": str(db_path_for(db, source)),
                    "question": iev["instruction"],
                    "oracle_sql": oracle["sql"],
                    "order_sensitive": bool(ver.get("order_sensitive", True)),
                    "ordered_sha256": ver.get("ordered_result_sha256"),
                    "unordered_sha256": ver.get("unordered_result_sha256"),
                    "expected_columns": list(ver.get("expected_columns") or []),
                    "expected_rows": (oracle.get("result") or {}).get("rows") or [],
                    "difficulty": row["task_ir"]["difficulty"].get("level"),
                    "structural_score": (row["task_ir"]["difficulty"]
                                         .get("measured_sql_features", {})
                                         .get("structural_score")),
                })

    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(out)
    if limit:
        out = out[:limit]
    return out
def sql_reward(task: dict[str, Any], sql: str | None,
               timeout_s: float = 30.0,
               max_rows: int = 100_000) -> dict[str, Any]:
    if not sql:
        return {"reward": 0.0, "status": "no_sql"}

    try:
        actual = EvaluationSQLite(Path(task["db_path"]),
                                  timeout_seconds=timeout_s,
                                  max_rows=max_rows).execute(sql)
    except Exception as exc:
        return {"reward": 0.0, "status": "exec_error",
                "detail": f"{type(exc).__name__}: {exc}"[:300]}

    ordered = task["order_sensitive"]
    hash_key = "ordered_result_sha256" if ordered else "unordered_result_sha256"
    expected_hash = task["ordered_sha256"] if ordered else task["unordered_sha256"]

    result_match = actual[hash_key] == expected_hash

    expected_rows = task["expected_rows"]
    if ordered:
        value_match = actual["rows"] == expected_rows
    else:
        value_match = (sorted(canonical_json(r) for r in actual["rows"])
                       == sorted(canonical_json(r) for r in expected_rows))

    question = str(task.get("question") or "").casefold()
    cols = task["expected_columns"]
    aliases_public = bool(cols) and all(str(c).casefold() in question for c in cols)

    if result_match:
        status, reward = "correct", 1.0
    elif value_match and not aliases_public:
        status, reward = "correct_values_only", 1.0
    elif value_match:
        status, reward = "wrong_column_names", 0.0
    else:
        status, reward = "wrong_result", 0.0

    return {
        "reward": reward,
        "status": status,
        "row_count": actual["row_count"],
        "elapsed_ms": actual["elapsed_ms"],
        "column_names_exact": actual["columns"] == cols,
    }


def extract_sql(text: str) -> str | None:
    if not text:
        return None
    tail = text
    marks = list(re.finditer(r"FINAL\s+ANSWER\s*:", text, re.IGNORECASE))
    if marks:
        tail = text[marks[-1].end():]
    fenced = re.findall(r"```sql\s*(.*?)```", tail, re.IGNORECASE | re.DOTALL)
    if not fenced:
        fenced = re.findall(r"```\s*((?:select|with)\s.*?)```",
                            tail, re.IGNORECASE | re.DOTALL)
    if not fenced and marks:
        fenced = re.findall(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if not fenced:
        return None
    return fenced[-1].strip() or None
