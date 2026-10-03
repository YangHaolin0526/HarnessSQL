"""Generate real tool-using trajectories with API or in-process local models.

Unlike ``pilot_trajectories.bootstrap.jsonl``, output from this module is sampled
from a model.  Oracle SQL is never placed in the prompt; it is used only by the
result-hash verifier after model execution.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .model_backends import (
    EmptyResponseExhausted,
    ModelBackend,
    add_backend_arguments,
    backend_from_args,
)
from .pipeline import (
    ReadOnlySQLite,
    describe_table_observation,
    discover_database,
    load_catalog,
    rank_tables,
    table_map,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_ROOT = ROOT / "benchmarks/spider2_repo/spider2-lite/resource/databases"
DEFAULT_CATALOG_DIR = ROOT / "artifacts/data_synthesis/sqlite/catalogs"
DEFAULT_TASK_FILE = ROOT / "artifacts/data_synthesis/sqlite/hard_pilot_10/pilot_tasks.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/data_synthesis/sqlite/teacher_trajectories"


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_schema",
            "description": "Rank tables from the offline metadata/sample/distribution catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List every table/view and exact or bounded row count.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Return columns, types, descriptions, distributions and sample rows.",
            "parameters": {
                "type": "object",
                "properties": {"table_name": {"type": "string"}},
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dialect_notes",
            "description": "Return relevant SQLite syntax and semantic notes.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute one read-only SQLite query and return columns, row count and preview.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 10000},
                },
                "required": ["query"],
            },
        },
    },
]


SYSTEM_PROMPT = """Solve the SQLite analytics task by investigating the database with tools.
Use only search_schema, list_tables, describe_table, dialect_notes and execute_sql. The database is read-only.
You must execute the complete final query successfully before answering. Check metric grain, distinctness,
denominators, date boundaries, nulls, ordering and ties. Do not use or ask for an oracle query.
End with exactly `FINAL ANSWER:` followed by one fenced SQL block containing the tested query."""


class CatalogToolEnvironment:
    def __init__(self, database_id: str, database_root: Path, catalog_dir: Path) -> None:
        self.database_id = database_id
        self.catalog, self.catalog_path = load_catalog(catalog_dir, database_id)
        self.tables = table_map(self.catalog)
        self.database = ReadOnlySQLite(discover_database(database_root, database_id))
        self.successful_queries: list[dict[str, Any]] = []

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | str:
        if name == "search_schema":
            limit = max(1, min(int(arguments.get("limit", 8)), 20))
            return rank_tables(self.catalog, str(arguments.get("query", "")))[:limit]
        if name == "list_tables":
            return [
                {
                    "name": table["name"],
                    "kind": table.get("kind"),
                    "row_count": table.get("row_count"),
                    "row_count_status": table.get("row_count_status"),
                }
                for table in self.catalog["tables"]
            ]
        if name == "describe_table":
            key = re.sub(r"[^a-z0-9]", "", str(arguments.get("table_name", "")).casefold())
            if key not in self.tables:
                raise ValueError(f"unknown table {arguments.get('table_name')!r}")
            return describe_table_observation(self.tables[key])
        if name == "dialect_notes":
            return (
                "SQLite supports CTEs and window functions. Use date()/datetime()/julianday()/strftime() "
                "for time logic, json_extract() for JSON text, NULLIF for protected division, and explicit "
                "CASE expressions for conditional aggregation. Integer division must be forced to REAL."
            )
        if name == "execute_sql":
            query = str(arguments.get("query", ""))
            result = self.database.execute(query)
            visible = {
                "status": result["status"],
                "columns": result["columns"],
                "row_count": result["row_count"],
                "preview": result["preview"],
                "ordered_result_sha256": result["ordered_result_sha256"],
                "elapsed_ms": result["elapsed_ms"],
            }
            self.successful_queries.append({"query": query, "result": visible})
            return visible
        raise ValueError(f"unknown tool {name!r}")


def parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    calls = []
    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.S):
        payload = json.loads(match.group(1))
        if not isinstance(payload, dict) or "name" not in payload:
            raise ValueError("textual tool call requires name and arguments")
        calls.append(
            {
                "id": "call_" + uuid.uuid4().hex[:16],
                "type": "function",
                "function": {
                    "name": payload["name"],
                    "arguments": json.dumps(payload.get("arguments", {}), ensure_ascii=False),
                },
            }
        )
    return calls


def parse_arguments(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    function = call.get("function") or {}
    name = str(function.get("name", ""))
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments or "{}")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    return name, arguments


def extract_final_sql(content: str) -> str | None:
    if "FINAL ANSWER:" not in content.upper():
        return None
    matches = re.findall(r"```(?:sql)?\s*(.*?)```", content, re.S | re.I)
    if not matches:
        return None
    return matches[-1].strip().rstrip(";")


def _expected_hash(task: dict[str, Any]) -> str:
    return task["instruction_environment_verifier"]["verifier"]["ordered_result_sha256"]


def run_trajectory(
    task: dict[str, Any],
    backend: ModelBackend,
    *,
    database_root: Path,
    catalog_dir: Path,
    max_turns: int = 24,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    sample_id = task["sample_id"]
    iev = task["instruction_environment_verifier"]
    database_id = iev["environment"]["database_id"]
    environment = CatalogToolEnvironment(database_id, database_root, catalog_dir)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Database id: {database_id}\nDialect: SQLite\nQuestion: {iev['instruction']}"
            ),
        },
    ]
    events = []
    usage_totals: dict[str, int] = {}
    status = "max_turns"
    final_sql = None
    final_result: dict[str, Any] | None = None
    expected_hash = _expected_hash(task)
    started = time.monotonic()
    for turn in range(1, max_turns + 1):
        try:
            reply = backend.complete(
                messages,
                tools=TOOLS,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except EmptyResponseExhausted as exc:
            for key, value in exc.usage.items():
                usage_totals[key] = usage_totals.get(key, 0) + value
            status = "empty_response_exhausted"
            events.append(
                {
                    "turn": turn,
                    "kind": "empty_response_exhausted",
                    "empty_response_attempts": exc.attempts,
                    "usage": exc.usage,
                }
            )
            break
        for key, value in reply.usage.items():
            if isinstance(value, int):
                usage_totals[key] = usage_totals.get(key, 0) + value
        calls = reply.tool_calls
        if not calls and not backend.supports_native_tools:
            try:
                calls = parse_text_tool_calls(reply.content)
            except Exception as exc:
                messages.append({"role": "assistant", "content": reply.content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Malformed tool call ({type(exc).__name__}: {exc}). Use the exact tool-call protocol.",
                    }
                )
                events.append({"turn": turn, "kind": "malformed_tool_call", "error": str(exc)})
                continue
        assistant_message = {"role": "assistant", "content": reply.content or ""}
        if calls:
            assistant_message["tool_calls"] = calls
        messages.append(assistant_message)
        if calls:
            for call in calls:
                call_id = call.get("id") or "call_" + uuid.uuid4().hex[:16]
                try:
                    name, arguments = parse_arguments(call)
                    output = environment.call(name, arguments)
                    content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
                    event = {"turn": turn, "kind": "tool", "name": name, "status": "success"}
                except Exception as exc:
                    name = (call.get("function") or {}).get("name", "unknown")
                    content = json.dumps(
                        {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    )
                    event = {
                        "turn": turn,
                        "kind": "tool",
                        "name": name,
                        "status": "error",
                        "error": str(exc),
                    }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": content[:50_000],
                    }
                )
                events.append(event)
            continue
        candidate = extract_final_sql(reply.content)
        if candidate is None:
            messages.append(
                {
                    "role": "user",
                    "content": "Continue with tools, or finish using the required FINAL ANSWER fenced-SQL format.",
                }
            )
            events.append({"turn": turn, "kind": "missing_final_format"})
            continue
        final_sql = candidate
        executed = next(
            (
                item["result"]
                for item in reversed(environment.successful_queries)
                if item["query"].strip().rstrip(";") == candidate
            ),
            None,
        )
        if executed is None:
            messages.append(
                {
                    "role": "user",
                    "content": "The final SQL was not executed verbatim. Execute it, inspect the result, then answer again.",
                }
            )
            events.append({"turn": turn, "kind": "final_not_executed"})
            continue
        final_result = executed
        if executed["ordered_result_sha256"] == expected_hash:
            status = "solved"
            break
        messages.append(
            {
                "role": "user",
                "content": (
                    "The tested query did not satisfy the hidden result verifier. Re-check population, grain, "
                    "denominator, boundaries and ties; investigate and submit a corrected tested query."
                ),
            }
        )
        events.append({"turn": turn, "kind": "verifier_mismatch"})
        final_sql = None
        final_result = None
    successful_execute_before_final = bool(final_result) and any(
        item["query"].strip().rstrip(";") == (final_sql or "")
        for item in environment.successful_queries
    )
    trajectory = {
        "trajectory_version": "spider2-model-rollout-v1",
        "sample_id": sample_id,
        "database_id": database_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "sampled_model_tool_rollout",
        "backend": type(backend).__name__,
        "model": backend.model,
        "empty_response_retries": getattr(backend, "empty_response_retries", None),
        "status": status,
        "training_eligibility": "eligible" if status == "solved" else "rejected",
        "oracle_sql_exposed_to_model": False,
        "messages": messages,
        "events": events,
        "final_sql": final_sql,
        "validation": {
            "successful_execute_before_final": successful_execute_before_final,
            "result_hash_match": bool(final_result)
            and final_result["ordered_result_sha256"] == expected_hash,
            "execute_attempts": len(environment.successful_queries),
        },
        "usage": usage_totals,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return trajectory


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_many(
    tasks: list[dict[str, Any]],
    backend: ModelBackend,
    *,
    database_root: Path,
    catalog_dir: Path,
    output_dir: Path,
    max_turns: int,
    temperature: float,
    max_tokens: int,
    resume: bool = True,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dir = output_dir / "tasks"
    task_dir.mkdir(exist_ok=True)
    trajectories = []
    for task in tasks:
        path = task_dir / f"{task['sample_id']}.json"
        if resume and path.exists():
            trajectory = json.loads(path.read_text())
        else:
            trajectory = run_trajectory(
                task,
                backend,
                database_root=database_root,
                catalog_dir=catalog_dir,
                max_turns=max_turns,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2) + "\n")
        trajectories.append(trajectory)
        (output_dir / "trajectories.jsonl").write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in trajectories) + "\n"
        )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(trajectories),
        "solved": sum(item["status"] == "solved" for item in trajectories),
        "backend": type(backend).__name__,
        "model": backend.model,
        "empty_response_retries": getattr(backend, "empty_response_retries", None),
        "oracle_sql_exposed_to_model": False,
        "resume": resume,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return trajectories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(parser)
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DATABASE_ROOT)
    parser.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ids", default="", help="optional comma-separated sample IDs")
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    tasks = load_tasks(args.task_file)
    selected = {value.strip() for value in args.ids.split(",") if value.strip()}
    if selected:
        tasks = [task for task in tasks if task["sample_id"] in selected]
    backend = backend_from_args(args)
    trajectories = run_many(
        tasks,
        backend,
        database_root=args.database_root,
        catalog_dir=args.catalog_dir,
        output_dir=args.output_dir,
        max_turns=args.max_turns,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        resume=not args.no_resume,
    )
    solved = sum(item["status"] == "solved" for item in trajectories)
    print(f"generated {len(trajectories)} trajectories; {solved} passed hidden result verification")
    return 0 if solved == len(trajectories) else 2


if __name__ == "__main__":
    raise SystemExit(main())
