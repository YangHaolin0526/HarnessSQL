from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .catalog import CatalogBuilder, CatalogOptions, default_paths
from .common import canonical_json, identifier_words, json_safe, normalize_identifier, quote_identifier, stable_hash
from .complexity import advanced_families, profile_sql
from .sqlite_hard_blueprints import HARD_PILOT_BLUEPRINTS
from .sqlite_curriculum_blueprints import CURRICULUM_PILOT_BLUEPRINTS


SAMPLE_VERSION = "harness-aware-sqlite-task-v2"
TRAJECTORY_VERSION = "codex-spiderdb-reference-v1"
ALLOWED_TOOLS = ["search_schema", "list_tables", "describe_table", "dialect_notes", "execute_sql"]
READ_ONLY_HEAD = re.compile(r"^\s*(?:WITH\b|SELECT\b)", re.IGNORECASE)
FORBIDDEN_SQL = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|VACUUM|REINDEX|PRAGMA)\b",
    re.IGNORECASE,
)


def normalize_db(value: str) -> str:
    return normalize_identifier(value)


def discover_database(database_root: Path, database_id: str) -> Path:
    wanted = normalize_db(database_id)
    for path in database_root.glob("*.sqlite"):
        if not path.name.startswith("._") and normalize_db(path.stem) == wanted:
            return path.resolve()
    raise FileNotFoundError(f"SQLite database not found for {database_id!r} under {database_root}")


def load_catalog(catalog_dir: Path, database_id: str) -> tuple[dict[str, Any], Path]:
    wanted = normalize_db(database_id)
    for path in catalog_dir.glob("*.catalog.json"):
        value = json.loads(path.read_text())
        if normalize_db(str(value.get("database_id", ""))) == wanted:
            return value, path.resolve()
    raise FileNotFoundError(f"catalog not found for {database_id!r} under {catalog_dir}")


def table_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {normalize_identifier(table["name"]): table for table in catalog["tables"]}


def column_map(table: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {normalize_identifier(column["name"]): column for column in table["columns"]}


def search_score(query: str, name: str, text: str, tags: Iterable[str] = ()) -> float:
    query_words = set(identifier_words(query))
    name_words = set(identifier_words(name))
    text_words = set(identifier_words(text))
    overlap = len(query_words & text_words) / max(1, len(query_words))
    name_overlap = len(query_words & name_words) / max(1, len(query_words))
    sequence = SequenceMatcher(None, normalize_identifier(query), normalize_identifier(name)).ratio()
    tag_overlap = len(query_words & set(tags)) / max(1, len(query_words))
    return round(0.45 * overlap + 0.25 * name_overlap + 0.2 * sequence + 0.1 * tag_overlap, 6)


def rank_tables(catalog: dict[str, Any], query: str) -> list[dict[str, Any]]:
    ranked = []
    for table in catalog["tables"]:
        text = " ".join(
            [table["name"]]
            + [column["name"] for column in table["columns"]]
            + [column.get("description", "") for column in table["columns"]]
            + [tag for column in table["columns"] for tag in column.get("semantic_tags", [])]
        )
        ranked.append(
            {
                "table": table["name"],
                "score": search_score(query, table["name"], text),
                "row_count": table.get("row_count"),
                "matched_columns": [
                    column["name"]
                    for column in table["columns"]
                    if set(identifier_words(query)) & set(identifier_words(column["name"]))
                ][:8],
            }
        )
    return sorted(ranked, key=lambda item: (-item["score"], item["table"].casefold()))


def rank_columns(table: dict[str, Any], query: str, required_type: str | None) -> list[dict[str, Any]]:
    ranked = []
    for column in table["columns"]:
        if required_type and column["type_family"] != required_type:
            continue
        text = " ".join(
            [column["name"], column.get("human_label", ""), column.get("description", "")]
            + column.get("semantic_tags", [])
        )
        ranked.append(
            {
                "column": column["name"],
                "score": search_score(query, column["name"], text, column.get("semantic_tags", [])),
                "type_family": column["type_family"],
                "semantic_tags": column.get("semantic_tags", []),
                "null_fraction": column["distribution"].get("null_fraction"),
                "sample_distinct_count": column["distribution"].get("sample_distinct_count"),
            }
        )
    return sorted(ranked, key=lambda item: (-item["score"], item["column"].casefold()))


def selected_rank(ranked: list[dict[str, Any]], key: str, selected: str) -> int | None:
    wanted = normalize_identifier(selected)
    for index, item in enumerate(ranked, 1):
        if normalize_identifier(str(item[key])) == wanted:
            return index
    return None


def find_catalog_join(catalog: dict[str, Any], join: list[str]) -> dict[str, Any] | None:
    left_table, left_column, right_table, right_column = map(normalize_identifier, join)
    for edge in catalog["joins"]:
        forward = (
            normalize_identifier(edge["left_table"]),
            normalize_identifier(edge["left_column"]),
            normalize_identifier(edge["right_table"]),
            normalize_identifier(edge["right_column"]),
        )
        reverse = (forward[2], forward[3], forward[0], forward[1])
        if (left_table, left_column, right_table, right_column) in (forward, reverse):
            return edge
    for edge in catalog.get("composite_joins", []):
        forward_components = {
            (
                normalize_identifier(edge["left_table"]),
                normalize_identifier(left),
                normalize_identifier(edge["right_table"]),
                normalize_identifier(right),
            )
            for left, right in zip(edge["left_columns"], edge["right_columns"])
        }
        reverse_components = {(item[2], item[3], item[0], item[1]) for item in forward_components}
        if (left_table, left_column, right_table, right_column) in (
            forward_components | reverse_components
        ):
            return {**edge, "composite": True}
    return None


class ReadOnlySQLite:
    def __init__(self, path: Path, timeout_seconds: float = 30.0, max_rows: int = 10_000) -> None:
        self.path = path.resolve()
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def static_audit(sql: str) -> dict[str, Any]:
        stripped = sql.strip().rstrip(";").strip()
        semicolons = stripped.count(";")
        join_count = len(re.findall(r"\bJOIN\b", stripped, flags=re.IGNORECASE))
        on_count = len(re.findall(r"\bON\b", stripped, flags=re.IGNORECASE))
        return {
            "read_only_head": bool(READ_ONLY_HEAD.match(stripped)),
            "forbidden_keyword": FORBIDDEN_SQL.search(stripped).group(0).upper() if FORBIDDEN_SQL.search(stripped) else None,
            "single_statement": semicolons == 0,
            "join_count": join_count,
            "join_predicate_count": on_count,
            "all_joins_have_predicates": on_count >= join_count,
        }

    @staticmethod
    def _authorizer(action: int, _arg1: str | None, _arg2: str | None, _db: str | None, _trigger: str | None) -> int:
        denied = {
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_CREATE_INDEX,
            sqlite3.SQLITE_CREATE_TABLE,
            sqlite3.SQLITE_CREATE_TEMP_INDEX,
            sqlite3.SQLITE_CREATE_TEMP_TABLE,
            sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
            sqlite3.SQLITE_CREATE_TEMP_VIEW,
            sqlite3.SQLITE_CREATE_TRIGGER,
            sqlite3.SQLITE_CREATE_VIEW,
            sqlite3.SQLITE_DROP_INDEX,
            sqlite3.SQLITE_DROP_TABLE,
            sqlite3.SQLITE_DROP_TEMP_INDEX,
            sqlite3.SQLITE_DROP_TEMP_TABLE,
            sqlite3.SQLITE_DROP_TEMP_TRIGGER,
            sqlite3.SQLITE_DROP_TEMP_VIEW,
            sqlite3.SQLITE_DROP_TRIGGER,
            sqlite3.SQLITE_DROP_VIEW,
            sqlite3.SQLITE_ALTER_TABLE,
            sqlite3.SQLITE_ATTACH,
            sqlite3.SQLITE_DETACH,
        }
        return sqlite3.SQLITE_DENY if action in denied else sqlite3.SQLITE_OK

    def execute(self, sql: str) -> dict[str, Any]:
        audit = self.static_audit(sql)
        if not all(
            [audit["read_only_head"], audit["single_statement"], audit["all_joins_have_predicates"]]
        ) or audit["forbidden_keyword"]:
            raise ValueError(f"static SQL audit failed: {audit}")
        started = time.monotonic()
        with self.connect() as connection:
            connection.set_authorizer(self._authorizer)
            connection.set_progress_handler(
                lambda: int(time.monotonic() - started > self.timeout_seconds), 10_000
            )
            try:
                plan = [
                    {"id": row[0], "parent": row[1], "detail": row[3]}
                    for row in connection.execute("EXPLAIN QUERY PLAN " + sql)
                ]
                cursor = connection.execute(sql)
                columns = [item[0] for item in cursor.description or []]
                raw_rows = cursor.fetchmany(self.max_rows + 1)
                if len(raw_rows) > self.max_rows:
                    raise ValueError(f"result exceeds max_rows={self.max_rows}")
            finally:
                connection.set_progress_handler(None, 0)
        rows = [[json_safe(value) for value in row] for row in raw_rows]
        ordered_payload = {"columns": columns, "rows": rows}
        unordered_rows = sorted((canonical_json(row) for row in rows))
        null_counts = {
            columns[index]: sum(row[index] is None for row in rows) for index in range(len(columns))
        }
        return {
            "status": "success",
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "null_counts": null_counts,
            "ordered_result_sha256": stable_hash(ordered_payload),
            "unordered_result_sha256": stable_hash({"columns": columns, "rows": unordered_rows}),
            "preview": [dict(zip(columns, row)) for row in rows[:10]],
            "query_plan": plan,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "static_audit": audit,
        }

    def join_exists(self, join: list[str]) -> tuple[bool, str | None]:
        left_table, left_column, right_table, right_column = join
        sql = (
            f"SELECT 1 FROM {quote_identifier(left_table)} AS l "
            f"JOIN {quote_identifier(right_table)} AS r "
            f"ON l.{quote_identifier(left_column)} = r.{quote_identifier(right_column)} "
            f"WHERE l.{quote_identifier(left_column)} IS NOT NULL LIMIT 1"
        )
        try:
            return self.execute(sql)["row_count"] == 1, None
        except Exception as exc:  # error is recorded as validation evidence
            return False, f"{type(exc).__name__}: {exc}"

    def composite_join_exists(self, edge: dict[str, Any]) -> tuple[bool, str | None]:
        predicates = " AND ".join(
            f"l.{quote_identifier(left)} = r.{quote_identifier(right)}"
            for left, right in zip(edge["left_columns"], edge["right_columns"])
        )
        sql = (
            f"SELECT 1 FROM {quote_identifier(edge['left_table'])} AS l "
            f"JOIN {quote_identifier(edge['right_table'])} AS r ON {predicates} LIMIT 1"
        )
        try:
            return self.execute(sql)["row_count"] == 1, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"


def validate_roles(catalog: dict[str, Any], blueprint: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    tables = table_map(catalog)
    evidence = []
    errors = []
    for role in blueprint["roles"]:
        selected = role["selected"]
        if role["kind"] == "table":
            ranked = rank_tables(catalog, role["query"])
            rank = selected_rank(ranked, "table", selected)
            if rank is None:
                errors.append(f"table role {role['role']} selected missing table {selected}")
            elif rank > 10:
                errors.append(
                    f"table role {role['role']} selected rank {rank} is outside constrained top-10 candidates"
                )
            evidence.append(
                {
                    **role,
                    "selected_rank": rank,
                    "selected_score": ranked[rank - 1]["score"] if rank else None,
                    "top_catalog_candidates": ranked[:5],
                }
            )
            continue
        table = tables.get(normalize_identifier(role["table"]))
        if table is None:
            errors.append(f"column role {role['role']} missing table {role['table']}")
            continue
        required_type = role.get("type")
        ranked = rank_columns(table, role["query"], required_type)
        rank = selected_rank(ranked, "column", selected)
        if rank is None:
            errors.append(
                f"column role {role['role']} missing/incompatible column {role['table']}.{selected}"
            )
        elif rank > 10:
            errors.append(
                f"column role {role['role']} selected rank {rank} is outside constrained top-10 candidates"
            )
        evidence.append(
            {
                **role,
                "selected_rank": rank,
                "selected_score": ranked[rank - 1]["score"] if rank else None,
                "top_catalog_candidates": ranked[:5],
            }
        )
    return evidence, errors


def validate_joins(
    catalog: dict[str, Any], database: ReadOnlySQLite, blueprint: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    tables = table_map(catalog)
    evidence = []
    errors = []
    for join in blueprint["joins"]:
        left_table, left_column, right_table, right_column = join
        left = tables.get(normalize_identifier(left_table))
        right = tables.get(normalize_identifier(right_table))
        if left is None or right is None:
            errors.append(f"join references missing table: {join}")
            continue
        left_col = column_map(left).get(normalize_identifier(left_column))
        right_col = column_map(right).get(normalize_identifier(right_column))
        if left_col is None or right_col is None:
            errors.append(f"join references missing column: {join}")
            continue
        compatible = left_col["type_family"] == right_col["type_family"] or {
            left_col["type_family"], right_col["type_family"]
        } <= {"text", "temporal"}
        edge = find_catalog_join(catalog, join)
        if edge and edge.get("composite"):
            executable, execution_error = database.composite_join_exists(edge)
        else:
            executable, execution_error = database.join_exists(join)
        if edge is None:
            errors.append(f"join is not present in the offline catalog graph: {join}")
        if not compatible or not executable:
            errors.append(f"join failed type/execution validation: {join}")
        evidence.append(
            {
                "left_table": left_table,
                "left_column": left_column,
                "right_table": right_table,
                "right_column": right_column,
                "type_compatible": compatible,
                "left_type": left_col["declared_type"],
                "right_type": right_col["declared_type"],
                "catalog_edge_found": edge is not None,
                "catalog_edge": edge,
                "live_join_executable": executable,
                "execution_error": execution_error,
            }
        )
    return evidence, errors


def validate_mutations(
    database: ReadOnlySQLite, blueprint: dict[str, Any], oracle: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence = []
    errors = []
    for mutation in blueprint["mutations"]:
        occurrences = blueprint["sql"].count(mutation["old"])
        if occurrences == 0:
            errors.append(f"mutation {mutation['id']} replacement target absent")
            continue
        mutated_sql = blueprint["sql"].replace(mutation["old"], mutation["new"])
        try:
            result = database.execute(mutated_sql)
            distinguishable = result["ordered_result_sha256"] != oracle["ordered_result_sha256"]
            if not distinguishable:
                errors.append(f"mutation {mutation['id']} was not distinguished")
            evidence.append(
                {
                    "id": mutation["id"],
                    "category": mutation["category"],
                    "replacement_occurrences": occurrences,
                    "execution_status": result["status"],
                    "row_count": result["row_count"],
                    "ordered_result_sha256": result["ordered_result_sha256"],
                    "distinguished_from_oracle": distinguishable,
                }
            )
        except Exception as exc:
            errors.append(f"mutation {mutation['id']} did not execute: {type(exc).__name__}: {exc}")
            evidence.append(
                {
                    "id": mutation["id"],
                    "category": mutation["category"],
                    "replacement_occurrences": occurrences,
                    "execution_status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "distinguished_from_oracle": False,
                }
            )
    return evidence, errors


def make_task_ir(blueprint: dict[str, Any]) -> dict[str, Any]:
    features = profile_sql(blueprint["sql"])
    families = advanced_families(features)
    return {
        "source": {
            "kind": "reusable_operator_template",
            "source_example_id": None,
            "note": "No Spider 2.0 evaluation question or gold SQL was copied.",
        },
        "template_id": blueprint["template_id"],
        "relation_structure": {
            "table_roles": [role["role"] for role in blueprint["roles"] if role["kind"] == "table"],
            "join_count": len(blueprint["joins"]),
        },
        "operators": blueprint["operators"],
        "output": {
            "columns": blueprint["expected_columns"],
            "row_bounds": blueprint["row_bounds"],
            "order_sensitive": True,
        },
        "difficulty": {
            "sql_join_count": len(blueprint["joins"]),
            "operator_count": len(blueprint["operators"]),
            "level": (
                f"spider2_{blueprint['difficulty_band']}"
                if blueprint.get("difficulty_band")
                else "spider2_hard" if blueprint.get("complexity_target") else (
                "hard" if len(blueprint["joins"]) >= 3 or len(blueprint["operators"]) >= 6 else "medium"
                )
            ),
            "measured_sql_features": features,
            "measured_advanced_families": families,
            "declared_advanced_families": blueprint.get("advanced_families", []),
            "reference_target": blueprint.get("complexity_target"),
        },
        "semantic_risks": blueprint.get("semantic_risks", []),
    }


def describe_table_observation(table: dict[str, Any]) -> str:
    lines = [
        f"Table: {table['name']}",
        f"Rows: {table.get('row_count')} ({table.get('row_count_status')})",
        "Columns:",
    ]
    for column in table["columns"]:
        distribution = column["distribution"]
        description = f" — {column['description']}" if column.get("description") else ""
        lines.append(
            f"- {column['name']} {column['declared_type']} "
            f"tags={','.join(column.get('semantic_tags', []))} "
            f"null={distribution.get('null_fraction')} distinct_sample={distribution.get('sample_distinct_count')}"
            f"{description}"
        )
    lines.append("Sample rows:")
    lines.append(json.dumps(table.get("sample_rows", [])[:3], ensure_ascii=False, default=str))
    return "\n".join(lines)[:18_000]


def make_reference_trajectory(
    blueprint: dict[str, Any], catalog: dict[str, Any], grounding: dict[str, Any], oracle: dict[str, Any]
) -> dict[str, Any]:
    tables = table_map(catalog)
    selected_tables = []
    for role in blueprint["roles"]:
        if role["kind"] == "table" and normalize_identifier(role["selected"]) not in {
            normalize_identifier(value) for value in selected_tables
        }:
            selected_tables.append(role["selected"])
    query = " ".join(role["query"] for role in blueprint["roles"])
    top_tables = rank_tables(catalog, query)[:8]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Solve one SQLite analytics task using only search_schema, list_tables, describe_table, "
                "dialect_notes, and execute_sql. Test the final read-only query before answering."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Database id: {blueprint['database_id']}\nDialect: SQLite\nQuestion: {blueprint['instruction']}"
            ),
        },
    ]

    call_id = f"{blueprint['sample_id']}_search"
    messages.append(
        {
            "role": "assistant",
            "content": "I will locate the relevant relations in the offline execution-grounded catalog.",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "search_schema", "arguments": json.dumps({"query": query, "limit": 8})},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "search_schema",
            "content": json.dumps(top_tables, ensure_ascii=False),
        }
    )
    for index, table_name in enumerate(selected_tables, 1):
        table = tables[normalize_identifier(table_name)]
        call_id = f"{blueprint['sample_id']}_describe_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": f"I need types, value distributions, and join keys for {table_name}.",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "describe_table",
                            "arguments": json.dumps({"table_name": table_name}),
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": "describe_table",
                "content": describe_table_observation(table),
            }
        )
    call_id = f"{blueprint['sample_id']}_dialect"
    messages.append(
        {
            "role": "assistant",
            "content": "The roles and executable join path are grounded; I will confirm SQLite date and JSON behavior.",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "dialect_notes", "arguments": "{}"},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "dialect_notes",
            "content": (
                "SQLite: use julianday()/datetime() for temporal calculations, json_extract() for JSON text, "
                "NULLIF for protected division, and modern SQLite window/aggregate functions."
            ),
        }
    )
    call_id = f"{blueprint['sample_id']}_execute"
    messages.append(
        {
            "role": "assistant",
            "content": "I will execute the complete candidate and check its columns, cardinality, and values.",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "execute_sql", "arguments": json.dumps({"query": blueprint["sql"], "max_rows": 100})},
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "execute_sql",
            "content": json.dumps(
                {
                    "status": "success",
                    "columns": oracle["columns"],
                    "row_count": oracle["row_count"],
                    "preview": oracle["preview"],
                    "ordered_result_sha256": oracle["ordered_result_sha256"],
                },
                ensure_ascii=False,
            ),
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": f"FINAL ANSWER:\n```sql\n{blueprint['sql'].rstrip()}\n```",
        }
    )
    return {
        "trajectory_version": TRAJECTORY_VERSION,
        "sample_id": blueprint["sample_id"],
        "source": "deterministic_executed_oracle_trace",
        "teacher_model": None,
        "training_eligibility": "bootstrap_only_pending_real_teacher_rollout",
        "allowed_tools": ALLOWED_TOOLS,
        "grounding_hash": stable_hash(grounding),
        "messages": messages,
        "validation": {
            "tool_names_valid": True,
            "execute_before_final": True,
            "successful_execute": True,
            "oracle_result_sha256": oracle["ordered_result_sha256"],
        },
    }


def write_csv(path: Path, columns: list[str], rows: list[list[Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    path.write_text(buffer.getvalue())


def build_sample(
    blueprint: dict[str, Any], database_root: Path, catalog_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog, catalog_path = load_catalog(catalog_dir, blueprint["database_id"])
    database_path = discover_database(database_root, blueprint["database_id"])
    database = ReadOnlySQLite(database_path)
    role_evidence, role_errors = validate_roles(catalog, blueprint)
    join_evidence, join_errors = validate_joins(catalog, database, blueprint)
    oracle = database.execute(blueprint["sql"])
    errors = role_errors + join_errors
    if oracle["columns"] != blueprint["expected_columns"]:
        errors.append(
            f"expected columns {blueprint['expected_columns']}, got {oracle['columns']}"
        )
    if blueprint.get("difficulty_band"):
        instruction = str(blueprint.get("instruction") or "").casefold()
        hidden_aliases = [
            str(column)
            for column in blueprint.get("expected_columns", [])
            if str(column).casefold() not in instruction
        ]
        if hidden_aliases:
            errors.append(
                "alias-sensitive verifier requirement is hidden from the instruction: "
                + ", ".join(hidden_aliases)
            )
    low, high = blueprint["row_bounds"]
    if not low <= oracle["row_count"] <= high:
        errors.append(f"row count {oracle['row_count']} outside [{low}, {high}]")
    if oracle["row_count"] == 0:
        errors.append("oracle result is empty")
    if oracle["rows"] and len({canonical_json(row) for row in oracle["rows"]}) == 1 and oracle["row_count"] > 1:
        errors.append("oracle result is degenerate: every row is identical")
    mutation_evidence, mutation_errors = validate_mutations(database, blueprint, oracle)
    errors.extend(mutation_errors)
    sql_features = profile_sql(blueprint["sql"])
    measured_families = advanced_families(sql_features)
    target = blueprint.get("complexity_target")
    complexity_errors: list[str] = []
    if target:
        if sql_features["structural_score"] < float(target["minimum_structural_score"]):
            complexity_errors.append(
                f"structural score {sql_features['structural_score']} below "
                f"{target['minimum_structural_score']}"
            )
        if sql_features["sql_tokens"] < float(target["minimum_sql_tokens"]):
            complexity_errors.append(
                f"SQL tokens {sql_features['sql_tokens']} below {target['minimum_sql_tokens']}"
            )
        if (
            target.get("maximum_structural_score") is not None
            and sql_features["structural_score"] > float(target["maximum_structural_score"])
        ):
            complexity_errors.append(
                f"structural score {sql_features['structural_score']} above "
                f"{target['maximum_structural_score']}"
            )
        if (
            target.get("maximum_sql_tokens") is not None
            and sql_features["sql_tokens"] > float(target["maximum_sql_tokens"])
        ):
            complexity_errors.append(
                f"SQL tokens {sql_features['sql_tokens']} above {target['maximum_sql_tokens']}"
            )
        if len(measured_families) < int(target["minimum_advanced_families"]):
            complexity_errors.append(
                f"only {len(measured_families)} measured advanced families: {measured_families}"
            )
    errors.extend(complexity_errors)
    grounding = {
        "method": "catalog_constrained_search_then_live_execution",
        "catalog_file": str(catalog_path),
        "catalog_version": catalog["catalog_version"],
        "role_candidates": role_evidence,
        "join_path": join_evidence,
        "environment_is_final_judge": True,
    }
    checks = {
        "catalog_loaded": True,
        "all_roles_exist_and_type_match": not role_errors,
        "all_joins_type_compatible_and_executable": not join_errors,
        "read_only_sql": oracle["static_audit"]["read_only_head"] and oracle["static_audit"]["forbidden_keyword"] is None,
        "single_statement": oracle["static_audit"]["single_statement"],
        "all_joins_have_on_predicates": oracle["static_audit"]["all_joins_have_predicates"],
        "execution_success": oracle["status"] == "success",
        "expected_columns_exact": oracle["columns"] == blueprint["expected_columns"],
        "result_non_empty": oracle["row_count"] > 0,
        "row_count_in_calibrated_bounds": low <= oracle["row_count"] <= high,
        "all_mutations_execute_and_differ": not mutation_errors,
        "spider2_complexity_target": not complexity_errors,
    }
    quality_score = round(sum(bool(value) for value in checks.values()) / len(checks), 4)
    task_ir = make_task_ir(blueprint)
    sample = {
        "sample_version": SAMPLE_VERSION,
        "sample_id": blueprint["sample_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "environment": "Spider 2.0-Lite local SQLite",
            "database_id": blueprint["database_id"],
            "database_path": str(database_path),
            "database_sha256": catalog["database"]["sha256"],
            "generation_mode": blueprint.get("generation", {}).get(
                "mode", "SQL-first model-authored task with programmatic catalog constraints"
            ),
            "official_question_used": False,
            "official_gold_sql_used": False,
            "reference_sql_use": (
                "aggregate complexity profiling only; no reference SQL text used for generation"
            ),
        },
        "task_ir": task_ir,
        "instruction_environment_verifier": {
            "instruction": blueprint["instruction"],
            "environment": {
                "backend": "sqlite",
                "database_id": blueprint["database_id"],
                "dialect": "SQLite",
                "catalog_file": str(catalog_path),
                "allowed_tools": ALLOWED_TOOLS,
                "read_only": True,
            },
            "verifier": {
                "kind": "execution_result_and_mutation",
                "order_sensitive": True,
                "expected_columns": blueprint["expected_columns"],
                "row_bounds": blueprint["row_bounds"],
                "ordered_result_sha256": oracle["ordered_result_sha256"],
                "unordered_result_sha256": oracle["unordered_result_sha256"],
                "mutation_count": len(mutation_evidence),
            },
        },
        "grounding": grounding,
        "oracle": {
            "dialect": "sqlite",
            "sql": blueprint["sql"].rstrip().rstrip(";").rstrip() + ";",
            "result": oracle,
        },
        "calibration": {
            "source": "live SQLite execution",
            "observed_row_count": oracle["row_count"],
            "accepted_row_bounds": blueprint["row_bounds"],
            "runtime_ms": oracle["elapsed_ms"],
            "non_null_output_columns": [name for name, count in oracle["null_counts"].items() if count < oracle["row_count"]],
        },
        "complexity_calibration": {
            "profile": sql_features,
            "measured_advanced_families": measured_families,
            "target": target,
            "errors": complexity_errors,
        },
        "mutation_tests": mutation_evidence,
        "validation": {
            "status": "passed" if not errors else "failed",
            "quality_score": quality_score,
            "quality_score_scope": "completed_automated_gates_only",
            "checks": checks,
            "errors": errors,
            "not_run": [
                "real_teacher_rollout",
                "independent_model_resolve",
                "independent_model_semantic_judge",
                "multi_seed_solvability",
            ],
        },
    }
    sample["content_sha256"] = stable_hash(
        {
            "task_ir": task_ir,
            "instruction": blueprint["instruction"],
            "database_sha256": catalog["database"]["sha256"],
            "sql": blueprint["sql"],
            "result": oracle["ordered_result_sha256"],
        }
    )
    trajectory = make_reference_trajectory(blueprint, catalog, grounding, oracle)
    return sample, trajectory


def load_blueprints(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text())
        rows = value.get("blueprints", []) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"blueprint file must contain a JSON array or JSONL objects: {path}")
    required = {
        "sample_id", "database_id", "template_id", "instruction", "sql", "expected_columns",
        "row_bounds", "operators", "roles", "joins", "mutations",
    }
    for index, row in enumerate(rows, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"blueprint {index} missing fields: {missing}")
    return rows


def generate_pilot(
    database_root: Path,
    catalog_dir: Path,
    output_dir: Path,
    count: int = 10,
    blueprints: list[dict[str, Any]] | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    available = blueprints or HARD_PILOT_BLUEPRINTS
    if count < 1 or count > len(available):
        raise ValueError(f"count must be between 1 and {len(available)}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    sql_dir = output_dir / "sql"
    result_dir = output_dir / "results"
    sql_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    selected = available[:count]
    if workers == 1:
        built = [build_sample(blueprint, database_root, catalog_dir) for blueprint in selected]
    else:
        # Each build opens an independent read-only SQLite connection, so the
        # expensive oracle/mutation executions are safe to parallelize.  map()
        # preserves blueprint order and therefore keeps artifacts deterministic.
        def parallel_build(blueprint: dict[str, Any]) -> tuple[Any, Exception | None]:
            try:
                return build_sample(blueprint, database_root, catalog_dir), None
            except Exception as exc:
                return None, exc

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            parallel_results = list(executor.map(parallel_build, selected))
        built = []
        for blueprint, (result, exc) in zip(selected, parallel_results):
            should_retry = exc is not None
            if result is not None:
                sample, _ = result
                errors = [str(value).casefold() for value in sample["validation"].get("errors", [])]
                should_retry = any("operationalerror: interrupted" in value for value in errors)
            if should_retry:
                # Complex SQLite mutations can trip their per-query progress
                # timeout when many independent validations saturate the CPU.
                # Replay only that candidate serially before declaring it bad.
                result = build_sample(blueprint, database_root, catalog_dir)
            built.append(result)
    samples = [sample for sample, _ in built]
    trajectories = [trajectory for _, trajectory in built]
    for sample in samples:
        (sql_dir / f"{sample['sample_id']}.sql").write_text(sample["oracle"]["sql"] + "\n")
        result = sample["oracle"]["result"]
        write_csv(result_dir / f"{sample['sample_id']}.csv", result["columns"], result["rows"])

    task_jsonl = "\n".join(json.dumps(sample, ensure_ascii=False, allow_nan=False) for sample in samples) + "\n"
    trajectory_jsonl = "\n".join(
        json.dumps(trajectory, ensure_ascii=False, allow_nan=False) for trajectory in trajectories
    ) + "\n"
    (output_dir / "pilot_tasks.jsonl").write_text(task_jsonl)
    (output_dir / "pilot_trajectories.bootstrap.jsonl").write_text(trajectory_jsonl)
    (output_dir / "pilot_tasks.pretty.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    passed = sum(sample["validation"]["status"] == "passed" for sample in samples)
    manifest = {
        "sample_version": SAMPLE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requested_count": count,
        "passed_count": passed,
        "failed_count": count - passed,
        "all_passed": passed == count,
        "catalog_directory": str(catalog_dir.resolve()),
        "task_file": "pilot_tasks.jsonl",
        "human_review_file": "SAMPLE_REVIEW.md",
        "bootstrap_trajectory_file": "pilot_trajectories.bootstrap.jsonl",
        "bootstrap_trajectory_warning": (
            "These traces are deterministic executed oracle bootstraps, not sampled teacher-model rollouts; "
            "do not mix them into the final teacher SFT split without an explicit ablation label."
        ),
        "validation_scope": {
            "completed": [
                "catalog grounding",
                "read-only static audit",
                "live execution",
                "result shape and non-degeneracy",
                "controlled mutation tests",
                "deterministic replay",
            ],
            "not_run_without_teacher_endpoint": [
                "real Codex CLI teacher rollout",
                "independent model re-solve",
                "independent model question-program semantic judge",
                "multi-model multi-seed solvability rate",
            ],
            "score_definition": "quality_score is the pass fraction of completed automated gates only",
        },
        "samples": [
            {
                "sample_id": sample["sample_id"],
                "database_id": sample["provenance"]["database_id"],
                "template_id": sample["task_ir"]["template_id"],
                "status": sample["validation"]["status"],
                "quality_score": sample["validation"]["quality_score"],
                "row_count": sample["oracle"]["result"]["row_count"],
                "runtime_ms": sample["oracle"]["result"]["elapsed_ms"],
                "mutations_distinguished": sum(
                    mutation["distinguished_from_oracle"] for mutation in sample["mutation_tests"]
                ),
                "structural_score": sample["complexity_calibration"]["profile"]["structural_score"],
                "sql_tokens": sample["complexity_calibration"]["profile"]["sql_tokens"],
                "advanced_family_count": len(
                    sample["complexity_calibration"]["measured_advanced_families"]
                ),
            }
            for sample in samples
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    write_report(output_dir / "QUALITY_REPORT.md", samples, manifest)
    write_sample_review(output_dir / "SAMPLE_REVIEW.md", samples)
    if passed != count:
        failures = {
            sample["sample_id"]: sample["validation"]["errors"]
            for sample in samples
            if sample["validation"]["status"] != "passed"
        }
        raise RuntimeError(f"{count - passed} pilot samples failed validation: {failures}")
    return samples


def write_report(path: Path, samples: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# SQLite pilot quality report",
        "",
        f"Automated gate result: **{manifest['passed_count']}/{manifest['requested_count']} passed**.",
        "",
        "This is the completed catalog/execution/mutation gate, not the full proposal Stage III acceptance. "
        "Real teacher rollout and independent-model re-solving are separate post-generation stages and are not "
        "claimed by this report.",
        "",
        "| Sample | Database | Template | Rows | Runtime ms | SQL score/tokens | Families | Mutations | Gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in samples:
        lines.append(
            "| {id} | {db} | {template} | {rows} | {runtime} | {structural}/{tokens} | {families} | {mutations}/{total} | {score:.2f} |".format(
                id=sample["sample_id"],
                db=sample["provenance"]["database_id"],
                template=sample["task_ir"]["template_id"],
                rows=sample["oracle"]["result"]["row_count"],
                runtime=sample["oracle"]["result"]["elapsed_ms"],
                mutations=sum(item["distinguished_from_oracle"] for item in sample["mutation_tests"]),
                total=len(sample["mutation_tests"]),
                score=sample["validation"]["quality_score"],
                structural=sample["complexity_calibration"]["profile"]["structural_score"],
                tokens=sample["complexity_calibration"]["profile"]["sql_tokens"],
                families=len(sample["complexity_calibration"]["measured_advanced_families"]),
            )
        )
    lines.extend(
        [
            "",
            "Checks applied to every sample:",
            "",
            "- catalog-constrained table/column role validation;",
            "- type-compatible, live-executable joins;",
            "- read-only/single-statement SQL audit and SQLite authorizer;",
            "- exact output-column and calibrated row-count checks;",
            "- result hashes and previews from full query execution;",
            "- three executable controlled mutations that must change the ordered result.",
            "",
            "> The bootstrap trajectories are deterministic oracle traces. They demonstrate the target harness "
            "format and are replay-grounded, but are deliberately labeled as not being real teacher-model rollouts.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def markdown_cell(value: Any) -> str:
    return str(value if value is not None else "NULL").replace("|", "\\|").replace("\n", " ")


def write_sample_review(path: Path, samples: list[dict[str, Any]]) -> None:
    lines = [
        "# SQLite pilot samples for review",
        "",
        "以下每条都包含自然语言任务、实际执行过的 SQLite oracle 和结果预览。完整验证证据见 "
        "`pilot_tasks.pretty.json`，完整结果见 `results/`。",
        "",
    ]
    for sample in samples:
        result = sample["oracle"]["result"]
        lines.extend(
            [
                f"## {sample['sample_id']} — {sample['provenance']['database_id']}",
                "",
                sample["instruction_environment_verifier"]["instruction"],
                "",
                "```sql",
                sample["oracle"]["sql"].rstrip(),
                "```",
                "",
                f"执行结果：{result['row_count']} 行；自动 gate score：{sample['validation']['quality_score']:.2f}；"
                f"mutation：{sum(item['distinguished_from_oracle'] for item in sample['mutation_tests'])}/"
                f"{len(sample['mutation_tests'])}。",
                "",
            ]
        )
        preview = result["preview"]
        if preview:
            columns = result["columns"]
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("|" + "|".join("---" for _ in columns) + "|")
            for row in preview[:10]:
                lines.append("| " + " | ".join(markdown_cell(row.get(column)) for column in columns) + " |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def verify_existing(task_file: Path, database_root: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in task_file.read_text().splitlines() if line.strip()]
    results = []
    for sample in rows:
        database_path = discover_database(database_root, sample["provenance"]["database_id"])
        actual = ReadOnlySQLite(database_path).execute(sample["oracle"]["sql"])
        expected = sample["instruction_environment_verifier"]["verifier"]
        passed = (
            actual["ordered_result_sha256"] == expected["ordered_result_sha256"]
            and actual["columns"] == expected["expected_columns"]
        )
        results.append(
            {
                "sample_id": sample["sample_id"],
                "passed": passed,
                "expected_hash": expected["ordered_result_sha256"],
                "actual_hash": actual["ordered_result_sha256"],
            }
        )
    return {"count": len(results), "passed": sum(item["passed"] for item in results), "results": results}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_database_root, default_metadata_root, default_catalog_dir = default_paths(root)
    default_artifacts = root / "artifacts" / "data_synthesis" / "sqlite"
    parser = argparse.ArgumentParser(description="Spider 2.0 SQLite harness-aware data synthesis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog", help="build offline catalogs")
    catalog_parser.add_argument("--database-root", type=Path, default=default_database_root)
    catalog_parser.add_argument("--metadata-root", type=Path, default=default_metadata_root)
    catalog_parser.add_argument("--catalog-dir", type=Path, default=default_catalog_dir)
    catalog_parser.add_argument("--databases", default="")
    catalog_parser.add_argument("--profile-rows", type=int, default=256)
    catalog_parser.add_argument("--query-timeout", type=float, default=4.0)
    catalog_parser.add_argument("--join-timeout", type=float, default=3.0)
    catalog_parser.add_argument("--no-hash", action="store_true")

    pilot_parser = subparsers.add_parser("pilot", help="generate and validate pilot samples")
    pilot_parser.add_argument("--database-root", type=Path, default=default_database_root)
    pilot_parser.add_argument("--catalog-dir", type=Path, default=default_catalog_dir)
    pilot_parser.add_argument("--output-dir", type=Path, default=default_artifacts / "hard_pilot_10")
    pilot_parser.add_argument("--count", type=int, default=10)
    pilot_parser.add_argument("--workers", type=int, default=1)
    pilot_parser.add_argument(
        "--blueprints-file",
        type=Path,
        default=None,
        help="optional JSON/JSONL candidate mappings using the pilot blueprint schema",
    )
    pilot_parser.add_argument(
        "--blueprint-set",
        choices=("hard", "curriculum"),
        default="hard",
        help="built-in blueprint set used when --blueprints-file is omitted",
    )

    all_parser = subparsers.add_parser("all", help="build every catalog, then generate pilot samples")
    all_parser.add_argument("--database-root", type=Path, default=default_database_root)
    all_parser.add_argument("--metadata-root", type=Path, default=default_metadata_root)
    all_parser.add_argument("--catalog-dir", type=Path, default=default_catalog_dir)
    all_parser.add_argument("--output-dir", type=Path, default=default_artifacts / "hard_pilot_10")
    all_parser.add_argument("--count", type=int, default=10)
    all_parser.add_argument(
        "--blueprint-set",
        choices=("hard", "curriculum"),
        default="hard",
        help="built-in blueprint set to validate after catalog construction",
    )
    all_parser.add_argument("--profile-rows", type=int, default=256)
    all_parser.add_argument("--reuse-catalogs", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="replay an existing task JSONL")
    verify_parser.add_argument("--database-root", type=Path, default=default_database_root)
    verify_parser.add_argument(
        "--task-file", type=Path, default=default_artifacts / "hard_pilot_10" / "pilot_tasks.jsonl"
    )
    verify_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON path for the deterministic replay report",
    )

    args = parser.parse_args()
    if args.command == "catalog":
        options = CatalogOptions(
            profile_rows=args.profile_rows,
            query_timeout_seconds=args.query_timeout,
            join_timeout_seconds=args.join_timeout,
            hash_database=not args.no_hash,
        )
        selected = {value.strip() for value in args.databases.split(",") if value.strip()} or None
        catalogs = CatalogBuilder(args.database_root, args.metadata_root, args.catalog_dir, options).build_all(selected)
        print(f"built {len(catalogs)} catalogs in {args.catalog_dir}")
        return
    if args.command == "pilot":
        if args.blueprints_file:
            blueprints = load_blueprints(args.blueprints_file)
        else:
            blueprints = (
                CURRICULUM_PILOT_BLUEPRINTS
                if args.blueprint_set == "curriculum"
                else HARD_PILOT_BLUEPRINTS
            )
        samples = generate_pilot(
            args.database_root,
            args.catalog_dir,
            args.output_dir,
            args.count,
            blueprints=blueprints,
            workers=args.workers,
        )
        print(f"generated {len(samples)} validated samples in {args.output_dir}")
        return
    if args.command == "all":
        if not args.reuse_catalogs:
            catalogs = CatalogBuilder(
                args.database_root,
                args.metadata_root,
                args.catalog_dir,
                CatalogOptions(profile_rows=args.profile_rows),
            ).build_all()
            print(f"built {len(catalogs)} catalogs in {args.catalog_dir}")
        blueprints = (
            CURRICULUM_PILOT_BLUEPRINTS
            if args.blueprint_set == "curriculum"
            else HARD_PILOT_BLUEPRINTS
        )
        samples = generate_pilot(
            args.database_root,
            args.catalog_dir,
            args.output_dir,
            args.count,
            blueprints=blueprints,
        )
        print(f"generated {len(samples)} validated samples in {args.output_dir}")
        return
    report = verify_existing(args.task_file, args.database_root)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["passed"] != report["count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
