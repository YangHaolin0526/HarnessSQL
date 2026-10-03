#!/usr/bin/env python3
"""Export the Spider 2.0-DBT DuckDB corpus into Spider-style SQLite databases.

The `data_synthesis` pipeline only speaks SQLite. The dbt tasks ship DuckDB files,
so this script materialises every distinct dbt database as one `.sqlite` file plus a
Spider-shaped metadata directory, which `data_synthesis.pipeline catalog` can consume
through `--database-root` / `--metadata-root` without any code change.

What it does per database:

* de-duplicates the 64 tasks into distinct dbt projects (f1001/f1002/f1003 share one DB);
* flattens DuckDB schemas into SQLite's single namespace, keeping the dbt layer as a tag;
* materialises dbt views as tables (a DuckDB view body will not parse in SQLite);
* maps every DuckDB type onto a SQLite affinity explicitly, instead of letting the
  default `COPY FROM DATABASE` turn DECIMAL/HUGEINT into VARCHAR;
* proves single-column key uniqueness from the data and writes PRIMARY KEY / UNIQUE
  INDEX, because dbt-built DuckDB files carry no constraints at all and catalog v2
  needs a proven-unique side before it will trust a join edge;
* harvests column descriptions out of the dbt `schema.yml` files (the 30 Spider SQLite
  databases have none).

Foreign keys are deliberately NOT fabricated into the DDL: `catalog.py::_infer_joins`
already derives join edges from name evidence plus live execution, and it can do that
once the unique side is backed by a real index.

Usage:
    python export_sqlite.py --output-root ../../artifacts/data_synthesis/dbt_sqlite/databases
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import duckdb
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE / "data"
DEFAULT_OUTPUT_ROOT = HERE.parents[1] / "artifacts" / "data_synthesis" / "dbt_sqlite" / "databases"

DECIMAL_RE = re.compile(r"^DECIMAL\((\d+),\s*(\d+)\)$")
INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}
FLOAT_TYPES = {"FLOAT", "REAL", "DOUBLE"}
TEXT_TYPES = {"VARCHAR", "CHAR", "TEXT", "STRING", "UUID", "BIT"}
NESTED_MARKERS = ("STRUCT(", "MAP(", "UNION(", "[]")
INT64_MAX = 2**63 - 1

CATALOG_REF_RE = re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\.((?:main|"main")[A-Za-z0-9_]*|"main_[A-Za-z0-9_]+")\.')
KEYISH_RE = re.compile(r"(^id$|_id$|_ids$|^.*_key$|^key$|_sk$|_pk$|_guid$|_uuid$|^.*_code$|^code$|_number$)")
SMALL_TABLE_ALL_COLUMNS = 100_000
MAX_KEY_CANDIDATES = 32


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


# --------------------------------------------------------------------------------------
# type mapping
# --------------------------------------------------------------------------------------


def map_type(duck_type: str, wide_int: bool = False) -> tuple[str, str, str]:
    """Return (sqlite_declared_type, duckdb_cast_template, note).

    The cast template contains `{c}`, already-quoted column reference.
    `wide_int` says a pre-scan found values outside int64 for this column.
    """
    t = (duck_type or "").strip().upper()
    if t == "BOOLEAN":
        return "INTEGER", "CAST({c} AS INTEGER)", "boolean->0/1"
    if t in INT_TYPES:
        return "INTEGER", "{c}", ""
    if t in {"HUGEINT", "UHUGEINT"}:
        if wide_int:
            return "REAL", "CAST({c} AS DOUBLE)", "hugeint exceeds int64; stored as REAL"
        return "INTEGER", "CAST({c} AS BIGINT)", "hugeint->int64"
    if t in FLOAT_TYPES:
        return "REAL", "CAST({c} AS DOUBLE)", ""
    decimal = DECIMAL_RE.match(t)
    if decimal:
        scale = int(decimal.group(2))
        if scale == 0 and not wide_int:
            return "INTEGER", "CAST({c} AS BIGINT)", f"{t}->int64"
        return "REAL", "CAST({c} AS DOUBLE)", f"{t}->REAL (SQLite has no fixed-point type)"
    if t == "DATE":
        return "DATE", "strftime({c}, '%Y-%m-%d')", ""
    if t.startswith("TIMESTAMP WITH TIME ZONE") or t == "TIMESTAMPTZ":
        return "TIMESTAMP", "strftime({c} AT TIME ZONE 'UTC', '%Y-%m-%d %H:%M:%S')", "normalised to UTC"
    if t.startswith("TIMESTAMP") or t == "DATETIME":
        return "TIMESTAMP", "strftime({c}, '%Y-%m-%d %H:%M:%S')", ""
    if t.startswith("TIME"):
        # Declaring the SQLite column TIME makes DuckDB's sqlite writer parse the value as a
        # timestamp on INSERT, and a bare "00:00:00" fails that parse. TEXT keeps it verbatim.
        return "TEXT", "CAST({c} AS VARCHAR)", "time-of-day kept as TEXT"
    if t == "INTERVAL":
        return "TEXT", "CAST({c} AS VARCHAR)", "interval->text"
    if t in {"BLOB", "BYTEA", "VARBINARY"}:
        return "BLOB", "{c}", ""
    if t == "JSON" or any(marker in t for marker in NESTED_MARKERS):
        return "TEXT", "CAST(to_json({c}) AS VARCHAR)", "nested type serialised as JSON text"
    if t in TEXT_TYPES or t.startswith("VARCHAR") or t.startswith("ENUM"):
        return "TEXT", "{c}", ""
    return "TEXT", "CAST({c} AS VARCHAR)", f"unmapped duckdb type {t}; stored as TEXT"


def needs_width_check(duck_type: str) -> bool:
    t = (duck_type or "").strip().upper()
    if t in {"HUGEINT", "UHUGEINT"}:
        return True
    decimal = DECIMAL_RE.match(t)
    return bool(decimal and int(decimal.group(2)) == 0 and int(decimal.group(1)) > 18)


# --------------------------------------------------------------------------------------
# dbt documentation harvesting
# --------------------------------------------------------------------------------------

DOC_BLOCK_RE = re.compile(r"\{%\s*docs\s+([A-Za-z0-9_]+)\s*%\}(.*?)\{%\s*enddocs\s*%\}", re.S)
DOC_REF_RE = re.compile(r"\{\{\s*doc\(\s*[\"']([A-Za-z0-9_]+)[\"']\s*\)\s*\}\}")


def harvest_dbt_docs(workspace: Path) -> dict[str, dict[str, Any]]:
    """Map normalised dbt entity name -> {'description': str, 'columns': {norm_col: str}}."""
    blocks: dict[str, str] = {}
    yml_files: list[Path] = []
    for base, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in {".git", "target", "logs"}]
        for name in files:
            path = Path(base) / name
            if name.endswith(".md"):
                try:
                    text = path.read_text(errors="replace")
                except OSError:
                    continue
                for key, body in DOC_BLOCK_RE.findall(text):
                    blocks.setdefault(key, " ".join(body.split()))
            elif name.endswith((".yml", ".yaml")):
                yml_files.append(path)

    def resolve(text: Any) -> str:
        if not isinstance(text, str):
            return ""
        rendered = DOC_REF_RE.sub(lambda m: blocks.get(m.group(1), ""), text)
        if "{{" in rendered or "{%" in rendered:
            rendered = re.sub(r"\{\{.*?\}\}|\{%.*?%\}", "", rendered, flags=re.S)
        return " ".join(rendered.split())

    docs: dict[str, dict[str, Any]] = {}

    def absorb(entity: Any) -> None:
        if not isinstance(entity, dict) or not entity.get("name"):
            return
        key = normalize(str(entity["name"]))
        entry = docs.setdefault(key, {"description": "", "columns": {}})
        described = resolve(entity.get("description"))
        if described and not entry["description"]:
            entry["description"] = described
        for column in entity.get("columns") or []:
            if isinstance(column, dict) and column.get("name"):
                text = resolve(column.get("description"))
                column_key = normalize(str(column["name"]))
                if text and not entry["columns"].get(column_key):
                    entry["columns"][column_key] = text

    for path in yml_files:
        try:
            document = yaml.safe_load(path.read_text(errors="replace"))
        except Exception:
            continue
        if not isinstance(document, dict):
            continue
        for section in ("models", "seeds", "snapshots"):
            for entity in document.get(section) or []:
                absorb(entity)
        for source in document.get("sources") or []:
            if not isinstance(source, dict):
                continue
            for table in source.get("tables") or []:
                absorb(table)
    return docs


def lookup_doc(docs: dict[str, dict[str, Any]], table_name: str) -> dict[str, Any]:
    candidates = [table_name]
    if table_name.endswith("_data"):
        candidates.append(table_name[: -len("_data")])
    if "__" in table_name:
        candidates.append(table_name.split("__", 1)[1])
    for candidate in candidates:
        hit = docs.get(normalize(candidate))
        if hit:
            return hit
    return {"description": "", "columns": {}}


# --------------------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------------------


def discover_families(data_root: Path) -> dict[str, dict[str, Any]]:
    """Group the 64 tasks by their dbt project database, pick one representative each."""
    families: dict[str, dict[str, Any]] = {}
    for task_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        gold = task_dir / "tests" / "gold.duckdb"
        if not gold.exists():
            continue
        workspace_dbs = sorted((task_dir / "workspace").glob("*.duckdb"))
        if not workspace_dbs:
            continue
        database_id = workspace_dbs[0].stem
        entry = families.setdefault(database_id, {"database_id": database_id, "tasks": [], "candidates": []})
        entry["tasks"].append(task_dir.name)
        entry["candidates"].append((gold.stat().st_size, task_dir.name, gold, workspace_dbs[0]))
    for entry in families.values():
        entry["tasks"].sort()
        size, task, gold, workspace = max(entry["candidates"])
        entry.update(task=task, gold=gold, workspace=workspace, gold_bytes=size)
        del entry["candidates"]
    return families


def relation_plan(con: duckdb.DuckDBPyConnection, alias: str) -> list[dict[str, Any]]:
    rows = con.execute(
        f"""
        SELECT schema_name, table_name AS name, 'table' AS kind, estimated_size
          FROM duckdb_tables() WHERE database_name = '{alias}'
        UNION ALL
        SELECT schema_name, view_name AS name, 'view' AS kind, NULL
          FROM duckdb_views() WHERE database_name = '{alias}' AND NOT internal
        ORDER BY 1, 2
        """
    ).fetchall()
    plan = []
    for schema, name, kind, estimated in rows:
        suffix = schema[len("main_"):] if schema.startswith("main_") else schema
        if schema == "main":
            flat = name
        elif name.startswith(suffix):
            flat = name
        else:
            flat = f"{suffix}__{name}"
        plan.append({"schema": schema, "name": name, "kind": kind, "flat": flat, "estimated_size": estimated})
    seen: dict[str, dict[str, Any]] = {}
    for item in plan:
        clash = seen.get(item["flat"])
        if clash is None:
            seen[item["flat"]] = item
            continue
        loser = item if item["schema"] != "main" else clash
        winner = clash if loser is item else item
        seen[winner["flat"]] = winner
        loser["flat"] = f"{loser['schema']}__{loser['name']}"
        seen[loser["flat"]] = loser
    return sorted(plan, key=lambda item: item["flat"])


def classify_layer(flat: str, source_names: set[str]) -> str:
    if normalize(flat) in source_names:
        return "source"
    lowered = flat.casefold()
    if lowered.startswith("stg_"):
        return "staging"
    if lowered.startswith(("int_", "int__")):
        return "intermediate"
    if lowered.startswith(("dim_", "fct_", "fact_", "rpt_", "mart_")):
        return "mart"
    if lowered.endswith("_data") or lowered.endswith("_seed"):
        return "source"
    return "mart"


# --------------------------------------------------------------------------------------
# per-database export
# --------------------------------------------------------------------------------------


def export_database(entry: dict[str, Any], output_root: Path, detect_keys: bool = True) -> dict[str, Any]:
    database_id = entry["database_id"]
    target = output_root / f"{database_id}.sqlite"
    metadata_dir = output_root / "sqlite" / database_id
    if target.exists():
        target.unlink()
    metadata_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    con = duckdb.connect(config={"memory_limit": "24GB", "threads": 8})
    con.execute("INSTALL sqlite; LOAD sqlite; INSTALL json; LOAD json")
    con.execute(f"ATTACH '{entry['gold']}' AS gold (READ_ONLY)")
    con.execute(f"ATTACH '{entry['workspace']}' AS ws (READ_ONLY)")
    # dbt renders ref() as `<project_db>.<schema>.<table>`, so the views inside gold.duckdb
    # refer to gold under its original catalog name (tickit's is `redshift`). Re-attach the
    # same file under every alias its own view bodies expect, or count(*) cannot bind.
    aliases: set[str] = set()
    for (view_sql,) in con.execute(
        "SELECT sql FROM duckdb_views() WHERE database_name = 'gold' AND NOT internal"
    ).fetchall():
        aliases.update(name for name, _ in CATALOG_REF_RE.findall(view_sql or ""))
    aliases -= {"gold", "ws"}
    for alias in sorted(aliases):
        con.execute(f"ATTACH '{entry['gold']}' AS {quote(alias)} (READ_ONLY)")
    entry["view_catalog_aliases"] = sorted(aliases)
    source_names = {
        normalize(row[0])
        for row in con.execute("SELECT table_name FROM duckdb_tables() WHERE database_name = 'ws'").fetchall()
    }
    plan = relation_plan(con, "gold")

    # ---- resolve columns, widths and key candidates -------------------------------------
    for item in plan:
        columns = con.execute(
            """
            SELECT column_name, data_type, is_nullable
              FROM duckdb_columns()
             WHERE database_name = 'gold' AND schema_name = ? AND table_name = ?
             ORDER BY column_index
            """,
            [item["schema"], item["name"]],
        ).fetchall()
        item["columns"] = [{"name": c, "duck_type": t, "nullable": bool(n)} for c, t, n in columns]
        item["layer"] = classify_layer(item["flat"], source_names)

    notes: list[str] = []
    for item in plan:
        ref = f"gold.{quote(item['schema'])}.{quote(item['name'])}"
        wide_columns: list[str] = []
        checks = [column for column in item["columns"] if needs_width_check(column["duck_type"])]
        if checks:
            selects = ", ".join(
                f"max(abs(CAST({quote(c['name'])} AS HUGEINT)))" for c in checks
            )
            try:
                maxima = con.execute(f"SELECT {selects} FROM {ref}").fetchone()
            except duckdb.Error:
                maxima = [None] * len(checks)
            for column, value in zip(checks, maxima):
                if value is not None and int(value) > INT64_MAX:
                    wide_columns.append(column["name"])
                    notes.append(f"{item['flat']}.{column['name']}: value exceeds int64, exported as REAL")
        # SQLite REAL has no NaN, so a NaN silently lands as NULL. Count them in the same
        # pass as the row count and record it instead of losing the fact.
        floaty = [
            column for column in item["columns"]
            if (column["duck_type"] or "").strip().upper() in {"FLOAT", "REAL", "DOUBLE"}
        ]
        nan_selects = "".join(
            f", count_if(isnan({quote(c['name'])})) , count_if(isinf({quote(c['name'])}))"
            for c in floaty
        )
        counts = con.execute(f"SELECT count(*){nan_selects} FROM {ref}").fetchone()
        item["row_count"] = int(counts[0])
        for offset, column in enumerate(floaty):
            nan_count, inf_count = int(counts[1 + 2 * offset] or 0), int(counts[2 + 2 * offset] or 0)
            if nan_count:
                notes.append(f"{item['flat']}.{column['name']}: {nan_count} NaN value(s) become NULL in SQLite")
            if inf_count:
                notes.append(f"{item['flat']}.{column['name']}: {inf_count} infinite value(s)")
        for column in item["columns"]:
            declared, cast, note = map_type(column["duck_type"], column["name"] in wide_columns)
            column["sqlite_type"] = declared
            column["cast"] = cast.format(c=quote(column["name"]))
            if note:
                column["note"] = note

    # ---- data-proven single-column keys -------------------------------------------------
    if detect_keys:
        for item in plan:
            item["unique_columns"] = []
            if item["row_count"] == 0:
                continue
            candidates = [
                column
                for column in item["columns"]
                if column["sqlite_type"] in {"INTEGER", "TEXT", "DATE", "TIMESTAMP"}
                and (item["row_count"] <= SMALL_TABLE_ALL_COLUMNS or KEYISH_RE.search(column["name"].casefold()))
            ][:MAX_KEY_CANDIDATES]
            if not candidates:
                continue
            ref = f"gold.{quote(item['schema'])}.{quote(item['name'])}"
            parts = []
            for index, column in enumerate(candidates):
                quoted = quote(column["name"])
                parts.append(f"count({quoted}) AS nn_{index}")
                parts.append(f"count(DISTINCT {quoted}) AS nd_{index}")
            try:
                stats = con.execute(f"SELECT {', '.join(parts)} FROM {ref}").fetchone()
            except duckdb.Error as exc:
                notes.append(f"{item['flat']}: key scan failed ({type(exc).__name__})")
                continue
            for index, column in enumerate(candidates):
                non_null, distinct = int(stats[2 * index] or 0), int(stats[2 * index + 1] or 0)
                if non_null == item["row_count"] and distinct == item["row_count"] and item["row_count"] > 1:
                    item["unique_columns"].append(column["name"])
    else:
        for item in plan:
            item["unique_columns"] = []

    # Uniqueness alone is not keyness: in a 3-row lookup table almost every column is
    # distinct. Only key-named columns are promoted into the DDL; the rest stay in the
    # sidecar as observations so nothing is lost and nothing is overclaimed.
    for item in plan:
        item["key_columns"] = [
            column for column in item["unique_columns"] if KEYISH_RE.search(column.casefold())
        ]
        primary = None
        for column in item["key_columns"]:
            lowered = column.casefold()
            score = (
                lowered == "id",
                lowered == f"{item['flat'].casefold()}_id",
                lowered.endswith("_id"),
                lowered.endswith(("_key", "_sk", "_pk")),
            )
            if primary is None or score > primary[1]:
                primary = (column, score)
        item["primary_key"] = primary[0] if primary else None

    # ---- create and fill ----------------------------------------------------------------
    connection = sqlite3.connect(target)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    for item in plan:
        column_sql = []
        for column in item["columns"]:
            piece = f"  {quote(column['name'])} {column['sqlite_type']}"
            if not column["nullable"]:
                piece += " NOT NULL"
            column_sql.append(piece)
        if item["primary_key"]:
            column_sql.append(f"  PRIMARY KEY ({quote(item['primary_key'])})")
        ddl = f"CREATE TABLE {quote(item['flat'])} (\n" + ",\n".join(column_sql) + "\n)"
        item["ddl"] = ddl
        connection.execute(ddl)
    connection.commit()
    connection.close()

    con.execute(f"ATTACH '{target}' AS out (TYPE SQLITE)")
    failures = []
    for item in plan:
        ref = f"gold.{quote(item['schema'])}.{quote(item['name'])}"
        projection = ", ".join(f"{column['cast']} AS {quote(column['name'])}" for column in item["columns"])
        try:
            con.execute(f"INSERT INTO out.{quote(item['flat'])} SELECT {projection} FROM {ref}")
        except duckdb.Error as exc:
            failures.append({"table": item["flat"], "error": f"{type(exc).__name__}: {exc}"[:400]})
    con.execute("DETACH out")

    # ---- indexes + metadata sidecars ----------------------------------------------------
    docs = harvest_dbt_docs(entry["workspace"].parent)
    connection = sqlite3.connect(target)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    index_count = 0
    for item in plan:
        for column in item["key_columns"]:
            if column == item["primary_key"]:
                continue
            name = f"ux_{item['flat']}_{column}"[:60]
            try:
                connection.execute(
                    f"CREATE UNIQUE INDEX {quote(name)} ON {quote(item['flat'])} ({quote(column)})"
                )
                index_count += 1
            except sqlite3.Error:
                pass
    connection.commit()

    connection.row_factory = sqlite3.Row
    described_tables = 0
    for item in plan:
        doc = lookup_doc(docs, item["flat"])
        names = [column["name"] for column in item["columns"]]
        descriptions = [doc["columns"].get(normalize(name), "") for name in names]
        if any(descriptions):
            described_tables += 1
        try:
            sample = [dict(row) for row in connection.execute(
                f"SELECT * FROM {quote(item['flat'])} LIMIT 5"
            ).fetchall()]
        except sqlite3.Error:
            sample = []
        payload = {
            "sample_rows": sample,
            "table_name": item["flat"],
            "table_fullname": f"{database_id}.{item['schema']}.{item['name']}",
            "column_names": names,
            "column_types": [column["sqlite_type"] for column in item["columns"]],
            "description": descriptions,
            "table_description": doc["description"],
            "dbt": {
                "layer": item["layer"],
                "duckdb_schema": item["schema"],
                "duckdb_object": item["name"],
                "duckdb_kind": item["kind"],
                "duckdb_column_types": [column["duck_type"] for column in item["columns"]],
                "primary_key_inferred": item["primary_key"],
                "key_columns_inferred": item["key_columns"],
                "unique_columns_observed": item["unique_columns"],
                "row_count": item["row_count"],
            },
        }
        (metadata_dir / f"{item['flat']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n"
        )
    connection.execute("ANALYZE")
    connection.commit()
    connection.close()
    con.close()

    exported_rows = sum(item["row_count"] for item in plan)
    return {
        "database_id": database_id,
        "tasks": entry["tasks"],
        "view_catalog_aliases": entry.get("view_catalog_aliases", []),
        "source_task": entry["task"],
        "sqlite_path": str(target),
        "sqlite_bytes": target.stat().st_size,
        "gold_duckdb_bytes": entry["gold_bytes"],
        "table_count": len(plan),
        "view_count": sum(1 for item in plan if item["kind"] == "view"),
        "row_count": exported_rows,
        "column_count": sum(len(item["columns"]) for item in plan),
        "primary_keys": sum(1 for item in plan if item["primary_key"]),
        "unique_indexes": index_count,
        "tables_with_descriptions": described_tables,
        "layers": {
            layer: sum(1 for item in plan if item["layer"] == layer)
            for layer in ("source", "staging", "intermediate", "mart")
        },
        "failures": failures,
        "notes": notes[:50],
        "elapsed_seconds": round(time.time() - started, 2),
        "tables": [
            {
                "name": item["flat"],
                "layer": item["layer"],
                "rows": item["row_count"],
                "columns": len(item["columns"]),
                "primary_key": item["primary_key"],
                "key_columns": item["key_columns"],
                "unique_columns": item["unique_columns"],
            }
            for item in plan
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--databases", default="", help="comma-separated database ids; empty means all")
    parser.add_argument("--no-keys", action="store_true", help="skip data-proven key detection")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    families = discover_families(args.data_root.resolve())
    wanted = {value.strip() for value in args.databases.split(",") if value.strip()}
    selected = [entry for key, entry in sorted(families.items()) if not wanted or key in wanted]
    print(f"{len(families)} distinct dbt databases discovered; exporting {len(selected)}", flush=True)

    results = []
    for index, entry in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {entry['database_id']} (from {entry['task']})", flush=True)
        try:
            summary = export_database(entry, output_root, detect_keys=not args.no_keys)
        except Exception as exc:  # noqa: BLE001 - one bad database must not sink the batch
            traceback.print_exc()
            results.append({"database_id": entry["database_id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.append(summary)
        print(
            f"    tables={summary['table_count']} rows={summary['row_count']:,} "
            f"pk={summary['primary_keys']} uidx={summary['unique_indexes']} "
            f"described={summary['tables_with_descriptions']} "
            f"size={summary['sqlite_bytes'] / 1e6:.1f}MB in {summary['elapsed_seconds']}s"
            + (f" FAILURES={len(summary['failures'])}" if summary["failures"] else ""),
            flush=True,
        )

    manifest_path = args.manifest or (output_root / "export_manifest.json")
    ok = [item for item in results if "error" not in item]
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_root": str(args.data_root.resolve()),
        "output_root": str(output_root),
        "database_count": len(ok),
        "table_count": sum(item["table_count"] for item in ok),
        "column_count": sum(item["column_count"] for item in ok),
        "row_count": sum(item["row_count"] for item in ok),
        "table_failures": sum(len(item["failures"]) for item in ok),
        "databases": results,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(
        f"\nwrote {len(ok)} sqlite databases, {manifest['table_count']} tables, "
        f"{manifest['row_count']:,} rows, {manifest['table_failures']} failed tables -> {manifest_path}"
    )
    if len(ok) != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
