#!/usr/bin/env python3
"""One isolated BigQuery/schema operation for the dsh SQL plugin."""

from __future__ import annotations

import base64
import datetime as dt
import decimal
import json
import math
import re
import sys
from pathlib import Path


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str))


def clip_text(value, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def clip_cell(value, limit: int):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) <= 2**53 - 1 else str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (decimal.Decimal, dt.date, dt.time, dt.datetime)):
        return clip_text(value, limit)
    if isinstance(value, bytes):
        return clip_text(base64.b64encode(value).decode("ascii"), limit)
    if isinstance(value, (list, tuple, dict)):
        return clip_text(json.dumps(value, ensure_ascii=False, default=str), limit)
    return clip_text(value, limit)


def schema_entries(schema_dir: str) -> list[dict]:
    root = Path(schema_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"schema directory does not exist: {root}")
    entries = []
    for path in sorted(root.rglob("*.json")):
        try:
            item = json.loads(path.read_text(errors="replace"))
        except Exception:
            continue
        if isinstance(item, dict) and item.get("table_fullname") and item.get("column_names"):
            item["_path"] = str(path)
            entries.append(item)
    return entries


def shard_key(fullname: str):
    match = re.match(r"^(.*_)(\d{8})$", fullname)
    return (match.group(1) + "*", match.group(2)) if match else (fullname, None)


def list_tables(payload: dict) -> None:
    grouped: dict[str, list[tuple[str | None, dict]]] = {}
    for item in schema_entries(payload["schemaDir"]):
        key, suffix = shard_key(item["table_fullname"])
        grouped.setdefault(key, []).append((suffix, item))
    lines = []
    for key, members in sorted(grouped.items()):
        suffixes = sorted(s for s, _ in members if s)
        if suffixes:
            lines.append(f"TABLE `{key}` ({len(members)} date-sharded tables, suffix {suffixes[0]}..{suffixes[-1]})")
        else:
            lines.append(f"TABLE `{key}`")
    emit({"ok": True, "text": f"{len(lines)} table objects/families in this task database:\n" + "\n".join(lines)})


def matches_table(item: dict, requested: str) -> bool:
    requested = requested.strip().strip("`")
    full = item["table_fullname"]
    short = item.get("table_name", full.rsplit(".", 1)[-1])
    if requested.endswith("*"):
        return full.startswith(requested[:-1])
    return requested in {full, short} or full.endswith("." + requested)


def schema_block(requested: str, matches: list[dict], sample_rows: int, max_cell: int) -> str:
    representative = matches[0]
    full = representative["table_fullname"]
    columns = representative.get("column_names", [])
    types = representative.get("column_types", [])
    descriptions = representative.get("description", []) or []
    defs = []
    for index, name in enumerate(columns):
        typ = types[index] if index < len(types) else "UNKNOWN"
        desc = descriptions[index] if index < len(descriptions) else None
        suffix = f" -- {clip_text(desc, 240)}" if desc else ""
        defs.append(f"  `{name}` {typ}{suffix}")
    family_note = ""
    if len(matches) > 1:
        family_note = f"-- `{requested}` is a wildcard family with {len(matches)} shards; representative schema: `{full}`\n"
    rows = representative.get("sample_rows") or []
    shown = [clip_text(json.dumps(row, ensure_ascii=False, default=str), max_cell * 4) for row in rows[:sample_rows]]
    samples = "\n".join(f"-- sample {i + 1}: {row}" for i, row in enumerate(shown)) or "-- no sample rows in metadata"
    return f"{family_note}CREATE TABLE `{full}` (\n" + ",\n".join(defs) + f"\n);\n{samples}"


def show_schema(payload: dict) -> None:
    entries = schema_entries(payload["schemaDir"])
    blocks = []
    for requested in (payload.get("tables") or [])[:20]:
        found = [item for item in entries if matches_table(item, requested)]
        if not found:
            blocks.append(f"-- {requested}: no such table in this task database")
            continue
        blocks.append(schema_block(requested, found, 3, int(payload.get("maxCell", 300))))
    emit({"ok": True, "text": "\n\n".join(blocks)})


def run_query(payload: dict) -> None:
    from google.cloud import bigquery
    from google.oauth2 import service_account

    credential_path = Path(payload["credentialPath"]).resolve()
    credentials = service_account.Credentials.from_service_account_file(str(credential_path))
    client = bigquery.Client(project=credentials.project_id, credentials=credentials)
    sql = payload["sql"]
    cap = int(payload["maximumBytesBilled"])
    timeout = max(float(payload["timeoutMs"]) / 1000.0, 1.0)
    job = None
    try:
        dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
        estimated = int(dry.total_bytes_processed or 0)
        statement_type = (getattr(dry, "statement_type", None) or "SELECT").upper()
        if statement_type != "SELECT":
            emit({"ok": False, "error": f"only SELECT queries are allowed; BigQuery classified this as {statement_type}"})
            return
        if estimated > cap:
            shard_hint = ""
            if re.search(r"`[^`]*\*[^`]*`", sql) and "_TABLE_SUFFIX" not in sql.upper():
                shard_hint = (
                    " This query uses a wildcard table without _TABLE_SUFFIX; add a constant "
                    "_TABLE_SUFFIX equality/range predicate so BigQuery prunes date shards."
                )
            emit({
                "ok": False,
                "error": (
                    f"BigQuery dry-run rejected query: estimated scan {estimated / 1024**3:.3f} GiB "
                    f"exceeds the per-query limit {cap / 1024**3:.3f} GiB. "
                    f"Narrow the date/partition/table filters.{shard_hint}"
                ),
                "estimatedBytes": estimated,
            })
            return
        job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(maximum_bytes_billed=cap, use_query_cache=True),
        )
        iterator = job.result(timeout=timeout)
        columns = [field.name for field in iterator.schema]
        max_rows = max(int(payload.get("maxRows", 50)), 0)
        max_cell = max(int(payload.get("maxCell", 300)), 20)
        rows = []
        for index, row in enumerate(iterator):
            if index >= max_rows:
                break
            rows.append([clip_cell(row[name], max_cell) for name in columns])
        emit({
            "ok": True,
            "columns": columns,
            "rows": rows,
            "total": int(iterator.total_rows or len(rows)),
            "estimatedBytes": estimated,
            "bytesProcessed": int(job.total_bytes_processed or 0),
        })
    except Exception as error:
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        emit({"ok": False, "error": clip_text(error, 2000)})


def main() -> None:
    payload = json.loads(sys.stdin.read())
    mode = payload.get("mode")
    if mode == "list":
        list_tables(payload)
    elif mode == "schema":
        show_schema(payload)
    elif mode == "query":
        run_query(payload)
    else:
        raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit({"ok": False, "error": clip_text(error, 2000)})
