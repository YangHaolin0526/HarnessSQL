from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator

from .common import (
    canonical_json,
    file_sha256,
    humanize_identifier,
    identifier_words,
    json_safe,
    normalize_identifier,
    percentile,
    quote_identifier,
)


CATALOG_VERSION = "spider2-sqlite-catalog-v2"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$")
GENERIC_KEYS = {"id", "key", "code", "no", "number", "index"}


def type_family(declared: str) -> str:
    value = (declared or "").upper()
    if "BLOB" in value or "BINARY" in value:
        return "blob"
    if any(token in value for token in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC", "BOOL")):
        return "numeric"
    if any(token in value for token in ("DATE", "TIME")):
        return "temporal"
    if any(token in value for token in ("CHAR", "TEXT", "CLOB", "JSON")):
        return "text"
    return "unknown"


def semantic_tags(name: str, declared: str) -> list[str]:
    words = set(identifier_words(name))
    compact = normalize_identifier(name)
    tags: set[str] = set()
    if (
        compact in GENERIC_KEYS
        or words & {"id", "key", "code", "ref", "number", "no", "index"}
        or compact.endswith(("id", "key", "code", "ref"))
    ):
        tags.add("identifier")
    if words & {"date", "datetime", "timestamp", "time", "year", "month", "day", "week"}:
        tags.add("temporal")
    if words & {
        "amount", "price", "cost", "sales", "revenue", "salary", "value", "fee", "total",
        "points", "score", "distance", "duration", "quantity", "qty", "count", "rate", "pct",
        "percent", "discount", "weight", "height", "runs", "wins", "losses", "average", "avg",
    }:
        tags.add("measure")
    if words & {
        "status", "type", "category", "segment", "state", "country", "city", "region", "name",
        "gender", "sex", "rating", "rank", "position", "channel", "platform", "class", "division",
        "department", "method", "mode", "role", "nationality", "genre",
    }:
        tags.add("dimension")
    family = type_family(declared)
    tags.add(family)
    if family == "numeric" and "identifier" not in tags:
        tags.add("measure_candidate")
    if family == "text" and "identifier" not in tags:
        tags.add("dimension_candidate")
    return sorted(tags)


def compatible_types(left: str, right: str) -> bool:
    a, b = type_family(left), type_family(right)
    if a == b:
        return a != "blob"
    return {a, b} <= {"text", "temporal"} or "unknown" in {a, b}


def singular(value: str) -> str:
    words = identifier_words(value)
    if not words:
        return normalize_identifier(value)
    last = words[-1]
    if last.endswith("ies") and len(last) > 3:
        last = last[:-3] + "y"
    elif last.endswith("s") and not last.endswith("ss") and len(last) > 2:
        last = last[:-1]
    return "".join(words[:-1] + [last])


@dataclass
class CatalogOptions:
    sample_rows: int = 5
    profile_rows: int = 256
    distinct_limit: int = 20
    exact_distinct_columns_per_table: int = 3
    query_timeout_seconds: float = 4.0
    join_timeout_seconds: float = 3.0
    max_join_candidates: int = 400
    hash_database: bool = True


class CatalogBuilder:
    """Build an execution-grounded, offline catalog for SQLite databases."""

    def __init__(
        self,
        database_root: Path,
        metadata_root: Path,
        output_dir: Path,
        options: CatalogOptions | None = None,
    ) -> None:
        self.database_root = database_root.resolve()
        self.metadata_root = metadata_root.resolve()
        self.output_dir = output_dir.resolve()
        self.options = options or CatalogOptions()

    def discover(self) -> list[Path]:
        return sorted(
            (path for path in self.database_root.glob("*.sqlite") if not path.name.startswith("._")),
            key=lambda path: path.name.casefold(),
        )

    def build_all(self, database_ids: set[str] | None = None) -> list[dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        selected = []
        wanted = {normalize_identifier(value) for value in database_ids or set()}
        for path in self.discover():
            if wanted and normalize_identifier(path.stem) not in wanted:
                continue
            selected.append(self.build_one(path))
        manifest = {
            "catalog_version": CATALOG_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "database_root": str(self.database_root),
            "database_count": len(selected),
            "databases": [
                {
                    "database_id": item["database_id"],
                    "catalog_file": f'{item["database_id"]}.catalog.json',
                    "table_count": item["summary"]["table_count"],
                    "inferred_join_count": item["summary"]["inferred_join_count"],
                    "composite_join_count": item["summary"]["composite_join_count"],
                    "database_sha256": item["database"]["sha256"],
                }
                for item in selected
            ],
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        return selected

    def build_one(self, database_path: Path) -> dict[str, Any]:
        database_id = database_path.stem
        metadata = self._load_metadata(database_id)
        uri = f"file:{database_path.resolve()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True) as connection:
            connection.row_factory = sqlite3.Row
            objects = connection.execute(
                "SELECT name, type, sql FROM sqlite_master "
                "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            tables = [self._profile_table(connection, row, metadata) for row in objects]
            joins = self._infer_joins(connection, tables)
            composite_joins = self._infer_composite_joins(connection, tables)

        explicit_fk_count = sum(len(table["foreign_keys"]) for table in tables)
        catalog = {
            "catalog_version": CATALOG_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "database_id": database_id,
            "dialect": "sqlite",
            "database": {
                "path": str(database_path.resolve()),
                "size_bytes": database_path.stat().st_size,
                "sha256": file_sha256(database_path) if self.options.hash_database else None,
            },
            "metadata": {
                "directory": str(self._metadata_dir(database_id) or ""),
                "table_files_loaded": len(metadata),
                "precedence": "live SQLite schema and values override checked-in metadata",
            },
            "profiling": {
                "sample_rows_per_table": self.options.sample_rows,
                "distribution_sample_rows": self.options.profile_rows,
                "distinct_value_limit": self.options.distinct_limit,
                "distribution_note": (
                    "Counts are exact when status=exact. Column distributions use deterministic rowid-spread "
                    "samples; low-cardinality distinct values are exact only when exact_distinct=true."
                ),
            },
            "summary": {
                "table_count": len(tables),
                "column_count": sum(len(table["columns"]) for table in tables),
                "explicit_foreign_key_count": explicit_fk_count,
                "inferred_join_count": len(joins),
                "composite_join_count": len(composite_joins),
                "tables_with_descriptions": sum(
                    1 for table in tables if any(column["description"] for column in table["columns"])
                ),
            },
            "tables": tables,
            "joins": joins,
            "composite_joins": composite_joins,
        }
        target = self.output_dir / f"{database_id}.catalog.json"
        target.write_text(json.dumps(json_safe(catalog), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return catalog

    def _metadata_dir(self, database_id: str) -> Path | None:
        target = normalize_identifier(database_id)
        for path in self.metadata_root.iterdir() if self.metadata_root.exists() else []:
            if path.is_dir() and normalize_identifier(path.name) == target:
                return path
        return None

    def _load_metadata(self, database_id: str) -> dict[str, dict[str, Any]]:
        directory = self._metadata_dir(database_id)
        if directory is None:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for path in sorted(directory.glob("*.json")):
            try:
                value = json.loads(path.read_text(errors="replace"))
            except (OSError, ValueError):
                continue
            if isinstance(value, dict) and value.get("table_name"):
                result[normalize_identifier(str(value["table_name"]))] = value
        return result

    @contextmanager
    def _deadline(self, connection: sqlite3.Connection, seconds: float) -> Iterator[None]:
        stop_at = time.monotonic() + seconds
        connection.set_progress_handler(lambda: int(time.monotonic() > stop_at), 10_000)
        try:
            yield
        finally:
            connection.set_progress_handler(None, 0)

    def _profile_table(
        self,
        connection: sqlite3.Connection,
        object_row: sqlite3.Row,
        metadata: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        name = str(object_row["name"])
        quoted = quote_identifier(name)
        pragma_name = name.replace("'", "''")
        metadata_entry = metadata.get(normalize_identifier(name), {})
        metadata_names = [str(value) for value in metadata_entry.get("column_names", []) or []]
        metadata_types = [str(value or "") for value in metadata_entry.get("column_types", []) or []]
        metadata_descriptions = [str(value or "") for value in metadata_entry.get("description", []) or []]
        schema_error = None
        try:
            columns_raw = connection.execute(f"PRAGMA table_info('{pragma_name}')").fetchall()
        except sqlite3.Error as exc:
            # Some converted Spider databases retain views whose upstream view is invalid. Keep the
            # object in the catalog and fall back to checked-in metadata instead of aborting the DB.
            schema_error = f"{type(exc).__name__}: {exc}"
            columns_raw = [
                (index, column, metadata_types[index] if index < len(metadata_types) else "", 0, None, 0)
                for index, column in enumerate(metadata_names)
            ]
        descriptions = {
            normalize_identifier(column): metadata_descriptions[index]
            for index, column in enumerate(metadata_names)
            if index < len(metadata_descriptions)
        }

        count_status = "exact"
        try:
            with self._deadline(connection, self.options.query_timeout_seconds):
                row_count = int(connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
        except sqlite3.Error as exc:
            row_count = None
            count_status = f"unavailable:{type(exc).__name__}"

        column_names = [str(row[1]) for row in columns_raw]
        samples = self._fetch_rows(connection, name, column_names, self.options.sample_rows)
        profile_rows, sampling_method = self._distribution_rows(
            connection, name, column_names, row_count, object_row["type"]
        )
        columns = []
        for row in columns_raw:
            column_name = str(row[1])
            declared_type = str(row[2] or "")
            values = [item.get(column_name) for item in profile_rows]
            columns.append(
                {
                    "name": column_name,
                    "human_label": humanize_identifier(column_name),
                    "declared_type": declared_type,
                    "type_family": type_family(declared_type),
                    "not_null": bool(row[3]),
                    "default": json_safe(row[4]),
                    "primary_key_position": int(row[5]),
                    "description": descriptions.get(normalize_identifier(column_name), ""),
                    "description_source": (
                        "spider2_metadata" if descriptions.get(normalize_identifier(column_name), "")
                        else "identifier_only"
                    ),
                    "semantic_tags": semantic_tags(column_name, declared_type),
                    "distribution": self._column_distribution(values, row_count),
                }
            )
        self._enrich_exact_distinct(connection, name, columns, row_count)

        foreign_keys = []
        try:
            for row in connection.execute(f"PRAGMA foreign_key_list('{pragma_name}')"):
                foreign_keys.append(
                    {
                        "id": int(row[0]),
                        "sequence": int(row[1]),
                        "target_table": str(row[2]),
                        "source_column": str(row[3]),
                        "target_column": str(row[4]),
                        "on_update": str(row[5]),
                        "on_delete": str(row[6]),
                    }
                )
        except sqlite3.Error:
            pass
        indexes = []
        try:
            for row in connection.execute(f"PRAGMA index_list('{pragma_name}')"):
                index_name = str(row[1])
                safe_index = index_name.replace("'", "''")
                index_columns = [str(item[2]) for item in connection.execute(f"PRAGMA index_info('{safe_index}')")]
                indexes.append({"name": index_name, "unique": bool(row[2]), "columns": index_columns})
        except sqlite3.Error:
            pass
        return {
            "name": name,
            "object_type": str(object_row["type"]),
            "schema_introspection_error": schema_error,
            "create_sql": str(object_row["sql"] or ""),
            "row_count": row_count,
            "row_count_status": count_status,
            "sampling_method": sampling_method,
            "profiled_row_count": len(profile_rows),
            "primary_key": [
                column["name"] for column in sorted(columns, key=lambda item: item["primary_key_position"])
                if column["primary_key_position"]
            ],
            "foreign_keys": foreign_keys,
            "indexes": indexes,
            "columns": columns,
            "sample_rows": samples,
            "metadata_sample_rows": json_safe((metadata_entry.get("sample_rows") or [])[: self.options.sample_rows]),
        }

    def _fetch_rows(
        self, connection: sqlite3.Connection, table: str, columns: list[str], limit: int
    ) -> list[dict[str, Any]]:
        if not columns or limit <= 0:
            return []
        projection = ", ".join(quote_identifier(column) for column in columns)
        try:
            with self._deadline(connection, self.options.query_timeout_seconds):
                rows = connection.execute(
                    f"SELECT {projection} FROM {quote_identifier(table)} LIMIT ?", (limit,)
                ).fetchall()
            return [json_safe(dict(row)) for row in rows]
        except sqlite3.Error:
            return []

    def _distribution_rows(
        self,
        connection: sqlite3.Connection,
        table: str,
        columns: list[str],
        row_count: int | None,
        object_type: str,
    ) -> tuple[list[dict[str, Any]], str]:
        cap = self.options.profile_rows
        if not columns or cap <= 0:
            return [], "disabled"
        if row_count is None or row_count <= cap or object_type != "table":
            return self._fetch_rows(connection, table, columns, cap), "full_or_prefix"
        quoted = quote_identifier(table)
        projection = ", ".join(quote_identifier(column) for column in columns)
        try:
            with self._deadline(connection, self.options.query_timeout_seconds):
                low, high = connection.execute(f"SELECT MIN(rowid), MAX(rowid) FROM {quoted}").fetchone()
            if low is None or high is None:
                raise sqlite3.OperationalError("rowid unavailable")
            if cap == 1:
                targets = [int(low)]
            else:
                targets = sorted({int(low + (high - low) * index / (cap - 1)) for index in range(cap)})
            placeholders = ",".join("?" for _ in targets)
            with self._deadline(connection, self.options.query_timeout_seconds):
                rows = connection.execute(
                    f"SELECT {projection} FROM {quoted} WHERE rowid IN ({placeholders}) ORDER BY rowid", targets
                ).fetchall()
            result = [json_safe(dict(row)) for row in rows]
            if len(result) < min(32, cap // 3):
                result.extend(self._fetch_rows(connection, table, columns, cap - len(result)))
                result = list({canonical_json(row): row for row in result}.values())[:cap]
                return result, "rowid_spread_plus_prefix"
            return result, "rowid_spread"
        except sqlite3.Error:
            return self._fetch_rows(connection, table, columns, cap), "prefix_fallback"

    def _column_distribution(self, values: list[Any], table_rows: int | None) -> dict[str, Any]:
        observed = len(values)
        non_null = [value for value in values if value is not None]
        keys = [canonical_json(value) for value in non_null]
        counts = Counter(keys)
        representatives = {canonical_json(value): value for value in non_null}
        top = [
            {"value": json_safe(representatives[key]), "sample_count": count}
            for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: self.options.distinct_limit]
        ]
        result: dict[str, Any] = {
            "sample_size": observed,
            "null_count": observed - len(non_null),
            "null_fraction": round((observed - len(non_null)) / observed, 6) if observed else None,
            "sample_distinct_count": len(counts),
            "sample_uniqueness_ratio": round(len(counts) / len(non_null), 6) if non_null else None,
            "top_values": top,
            "exact_distinct": False,
            "table_row_count": table_rows,
        }
        numeric = [float(value) for value in non_null if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if numeric and len(numeric) >= max(1, int(0.8 * len(non_null))):
            result["numeric"] = {
                "min": min(numeric),
                "p25": percentile(numeric, 0.25),
                "median": percentile(numeric, 0.5),
                "p75": percentile(numeric, 0.75),
                "max": max(numeric),
                "mean": sum(numeric) / len(numeric),
            }
        strings = [value for value in non_null if isinstance(value, str)]
        if strings and len(strings) >= max(1, int(0.8 * len(non_null))):
            lengths = [len(value) for value in strings]
            result["text"] = {
                "min_length": min(lengths),
                "max_length": max(lengths),
                "mean_length": sum(lengths) / len(lengths),
                "empty_fraction": round(sum(not value for value in strings) / len(strings), 6),
            }
            dates = [value for value in strings if DATE_RE.match(value)]
            if len(dates) >= max(1, int(0.8 * len(strings))):
                result["temporal"] = {"sample_min": min(dates), "sample_max": max(dates)}
        return json_safe(result)

    def _enrich_exact_distinct(
        self,
        connection: sqlite3.Connection,
        table: str,
        columns: list[dict[str, Any]],
        row_count: int | None,
    ) -> None:
        if not row_count or row_count > 2_000_000:
            return
        candidates = []
        for column in columns:
            distribution = column["distribution"]
            sample_size = distribution["sample_size"]
            sample_distinct = distribution["sample_distinct_count"]
            if not sample_size or sample_distinct > self.options.distinct_limit:
                continue
            tags = set(column["semantic_tags"])
            uniqueness = distribution.get("sample_uniqueness_ratio")
            if "identifier" in tags and (uniqueness is None or uniqueness > 0.5):
                continue
            score = (1 if "dimension" in tags else 0) + (1 - sample_distinct / max(sample_size, 1))
            candidates.append((score, column))
        candidates.sort(key=lambda item: (-item[0], item[1]["name"].casefold()))
        for _, column in candidates[: self.options.exact_distinct_columns_per_table]:
            quoted_table = quote_identifier(table)
            quoted_column = quote_identifier(column["name"])
            try:
                with self._deadline(connection, self.options.query_timeout_seconds):
                    rows = connection.execute(
                        f"SELECT {quoted_column}, COUNT(*) AS n FROM {quoted_table} "
                        f"GROUP BY {quoted_column} ORDER BY n DESC, {quoted_column} "
                        f"LIMIT {self.options.distinct_limit + 1}"
                    ).fetchall()
            except sqlite3.Error:
                continue
            truncated = len(rows) > self.options.distinct_limit
            rows = rows[: self.options.distinct_limit]
            column["distribution"]["top_values"] = [
                {"value": json_safe(row[0]), "count": int(row[1])} for row in rows
            ]
            column["distribution"]["exact_distinct"] = not truncated
            column["distribution"]["distinct_count"] = len(rows) if not truncated else None
            column["distribution"]["distinct_values_truncated"] = truncated

    def _column_name_score(self, left_table: str, left: str, right_table: str, right: str) -> float:
        left_norm, right_norm = normalize_identifier(left), normalize_identifier(right)
        left_words, right_words = identifier_words(left), identifier_words(right)
        left_generic = left_norm in GENERIC_KEYS
        right_generic = right_norm in GENERIC_KEYS
        if left_generic and right_generic:
            return 0.0
        if left_norm == right_norm and not left_generic:
            return 1.0
        left_base = "".join(word for word in left_words if word not in {"id", "key", "code", "ref", "no"})
        right_base = "".join(word for word in right_words if word not in {"id", "key", "code", "ref", "no"})
        if left_base and left_base == right_base:
            return 0.94
        directional = {"arrival", "departure", "source", "target", "from", "to", "origin", "destination"}
        left_core = {
            word for word in left_words
            if word not in {"id", "key", "code", "ref", "no", "number"} | directional
        }
        right_core = {
            word for word in right_words
            if word not in {"id", "key", "code", "ref", "no", "number"} | directional
        }
        if left_core and right_core and left_core == right_core:
            return 0.9
        if left_core and right_core:
            token_overlap = len(left_core & right_core) / max(len(left_core), len(right_core))
            if token_overlap >= 0.5:
                return 0.78 + 0.12 * token_overlap
        if left_generic and (right_base == singular(left_table) or normalize_identifier(right).startswith(singular(left_table))):
            return 0.92
        if right_generic and (left_base == singular(right_table) or normalize_identifier(left).startswith(singular(right_table))):
            return 0.92
        return SequenceMatcher(None, left_norm, right_norm).ratio() * 0.82

    @staticmethod
    def _relationship_name_evidence(
        left_table: str, left_column: str, right_table: str, right_column: str
    ) -> bool:
        """Recognize directional FK-like names such as ``team_batting -> team.team_id``."""

        ignored = {"id", "key", "code", "ref", "no", "number"}

        def base(column: str) -> str:
            return "".join(word for word in identifier_words(column) if word not in ignored)

        left_words = set(identifier_words(left_column))
        right_words = set(identifier_words(right_column))
        left_table_words = identifier_words(left_table)
        right_table_words = identifier_words(right_table)
        left_table_token = singular(left_table_words[-1]) if left_table_words else ""
        right_table_token = singular(right_table_words[-1]) if right_table_words else ""
        left_base = base(left_column)
        right_base = base(right_column)
        left_table_key = left_base in {singular(left_table), left_table_token} or (
            normalize_identifier(left_column) in GENERIC_KEYS
        )
        right_table_key = right_base in {singular(right_table), right_table_token} or (
            normalize_identifier(right_column) in GENERIC_KEYS
        )
        return bool(
            (right_table_key and right_table_token in left_words)
            or (left_table_key and left_table_token in right_words)
        )

    def _infer_joins(self, connection: sqlite3.Connection, tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
        explicit: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        table_by_norm = {normalize_identifier(table["name"]): table for table in tables}
        for table in tables:
            for fk in table["foreign_keys"]:
                target = table_by_norm.get(normalize_identifier(fk["target_table"]))
                if target is None:
                    continue
                key = (table["name"], fk["source_column"], target["name"], fk["target_column"])
                explicit[key] = {"explicit_foreign_key": True}

        raw_candidates: list[tuple[float, str, dict[str, Any], str, dict[str, Any], bool]] = []
        for left_index, left_table in enumerate(tables):
            for right_table in tables[left_index + 1 :]:
                pair_candidates = []
                for left_column in left_table["columns"]:
                    for right_column in right_table["columns"]:
                        if not compatible_types(left_column["declared_type"], right_column["declared_type"]):
                            continue
                        name_score = self._column_name_score(
                            left_table["name"], left_column["name"], right_table["name"], right_column["name"]
                        )
                        keyish = (
                            "identifier" in left_column["semantic_tags"]
                            and "identifier" in right_column["semantic_tags"]
                        )
                        relationship_name = self._relationship_name_evidence(
                            left_table["name"], left_column["name"],
                            right_table["name"], right_column["name"],
                        )
                        is_explicit = (
                            (left_table["name"], left_column["name"], right_table["name"], right_column["name"]) in explicit
                            or (right_table["name"], right_column["name"], left_table["name"], left_column["name"]) in explicit
                        )
                        # Executability by itself is weak evidence: two unrelated dimensions or measures
                        # often share at least one value. Inferred edges therefore need identifier semantics.
                        if not is_explicit and not (
                            (keyish and name_score >= 0.9) or relationship_name
                        ):
                            continue
                        if relationship_name:
                            name_score = max(name_score, 0.92)
                        pair_candidates.append(
                            (name_score + (0.4 if is_explicit else 0), left_table, left_column, right_table, right_column, is_explicit)
                        )
                pair_candidates.sort(key=lambda item: -item[0])
                raw_candidates.extend(pair_candidates[:3])
        raw_candidates.sort(key=lambda item: (-item[0], item[1]["name"], item[3]["name"]))

        joins = []
        live_unique_cache: dict[tuple[str, str], bool] = {}
        for _, left_table, left_column, right_table, right_column, is_explicit in raw_candidates[
            : self.options.max_join_candidates
        ]:
            name_score = self._column_name_score(
                left_table["name"], left_column["name"], right_table["name"], right_column["name"]
            )
            if self._relationship_name_evidence(
                left_table["name"], left_column["name"], right_table["name"], right_column["name"]
            ):
                name_score = max(name_score, 0.92)
            left_values = self._profile_values(left_table, left_column["name"])
            right_values = self._profile_values(right_table, right_column["name"])
            left_set = {canonical_json(value) for value in left_values if value is not None}
            right_set = {canonical_json(value) for value in right_values if value is not None}
            sample_overlap = len(left_set & right_set) / max(1, min(len(left_set), len(right_set)))
            executable, execution_error = self._join_exists(
                connection, left_table["name"], left_column["name"], right_table["name"], right_column["name"]
            )
            left_unique = self._unique_signal(left_table, left_column) or self._live_unique_signal(
                connection, left_table, left_column, live_unique_cache
            )
            right_unique = self._unique_signal(right_table, right_column) or self._live_unique_signal(
                connection, right_table, right_column, live_unique_cache
            )
            relationship = "many_to_many"
            if left_unique and right_unique:
                relationship = "one_to_one"
            elif right_unique:
                relationship = "many_to_one"
            elif left_unique:
                relationship = "one_to_many"
            # A single inferred equality between two non-unique columns is not a
            # safe join key.  It is usually a fact-to-fact fanout (or two
            # unrelated ID domains that happen to overlap).  True composite
            # relations are recorded separately with all companion columns.
            if not is_explicit and not (left_unique or right_unique):
                continue
            score = min(
                1.0,
                0.45 * name_score
                + (0.25 if executable else 0)
                + (0.2 if is_explicit else 0)
                + 0.1 * min(1.0, sample_overlap * 2)
                + (0.05 if left_unique or right_unique else 0),
            )
            if not is_explicit and (not executable or score < 0.55):
                continue
            joins.append(
                {
                    "left_table": left_table["name"],
                    "left_column": left_column["name"],
                    "right_table": right_table["name"],
                    "right_column": right_column["name"],
                    "explicit_foreign_key": is_explicit,
                    "type_compatible": True,
                    "name_similarity": round(name_score, 4),
                    "profile_sample_overlap": round(sample_overlap, 4),
                    "join_executable": executable,
                    "execution_error": execution_error,
                    "relationship_estimate": relationship,
                    "confidence": round(score, 4),
                    "evidence": [
                        "explicit_foreign_key" if is_explicit else "inferred_from_identifier_names",
                        "bounded_join_returned_a_row" if executable else "bounded_join_did_not_return_a_row",
                        "types_compatible",
                        "sample_value_overlap" if sample_overlap else "no_overlap_in_small_profile_sample",
                    ],
                }
            )
        joins.sort(key=lambda item: (-item["confidence"], item["left_table"], item["right_table"]))
        return joins

    def _infer_composite_joins(
        self, connection: sqlite3.Connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find exact-name multi-column key candidates and execute the full predicate.

        Components are never advertised as independently safe edges.  Consumers
        must use every column listed in ``left_columns``/``right_columns``.
        """

        groups: list[dict[str, Any]] = []
        for left_index, left_table in enumerate(tables):
            left_columns = {
                normalize_identifier(column["name"]): column
                for column in left_table["columns"]
                if "identifier" in column["semantic_tags"]
                and normalize_identifier(column["name"]) not in GENERIC_KEYS
            }
            for right_table in tables[left_index + 1 :]:
                right_columns = {
                    normalize_identifier(column["name"]): column
                    for column in right_table["columns"]
                    if "identifier" in column["semantic_tags"]
                    and normalize_identifier(column["name"]) not in GENERIC_KEYS
                }
                common = sorted(set(left_columns) & set(right_columns))
                if len(common) < 2:
                    continue
                # Four columns are enough to express the common event keys in
                # Spider 2.0 while avoiding accidental inclusion of attributes.
                common = common[:4]
                if not all(
                    compatible_types(
                        left_columns[name]["declared_type"],
                        right_columns[name]["declared_type"],
                    )
                    for name in common
                ):
                    continue
                left_names = [left_columns[name]["name"] for name in common]
                right_names = [right_columns[name]["name"] for name in common]
                predicates = " AND ".join(
                    f"l.{quote_identifier(left)} = r.{quote_identifier(right)}"
                    for left, right in zip(left_names, right_names)
                )
                nonnull = " AND ".join(
                    f"l.{quote_identifier(column)} IS NOT NULL" for column in left_names
                )
                sql = (
                    f"SELECT 1 FROM {quote_identifier(left_table['name'])} AS l "
                    f"JOIN {quote_identifier(right_table['name'])} AS r ON {predicates} "
                    f"WHERE {nonnull} LIMIT 1"
                )
                try:
                    with self._deadline(connection, self.options.join_timeout_seconds):
                        executable = connection.execute(sql).fetchone() is not None
                    error = None
                except sqlite3.Error as exc:
                    executable = False
                    error = type(exc).__name__
                if not executable:
                    continue
                groups.append(
                    {
                        "left_table": left_table["name"],
                        "left_columns": left_names,
                        "right_table": right_table["name"],
                        "right_columns": right_names,
                        "type_compatible": True,
                        "join_executable": True,
                        "execution_error": error,
                        "relationship_estimate": "unknown_until_full_grain_audit",
                        "confidence": 0.82,
                        "must_use_all_columns": True,
                        "evidence": [
                            "two_or_more_exact_identifier_names",
                            "full_composite_predicate_returned_a_row",
                            "types_compatible",
                            "single_components_not_promoted",
                        ],
                    }
                )
        groups.sort(
            key=lambda item: (
                item["left_table"].casefold(), item["right_table"].casefold(), item["left_columns"]
            )
        )
        return groups

    @staticmethod
    def _profile_values(table: dict[str, Any], column_name: str) -> list[Any]:
        for column in table["columns"]:
            if column["name"] == column_name:
                return [item["value"] for item in column["distribution"].get("top_values", [])]
        return []

    @staticmethod
    def _unique_signal(table: dict[str, Any], column: dict[str, Any]) -> bool:
        primary_key_columns = sum(bool(item["primary_key_position"]) for item in table["columns"])
        if column["primary_key_position"] and primary_key_columns == 1:
            return True
        if any(index["unique"] and index["columns"] == [column["name"]] for index in table["indexes"]):
            return True
        ratio = column["distribution"].get("sample_uniqueness_ratio")
        return bool(ratio is not None and ratio >= 0.995 and (table["row_count"] or 0) <= column["distribution"]["sample_size"])

    def _live_unique_signal(
        self,
        connection: sqlite3.Connection,
        table: dict[str, Any],
        column: dict[str, Any],
        cache: dict[tuple[str, str], bool],
    ) -> bool:
        key = (table["name"], column["name"])
        if key in cache:
            return cache[key]
        row_count = table.get("row_count")
        if not row_count or row_count > 250_000:
            cache[key] = False
            return False
        table_name = quote_identifier(table["name"])
        column_name = quote_identifier(column["name"])
        sql = (
            f"SELECT 1 FROM {table_name} WHERE {column_name} IS NOT NULL "
            f"GROUP BY {column_name} HAVING COUNT(*) > 1 LIMIT 1"
        )
        try:
            with self._deadline(connection, self.options.join_timeout_seconds):
                has_duplicate = connection.execute(sql).fetchone() is not None
            value = not has_duplicate
        except sqlite3.Error:
            value = False
        cache[key] = value
        return value

    def _join_exists(
        self,
        connection: sqlite3.Connection,
        left_table: str,
        left_column: str,
        right_table: str,
        right_column: str,
    ) -> tuple[bool, str | None]:
        sql = (
            f"SELECT 1 FROM {quote_identifier(left_table)} AS l "
            f"JOIN {quote_identifier(right_table)} AS r "
            f"ON l.{quote_identifier(left_column)} = r.{quote_identifier(right_column)} "
            f"WHERE l.{quote_identifier(left_column)} IS NOT NULL LIMIT 1"
        )
        try:
            with self._deadline(connection, self.options.join_timeout_seconds):
                return connection.execute(sql).fetchone() is not None, None
        except sqlite3.Error as exc:
            return False, type(exc).__name__


def default_paths(root: Path) -> tuple[Path, Path, Path]:
    spider = root / "benchmarks" / "spider2_repo" / "spider2-lite" / "resource" / "databases"
    return spider, spider / "sqlite", root / "artifacts" / "data_synthesis" / "sqlite" / "catalogs"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    database_root, metadata_root, output_dir = default_paths(root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-root", type=Path, default=database_root)
    parser.add_argument("--metadata-root", type=Path, default=metadata_root)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--databases", default="", help="comma-separated database ids; empty means every .sqlite")
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--profile-rows", type=int, default=256)
    parser.add_argument("--query-timeout", type=float, default=4.0)
    parser.add_argument("--join-timeout", type=float, default=3.0)
    parser.add_argument("--max-join-candidates", type=int, default=400)
    parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args()
    options = CatalogOptions(
        sample_rows=args.sample_rows,
        profile_rows=args.profile_rows,
        query_timeout_seconds=args.query_timeout,
        join_timeout_seconds=args.join_timeout,
        max_join_candidates=args.max_join_candidates,
        hash_database=not args.no_hash,
    )
    builder = CatalogBuilder(args.database_root, args.metadata_root, args.output_dir, options)
    database_ids = {value.strip() for value in args.databases.split(",") if value.strip()} or None
    catalogs = builder.build_all(database_ids)
    print(f"built {len(catalogs)} catalogs in {args.output_dir}")


if __name__ == "__main__":
    main()
