#!/usr/bin/env python3
"""Compare the exported dbt SQLite corpus with the 30 Spider 2.0-Lite SQLite catalogs.

Also emits `duplicate_tables.json`: pairs of tables inside one database that hold the same
entity at two dbt layers (a source table and its `stg_` rename, a mart and the seed it was
built from). Those pairs join trivially and make a generated question ambiguous — the join
is true by construction rather than by anything the data says — so question generation
should treat them as mutually exclusive rather than as extra schema breadth.

    python corpus_report.py [--markdown out.md]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPIDER_CATALOGS = ROOT / "artifacts" / "data_synthesis" / "sqlite" / "catalogs"
DBT_ROOT = ROOT / "artifacts" / "data_synthesis" / "dbt_sqlite"
DBT_CATALOGS = DBT_ROOT / "catalogs"
EXPORT_MANIFEST = DBT_ROOT / "databases" / "export_manifest.json"

DUPLICATE_JACCARD = 0.6
DUPLICATE_CONFIDENT_ROWS = 20
DUPLICATE_CONFIDENT_JACCARD = 0.7


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def load(directory: Path) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(directory / "*.catalog.json")))]


def aggregate(catalogs: list[dict]) -> dict:
    out = dict(databases=len(catalogs), tables=0, usable=0, large=0, columns=0, rows=0,
               joins=0, composite=0, tables_with_docs=0, documented_columns=0,
               tables_with_pk=0, widths=[])
    for catalog in catalogs:
        summary = catalog["summary"]
        out["tables"] += summary["table_count"]
        out["columns"] += summary["column_count"]
        out["joins"] += summary["inferred_join_count"]
        out["composite"] += summary["composite_join_count"]
        out["tables_with_docs"] += summary["tables_with_descriptions"]
        out["widths"].append(summary["table_count"])
        for table in catalog["tables"]:
            rows = table.get("row_count") or 0
            out["rows"] += rows
            out["usable"] += rows >= 100
            out["large"] += rows >= 10_000
            out["tables_with_pk"] += bool(table.get("primary_key"))
            out["documented_columns"] += sum(1 for c in table["columns"] if c.get("description"))
    return out


def duplicate_pairs(catalogs: list[dict]) -> dict:
    found: dict[str, list[dict]] = {}
    for catalog in catalogs:
        tables = [
            (t["name"], t.get("row_count") or 0, {normalize(c["name"]) for c in t["columns"]})
            for t in catalog["tables"]
        ]
        pairs = []
        for (a, rows_a, cols_a), (b, rows_b, cols_b) in itertools.combinations(tables, 2):
            if rows_a != rows_b or rows_a == 0:
                continue
            union = len(cols_a | cols_b)
            if not union:
                continue
            jaccard = len(cols_a & cols_b) / union
            if jaccard < DUPLICATE_JACCARD:
                continue
            pairs.append({
                "a": a, "b": b, "rows": rows_a, "column_jaccard": round(jaccard, 3),
                "high_confidence": rows_a >= DUPLICATE_CONFIDENT_ROWS
                                   and jaccard >= DUPLICATE_CONFIDENT_JACCARD,
            })
        if pairs:
            found[catalog["database_id"]] = pairs
    return found


def layer_relation(catalogs: list[dict], layers: dict[str, dict[str, str]]) -> Counter:
    counter: Counter = Counter()
    for catalog in catalogs:
        mapping = layers.get(catalog["database_id"], {})
        for join in catalog["joins"]:
            left = mapping.get(join["left_table"], "unknown")
            right = mapping.get(join["right_table"], "unknown")
            counter[tuple(sorted((left, right)))] += 1
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()

    spider_catalogs, dbt_catalogs = load(SPIDER_CATALOGS), load(DBT_CATALOGS)
    spider, dbt = aggregate(spider_catalogs), aggregate(dbt_catalogs)
    manifest = json.loads(EXPORT_MANIFEST.read_text())
    layers = {
        d["database_id"]: {t["name"]: t["layer"] for t in d["tables"]}
        for d in manifest["databases"] if "error" not in d
    }
    layer_mix = Counter()
    for d in manifest["databases"]:
        if "error" not in d:
            layer_mix.update(d["layers"])

    duplicates = duplicate_pairs(dbt_catalogs)
    (DBT_ROOT / "duplicate_tables.json").write_text(json.dumps(duplicates, indent=1) + "\n")
    confident = sum(1 for pairs in duplicates.values() for p in pairs if p["high_confidence"])

    lines = []
    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    metrics = [
        ("databases", "databases"), ("tables", "tables"),
        ("tables with >=100 rows", "usable"), ("tables with >=10k rows", "large"),
        ("columns", "columns"), ("rows", "rows"),
        ("inferred join edges", "joins"), ("composite joins", "composite"),
        ("tables with column docs", "tables_with_docs"),
        ("documented columns", "documented_columns"),
        ("tables with a primary key", "tables_with_pk"),
    ]
    emit("| metric | spider2-lite | spider2-dbt | ratio | combined |")
    emit("|---|---:|---:|---:|---:|")
    for label, key in metrics:
        a, b = spider[key], dbt[key]
        ratio = f"{b / a:.2f}x" if a else "n/a"
        emit(f"| {label} | {a:,} | {b:,} | {ratio} | {a + b:,} |")
    emit()
    for label, data in (("spider2-lite", spider), ("spider2-dbt", dbt)):
        w = data["widths"]
        emit(f"- {label} tables per database: min={min(w)} p50={int(statistics.median(w))} "
             f"p90={int(statistics.quantiles(w, n=10)[8])} max={max(w)}")
    emit()
    emit(f"- dbt layer mix: {dict(layer_mix)}")
    emit(f"- duplicate entity pairs (same rows, >={DUPLICATE_JACCARD} column overlap): "
         f"{sum(len(v) for v in duplicates.values())} across {len(duplicates)} databases, "
         f"{confident} high-confidence")
    relation = layer_relation(dbt_catalogs, layers)
    cross = sum(v for k, v in relation.items() if k[0] != k[1])
    emit(f"- join edges crossing dbt layers: {cross} of {sum(relation.values())} "
         f"({100 * cross / max(1, sum(relation.values())):.1f}%); "
         f"top pairs {relation.most_common(5)}")
    emit()
    emit("| database | tables | usable | rows | joins | composite | doc'd tables |")
    emit("|---|---:|---:|---:|---:|---:|---:|")
    for catalog in sorted(dbt_catalogs, key=lambda c: -sum((t.get("row_count") or 0) for t in c["tables"])):
        rows = sum((t.get("row_count") or 0) for t in catalog["tables"])
        usable = sum(1 for t in catalog["tables"] if (t.get("row_count") or 0) >= 100)
        s = catalog["summary"]
        emit(f"| {catalog['database_id']} | {s['table_count']} | {usable} | {rows:,} | "
             f"{s['inferred_join_count']} | {s['composite_join_count']} | {s['tables_with_descriptions']} |")

    if args.markdown:
        args.markdown.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.markdown}")


if __name__ == "__main__":
    main()
