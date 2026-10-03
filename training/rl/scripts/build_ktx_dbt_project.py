#!/usr/bin/env python3
"""Build a writable KTX project that serves the dbt databases as well.

Why this exists
---------------
The RL rollout does not read a .sqlite file directly. It runs Codex against the
KTX MCP server and the agent reaches the database only through
``sql_execution(connectionId=..., sql=...)``. KTX resolves ``connectionId``
against the ``connections:`` map in its project's ``ktx.yaml``. An id that is
not in that map is not an error the reward pipeline can see: the tool call
fails, the agent never produces a tested query, and ``sql_reward`` returns 0.0
with status ``no_sql``. A whole task pool can therefore score a clean,
plausible-looking zero purely because its databases were never registered.

The 1,800 Spider2-lite tasks use the 30 ``spider2-sqlite-*`` connections in the
existing project, which lives in a **read-only** account. So this script copies
that project into a writable location and appends one ``dbt-sqlite-<slug>``
connection per dbt database, with ``enabled_tables`` enumerated live from each
file's ``sqlite_master``.

``.ktx/cache``, ``.ktx/runtime`` and logs are dropped: runtime is per-instance
daemon state that races between concurrent rollouts, cache is ingest-time
scratch that query serving never reads, and the logs are 20 MB of dead weight.
This mirrors ``codex_rollout._KTX_IGNORE``, which excludes the same paths when
it fans the project out per rollout.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

IGNORE = shutil.ignore_patterns("runtime", "cache", "logs", "*.log",
                                "*.sqlite-shm", "*.sqlite-wal")


def slug(database_id: str) -> str:
    return database_id.lower().replace("_", "-")


def tables_of(db: Path) -> list[str]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="Existing KTX project containing the Spider2 connections.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dbt-db-dir", type=Path, required=True)
    ap.add_argument("--timeout-ms", type=int, default=70000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (args.src / "ktx.yaml").is_file():
        print(f"FAIL: no ktx.yaml under {args.src}", file=sys.stderr)
        return 2
    dbs = sorted(p for p in args.dbt_db_dir.glob("*.sqlite"))
    if not dbs:
        print(f"FAIL: no .sqlite under {args.dbt_db_dir}", file=sys.stderr)
        return 2

    if args.out.exists():
        if not args.force:
            print(f"FAIL: {args.out} exists; pass --force", file=sys.stderr)
            return 2
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.src, args.out, ignore=IGNORE)
    print(f"copied project {args.src} -> {args.out}")

    yaml_path = args.out / "ktx.yaml"
    text = yaml_path.read_text()
    if not text.endswith("\n"):
        text += "\n"

    existing = set()
    for line in text.splitlines():
        if len(line) > 2 and line[:2] == "  " and line[2] != " " and line.rstrip().endswith(":"):
            existing.add(line.strip()[:-1])
    print(f"existing connections: {len(existing)}")

    blocks, added, skipped, total_tables = [], [], [], 0
    for db in dbs:
        conn_id = "dbt-sqlite-" + slug(db.stem)
        if conn_id in existing:
            skipped.append(conn_id)
            continue
        tbls = tables_of(db)
        if not tbls:
            print(f"  WARN {db.name}: no tables, skipping")
            continue
        total_tables += len(tbls)
        lines = [f"  {conn_id}:",
                 "    driver: sqlite",
                 f"    path: {db.resolve()}",
                 f"    query_timeout_ms: {args.timeout_ms}",
                 "    enabled_tables:"]
        lines += [f"      - {t}" for t in tbls]
        blocks.append("\n".join(lines))
        added.append(conn_id)

    if not blocks:
        print("nothing to add")
    else:
        # The two-space blocks belong to the `connections:` mapping, and
        # `connections:` is NOT the only top-level key -- storage, llm, ingest,
        # agent, scan and setup follow it. Appending at end-of-file would nest
        # every dbt connection under the last of those instead, which YAML
        # accepts silently and KTX then serves with 49 connections missing. So
        # insert immediately before the next top-level key.
        lines = text.splitlines()
        tops = [(i, ln.split(":")[0]) for i, ln in enumerate(lines)
                if ln and not ln[0].isspace() and ln.rstrip().endswith(":")]
        conn_at = [i for i, k in tops if k == "connections"]
        if len(conn_at) != 1:
            print(f"FAIL: want exactly one top-level 'connections:', found {conn_at}",
                  file=sys.stderr)
            return 2
        after = [i for i, _ in tops if i > conn_at[0]]
        insert_at = after[0] if after else len(lines)
        print(f"connections: at line {conn_at[0]+1}, "
              f"inserting before line {insert_at+1} "
              f"({[k for i,k in tops if i==insert_at] or ['EOF']})")
        merged = lines[:insert_at] + "\n".join(blocks).splitlines() + lines[insert_at:]
        yaml_path.write_text("\n".join(merged) + "\n")
        print(f"inserted {len(added)} dbt connections, {total_tables} tables")
    if skipped:
        print(f"skipped {len(skipped)} already present")

    # Re-parse and assert every dbt db resolves to a real readable file.
    import yaml
    conf = yaml.safe_load(yaml_path.read_text())
    conns = conf["connections"]
    dbt = {k: v for k, v in conns.items() if k.startswith("dbt-sqlite-")}
    print(f"\nVERIFY: {len(conns)} total connections, {len(dbt)} dbt")
    bad = 0
    for k, v in sorted(dbt.items()):
        p = Path(v["path"])
        if v.get("driver") != "sqlite" or not p.is_file():
            print(f"  BAD {k}: driver={v.get('driver')} path={p} exists={p.is_file()}")
            bad += 1
        elif not v.get("enabled_tables"):
            print(f"  BAD {k}: no enabled_tables")
            bad += 1
    if len(dbt) != len(dbs):
        print(f"  BAD: {len(dbs)} db files but {len(dbt)} dbt connections")
        bad += 1
    # The insert must not have swallowed or displaced the pool that already
    # trains, nor leaked keys into a sibling top-level section.
    src_conf = yaml.safe_load((args.src / "ktx.yaml").read_text())
    lost = set(src_conf["connections"]) - set(conns)
    if lost:
        print(f"  BAD: {len(lost)} original connections lost: {sorted(lost)[:5]}")
        bad += 1
    for k in ("storage", "llm", "ingest", "agent", "scan", "setup"):
        if k in src_conf and conf.get(k) != src_conf[k]:
            print(f"  BAD: top-level '{k}' section was modified")
            bad += 1
    spider2 = [k for k in conns if k.startswith("spider2-sqlite-")]
    print(f"  original connections preserved: {len(src_conf['connections'])}")
    print(f"  spider2-sqlite-*: {len(spider2)}   dbt-sqlite-*: {len(dbt)}")
    for must in ("ktx.yaml", ".ktx/db.sqlite"):
        if not (args.out / must).exists():
            print(f"  BAD: missing {must}")
            bad += 1
    print("VERIFY: FAIL" if bad else "VERIFY: OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
