#!/usr/bin/env python3
"""Replay every task's own oracle SQL through sql_reward; list the ones that fail.

Run this inside whatever stack computes reward during RL, because that is the
only stack whose answer counts. A stored ordered_result_sha256 is a claim about
what a particular SQLite build produced; a different build can legitimately
disagree in the last bits of a float aggregate, and the reward path compares
exactly.

Emits, into --out:
  unreproducible_<source>.txt   one sample_id per line, for the data builder
  replay_<source>.json          per-task status, for auditing
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text2sql import task_data  # noqa: E402
from text2sql.task_data import load_tasks, sql_reward  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sources", nargs="+", default=["dbt", "spider2"])
    ap.add_argument("--only-ids-from", type=Path, nargs="*", default=[],
                    help="prompt-data JSONL(s); replay only the sample_ids they "
                         "contain and require 100%% reproducible")
    args = ap.parse_args()

    keep = None
    if args.only_ids_from:
        keep = set()
        for f in args.only_ids_from:
            n = 0
            for line in Path(f).read_text().splitlines():
                line = line.strip()
                if line:
                    keep.add(json.loads(line)["metadata"]["sample_id"]); n += 1
            print(f"{f}: {n} rows")
        print(f"restricting replay to {len(keep)} sample_ids")
    args.out.mkdir(parents=True, exist_ok=True)

    import sqlite3
    print(f"sqlite {sqlite3.sqlite_version}")
    if task_data.EvaluationSQLite is None:
        print("FAIL: EvaluationSQLite did not import", file=sys.stderr)
        return 2

    grand = {}
    for source in args.sources:
        tasks = load_tasks(sources=[source])
        if keep is not None:
            tasks = [t for t in tasks if t["sample_id"] in keep]
            if not tasks:
                print(f"\n{source}: no ids in the restricted set, skipping")
                continue
        print(f"\n{'='*62}\n{source}: {len(tasks)} tasks\n{'='*62}", flush=True)

        stats, bad, recs = collections.Counter(), [], []
        t0 = time.time()
        for i, t in enumerate(tasks, 1):
            r = sql_reward(t, t["oracle_sql"])
            stats[r["status"]] += 1
            recs.append({"sample_id": t["sample_id"],
                         "database_id": t["database_id"],
                         "status": r["status"],
                         "reward": r["reward"],
                         "row_count": r.get("row_count"),
                         "column_names_exact": r.get("column_names_exact")})
            if r["reward"] != 1.0:
                bad.append(t)
            if i % 200 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}  bad={len(bad)}  "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

        ok = len(tasks) - len(bad)
        print(f"\nstatus: {dict(stats)}")
        print(f"reproducible: {ok}/{len(tasks)} ({100*ok/len(tasks):.2f}%)")
        print(f"unreproducible: {len(bad)} ({100*len(bad)/len(tasks):.2f}%)")
        if bad:
            per_db = collections.Counter(t["database_id"] for t in bad)
            print(f"spread over {len(per_db)} databases: {per_db.most_common(10)}")

        bp = args.out / f"unreproducible_{source}.txt"
        bp.write_text("".join(t["sample_id"] + "\n" for t in bad))
        (args.out / f"replay_{source}.json").write_text(json.dumps(recs, indent=1))
        print(f"wrote {len(bad)} ids -> {bp}")
        grand[source] = {"total": len(tasks), "bad": len(bad),
                         "sqlite": sqlite3.sqlite_version,
                         "status": dict(stats)}

    (args.out / "summary.json").write_text(json.dumps(grand, indent=1))
    print(f"\n{'='*62}")
    for s, v in grand.items():
        print(f"{s:9s} {v['total']-v['bad']}/{v['total']} reproducible "
              f"({100*(v['total']-v['bad'])/v['total']:.2f}%)")

    if keep is not None:
        # In this mode the pool has already been filtered, so anything left
        # unreproducible means the filter and the reward stack disagree.
        total_bad = sum(v["bad"] for v in grand.values())
        seen = sum(v["total"] for v in grand.values())
        if seen != len(keep):
            print(f"\nFAIL: replayed {seen} of {len(keep)} requested ids",
                  file=sys.stderr)
            return 1
        if total_bad:
            print(f"\nFAIL: {total_bad} tasks in the FILTERED pool still "
                  f"cannot reach reward 1.0", file=sys.stderr)
            return 1
        print(f"\nOK: all {seen} tasks in the filtered pool reach reward 1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
