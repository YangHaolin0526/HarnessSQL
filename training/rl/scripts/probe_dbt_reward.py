#!/usr/bin/env python3
"""Replay oracle SQL through sql_reward and assert it scores 1.0.

The reward path has three independent places where the dbt pool can be wrong
in a way that produces a plausible zero rather than an error:

* ``db_path_for`` resolves to a file that exists but is the wrong database
  (dbt rows may carry an authoring-machine path in ``provenance.database_path``),
* the stored ``ordered_result_sha256`` was computed against a different build
  of the database than the one on this filesystem,
* ``expected_columns`` / ``order_sensitive`` are shaped differently in the dbt
  rows than in the spider2 rows, so the comparison branch never matches.

Running each task's own oracle SQL is the tightest available check: it must
score exactly 1.0. Anything less means RL would train against a reward that
cannot be earned, which looks identical to "the model is bad".

This only touches SQLite locally -- no model, no GPU, no KTX.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text2sql import task_data  # noqa: E402
from text2sql.task_data import load_tasks, sql_reward  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="dbt")
    ap.add_argument("--per-db", type=int, default=2,
                    help="tasks to replay per database (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"SYNTH     {task_data.SYNTH}")
    print(f"DB_DIR    {task_data.DB_DIR}")
    print(f"DBT_SYNTH {task_data.DBT_SYNTH}")
    print(f"DBT_DB_DIR{task_data.DBT_DB_DIR}")
    if task_data.EvaluationSQLite is None:
        print("FAIL: EvaluationSQLite did not import; check SQL_CLI on sys.path",
              file=sys.stderr)
        return 2

    tasks = load_tasks(sources=[args.source])
    print(f"\n{args.source}: {len(tasks)} tasks")

    by_db = collections.defaultdict(list)
    for t in tasks:
        by_db[t["database_id"]].append(t)
    print(f"{len(by_db)} databases")

    missing = sorted({t["database_id"] for t in tasks
                      if not Path(t["db_path"]).is_file()})
    if missing:
        print(f"FAIL: {len(missing)} databases have no file: {missing[:5]}",
              file=sys.stderr)
        return 2
    print("all database files present")

    rng = random.Random(args.seed)
    picked = []
    for db, ts in sorted(by_db.items()):
        picked += ts if args.per_db == 0 else rng.sample(ts, min(args.per_db, len(ts)))
    print(f"\nreplaying oracle SQL for {len(picked)} tasks...\n")

    stats = collections.Counter()
    bad = []
    t0 = time.time()
    for i, t in enumerate(picked, 1):
        r = sql_reward(t, t["oracle_sql"])
        stats[r["status"]] += 1
        if r["reward"] != 1.0:
            bad.append((t, r))
        if i % 25 == 0 or i == len(picked):
            print(f"  {i}/{len(picked)}  ok={stats['correct']+stats['correct_values_only']}"
                  f"  elapsed={time.time()-t0:.0f}s")

    print(f"\nstatus breakdown: {dict(stats)}")
    ok = stats["correct"] + stats["correct_values_only"]
    print(f"oracle reward 1.0: {ok}/{len(picked)}  ({100*ok/len(picked):.1f}%)")

    if bad:
        print(f"\n{len(bad)} FAILING tasks (first 10):")
        for t, r in bad[:10]:
            print(f"  {t['sample_id']}  db={t['database_id']}  status={r['status']}"
                  f"  detail={str(r.get('detail'))[:120]}")
        by_db_bad = collections.Counter(t["database_id"] for t, _ in bad)
        print(f"\nfailures concentrated in {len(by_db_bad)} dbs: "
              f"{by_db_bad.most_common(8)}")
        print("\nPROBE: FAIL")
        return 1
    print("\nPROBE: OK -- every replayed oracle earns reward 1.0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
