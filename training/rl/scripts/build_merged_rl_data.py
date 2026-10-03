#!/usr/bin/env python3
"""Build the merged slime prompt-data JSONL from both SQLite task pools.

Replaces ``build_prompt_data.py`` for the second RL run. Two changes:

* **Both pools.** ``load_tasks()`` returns the configured Spider2-lite and DBT
  task batches rather than assuming one fixed dataset size.
* **Stratified holdout.** The two sources are held out by explicit,
  independently configurable quotas instead of relying on a global shuffle.

``metadata.sample_id`` is still the only field ``generate()`` consumes -- it
looks the full task (db path, oracle hash, expected columns) back up from the
task index by id. ``metadata.source`` is carried so reward can be broken down
per pool afterwards, which the first run could not do.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from text2sql.task_data import load_tasks  # noqa: E402


def row(t):
    return {
        # A chat-message LIST, not a bare string: slime's Dataset.__init__
        # asserts the raw prompt is a list before apply_chat_template.
        "prompt": [{"role": "user", "content": t["question"]}],
        # Binary execution reward is computed by generate(); the label is
        # carried only so the field exists and debug dumps are readable.
        "label": t["oracle_sql"],
        "metadata": {
            "sample_id": t["sample_id"],
            "source": t["source"],
            "database_id": t["database_id"],
            "connection_id": t["connection_id"],
            "batch": t["batch"],
            "difficulty": t["difficulty"],
        },
    }


def describe(name, tasks):
    per_src = collections.Counter(t["source"] for t in tasks)
    print(f"  {name}: {len(tasks)}  " +
          "  ".join(f"{k}={v}" for k, v in sorted(per_src.items())))
    for src in sorted(per_src):
        sub = [t for t in tasks if t["source"] == src]
        dbs = collections.Counter(t["database_id"] for t in sub)
        print(f"      {src}: {len(dbs)} dbs, "
              f"per-db min={min(dbs.values())} max={max(dbs.values())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shuffle-seed", type=int, default=1234)
    ap.add_argument("--holdout-spider2", type=int, default=0)
    ap.add_argument("--holdout-dbt", type=int, default=0)
    ap.add_argument("--expect-spider2", type=int, default=0,
                    help="optional exact-count guard; 0 disables it")
    ap.add_argument("--expect-dbt", type=int, default=0,
                    help="optional exact-count guard; 0 disables it")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="files of sample_ids to drop, one per line "
                         "(from scripts/replay_all_oracles.py)")
    args = ap.parse_args()

    tasks = load_tasks(shuffle_seed=args.shuffle_seed)
    by_src = collections.defaultdict(list)
    for t in tasks:
        by_src[t["source"]].append(t)

    print("loaded:")
    describe("all", tasks)

    # A silently short pool is the failure mode that matters: it looks like a
    # working run and quietly trains on less data than intended.
    for src, want in (("spider2", args.expect_spider2), ("dbt", args.expect_dbt)):
        got = len(by_src.get(src, []))
        if want and got != want:
            print(f"FAIL: source {src} has {got} tasks, expected {want}",
                  file=sys.stderr)
            return 2

    ids = [t["sample_id"] for t in tasks]
    if len(set(ids)) != len(ids):
        dupes = [k for k, v in collections.Counter(ids).items() if v > 1]
        print(f"FAIL: {len(dupes)} duplicate sample_ids, e.g. {dupes[:5]}",
              file=sys.stderr)
        return 2

    # Drop tasks whose own oracle SQL cannot score 1.0 in the stack that
    # computes reward. Such a task is not merely hard, it is unwinnable: every
    # sample in its GRPO group gets reward 0, the group advantage is exactly 0,
    # so it contributes no gradient while still consuming a rollout slot and
    # diluting the batch. The verified cause is a SQLite version difference
    # between where the result hashes were computed (3.37.2) and where reward
    # runs (3.45.1 in the slime container) reordering float aggregates -- the
    # databases themselves are md5-identical, so this is not bad data.
    drop: set[str] = set()
    for f in args.exclude:
        got = {ln.strip() for ln in Path(f).read_text().splitlines() if ln.strip()}
        print(f"exclude {f}: {len(got)} ids")
        drop |= got
    if drop:
        stale = drop - set(ids)
        if stale:
            print(f"FAIL: {len(stale)} excluded ids are not in the pool "
                  f"(stale list?), e.g. {sorted(stale)[:5]}", file=sys.stderr)
            return 2
        before = len(tasks)
        tasks = [t for t in tasks if t["sample_id"] not in drop]
        by_src = collections.defaultdict(list)
        for t in tasks:
            by_src[t["source"]].append(t)
        print(f"dropped {before - len(tasks)} unreproducible tasks")
        print("after exclusion:")
        describe("all", tasks)
        ids = [t["sample_id"] for t in tasks]

    quota = {"spider2": args.holdout_spider2, "dbt": args.holdout_dbt}
    rng = random.Random(args.shuffle_seed)
    eval_rows, train_rows = [], []
    for src, pool in sorted(by_src.items()):
        n = quota.get(src, 0)
        if n > len(pool):
            print(f"FAIL: holdout {n} > pool {len(pool)} for {src}", file=sys.stderr)
            return 2
        idx = set(rng.sample(range(len(pool)), n))
        eval_rows += [t for i, t in enumerate(pool) if i in idx]
        train_rows += [t for i, t in enumerate(pool) if i not in idx]

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)

    e_ids, t_ids = {t["sample_id"] for t in eval_rows}, {t["sample_id"] for t in train_rows}
    if e_ids & t_ids:
        print(f"FAIL: {len(e_ids & t_ids)} ids in both splits", file=sys.stderr)
        return 2
    if len(e_ids) + len(t_ids) != len(ids):
        print("FAIL: split does not partition the pool", file=sys.stderr)
        return 2

    print("\nsplit:")
    describe("train", train_rows)
    describe("eval ", eval_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ep = args.out.with_name(f"{args.out.stem}.eval{args.out.suffix}")
    ep.write_text("".join(json.dumps(row(t), ensure_ascii=False) + "\n" for t in eval_rows))
    args.out.write_text("".join(json.dumps(row(t), ensure_ascii=False) + "\n" for t in train_rows))
    print(f"\nwrote {len(eval_rows)} eval rows  -> {ep}")
    print(f"wrote {len(train_rows)} train rows -> {args.out}")

    # Re-read what was written; a mangled row would otherwise only surface as a
    # dataloader crash minutes into a multi-hour job.
    for p, n in ((args.out, len(train_rows)), (ep, len(eval_rows))):
        got = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        assert len(got) == n, f"{p}: wrote {n} read {len(got)}"
        for r in got:
            assert isinstance(r["prompt"], list) and r["prompt"][0]["content"], p
            assert r["label"], p
            assert r["metadata"]["sample_id"] and r["metadata"]["source"], p
    print("VERIFY: OK (both files re-read and field-checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
