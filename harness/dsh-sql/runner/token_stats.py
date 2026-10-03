#!/usr/bin/env python3
"""Turn / token statistics for a dsh-sql run.

Reads the per-task dsh session logs (`<run>/homes/<id>/sessions/**/session.jsonl.zstd`),
which record one `assistant/message` per model call with its `usage`, plus every
`tool/call`. Correctness comes from `<run>/submission-ids.csv`, written by the
official evaluator.

Note on input tokens: dsh resends the whole conversation on every step, so the
per-task input total is the *billed* prefill summed over steps, not unique text.
`peak context` (the last step's inputTokens) is the high-water mark of one request.

    python3 token_stats.py <run-dir> [<run-dir> ...]
"""
import collections
import glob
import json
import os
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

import zstandard


def read_session(path):
    raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb")).read()
    steps, tools, turns = [], collections.Counter(), set()
    for line in raw.decode("utf8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind, data = rec.get("type"), rec.get("data", {})
        if kind == "assistant/message":
            usage = data.get("usage") or {}
            steps.append((usage.get("inputTokens", 0), usage.get("outputTokens", 0)))
        elif kind == "tool/call":
            tools[data.get("name")] += 1
        elif kind == "turn/start":
            turns.add(data.get("turn"))
    return dict(
        steps=len(steps),
        turns=len(turns) or 1,
        inp=sum(s[0] for s in steps),
        out=sum(s[1] for s in steps),
        peak=max((s[0] for s in steps), default=0),
        tools=tools,
        ntools=sum(tools.values()),
    )


def load_run(run):
    tasks = {}
    for home in sorted(glob.glob(os.path.join(run, "homes", "*"))):
        found = glob.glob(os.path.join(home, "sessions", "*", "session-*", "session.jsonl.zstd"))
        if found:
            tasks[os.path.basename(home)] = found
    with ThreadPoolExecutor(16) as pool:
        per = {}
        for task, files in tasks.items():
            merged = dict(steps=0, turns=0, inp=0, out=0, peak=0, ntools=0, tools=collections.Counter())
            for one in pool.map(read_session, files):
                for key in ("steps", "turns", "inp", "out", "ntools"):
                    merged[key] += one[key]
                merged["peak"] = max(merged["peak"], one["peak"])
                merged["tools"] += one["tools"]
            per[task] = merged
    outcomes = {
        os.path.basename(f)[:-5]: json.load(open(f))
        for f in glob.glob(os.path.join(run, "results", "*.json"))
    }
    correct = set()
    ids = os.path.join(run, "submission-ids.csv")
    if os.path.exists(ids):
        for line in open(ids):
            line = line.strip()
            if line and line != "instance_id":
                correct.add(line.replace("sf_", ""))
    return per, outcomes, correct


def report(run):
    per, outcomes, correct = load_run(run)
    steps = [m["steps"] for m in per.values()]
    inp = [m["inp"] for m in per.values()]
    out = [m["out"] for m in per.values()]
    peak = [m["peak"] for m in per.values()]
    ntools = [m["ntools"] for m in per.values()]
    print("=" * 78)
    print(f"{os.path.basename(run.rstrip('/'))}  tasks={len(per)}  correct={len(correct & set(per))}")
    print(f"  model calls / task : mean {st.mean(steps):6.1f}  median {st.median(steps):5.0f}  min {min(steps)}  max {max(steps)}")
    print(f"  tool calls  / task : mean {st.mean(ntools):6.1f}  median {st.median(ntools):5.0f}  min {min(ntools)}  max {max(ntools)}")
    print(f"  input tokens       : total {sum(inp):>14,}  mean/task {st.mean(inp):>10,.0f}  median {st.median(inp):>10,.0f}")
    print(f"  output tokens      : total {sum(out):>14,}  mean/task {st.mean(out):>10,.0f}  median {st.median(out):>10,.0f}")
    print(f"  all tokens         : total {sum(inp) + sum(out):>14,}")
    print(f"  peak context       : mean {st.mean(peak):>10,.0f}  median {st.median(peak):>10,.0f}  max {max(peak):,}")
    mix = collections.Counter()
    for m in per.values():
        mix += m["tools"]
    print("  tool mix           :", dict(mix.most_common()))

    by = collections.defaultdict(list)
    for task, m in per.items():
        by[outcomes.get(task, {}).get("outcome", "?")].append(task)
    for outcome, keys in sorted(by.items()):
        s = [per[k]["steps"] for k in keys]
        print(f"    outcome {outcome:8s} n={len(keys):3d}  steps mean {st.mean(s):6.1f}  tokens {sum(per[k]['inp'] + per[k]['out'] for k in keys):>12,}")
    if correct:
        for tag, keys in (
            ("correct", [k for k in per if k in correct]),
            ("wrong  ", [k for k in per if k not in correct]),
        ):
            if not keys:
                continue
            s = [per[k]["steps"] for k in keys]
            e = [per[k]["tools"].get("sql_exec", 0) for k in keys]
            tok = [per[k]["inp"] + per[k]["out"] for k in keys]
            print(f"    {tag} n={len(keys):3d}  steps mean {st.mean(s):6.1f}  sql_exec mean {st.mean(e):5.1f}  tokens/task {st.mean(tok):>12,.0f}")
        n = len(correct & set(per))
        print(f"  tokens per correct answer: {(sum(inp) + sum(out)) / max(1, n):,.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for run in sys.argv[1:]:
        report(run)
