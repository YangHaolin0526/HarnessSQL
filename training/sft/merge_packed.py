#!/usr/bin/env python3
"""Merge DBT-derived and Spider2-derived packed ShareGPT datasets.

Both sides are already in the identical packed_clean format produced by
build_sft_from_dsh.py. This concatenates them after schema and overlap checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbt", type=Path, required=True)
    parser.add_argument("--spider2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dbt = json.loads(args.dbt.read_text(encoding="utf-8"))
    lite = json.loads(args.spider2.read_text(encoding="utf-8"))
    print(f"dbt735: {len(dbt)} samples | spider2.0-lite: {len(lite)} samples", flush=True)

    # sanity: both share the exact same system prompt and conversation schema
    assert len(dbt) and len(lite), "one side is empty"
    for name, data in (("dbt", dbt), ("lite", lite)):
        for x in data:
            assert set(x.keys()) == {"id", "system", "conversations"}, f"{name} bad keys: {set(x.keys())}"
            assert x["system"], f"{name} empty system"

    # overlap guard
    ids = [x["id"] for x in dbt] + [x["id"] for x in lite]
    assert len(ids) == len(set(ids)), "duplicate ids across the two datasets"

    merged = dbt + lite
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(merged)} merged samples -> {args.output}", flush=True)
    print(f"  dbt735:   {len(dbt)}", flush=True)
    print(f"  lite:     {len(lite)}", flush=True)
    print(f"  total:    {len(merged)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
