#!/usr/bin/env python3
"""Convert selected DBT DSH session trajectories into ShareGPT packed format.

Uses the same packed conversation schema as the Spider2-derived task pipeline,
so both sources have the same shape:

  [{ "id", "system", "conversations": [ {"from":"human"/"gpt", "value": ...} ... ] }]

Input paths are explicit command-line arguments; no cluster layout is assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import zstandard

SYSTEM = "You are a professional Text-to-SQL agent specialized in SQLite databases with multi-turn tool execution capabilities."


def parse_session_file(session_path: Path):
    try:
        if session_path.suffix in {".zst", ".zstd"}:
            with session_path.open("rb") as source:
                text = zstandard.ZstdDecompressor().stream_reader(source).read().decode("utf-8")
        else:
            text = session_path.read_text(encoding="utf-8")
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, ValueError, zstandard.ZstdError):
        return None

    initial_prompt = ""
    turns = []  # list of {"role": "assistant"/"tool", ...}

    for line in lines:
        t = line.get("type", "")
        data = line.get("data", {})

        # User message (the task prompt)
        if t == "user/message":
            content_list = data.get("content", [])
            for c in content_list:
                if c.get("type") == "text" and "Spider 2.0 Task" in c.get("text", ""):
                    initial_prompt = c.get("text", "").strip()

        elif t == "assistant/message":
            msg = data.get("message", {})
            content = msg.get("content", [])
            text_parts = []
            tool_calls = []
            for c in content:
                c_type = c.get("type")
                if c_type == "text":
                    text_parts.append(c.get("text", ""))
                elif c_type == "tool-call":
                    t_id = c.get("id", "call_0")
                    t_name = c.get("name", "")
                    t_args = c.get("arguments", "{}")
                    if isinstance(t_args, dict):
                        t_args_str = json.dumps(t_args, ensure_ascii=False)
                    else:
                        t_args_str = str(t_args)
                    tool_calls.append({
                        "id": t_id,
                        "function": {"name": t_name, "arguments": t_args_str},
                    })
            text_body = "".join(text_parts).strip()
            turns.append({"role": "assistant", "content": text_body, "tool_calls": tool_calls or None})

        elif t == "tool/result":
            msg = data.get("message", {})
            call_id = data.get("turn") or msg.get("source", {}).get("callId", "call_0")
            content = msg.get("content", [])
            res_text = ""
            for c in content:
                if c.get("type") == "tool-result":
                    for sub in c.get("content", []):
                        if sub.get("type") == "text":
                            res_text += sub.get("text", "")
            turns.append({"role": "tool", "tool_call_id": call_id, "content": res_text.strip()})

    if not initial_prompt or not turns:
        return None

    # Require the trajectory to contain a sql_submit call
    has_submit = any(
        turn.get("role") == "assistant" and turn.get("tool_calls")
        and any(tc["function"]["name"] == "sql_submit" for tc in turn["tool_calls"])
        for turn in turns
    )
    if not has_submit:
        return None

    return {"initial_prompt": initial_prompt, "turns": turns}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sessions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    indexes = []
    for line in args.index.open("r", encoding="utf-8"):
        line = line.strip()
        if line:
            indexes.append(json.loads(line))
    print(f"Loaded {len(indexes)} index rows from {args.index}", flush=True)

    packed = []
    skipped = 0
    for row in indexes:
        sample_id = row["sample_id"]
        candidates = [args.sessions_dir / f"{sample_id}{suffix}"
                      for suffix in (".jsonl", ".jsonl.zst", ".jsonl.zstd")]
        session = next((path for path in candidates if path.is_file()), candidates[0])
        if not session.is_file():
            print(f"  MISSING session {sample_id}", flush=True)
            skipped += 1
            continue
        traj = parse_session_file(session)
        if not traj:
            print(f"  PARSE_FAIL {sample_id}", flush=True)
            skipped += 1
            continue

        init_prompt = traj["initial_prompt"]
        turns = traj["turns"]
        convs = [{"from": "human", "value": init_prompt}]
        for turn in turns:
            if turn["role"] == "assistant":
                val = turn["content"]
                if turn.get("tool_calls"):
                    for tc in turn["tool_calls"]:
                        fn = tc["function"]["name"]
                        args = tc["function"]["arguments"]
                        val += f"\n<tool_call>\n{{\"name\": \"{fn}\", \"arguments\": {args}}}\n</tool_call>"
                convs.append({"from": "gpt", "value": val.strip()})
            elif turn["role"] == "tool":
                val = f"<tool_response>\n{turn['content']}\n</tool_response>"
                convs.append({"from": "human", "value": val})

        packed.append({
            "id": f"{sample_id}_packed",
            "system": SYSTEM,
            "conversations": convs,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(packed, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(packed)} packed trajectories to {args.output} (skipped {skipped})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
