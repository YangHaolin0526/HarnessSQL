"""Convert completed dsh session logs into packed and per-turn SFT datasets."""

import argparse
import glob
import json
from pathlib import Path

import zstandard


def read_session_lines(session_path: str) -> list[dict]:
    path = Path(session_path)
    if path.suffix in {".zst", ".zstd"}:
        with path.open("rb") as source:
            text = zstandard.ZstdDecompressor().stream_reader(source).read().decode("utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def parse_session_file(session_path):
    try:
        lines = read_session_lines(session_path)
    except (OSError, ValueError, zstandard.ZstdError):
        return None

    # Tools definition in DSH
    tools_spec = [
        {
            "name": "sql_list_tables",
            "description": "List every table and view in the database, with its row count.",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "sql_schema",
            "description": "Show the CREATE statement and a few sample rows for one or more tables.",
            "parameters": {"type": "object", "properties": {"tables": {"type": "array", "items": {"type": "string"}}}, "required": ["tables"]}
        },
        {
            "name": "sql_exec",
            "description": "Run one read-only SQLite query and see its result.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        },
        {
            "name": "sql_submit",
            "description": "Submit your final answer query.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    ]

    initial_prompt = ""
    turns = [] # list of {"role": "assistant"/"user", "content": ..., "tool_calls": [...], "tool_call_id": ...}

    for line in lines:
        t = line.get("type", "")
        data = line.get("data", {})

        # User message
        if t == "user/message":
            content_list = data.get("content", [])
            for c in content_list:
                if c.get("type") == "text" and "Spider 2.0 Task" in c.get("text", ""):
                    initial_prompt = c.get("text", "").strip()

        # Assistant message with tool call or text
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
                        "type": "function",
                        "function": {
                            "name": t_name,
                            "arguments": t_args_str
                        }
                    })

            text_body = "".join(text_parts).strip()
            turns.append({
                "role": "assistant",
                "content": text_body,
                "tool_calls": tool_calls if tool_calls else None
            })

        # Tool result
        elif t == "tool/result":
            msg = data.get("message", {})
            call_id = data.get("turn") # or msg.get("source", {}).get("callId", "")
            if not call_id:
                call_id = msg.get("source", {}).get("callId", "call_0")
            content = msg.get("content", [])
            res_text = ""
            for c in content:
                if c.get("type") == "tool-result":
                    for sub in c.get("content", []):
                        if sub.get("type") == "text":
                            res_text += sub.get("text", "")

            turns.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": res_text.strip()
            })

    if not initial_prompt or not turns:
        return None

    # Verify that the trajectory has a valid final submission
    has_submit = False
    for turn in turns:
        if turn.get("role") == "assistant" and turn.get("tool_calls"):
            for tc in turn["tool_calls"]:
                if tc["function"]["name"] == "sql_submit":
                    has_submit = True
                    break

    if not has_submit:
        return None

    return {
        "initial_prompt": initial_prompt,
        "turns": turns
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", required=True,
                        help="Glob matching dsh session.jsonl, .zst, or .zstd files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    session_files = sorted(glob.glob(args.sessions, recursive=True))
    print(f"Found {len(session_files)} total raw session files.")

    parsed_trajectories = []
    seen_tasks = set()

    for index, s_path in enumerate(session_files, 1):
        traj = parse_session_file(s_path)
        if traj:
            # Extract task ID from prompt
            lines = traj["initial_prompt"].split("\n")
            first_line = lines[0] if lines else ""
            task_id = first_line.split()[3] if len(first_line.split()) >= 4 else Path(s_path).parent.name

            if task_id not in seen_tasks:
                seen_tasks.add(task_id)
                parsed_trajectories.append((task_id, traj))
        if index % 100 == 0 or index == len(session_files):
            print(f"Parsed {index}/{len(session_files)} session files", flush=True)

    print(f"Successfully extracted {len(parsed_trajectories)} unique complete trajectories with sql_submit.")

    # 1. Build Packed ShareGPT format
    packed_dataset = []
    # 2. Build Split ShareGPT format
    split_dataset = []

    for task_id, traj in parsed_trajectories:
        init_prompt = traj["initial_prompt"]
        turns = traj["turns"]

        # Build full conversation for packed
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

        packed_dataset.append({
            "id": f"{task_id}_packed",
            "system": "You are a professional Text-to-SQL agent specialized in SQLite databases with multi-turn tool execution capabilities.",
            "conversations": convs
        })

        # Build step-by-step prefix split
        curr_convs = [{"from": "human", "value": init_prompt}]
        for i, turn in enumerate(turns):
            if turn["role"] == "assistant":
                val = turn["content"]
                if turn.get("tool_calls"):
                    for tc in turn["tool_calls"]:
                        fn = tc["function"]["name"]
                        args = tc["function"]["arguments"]
                        val += f"\n<tool_call>\n{{\"name\": \"{fn}\", \"arguments\": {args}}}\n</tool_call>"

                # Snapshot prefix + current assistant turn as an independent sample
                sample_convs = [dict(c) for c in curr_convs] + [{"from": "gpt", "value": val.strip()}]
                split_dataset.append({
                    "id": f"{task_id}_step_{len(split_dataset)}",
                    "system": "You are a professional Text-to-SQL agent specialized in SQLite databases with multi-turn tool execution capabilities.",
                    "conversations": sample_convs
                })
                curr_convs.append({"from": "gpt", "value": val.strip()})
            elif turn["role"] == "tool":
                val = f"<tool_response>\n{turn['content']}\n</tool_response>"
                curr_convs.append({"from": "human", "value": val})

    print(f"Packed dataset size: {len(packed_dataset)}")
    print(f"Split dataset size: {len(split_dataset)}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "train.packed.json").open("w", encoding="utf-8") as f:
        json.dump(packed_dataset, f, indent=2, ensure_ascii=False)

    with (out_dir / "train.split.json").open("w", encoding="utf-8") as f:
        json.dump(split_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved datasets to {out_dir / 'train.packed.json'} and {out_dir / 'train.split.json'}")

if __name__ == "__main__":
    main()
