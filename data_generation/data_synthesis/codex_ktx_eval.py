#!/usr/bin/env python3
"""Run and evaluate synthesized tasks in the real Codex + KTX harness.

The model-facing subprocess receives only the question, dialect, KTX connection
id, and read-only KTX tools.  Oracle SQL and result hashes are loaded only by
the parent-side evaluator after ``codex exec`` has exited.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import signal
import shutil
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import canonical_json
from .pipeline import ReadOnlySQLite


DEFAULT_CODEX_BIN = Path(shutil.which("codex") or "codex")
DEFAULT_NODE_BIN = Path(shutil.which("node") or "node")

KTX_TOOLS = [
    "discover_data",
    "wiki_search",
    "wiki_read",
    "sl_read_source",
    "sql_dialect_notes",
    "sql_execution",
]

CONTINUE_PROMPT = """Continue the same task now. Your prior turn ended without a clean final SQL answer.
Immediately use the next relevant KTX tool, test a complete candidate with sql_execution, and finish with
exactly FINAL ANSWER: plus one fenced sql block. Never end on an intention sentence."""

_print_lock = threading.Lock()


class EvaluationSQLite(ReadOnlySQLite):
    """Read-only evaluator that permits intentional CROSS JOIN/dense grids.

    The synthesis gate requires every JOIN to spell out a predicate, but a model
    may express a legitimate category×calendar grid with ``CROSS JOIN``.  SQLite's
    authorizer still blocks every write operation here.
    """

    @staticmethod
    def static_audit(sql: str) -> dict[str, Any]:
        audit = ReadOnlySQLite.static_audit(sql)
        audit["all_joins_have_predicates"] = True
        return audit


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).casefold()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _connection_map(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for key, entry in (manifest.get("connections") or {}).items():
        if entry.get("backend") == "sqlite" and entry.get("db"):
            result[str(entry["db"])] = str(entry["id"])
        elif key.startswith("sqlite:") and entry.get("id"):
            result[key.split(":", 1)[1]] = str(entry["id"])
    for entry in (manifest.get("tasks") or {}).values():
        if entry.get("backend") == "sqlite" and entry.get("db") and entry.get("id"):
            result.setdefault(str(entry["db"]), str(entry["id"]))
    return result


def load_tasks(task_file: Path, manifest_path: Path, ids: set[str] | None = None) -> tuple[list[dict], dict[str, dict]]:
    connections = _connection_map(manifest_path)
    public_tasks: list[dict[str, Any]] = []
    hidden: dict[str, dict[str, Any]] = {}
    for full in _load_jsonl(task_file):
        sample_id = str(full["sample_id"])
        if ids and sample_id not in ids:
            continue
        iev = full["instruction_environment_verifier"]
        database_id = str(iev["environment"]["database_id"])
        connection_id = connections.get(database_id)
        if not connection_id:
            raise KeyError(f"no Spider 2.0 KTX connection for database {database_id!r}")
        public_tasks.append(
            {
                "instance_id": sample_id,
                "database_id": database_id,
                "connection_id": connection_id,
                "backend": "sqlite",
                "question": str(iev["instruction"]),
            }
        )
        hidden[sample_id] = {
            "database_path": str(full["provenance"]["database_path"]),
            "database_sha256": full["provenance"].get("database_sha256"),
            "verifier": iev["verifier"],
            "oracle_sql": str(full.get("oracle", {}).get("sql") or ""),
            "expected_columns": list(full.get("oracle", {}).get("result", {}).get("columns") or []),
            "expected_rows": list(full.get("oracle", {}).get("result", {}).get("rows") or []),
        }
    public_tasks.sort(key=lambda row: row["instance_id"])
    return public_tasks, hidden


def initial_prompt(task: dict[str, Any]) -> str:
    return f'''You are solving one difficult Spider 2.0-style SQLite text-to-SQL task in the real Codex runtime with the KTX MCP context layer.

Task id: {task['instance_id']}
KTX connection id: {task['connection_id']}
Dialect: SQLite
Question: {task['question']}

Mandatory workflow:
1. Use only the KTX MCP tools. Never use shell, filesystem, Python, web search, evaluator files, prior submissions, or benchmark answers.
2. Exact allowed tool names: discover_data, wiki_search, wiki_read, sl_read_source, sql_dialect_notes, sql_execution.
3. Always pass connectionId exactly as "{task['connection_id']}".
4. Start with discover_data(connectionId, query). Omit optional numeric arguments.
5. Inspect relevant sources with sl_read_source using sourceName values returned by discover_data. If discovery is insufficient, inspect SQLite metadata through read-only sql_execution.
6. Call sql_dialect_notes before drafting SQL.
7. Build one read-only SELECT/WITH query. Test it with sql_execution and revise any execution or semantic mistakes. Check metric grain, distinctness, date boundaries, nulls, denominators, requested rounding, ordering, and ties.
8. Do not finish before at least one successful sql_execution of the complete candidate.
9. End exactly with `FINAL ANSWER:` and one fenced sql block containing the tested query. Do not compute the answer outside SQL.
'''


def write_codex_config(home: Path, workdir: Path, args: argparse.Namespace) -> None:
    home.mkdir(parents=True, exist_ok=True)
    ktx_args = [str(args.ktx_bin), "--project-dir", str(args.ktx_project), "mcp", "stdio"]
    arg_text = ", ".join(json.dumps(value) for value in ktx_args)
    tool_text = ", ".join(json.dumps(value) for value in KTX_TOOLS)
    config = f'''model = {json.dumps(args.model)}
model_provider = "benchmark-proxy"
approval_policy = "never"
web_search = "disabled"

[features]
apps = false
goals = false
hooks = false
multi_agent = false
personality = false
plugins = false
remote_plugin = false
shell_snapshot = false
shell_tool = false

[agents]
enabled = false

[model_providers.benchmark-proxy]
name = "isolated benchmark model proxy"
base_url = {json.dumps(args.proxy_url)}
wire_api = "responses"
requires_openai_auth = false

[mcp_servers.ktx]
command = {json.dumps(str(args.node_bin))}
args = [{arg_text}]
required = true
startup_timeout_sec = 60.0
tool_timeout_sec = {float(args.query_timeout + 30):.1f}
enabled_tools = [{tool_text}]

[projects.{json.dumps(str(workdir))}]
trust_level = "trusted"
'''
    (home / "config.toml").write_text(config, encoding="utf-8")


def extract_sql(text: str) -> str | None:
    if not text:
        return None
    fenced = re.findall(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip().rstrip(";") + ";"
    marker = re.search(r"FINAL\s+ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if marker:
        candidate = marker.group(1).strip().strip("`").strip()
        if re.match(r"^(SELECT|WITH)\b", candidate, flags=re.IGNORECASE):
            return candidate.rstrip(";") + ";"
    return None


def _execution_succeeded(item: dict[str, Any]) -> bool:
    if item.get("status") != "completed" or item.get("error"):
        return False
    result = item.get("result")
    if not isinstance(result, dict):
        return False
    structured = result.get("structured_content")
    if isinstance(structured, dict) and "rowCount" in structured and "headers" in structured:
        return True
    for block in result.get("content") or []:
        if not isinstance(block, dict) or not isinstance(block.get("text"), str):
            continue
        try:
            payload = json.loads(block["text"])
        except ValueError:
            continue
        if isinstance(payload, dict) and "rowCount" in payload and "headers" in payload:
            return True
    return False


def parse_codex_events(output: str) -> dict[str, Any]:
    messages: list[str] = []
    successful_sql: list[str] = []
    tools: Counter[str] = Counter()
    errors: list[str] = []
    reasoning_items = 0
    nonempty_agent_messages = 0
    usage = {"input_tokens": 0, "output_tokens": 0}
    thread_id: str | None = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "thread.started":
            thread_id = str(event.get("thread_id") or "") or thread_id
        elif event.get("type") == "item.completed":
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "agent_message":
                text = str(item.get("text") or "")
                messages.append(text)
                nonempty_agent_messages += int(bool(text.strip()))
            elif item_type == "reasoning":
                reasoning_items += 1
            elif item_type == "mcp_tool_call":
                name = str(item.get("tool") or "")
                tools[f"{name}:{item.get('status')}"] += 1
                if name.endswith("sql_execution") and _execution_succeeded(item):
                    arguments = item.get("arguments") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            arguments = {}
                    sql = arguments.get("sql") if isinstance(arguments, dict) else None
                    if isinstance(sql, str) and sql.strip():
                        successful_sql.append(sql.strip().rstrip(";") + ";")
            elif item_type == "error":
                errors.append(str(item.get("message") or "error"))
        elif event.get("type") == "turn.completed":
            turn_usage = event.get("usage") or {}
            for key in usage:
                usage[key] = max(usage[key], int(turn_usage.get(key, 0) or 0))
        elif event.get("type") in {"turn.failed", "error"}:
            errors.append(json.dumps(event, ensure_ascii=False)[:2000])
    return {
        "last_message": messages[-1] if messages else "",
        "successful_sql": successful_sql,
        "tool_counts": dict(tools),
        "errors": errors,
        "reasoning_items": reasoning_items,
        "nonempty_agent_messages": nonempty_agent_messages,
        "usage": usage,
        "thread_id": thread_id,
    }


def _is_empty_model_response(parsed: dict[str, Any], return_code: int) -> bool:
    """A successful turn with no actionable text, tool call, or explicit error."""

    return bool(
        return_code == 0
        and not parsed.get("nonempty_agent_messages")
        and not parsed.get("tool_counts")
        and not parsed.get("errors")
    )


def _is_model_protocol_failure(errors: list[str]) -> bool:
    """Recognize malformed model-emitted tool payloads, not transport failures."""

    text = "\n".join(errors).casefold()
    markers = (
        "unterminated string starting at",
        "failed to parse tool call arguments",
        "invalid tool call arguments",
    )
    return any(marker in text for marker in markers)


def _run(command: list[str], prompt: str, env: dict[str, str], timeout: int, cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(cwd),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        # Codex starts its native binary and an MCP server below the CLI wrapper.
        # Killing only the wrapper leaves both children running and can race the
        # continuation.  Each attempt therefore owns a fresh process group.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
        return 124, stdout, stderr + f"\nrunner timeout after {timeout}s"


_STARTUP_FAILURE = re.compile(
    r"Failed to initialize session|failed to spawn thread|Resource temporarily unavailable|MCP servers failed to initialize",
    re.IGNORECASE,
)


def _run_resilient(command: list[str], prompt: str, env: dict[str, str], timeout: int, cwd: Path) -> tuple[int, str, str]:
    result = (1, "", "")
    for attempt in range(3):
        result = _run(command, prompt, env, timeout, cwd)
        rc, stdout, stderr = result
        if rc == 0 or stdout.strip() or not _STARTUP_FAILURE.search(stderr):
            return result
        if attempt < 2:
            time.sleep(10 * (attempt + 1))
    return result


def run_one(task: dict[str, Any], args: argparse.Namespace, index: int, total: int) -> dict[str, Any]:
    task_id = task["instance_id"]
    task_dir = args.output_dir / "tasks" / task_id
    result_path = task_dir / "result.json"
    submission_path = args.output_dir / "submission" / f"{task_id}.sql"
    if args.resume and result_path.exists():
        old = json.loads(result_path.read_text(encoding="utf-8"))
        if old.get("status") not in {"infra_error", "infra_stall", "setup_error"}:
            with _print_lock:
                print(f"[{task_id}] ({index}/{total}) resume existing {old.get('status')}", flush=True)
            return old

    task_dir.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = args.output_dir / "work" / task_id
    workdir.mkdir(parents=True, exist_ok=True)
    home = args.output_dir / "codex_homes" / task_id
    prompt = initial_prompt(task)
    (task_dir / "model_prompt.txt").write_text(prompt, encoding="utf-8")
    write_codex_config(home, workdir, args)

    env = os.environ.copy()
    env["CODEX_HOME"] = str(home)
    base = [
        str(args.codex_bin),
        "exec",
        "--json",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "-C",
        str(workdir),
        "-c",
        f"model_context_window={args.context_window}",
        "-",
    ]

    all_sql: list[str] = []
    all_tools: Counter[str] = Counter()
    errors: list[str] = []
    messages: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    reasoning_items = 0
    started = time.monotonic()
    rc = 1
    stalled = False
    final_sql: str | None = None
    thread_id: str | None = None
    empty_response_count = 0
    empty_response_exhausted = False
    empty_response_events: list[dict[str, Any]] = []
    logical_attempts = 0
    task_budget_exhausted = False

    for attempt in range(args.continuations + 1):
        logical_attempts += 1
        turn_prompt = prompt if attempt == 0 else CONTINUE_PROMPT
        stdout = stderr = ""
        parsed: dict[str, Any] = {}
        for empty_attempt in range(1, args.empty_response_retries + 1):
            remaining_seconds = args.task_timeout - (time.monotonic() - started)
            if remaining_seconds <= 0:
                rc = 124
                stderr = f"total task timeout after {args.task_timeout}s"
                parsed = parse_codex_events("")
                task_budget_exhausted = True
                break
            resume_target = [thread_id] if thread_id else ["--last"]
            command = base if attempt == 0 and empty_attempt == 1 else [
                str(args.codex_bin),
                "exec",
                "resume",
                *resume_target,
                "--json",
                "--skip-git-repo-check",
                "-c",
                f"model_context_window={args.context_window}",
                "-",
            ]
            rc, stdout, stderr = _run_resilient(
                command, turn_prompt, env, max(1, math.ceil(remaining_seconds)), workdir
            )
            parsed = parse_codex_events(stdout)
            thread_id = parsed.get("thread_id") or thread_id
            task_budget_exhausted = time.monotonic() - started >= args.task_timeout
            if not _is_empty_model_response(parsed, rc):
                break
            empty_response_count += 1
            retry_dir = task_dir / "empty_response_retries"
            retry_dir.mkdir(exist_ok=True)
            retry_stem = f"attempt_{attempt}_empty_{empty_attempt}"
            retry_jsonl = retry_dir / f"{retry_stem}.jsonl"
            retry_stderr = retry_dir / f"{retry_stem}.stderr.log"
            retry_jsonl.write_text(stdout, encoding="utf-8")
            retry_stderr.write_text(stderr, encoding="utf-8")
            empty_response_events.append(
                {
                    "logical_attempt": attempt + 1,
                    "empty_attempt": empty_attempt,
                    "return_code": rc,
                    "thread_id": thread_id,
                    "jsonl": str(retry_jsonl.resolve()),
                    "stderr": str(retry_stderr.resolve()),
                }
            )
            if empty_attempt < args.empty_response_retries and not task_budget_exhausted:
                reasoning_items += parsed["reasoning_items"]
                for key in usage:
                    usage[key] += parsed["usage"][key]
                time.sleep(min(2 * empty_attempt, 5))
        empty_response_exhausted = _is_empty_model_response(parsed, rc)
        (task_dir / f"attempt_{attempt}.jsonl").write_text(stdout, encoding="utf-8")
        (task_dir / f"attempt_{attempt}.stderr.log").write_text(stderr, encoding="utf-8")
        if parsed["last_message"].strip():
            messages.append(parsed["last_message"])
        all_sql.extend(parsed["successful_sql"])
        all_tools.update(parsed["tool_counts"])
        errors.extend(parsed["errors"])
        reasoning_items += parsed["reasoning_items"]
        for key in usage:
            usage[key] += parsed["usage"][key]
        final_sql = extract_sql(parsed["last_message"])
        stalled = rc == 124 and not stdout.strip() and not all_sql
        if empty_response_exhausted:
            errors.append(
                f"empty response exhausted after {args.empty_response_retries} attempts "
                f"during logical attempt {attempt + 1}"
            )
        if final_sql or stalled or empty_response_exhausted or task_budget_exhausted:
            break

    status = "clean_final" if final_sql else "no_sql"
    if final_sql is None and all_sql:
        final_sql = all_sql[-1]
        status = "recovered_execution"
    if empty_response_exhausted and final_sql is None:
        status = "empty_response_exhausted"
    if task_budget_exhausted and final_sql is None and not all_sql:
        status = "model_timeout"
        stalled = False
    elif stalled:
        status = "infra_stall"
    elif rc != 0 and final_sql is None and not all_sql:
        status = "model_protocol_error" if _is_model_protocol_failure(errors) else "infra_error"

    if final_sql:
        submission_path.write_text(final_sql + ("" if final_sql.endswith("\n") else "\n"), encoding="utf-8")
    executed_norm = {_normalize_sql(sql) for sql in all_sql}
    result = {
        "instance_id": task_id,
        "database_id": task["database_id"],
        "connection_id": task["connection_id"],
        "harness": "codex+ktx",
        "model": args.model,
        "status": status,
        "sql": final_sql,
        "final_sql_executed_in_harness": bool(final_sql) and _normalize_sql(final_sql) in executed_norm,
        "attempts": logical_attempts,
        "return_code": rc,
        "infra_stall": stalled,
        "empty_response_retries_limit": args.empty_response_retries,
        "empty_response_count": empty_response_count,
        "empty_response_exhausted": empty_response_exhausted,
        "empty_response_events": empty_response_events,
        "task_timeout_total_seconds": args.task_timeout,
        "task_budget_exhausted": task_budget_exhausted,
        "tool_counts": dict(all_tools),
        "successful_execute_count": len(all_sql),
        "successful_executes": all_sql,
        "codex_reasoning_items": reasoning_items,
        "thread_id": thread_id,
        **usage,
        "elapsed_s": round(time.monotonic() - started, 3),
        "last_message": messages[-1] if messages else "",
        "errors": errors[-20:],
    }
    _json_dump(result_path, result)
    with _print_lock:
        print(
            f"[{task_id}] ({index}/{total}) {status} exec={len(all_sql)} "
            f"tools={sum(all_tools.values())} elapsed={result['elapsed_s']}s",
            flush=True,
        )
    return result


def _evaluate_one(public: dict[str, Any], hidden: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    sql = result.get("sql")
    raw_status = result.get("status")
    model_status = raw_status
    question = str(public.get("question") or "").casefold()
    expected_columns = list(hidden.get("expected_columns") or [])
    aliases_public = bool(expected_columns) and all(
        str(column).casefold() in question for column in expected_columns
    )
    if raw_status == "clean_final":
        model_status = "clean_tested_final" if result.get("final_sql_executed_in_harness") else "clean_untested_final"
    evaluation: dict[str, Any] = {
        "sample_id": public["instance_id"],
        "model_status": model_status,
        "raw_model_status": raw_status,
        "has_sql": bool(sql),
        "final_sql_executed_in_harness": bool(result.get("final_sql_executed_in_harness")),
        "execution_success": False,
        "result_match": False,
        "value_result_match": False,
        "semantic_result_match": False,
        "aliases_public": aliases_public,
        "match_mode": "none",
        "error": None,
    }
    if not sql:
        return evaluation
    try:
        actual = EvaluationSQLite(Path(hidden["database_path"]), timeout_seconds=90, max_rows=100_000).execute(sql)
        verifier = hidden["verifier"]
        hash_key = "ordered_result_sha256" if verifier.get("order_sensitive", True) else "unordered_result_sha256"
        expected_hash = verifier[hash_key]
        evaluation.update(
            {
                "execution_success": True,
                "result_match": actual[hash_key] == expected_hash,
                "value_result_match": (
                    actual["rows"] == hidden["expected_rows"]
                    if verifier.get("order_sensitive", True)
                    else sorted(canonical_json(row) for row in actual["rows"])
                    == sorted(canonical_json(row) for row in hidden["expected_rows"])
                ),
                "column_names_exact": actual["columns"] == hidden["expected_columns"],
                "hash_mode": "ordered" if hash_key.startswith("ordered") else "unordered",
                "actual_result_sha256": actual[hash_key],
                "expected_result_sha256": expected_hash,
                "columns": actual["columns"],
                "row_count": actual["row_count"],
                "elapsed_ms": actual["elapsed_ms"],
            }
        )
        if evaluation["result_match"]:
            evaluation["semantic_result_match"] = True
            evaluation["match_mode"] = "exact_result"
        elif evaluation["value_result_match"] and not aliases_public:
            evaluation["semantic_result_match"] = True
            evaluation["match_mode"] = "values_only_public_aliases_not_required"
    except Exception as exc:  # evaluator errors are retained as task outcomes
        evaluation["error"] = f"{type(exc).__name__}: {exc}"
    return evaluation


def _load_attempts(task_dir: Path) -> list[dict[str, Any]]:
    attempts = []
    for path in sorted(task_dir.glob("attempt_*.jsonl")):
        events = []
        invalid_lines = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                invalid_lines += 1
        attempts.append({"file": str(path.resolve()), "events": events, "invalid_lines": invalid_lines})
    return attempts


def _mark_replay_refusals(payload: dict[str, Any]) -> dict[str, Any]:
    refusal_markers = (
        "nothing to reproduce",
        "nothing to copy",
        "no thinking block",
        "no reasoning block",
        "no prior thinking",
        "didn't produce any thinking",
        "did not produce any thinking",
        "there was no thinking",
        "there wasn't any",
        "reasoning budget went unused",
    )
    for turn in payload.get("turns") or []:
        recovered = str(turn.get("reasoning_recovered") or "").casefold()
        if turn.get("reasoning_source") == "replay" and any(
            marker in recovered for marker in refusal_markers
        ):
            # Preserve the exact replay text for audit, but do not count a
            # fluent refusal/hallucinated "ok -> Done" conversation as
            # recovered task reasoning.
            turn["reasoning_source"] = "replay_refused"
    return payload


def _cot_payload(
    task_id: str,
    cot_dir: Path | None,
    recovered_dir: Path | None,
    expected_model: str | None = None,
) -> dict[str, Any]:

    recovered_path = recovered_dir / f"{task_id}.json" if recovered_dir else None
    if recovered_path and recovered_path.exists():
        payload = json.loads(recovered_path.read_text(encoding="utf-8"))
        if expected_model:
            payload["turns"] = [
                turn for turn in payload.get("turns", [])
                if turn.get("model") == expected_model
            ]
        return _mark_replay_refusals(payload)
    raw_path = cot_dir / f"cot-{task_id}.jsonl" if cot_dir else None
    if raw_path and raw_path.exists():
        turns = _load_jsonl(raw_path)
        if expected_model:
            turns = [
                turn for turn in turns
                if turn.get("model") == expected_model
            ]
        return {"task_id": task_id, "turns": turns, "replay_pending": True}
    return {"task_id": task_id, "turns": []}


def assemble(
    tasks: list[dict[str, Any]],
    hidden: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    cot_dir: Path | None,
    recovered_cot_dir: Path | None,
) -> dict[str, Any]:
    trajectories_dir = output_dir / "trajectories"
    correct_dir = trajectories_dir / "correct"
    incorrect_dir = trajectories_dir / "incorrect"
    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)
    evaluations = []
    trajectories = []
    leakage_rows = []

    for public in tasks:
        task_id = public["instance_id"]
        task_dir = output_dir / "tasks" / task_id
        result_path = task_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {
            "instance_id": task_id,
            "status": "missing_run",
            "sql": None,
        }
        evaluation = _evaluate_one(public, hidden[task_id], result)
        cot = _cot_payload(task_id, cot_dir, recovered_cot_dir, result.get("model"))
        prompt = initial_prompt(public)
        oracle = hidden[task_id]["oracle_sql"]
        exact_oracle_in_prompt = bool(oracle.strip()) and _normalize_sql(oracle) in _normalize_sql(prompt)
        leakage = {
            "sample_id": task_id,
            "oracle_sql_exposed_to_model": exact_oracle_in_prompt,
            "gold_result_hash_exposed_to_model": False,
            "model_input_fields": ["sample_id", "database_id", "connection_id", "dialect", "question", "tool_observations"],
            "forbidden_model_input_fields": ["oracle", "verifier", "ordered_result_sha256", "unordered_result_sha256"],
            "shell_enabled": False,
            "filesystem_enabled": False,
            "database_access": "read-only KTX MCP only",
            "evaluator_started_after_rollout": True,
        }
        trajectory = {
            "trajectory_version": "synthesized-sqlite-codex-ktx-v2",
            "sample_id": task_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "harness": "codex+ktx",
            "model": result.get("model"),
            "task": public,
            "leakage_audit": leakage,
            "run": result,
            "attempts": _load_attempts(task_dir),
            "cot": cot,
            "evaluation": evaluation,
            "outcome": "correct" if evaluation["semantic_result_match"] else "incorrect",
        }
        semantic_correct = evaluation["semantic_result_match"]
        target_dir = correct_dir if semantic_correct else incorrect_dir
        stale_path = (incorrect_dir if semantic_correct else correct_dir) / f"{task_id}.json"
        if stale_path.exists():
            stale_path.unlink()
        _json_dump(target_dir / f"{task_id}.json", trajectory)
        trajectories.append(trajectory)
        evaluations.append(evaluation)
        leakage_rows.append(leakage)

    (output_dir / "trajectories.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trajectories), encoding="utf-8"
    )
    total_turns = 0
    signature_turns = 0
    usable_turns = 0
    tasks_with_cot = 0
    direct_turns = 0
    replay_turns = 0
    replay_refusals = 0
    direct_reasoning_chars = 0
    recovered_reasoning_chars = 0
    for trajectory in trajectories:
        turns = trajectory["cot"].get("turns") or []
        total_turns += len(turns)
        task_has_cot = False
        for turn in turns:
            if turn.get("signature"):
                signature_turns += 1
            source = turn.get("reasoning_source")
            direct_text = (turn.get("thinking_summary") or "").strip()
            recovered_text = (turn.get("reasoning_recovered") or "").strip()
            usable = bool(direct_text or (recovered_text and source != "replay_refused"))
            direct_turns += int(source == "direct")
            replay_turns += int(source == "replay")
            replay_refusals += int(source == "replay_refused")
            direct_reasoning_chars += len(direct_text)
            recovered_reasoning_chars += len(recovered_text) if source != "replay_refused" else 0
            usable_turns += int(usable)
            task_has_cot = task_has_cot or usable
        tasks_with_cot += int(task_has_cot)

    completed = [row for row in evaluations if row["model_status"] not in {"infra_error", "infra_stall", "missing_run"}]
    run_rows = [trajectory["run"] for trajectory in trajectories]
    cot_capture_configured = cot_dir is not None or recovered_cot_dir is not None
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "harness": "codex+ktx",
        "model": next((t["run"].get("model") for t in trajectories if t["run"].get("model")), None),
        "tasks_total": len(tasks),
        "tasks_completed_without_infra_failure": len(completed),
        "correct": sum(row["result_match"] for row in evaluations),
        "accuracy_all_10": sum(row["result_match"] for row in evaluations) / len(tasks) if tasks else 0.0,
        "accuracy_completed": sum(row["result_match"] for row in completed) / len(completed) if completed else None,
        "semantic_correct": sum(row["semantic_result_match"] for row in evaluations),
        "semantic_accuracy_all_tasks": (
            sum(row["semantic_result_match"] for row in evaluations) / len(tasks) if tasks else 0.0
        ),
        "value_only_correct": sum(row["value_result_match"] for row in evaluations),
        "value_only_accuracy_all_10": sum(row["value_result_match"] for row in evaluations) / len(tasks) if tasks else 0.0,
        "execution_successes": sum(row["execution_success"] for row in evaluations),
        "clean_tested_finals": sum(
            row["model_status"] == "clean_tested_final" for row in evaluations
        ),
        "resource_usage": {
            "attempts": sum(int(row.get("attempts", 0) or 0) for row in run_rows),
            "tool_events": sum(sum((row.get("tool_counts") or {}).values()) for row in run_rows),
            "successful_sql_executions": sum(int(row.get("successful_execute_count", 0) or 0) for row in run_rows),
            "input_tokens_reported": sum(int(row.get("input_tokens", 0) or 0) for row in run_rows),
            "output_tokens_reported": sum(int(row.get("output_tokens", 0) or 0) for row in run_rows),
            "per_task_elapsed_seconds_sum": round(sum(float(row.get("elapsed_s", 0) or 0) for row in run_rows), 3),
        },
        "empty_responses": {
            "retry_limit_per_turn": max(
                (int(row.get("empty_response_retries_limit", 0) or 0) for row in run_rows),
                default=0,
            ),
            "tasks_with_empty_response": sum(
                int(row.get("empty_response_count", 0) or 0) > 0 for row in run_rows
            ),
            "tasks_exhausted": sum(bool(row.get("empty_response_exhausted")) for row in run_rows),
            "empty_response_count": sum(
                int(row.get("empty_response_count", 0) or 0) for row in run_rows
            ),
        },
        "cot": {
            "capture_configured": cot_capture_configured,
            "tasks_with_usable_cot": tasks_with_cot,
            "task_acquisition_rate": (
                tasks_with_cot / len(tasks) if tasks and cot_capture_configured else None
            ),
            "model_turns": total_turns,
            "turns_with_signature": signature_turns,
            "signature_rate": signature_turns / total_turns if total_turns else None,
            "turns_with_usable_cot": usable_turns,
            "turn_acquisition_rate": usable_turns / total_turns if total_turns else None,
            "turns_with_direct_thinking": direct_turns,
            "turns_with_replay": replay_turns,
            "replay_refusals": replay_refusals,
            "direct_reasoning_chars": direct_reasoning_chars,
            "recovered_reasoning_chars": recovered_reasoning_chars,
            "total_reasoning_chars": direct_reasoning_chars + recovered_reasoning_chars,
        },
        "oracle_sql_exposed_to_model": any(row["oracle_sql_exposed_to_model"] for row in leakage_rows),
        "outcomes": evaluations,
    }
    _json_dump(output_dir / "evaluation_summary.json", summary)
    _json_dump(output_dir / "leakage_audit.json", leakage_rows)
    cot_summary = summary["cot"]
    resource_usage = summary["resource_usage"]
    empty_summary = summary["empty_responses"]
    report = [
        f"# Synthesized SQLite tasks — Codex + KTX evaluation ({summary['tasks_total']} tasks)",
        "",
        f"- model: `{summary['model']}`",
        f"- completed without infrastructure failure: {summary['tasks_completed_without_infra_failure']}/{summary['tasks_total']}",
        f"- result-hash accuracy (all tasks): {summary['correct']}/{summary['tasks_total']} "
        f"({summary['accuracy_all_10']:.1%})",
        f"- alias-policy-aware semantic accuracy: {summary['semantic_correct']}/{summary['tasks_total']} "
        f"({summary['semantic_accuracy_all_tasks']:.1%})",
        f"- value-only diagnostic accuracy: {summary['value_only_correct']}/{summary['tasks_total']} "
        f"({summary['value_only_accuracy_all_10']:.1%})",
        f"- successful final-SQL executions: {summary['execution_successes']}/{summary['tasks_total']}",
        f"- KTX tool events / successful SQL executions: {resource_usage['tool_events']} / "
        f"{resource_usage['successful_sql_executions']}",
        f"- reported input/output tokens: {resource_usage['input_tokens_reported']:,} / "
        f"{resource_usage['output_tokens_reported']:,}",
        f"- empty responses / exhausted tasks / per-turn limit: "
        f"{empty_summary['empty_response_count']} / {empty_summary['tasks_exhausted']} / "
        f"{empty_summary['retry_limit_per_turn']}",
        f"- oracle SQL exposed to model: `{str(summary['oracle_sql_exposed_to_model']).lower()}`",
        (
            f"- usable CoT task acquisition: {cot_summary['tasks_with_usable_cot']}/{summary['tasks_total']} "
            f"({cot_summary['task_acquisition_rate']:.1%})"
            if cot_summary["capture_configured"]
            else "- external CoT capture: not configured for this local-model branch"
        ),
        *(
            [
                f"- usable CoT turn acquisition: {cot_summary['turns_with_usable_cot']}/"
                f"{cot_summary['model_turns']} ({cot_summary['turn_acquisition_rate']:.1%})",
                f"- turns carrying a thinking signature: {cot_summary['turns_with_signature']}/"
                f"{cot_summary['model_turns']} ({cot_summary['signature_rate']:.1%})",
                f"- direct/replayed/refused turns: {cot_summary['turns_with_direct_thinking']} / "
                f"{cot_summary['turns_with_replay']} / {cot_summary['replay_refusals']}",
                f"- direct/recovered reasoning characters: {cot_summary['direct_reasoning_chars']:,} / "
                f"{cot_summary['recovered_reasoning_chars']:,}",
            ]
            if cot_summary["capture_configured"] and cot_summary["model_turns"]
            else []
        ),
        "",
        "| sample | rollout status | executed in harness | evaluator execution | exact match | semantic match | rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in evaluations:
        report.append(
            f"| {row['sample_id']} | {row['model_status']} | "
            f"{str(row['final_sql_executed_in_harness']).lower()} | "
            f"{str(row['execution_success']).lower()} | {str(row['result_match']).lower()} | "
            f"{str(row['semantic_result_match']).lower()} | "
            f"{row.get('row_count', '')} |"
        )
    report.extend(
        [
            "",
            "Accuracy is execution-result equivalence against the hidden ordered/unordered hash selected by each task verifier. "
            "Semantic match additionally accepts value-equivalent output only when exact aliases were not published in the "
            "question. The model-facing prompt and KTX process never receive oracle SQL or verifier hashes.",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--ids", default="", help="comma-separated sample ids")
    parser.add_argument("--model", required=True, help="model id expected by the configured proxy")
    parser.add_argument("--proxy-url", required=True, help="Responses-compatible proxy URL ending in /v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--continuations", type=int, default=3)
    parser.add_argument(
        "--empty-response-retries",
        type=int,
        default=3,
        help="maximum total attempts for one empty model turn before it is skipped",
    )
    parser.add_argument("--context-window", type=int, default=131072)
    parser.add_argument("--task-timeout", type=int, default=900)
    parser.add_argument("--query-timeout", type=int, default=90)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--cot-dir", type=Path)
    parser.add_argument("--recovered-cot-dir", type=Path)
    parser.add_argument("--ktx-project", type=Path, required=True)
    parser.add_argument("--ktx-manifest", type=Path, required=True)
    parser.add_argument("--ktx-bin", type=Path, required=True)
    parser.add_argument("--codex-bin", type=Path, default=DEFAULT_CODEX_BIN)
    parser.add_argument("--node-bin", type=Path, default=DEFAULT_NODE_BIN)
    args = parser.parse_args()
    if args.empty_response_retries < 1:
        parser.error("--empty-response-retries must be at least 1")
    for label, path in (("task file", args.task_file), ("KTX project", args.ktx_project),
                        ("KTX manifest", args.ktx_manifest), ("KTX executable", args.ktx_bin),
                        ("Codex executable", args.codex_bin), ("Node executable", args.node_bin)):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = {value.strip() for value in args.ids.split(",") if value.strip()} or None
    tasks, hidden = load_tasks(args.task_file, args.ktx_manifest, selected)

    if not args.assemble_only:
        print(
            f"Codex+KTX synthesized-task evaluation | model={args.model} tasks={len(tasks)} workers={args.workers} "
            f"context={args.context_window}",
            flush=True,
        )
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, task, args, index, len(tasks)): task["instance_id"]
                for index, task in enumerate(tasks, 1)
            }
            for future in concurrent.futures.as_completed(futures):
                task_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    result = {
                        "instance_id": task_id,
                        "model": args.model,
                        "harness": "codex+ktx",
                        "status": "runner_exception",
                        "sql": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    results.append(result)
                    _json_dump(args.output_dir / "tasks" / task_id / "result.json", result)
                    print(f"[{task_id}] runner_exception: {exc}", flush=True)
                _json_dump(args.output_dir / "run_summary.json", sorted(results, key=lambda row: row["instance_id"]))

    summary = assemble(
        tasks,
        hidden,
        args.output_dir,
        cot_dir=args.cot_dir,
        recovered_cot_dir=args.recovered_cot_dir,
    )
    print(json.dumps({key: summary[key] for key in ("tasks_total", "correct", "accuracy_all_10", "cot")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
