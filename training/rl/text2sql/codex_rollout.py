"""Codex+KTX rollout harness for Text-to-SQL RL.

Spawns a fresh Codex subprocess per rollout, captures SSE events, parses them
into trajectory segments suitable for `slime.agent.trajectory.merge_turns()`,
and computes the execution reward. Multi-turn by design: one rollout contains
as many model calls as the agent needs to reach a final SQL answer, up to the
configured turn budget.

**Key parameters that differ from single-turn RL:**

  --rollout-max-response-len       per-call generation budget (NOT total)
  --rollout-max-prompt-len         last turn's prompt (holds full history)
  --max-tokens-per-gpu             merged trajectory length (training side)

  From qwen_traj_findings.md measured distributions (135 sqlite tasks, 27B models):
    per-turn output: p50=129, p90=776, p95=1915, p99=8956 -> 2048-4096
    max prompt:      p50=12567, p90=18888, p95=23857, p99=31488 -> 32K
    total length:    p50=12954, p90=20062, p95=25354, p99=32160 -> 32768

  The existing `10_train_rl_pinned.sh` sets `--max-tokens-per-gpu 4096`, which
  cannot hold a single trajectory (shortest real trace = 10,626 tokens). That
  script was a dapo-math single-turn config ported verbatim; these are the first
  agentic params tuned against a real multi-turn distribution.

**Truncation.**  KTX's `sql_execution` tool accepts `maxRows` (default 1000, max
10,000); measured tool observations hit p99=2,263 and max=7,063 tokens. The user
requested 1024-1536 token caps. We set `maxRows=50` in the codex config (enough
to judge correctness; far below the token target), but that is a *row* budget,
not a token budget. A 50-row × 20-column result can still exceed 1536 tokens. A
post-execution character truncation with an explicit `[truncated]` marker would
be the principled fix, but that requires patching either KTX or the Codex MCP
bridge -- deferred until the speed test confirms truncation is the bottleneck.

**Colocated rollout + offload.** `10_train_rl_pinned.sh` sets `--colocate`
without `--offload`, which crashes when rollout and training share 8 GPUs -- the
trainer's weights stay resident and rollout OOMs. Must add `--offload` (or drop
`--colocate` and let rollout use a separate SGLang instance, which is slower but
simpler for this speed test).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

NODE_BIN = Path(os.environ.get("NODE_BIN") or shutil.which("node") or "node")
CODEX_BIN = Path(os.environ.get("CODEX_BIN") or shutil.which("codex") or "codex")
KTX_BIN = Path(os.environ["HARNESS_SQL_KTX_BIN"]) if os.environ.get("HARNESS_SQL_KTX_BIN") else None
KTX_PROJECT = (Path(os.environ["HARNESS_SQL_KTX_PROJECT"])
               if os.environ.get("HARNESS_SQL_KTX_PROJECT") else None)


def require_path(path: Path | None, description: str) -> Path:
    if path is None or not path.exists():
        raise RuntimeError(f"{description} is not configured or does not exist; see .env.example")
    return path.resolve()

# 12 tools registered; measured usage shows 6 are actually called.
KTX_TOOLS = [
    "discover_data", "wiki_search", "wiki_read",
    "sl_read_source", "sql_dialect_notes", "sql_execution",
]


def toml_string(s: str) -> str:
    """Minimal TOML basic-string escaper -- no \\u handling needed here."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def write_codex_config(
    home: Path,
    workdir: Path,
    proxy_url: str,
    served_model_name: str,
    api_key: str,
    query_timeout: int = 30,
    ktx_project: Path | None = None,
) -> None:
    """Write a minimal Codex config pointing at the vLLM compatibility proxy.

    The proxy passes `model` through to vLLM, so `served_model_name` must match
    the vLLM `--served-model-name` (e.g. "Qwen3.5-27B", not a HF path).

    Note there is deliberately no tool-argument default here: `McpServerConfig`
    (codex-rs/config/src/mcp_types.rs:150-215) exposes only enabled_tools /
    disabled_tools / timeouts / approval mode -- it has no mechanism to inject a
    default argument such as `maxRows` into a tool call. Bounding KTX result
    size has to happen either in the prompt (unreliable -- the model chooses)
    or inside KTX itself. See the truncation note in the module docstring.
    """
    home.mkdir(parents=True, exist_ok=True)
    ktx_bin = require_path(KTX_BIN, "HARNESS_SQL_KTX_BIN")
    project = require_path(ktx_project or KTX_PROJECT, "HARNESS_SQL_KTX_PROJECT")
    node_bin = require_path(NODE_BIN, "NODE_BIN")
    ktx_args = [str(ktx_bin), "--project-dir", str(project), "mcp", "stdio"]
    arg_text = ", ".join(toml_string(x) for x in ktx_args)
    tool_text = ", ".join(toml_string(x) for x in KTX_TOOLS)
    config = f'''model = {toml_string(served_model_name)}
model_provider = "vllm-proxy"
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

[model_providers.vllm-proxy]
name = "local model via vLLM compatibility proxy"
base_url = {toml_string(proxy_url)}
wire_api = "responses"
requires_openai_auth = false

[model_providers.vllm-proxy.http_headers]
Authorization = {toml_string(f"Bearer {api_key}")}

[mcp_servers.ktx]
command = {toml_string(str(node_bin))}
args = [{arg_text}]
required = true
startup_timeout_sec = 60.0
tool_timeout_sec = {float(query_timeout + 30):.1f}
enabled_tools = [{tool_text}]

[projects.{toml_string(str(workdir))}]
trust_level = "trusted"
'''
    (home / "config.toml").write_text(config)


# `.ktx/db.sqlite` (64 MB) is the ingested catalog that discover_data and
# sl_read_source read, so it HAS to reach each rollout -- excluding all of
# `.ktx/` is what previously left every rollout with an empty catalog. But
# excluding it was also how concurrent codex boots were stopped from racing on
# KTX's runtime state, so exclude exactly the volatile and useless parts
# instead: `runtime/` is the per-instance daemon state that raced, `cache/` is
# 50 MB of ingest-time scratch that query serving never reads, and logs are
# dead weight. Concurrent copy+boot+discover_data probes verified this subset.
_KTX_IGNORE = shutil.ignore_patterns(
    "runtime", "cache", "logs", "*.log", "*.sqlite-shm")

# Cloning a project straight from shared network storage for every rollout is
# expensive at high concurrency. Stage it once per process onto node-local
# scratch instead, then fan out locally.
_LOCAL_TEMPLATE: Path | None = None
_TEMPLATE_LOCK = threading.Lock()


def _local_project_template() -> Path:
    """Node-local copy of the KTX project, built once per process."""
    global _LOCAL_TEMPLATE
    if _LOCAL_TEMPLATE is not None:
        return _LOCAL_TEMPLATE
    with _TEMPLATE_LOCK:
        if _LOCAL_TEMPLATE is None:
            staging = Path(tempfile.mkdtemp(prefix="ktx_template_")) / "project"
            shutil.copytree(require_path(KTX_PROJECT, "HARNESS_SQL_KTX_PROJECT"),
                            staging, ignore=_KTX_IGNORE)
            _LOCAL_TEMPLATE = staging
    return _LOCAL_TEMPLATE


def initial_prompt(task: dict[str, Any]) -> str:
    """First-turn system+user message for a Text-to-SQL rollout.

    The wording is the sweep harness's, verbatim where it matters
    (``codex-qwen35-sweep/run_ktx_codex_spider2.py:255``). That is the prompt
    the 95-100%-reward runs used; the RL harness had paraphrased it and lost
    two clauses with directly observed consequences.

    The costly one was in rule 6. The sweep ends it with "If discover_data
    returns no usable sources, query the information schema through
    sql_execution instead of guessing." Direct MCP probes show
    ``discover_data`` returns ``{"refs":[]}`` and ``sl_read_source`` reports no
    semantic-layer source for THIS project -- while ``sql_execution`` works
    perfectly (``SELECT COUNT(*) FROM customer`` -> 599 on Pagila). Without the
    escape clause the agent is forbidden from guessing table names and has no
    legal way to learn them, so it falls back to its priors and writes generic
    orders/customers SQL, producing widespread ``no such table`` errors on
    databases such as Pagila and complex_oracle.

    The second was in rule 2: the sweep forbids dynamically constructing a tool
    name. Without it the run produced ``mcp__ktx__wikipedia`` and
    ``mcp__ktx__sl_dialect_notes``, neither of which exists.

    Rule 6 then goes further than the sweep, because the sweep's own wording is
    written for BigQuery/Snowflake: "query the information schema" has no
    meaning on SQLite, and a model following it literally runs
    ``SELECT ... FROM information_schema.tables``, gets "no such table", and
    concludes the schema is unreachable. Probing KTX directly shows which
    routes actually work on this dialect:

        SELECT name, sql FROM sqlite_master WHERE type='table'   -> 16 tables + DDL
        SELECT * FROM pragma_table_info('customer')              -> columns
        PRAGMA table_info(customer)   -> rejected, "read/write operation: Pragma"
        information_schema.tables     -> no such table

    So the two most natural SQLite introspection moves are both dead ends and
    only the less obvious forms work. The prompt now names them outright.

    Naming both routes was itself too loose: the model then swept
    ``pragma_table_info`` table by table and ran out of turns before answering.
    Rule 6 therefore says outright that one ``sqlite_master`` query returns
    every CREATE statement and that per-table pragma calls are not for
    enumeration.
    """
    return f'''You are solving exactly one Spider 2.0-Lite text-to-SQL task with Codex and the KTX MCP server.

Task id: {task['sample_id']}
KTX connection id: {task['connection_id']}
Database: {task['database_id']} (SQLite)
Question: {task['question']}

Mandatory workflow:
1. Use only the ktx MCP tools. Never use shell, filesystem, Python, web search, or prior submissions.
2. Exact allowed tool names: discover_data, wiki_search, wiki_read, sl_read_source, sql_dialect_notes, sql_execution. Never invent, abbreviate, pluralize, or dynamically construct a tool name. There is no tool named slug.
3. Always pass connectionId exactly as "{task['connection_id']}".
4. Start with discover_data using only the scalar fields connectionId and query. query must be a non-empty description. Omit kinds and limit entirely.
5. Never send an optional numeric argument. Omit maxRows on sql_execution and limit on discover_data and wiki_search; their defaults are correct. A number written as a string is rejected by the tool.
6. Inspect each relevant source with sl_read_source, whose arguments are the scalar strings connectionId and sourceName. Only use a sourceName that discover_data actually returned as a ref id; do not guess names. If discover_data returns no usable sources, enumerate the schema through sql_execution instead of guessing, and never conclude the tables do not exist without doing so. On SQLite, ONE call gets you the whole schema: `SELECT name, sql FROM sqlite_master WHERE type='table'` returns every table name together with its full CREATE statement. Read that once and move on to writing the query. Do NOT call pragma_table_info table by table -- that burns your turn budget before you answer. Use `SELECT * FROM pragma_table_info('<table>')` only if one table's CREATE statement came back truncated. There is no information_schema on SQLite, and a bare `PRAGMA ...` statement is rejected as a write operation.
7. Call sql_dialect_notes before drafting SQL.
8. Build one read-only SELECT/WITH query. Test it with sql_execution, passing only the scalar connectionId and the scalar sql. Never encode a JSON value as a string. Revise on execution errors and call sql_execution again as needed.
9. Do not stop after saying what you will do. Do not finish before at least one successful sql_execution. Never end a turn on an intention sentence.
10. End exactly with `FINAL ANSWER:` and one fenced sql block containing the tested query. The query itself must compute the requested result, preserving requested columns, grain, filters, dates, ordering, and limits. Do not compute the answer outside SQL.
'''


CONTINUE_PROMPT = '''Continue the same Spider2 task now. Your prior turn ended without a clean final SQL answer. Do not restate a plan. Immediately call the ktx MCP tool you said you would use next, test a complete candidate with sql_execution, and finish with exactly FINAL ANSWER: plus one fenced sql block. Never end on an intention sentence.'''


def _sql_execution_succeeded(item: dict[str, Any]) -> bool:
    """KTX sql_execution returns structured {headers, rows, rowCount} on success."""
    if item.get("status") != "completed" or item.get("error"):
        return False
    result = item.get("result")
    if not isinstance(result, dict):
        return False
    structured = result.get("structured_content")
    if isinstance(structured, dict) and "rowCount" in structured:
        return True
    for block in result.get("content") or []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            try:
                payload = json.loads(block["text"])
            except ValueError:
                continue
            if isinstance(payload, dict) and "rowCount" in payload:
                return True
    return False


def parse_codex_jsonl(output: str) -> dict[str, Any]:
    """Parse Codex --json SSE output into trajectory metadata.

    Returns per-turn usage, tool call counts, successful sql_execution calls,
    the last assistant message, and any errors. Multi-turn by design: one
    rollout can contain many turns, and we reconstruct the history from events.
    """
    turns: list[dict[str, Any]] = []
    current_turn: dict[str, Any] = {"messages": [], "tool_calls": [], "usage": {}}
    successful_executes: list[str] = []
    tool_counts: Counter[str] = Counter()
    errors: list[str] = []

    for line in output.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue

        if event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                current_turn["messages"].append(str(item.get("text") or ""))
            elif item.get("type") == "mcp_tool_call":
                name = str(item.get("tool") or "")
                status = item.get("status")
                tool_counts[f"{name}:{status}"] += 1
                current_turn["tool_calls"].append(item)
                if name == "sql_execution" and _sql_execution_succeeded(item):
                    args = item.get("arguments") or {}
                    sql = args.get("sql") if isinstance(args, dict) else None
                    if isinstance(sql, str) and sql.strip():
                        successful_executes.append(sql.strip().rstrip(";") + ";")
            elif item.get("type") == "error":
                errors.append(str(item.get("message") or "error"))

        elif event.get("type") == "turn.completed":
            turn_usage = event.get("usage") or {}
            current_turn["usage"] = {
                "input_tokens": int(turn_usage.get("input_tokens") or 0),
                "output_tokens": int(turn_usage.get("output_tokens") or 0),
            }
            turns.append(current_turn)
            current_turn = {"messages": [], "tool_calls": [], "usage": {}}

        elif event.get("type") in ("turn.failed", "error"):
            errors.append(json.dumps(event, ensure_ascii=False)[:1000])

    last_msg = turns[-1]["messages"][-1] if turns and turns[-1]["messages"] else ""
    # Two different senses of "turn", kept separate on purpose:
    #   n_rounds      codex `exec` invocations (1 + continuations we drove)
    #   n_model_calls individual model responses -- THIS is what slime counts as
    #                 a TurnRecord, and what qwen_traj_findings.md measured
    #                 (p50=16 for Qwen3.5-27B). A single round routinely holds
    #                 a dozen model calls, so reporting rounds as "turns" made
    #                 the smoke test look like it had collapsed to one call.
    n_model_calls = sum(len(t["messages"]) + len(t["tool_calls"]) for t in turns)
    return {
        "turns": turns,
        "n_rounds": len(turns),
        "n_turns": n_model_calls,
        "n_tool_calls": sum(len(t["tool_calls"]) for t in turns),
        "last_message": last_msg,
        "successful_executes": successful_executes,
        "tool_counts": dict(tool_counts),
        "errors": errors,
    }


def run_codex_subprocess(
    task: dict[str, Any],
    proxy_url: str,
    served_model_name: str,
    api_key: str | None = None,
    max_turns: int = 20,
    timeout_per_turn: int = 180,
) -> tuple[str, str]:
    """Drive codex through one task and return raw (stdout, stderr).

    This is the transport half of a rollout: it owns the fresh CODEX_HOME, the
    config file, and the continuation loop, but computes no reward. RL calls it
    directly because the adapter -- not this function -- is what captures the
    token ids; ``run_codex_rollout`` wraps it for standalone evaluation.

    ``api_key`` becomes the bearer token codex sends on every request. Under RL
    that is the slime adapter's session id, which is how a multi-turn trajectory
    gets grouped into one chain.

    Each rollout gets its OWN COPY of the KTX project dir. Sharing one project
    across concurrent rollouts is not merely slower -- it fails hard: booting
    4 codex instances against a shared project reproducibly kills half of them
    with ``ktx: handshaking with MCP server failed: connection closed``, ~3.2s
    in, before a single model call. That was the entire "reward drops with
    concurrency" effect in the sweep (0 / 3 / 3 / 8 dead rollouts at 1 / 4 / 8 /
    16 workers). KTX writes ``.ktx/mcp.json`` and a lock under the project root,
    so concurrent boots race. The sweep's own launcher copies the project per
    node for this reason (``node_launcher.sh:60-66``).
    """
    with tempfile.TemporaryDirectory(prefix="codex_home_") as home_str:
        home = Path(home_str)
        workdir = home / "work"
        workdir.mkdir()

        # Private KTX project for this rollout. ktx.yaml + semantic layer are a
        # few MB; the per-rollout copy is far cheaper than a failed rollout.
        #
        # Private KTX project for this rollout, cloned from a node-local
        # template rather than from the shared filesystem -- see
        # _local_project_template().
        ktx_project = home / "ktx_project"
        shutil.copytree(_local_project_template(), ktx_project, ignore=_KTX_IGNORE)

        # The session id must reach the adapter as a Bearer token: that is how
        # slime groups a multi-turn trajectory into one chain. It CANNOT be
        # passed via OPENAI_API_KEY -- `requires_openai_auth = false` makes
        # codex omit the Authorization header entirely (verified with a probe
        # server: AUTH: None on every request). Every rollout then fell back to
        # the adapter's `"default"` session id, shared one asyncio Session.lock,
        # and serialized: 64 concurrent rollouts but `#running-req: 1` on the
        # engine throughout. `http_headers` sets the header unconditionally.
        write_codex_config(home, workdir, proxy_url, served_model_name,
                           api_key=api_key or "default",
                           ktx_project=ktx_project)

        env = os.environ.copy()
        env["CODEX_HOME"] = str(home)
        env["KTX_PROJECT_DIR"] = str(ktx_project)
        if api_key:
            env["OPENAI_API_KEY"] = api_key
        # An inherited HTTP proxy can swallow the loopback call to the local
        # adapter, so exclude it explicitly.
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(k, None)
        env["NO_PROXY"] = env["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

        cmd = [
            str(require_path(CODEX_BIN, "CODEX_BIN")), "exec", "--json", "--sandbox", "read-only",
            "--skip-git-repo-check", "-C", str(workdir), "-",
        ]

        prompt = initial_prompt(task)
        all_stdout: list[str] = []
        all_stderr: list[str] = []

        for _ in range(max_turns):
            try:
                proc = subprocess.run(
                    cmd, input=prompt, text=True, capture_output=True,
                    env=env, timeout=timeout_per_turn, cwd=str(workdir),
                )
                rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired as exc:
                rc = 124
                stdout = (exc.stdout.decode("utf-8", errors="replace")
                          if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
                stderr = (exc.stderr.decode("utf-8", errors="replace")
                          if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
                stderr += f"\nrollout timeout after {timeout_per_turn}s"

            all_stdout.append(stdout)
            all_stderr.append(stderr)

            parsed = parse_codex_jsonl(stdout)
            # Codex always emits "Model metadata for <name> not found" against a
            # locally served model; treating that as fatal used to end every
            # rollout after one round, mid-deliberation.
            fatal = [e for e in parsed["errors"] if "Model metadata" not in e]
            if rc != 0 or fatal or "FINAL ANSWER:" in parsed["last_message"].upper():
                break
            prompt = CONTINUE_PROMPT

        return "\n".join(all_stdout), "\n".join(all_stderr)


def run_codex_rollout(
    task: dict[str, Any],
    proxy_url: str,
    served_model_name: str,
    max_turns: int = 20,
    timeout_per_turn: int = 180,
    query_timeout: int = 30,
) -> dict[str, Any]:
    """Standalone rollout: run codex, extract SQL, score it.

    Used by the smoke test and the concurrency sweep. The RL path does NOT go
    through here -- it calls ``run_codex_subprocess`` and scores separately so
    the adapter can hand back token segments.
    """
    started = time.monotonic()
    full_stdout, full_stderr = run_codex_subprocess(
        task, proxy_url, served_model_name,
        max_turns=max_turns, timeout_per_turn=timeout_per_turn,
    )
    elapsed = time.monotonic() - started

    parsed = parse_codex_jsonl(full_stdout)
    from .task_data import extract_sql, sql_reward
    sql = extract_sql(parsed["last_message"])
    # A run that solved the task but fumbled the FINAL ANSWER format still
    # carries signal; fall back to the last SQL KTX actually executed.
    if not sql and parsed["successful_executes"]:
        sql = parsed["successful_executes"][-1]
    reward_info = sql_reward(task, sql, timeout_s=query_timeout)

    return {
        "sample_id": task["sample_id"],
        "n_turns": parsed["n_turns"],
        "stdout": full_stdout,
        "stderr": full_stderr,
        "parsed": parsed,
        "sql": sql,
        "reward": reward_info["reward"],
        "reward_detail": reward_info,
        "elapsed_s": elapsed,
    }
