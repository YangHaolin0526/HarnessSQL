"""slime rollout function for Text-to-SQL RL with the codex+ktx harness.

Wiring:

    slime rollout loop
      -> generate(args, sample, sampling_params)          [this file]
         -> PatchedOpenAIAdapter on a background aiohttp thread
         -> codex exec subprocess (config.toml -> adapter as an OpenAI provider)
            -> KTX MCP over stdio (sqlite, read-only)
            -> every model call goes back through the adapter, which renders the
               chat template, calls sglang /generate, and records token ids
      -> execution reward against the precomputed oracle hash
      -> merge_turns / fan_out_sample_segments -> list[Sample]

Why the adapter rather than the sweep's standalone proxy: the proxy returns
*text*, so training would have to re-tokenize the transcript and hope the
retokenization matches what was sampled. The adapter keeps the actual token ids
and per-token logprobs from the rollout engine, which is what GRPO needs.

Codex is configured entirely through ``$CODEX_HOME/config.toml`` -- it has no
``--api-base`` / ``--api-key`` / ``--tools`` flags. ``codex_rollout.py`` writes
that file; this module reuses it and only swaps the base_url to the adapter.

Length parameters come from measured trajectories (``qwen_traj_findings.md``,
Qwen3.5-27B over 135 sqlite tasks): per-turn output p99=1,484; peak prompt
max=29,199; whole trajectory max=29,831. The old ``10_train_rl_pinned.sh``
values (response 1024, max-tokens-per-gpu 4096) cannot hold a single real
trajectory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from slime.agent.trajectory import fan_out_sample_segments
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from .aiohttp_threaded import run_app_in_thread
from .codex_rollout import parse_codex_jsonl, run_codex_subprocess
from .patched_openai_adapter import apply_patches
from .task_data import extract_sql, load_tasks, sql_reward

logger = logging.getLogger(__name__)

# Wall-clock guard for one whole trajectory. Measured median latency is far
# below this; the guard exists so a single wedged rollout cannot stall the
# training step. Exceeding it aborts that sample only.
SQL_GENERATE_GUARD_SEC = int(os.environ.get("SQL_GENERATE_GUARD_SEC", "900"))
SQL_MAX_TURNS = int(os.environ.get("SQL_MAX_TURNS", "24"))
SQL_TURN_TIMEOUT_SEC = int(os.environ.get("SQL_TURN_TIMEOUT_SEC", "300"))
SQL_QUERY_TIMEOUT_SEC = int(os.environ.get("SQL_QUERY_TIMEOUT_SEC", "30"))
MAX_TOOL_OUTPUT_CHARS = int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "6000"))

# Ceiling on rollouts in flight at once, independent of the training batch size.
#
# SGLang's hybrid Qwen3.5 architecture allocates a fixed pool of Mamba/linear-
# attention recurrent state slots -- this run reported
# `Mamba Cache is allocated. max_mamba_cache_size: 287` and
# `max_running_requests=57`. Once the session-lock bug was fixed and all 64
# trajectories genuinely ran in parallel, the schedulers crashed in
# `memory_pool.py:609` (`req_index_to_mamba_index_mapping[select_index] = ...`)
# and the router then returned 500 for every request, so every rollout scored
# `no_sql`. Capping in-flight rollouts keeps the engine inside the limit while
# leaving rollout-batch-size x n-samples (the GRPO group structure) untouched:
# the excess simply waits instead of crashing the engine.
SQL_MAX_CONCURRENT_ROLLOUTS = int(os.environ.get("SQL_MAX_CONCURRENT_ROLLOUTS", "32"))
_ROLLOUT_GATE: asyncio.Semaphore | None = None


def _rollout_gate() -> asyncio.Semaphore:
    global _ROLLOUT_GATE
    if _ROLLOUT_GATE is None:
        _ROLLOUT_GATE = asyncio.Semaphore(SQL_MAX_CONCURRENT_ROLLOUTS)
    return _ROLLOUT_GATE

SHIM_BIND_HOST = os.environ.get("SHIM_BIND_HOST", "0.0.0.0")
SHIM_PORT = int(os.environ.get("SHIM_PORT", "18002"))

# Rollouts run codex as a blocking subprocess, so each occupies a thread for its
# entire lifetime (minutes). asyncio's default to_thread executor holds only
# min(32, cpu+4) threads, which caps true concurrency well below the 64
# trajectories a rollout step launches. These threads sit in waitpid, not in
# Python, so a large pool costs almost nothing.
SQL_ROLLOUT_THREADS = int(os.environ.get("SQL_ROLLOUT_THREADS", "256"))
_EXECUTOR: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(
            max_workers=SQL_ROLLOUT_THREADS, thread_name_prefix="sql-rollout"
        )
    return _EXECUTOR


async def _run_blocking(fn, /, **kwargs):
    """Run a blocking callable on the dedicated rollout executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor(), partial(fn, **kwargs))


class _State(metaclass=SingletonMeta):
    """Tokenizer + patched adapter + task index, built once per rollout worker."""

    def __init__(self, args) -> None:
        apply_patches(MAX_TOOL_OUTPUT_CHARS)

        # Import here so the patch is installed before the class is constructed.
        from slime.agent.adapters import OpenAIAdapter

        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

        # Codex runs as a local subprocess, so unlike the SWE example's remote
        # sandboxes it can reach the adapter on loopback. SLIME_HEAD_HOST stays
        # available as an override for a future remote-sandbox setup.
        public_host = os.environ.get("SLIME_HEAD_HOST", "127.0.0.1")

        self.adapter = OpenAIAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=SHIM_BIND_HOST,
            port=SHIM_PORT,
            thread_name="openai-adapter",
            runner_kwargs={"handler_cancellation": True},
        )
        self.adapter_url = f"http://{public_host}:{self.app_handle.port}/v1"

        self.tasks = {t["sample_id"]: t for t in load_tasks()}
        logger.info(
            "[text2sql] adapter=%s tasks=%d max_context_len=%s tool_output_cap=%d",
            self.adapter_url, len(self.tasks), self.max_context_len,
            MAX_TOOL_OUTPUT_CHARS,
        )


def _task_for(state: _State, sample: Sample) -> dict[str, Any] | None:
    md = sample.metadata or {}
    sid = md.get("sample_id") or getattr(sample, "label", None)
    if isinstance(sid, str) and sid in state.tasks:
        return state.tasks[sid]
    return None


def _abort(sample: Sample, reason: str) -> list[Sample]:
    sample.status = Sample.Status.ABORTED
    sample.reward = 0.0
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason}
    logger.warning("[text2sql] aborted: %s", reason)
    return [sample]


# Live count of rollouts between open_session and finish_session. Two bugs so
# far (the 32-slot default thread pool, and all rollouts sharing the "default"
# session lock) presented as healthy-looking logs while trajectories ran
# effectively serially, and both were caught only by counting processes by hand
# afterwards. Logging the peak makes the failure visible in the run itself.
_INFLIGHT = 0
_INFLIGHT_PEAK = 0


async def generate(args, sample: Sample, sampling_params: dict[str, Any]):
    """Per-sample agentic rollout: codex+ktx -> SQL -> execution reward."""
    # Wait for a slot before touching the engine. Held for the whole
    # trajectory, so at most SQL_MAX_CONCURRENT_ROLLOUTS sessions are ever
    # open against SGLang's fixed Mamba state pool.
    async with _rollout_gate():
        return await _generate_one(args, sample, sampling_params)


async def _generate_one(args, sample: Sample, sampling_params: dict[str, Any]):
    global _INFLIGHT, _INFLIGHT_PEAK
    state = _State(args)
    task = _task_for(state, sample)
    if task is None:
        return _abort(sample, "unknown_sample_id")

    if sample.session_id:
        session_id = sample.session_id
    elif sample.index is not None and sample.group_index is not None:
        session_id = f"sql-{task['sample_id']}-{sample.index}-{sample.group_index}"
    else:
        session_id = f"sql-{task['sample_id']}-{secrets.token_hex(8)}"
    sample.session_id = session_id

    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )

    _INFLIGHT += 1
    if _INFLIGHT > _INFLIGHT_PEAK:
        _INFLIGHT_PEAK = _INFLIGHT
        if _INFLIGHT_PEAK % 8 == 0 or _INFLIGHT_PEAK <= 4:
            logger.info("[text2sql] concurrent rollouts peak=%d", _INFLIGHT_PEAK)

    t0 = time.time()
    try:
        async with asyncio.timeout(SQL_GENERATE_GUARD_SEC):
            # Codex authenticates to the adapter with the session id as its
            # bearer token; that is how the adapter groups a multi-turn
            # trajectory into one chain.
            #
            # NOT asyncio.to_thread: that shares one default executor capped at
            # min(32, cpu+4) threads, and each rollout holds its thread for the
            # WHOLE multi-minute trajectory. With 64 concurrent rollouts that
            # silently serialized them into two waves -- measured 0.43
            # model-calls/s against the standalone sweep's 2.13 at one eighth
            # the concurrency, and every rollout then tripped the 900s guard.
            # A dedicated executor sized to the rollout count keeps them truly
            # concurrent; the threads are I/O-blocked on a subprocess, not
            # holding the GIL.
            stdout, _stderr = await _run_blocking(
                run_codex_subprocess,
                task=task,
                proxy_url=state.adapter_url,
                served_model_name=getattr(args, "sglang_served_model_name", None)
                or "slime-actor",
                api_key=session_id,
                max_turns=SQL_MAX_TURNS,
                timeout_per_turn=SQL_TURN_TIMEOUT_SEC,
            )

            parsed = parse_codex_jsonl(stdout)
            sql = extract_sql(parsed["last_message"])
            sql_source = "final_answer" if sql else "none"
            # Fall back to the last SQL that KTX actually executed successfully:
            # a run that solved the task but fumbled the FINAL ANSWER format
            # still carries signal, and scoring it 0 would train the model away
            # from correct SQL for a formatting reason.
            if not sql and parsed["successful_executes"]:
                sql = parsed["successful_executes"][-1]
                sql_source = "ktx_fallback"

            # Same executor, same reason: the reward runs a real SQLite query
            # (measured p99 18.5s, max 30s) and must not occupy a slot in the
            # shared default pool that rollouts also need.
            reward_info = await _run_blocking(
                sql_reward, task=task, sql=sql, timeout_s=SQL_QUERY_TIMEOUT_SEC
            )
            segments = await state.adapter.finish_session(session_id)
            if not segments:
                return _abort(sample, "adapter_session_empty")

            elapsed = time.time() - t0
            metadata = {
                **(sample.metadata or {}),
                "sample_id": task["sample_id"],
                "database_id": task["database_id"],
                "reward_status": reward_info["status"],
                "reward_detail": str(reward_info.get("detail"))[:300],
                "sql_source": sql_source,
                "n_turns": parsed["n_turns"],
                "elapsed_sec": elapsed,
            }
            fanned = fan_out_sample_segments(
                sample, segments, float(reward_info["reward"]),
                state.tokenizer, metadata=metadata,
            )
            # `status` alone cannot tell a broken harness from a model that
            # writes invalid SQL. Large-rollout diagnostics returned many
            # exec_error results while concurrent oracle replay stayed clean,
            # locating the errors in generated SQL rather than the reward path.
            # sql_reward puts the SQLite exception in
            # `detail`; without it here there is no way to tell "no such column"
            # (a truncated-schema problem) from "syntax error" (a sampling
            # problem), which need opposite fixes.
            logger.info(
                "[text2sql] %s reward=%.1f status=%s turns=%d segments=%d "
                "%.1fs inflight=%d src=%s sql=%r detail=%s",
                task["sample_id"], reward_info["reward"], reward_info["status"],
                parsed["n_turns"], len(fanned), elapsed, _INFLIGHT,
                sql_source, (sql or "")[:160].replace("\n", " "),
                str(reward_info.get("detail"))[:200],
            )
            return fanned

    except asyncio.TimeoutError:
        return _abort(sample, "wall_clock_timeout")
    except Exception as exc:
        logger.error("[text2sql] %s failed: %s\n%s",
                     task["sample_id"], exc, traceback.format_exc())
        return _abort(sample, f"exception:{type(exc).__name__}")
    finally:
        _INFLIGHT -= 1
        # Close before the next train step's release_memory_occupation; a
        # straggler from this trajectory would otherwise race sglang's idle
        # assertion.
        #
        # Best-effort, always. An exception raised in a `finally` replaces the
        # value the `try` already produced, so a failure here does not cost a
        # sample -- it discards a scored rollout and propagates out of
        # generate_and_rm into train.py, ending the run. A prior timeout burst
        # exposed this when closing half-open sessions raised "got Future
        # attached to a different loop".
        # That specific bug is fixed in patched_openai_adapter; this guard is
        # the invariant, so the next cleanup bug costs a log line instead.
        try:
            await state.adapter.finish_session(session_id)  # idempotent
        except Exception as exc:  # noqa: BLE001 - cleanup must not be fatal
            logger.warning("[text2sql] finish_session(%s) failed, continuing: %s: %s",
                           session_id, type(exc).__name__, exc)
