"""Codex-compatible patches for slime's ``OpenAIAdapter``.

``OpenAIAdapter`` already speaks the Responses API and captures token ids
natively, which is exactly what RL needs -- but four details make it unable to
drive the Codex CLI + KTX MCP harness unmodified. Each was verified by reading
``slime/agent/adapters/openai.py`` and ``codex-rs``, not assumed:

1. **MCP namespace tools are silently dropped.** ``_normalize_tool`` (line 145)
   opens with ``if tool.get("type") != "function": return None``. Codex
   advertises its MCP surface as ``{"type": "namespace", "name": "mcp__ktx",
   "tools": [...]}``, so every KTX tool disappears and the model is offered an
   empty toolset. It then cannot call ``sql_execution`` at all.

2. **Reasoning items poison the prompt.** (Now largely moot: the training
   script no longer sets --sglang-reasoning-parser, so SGLang output is not
   split and no reasoning item is produced in the first place. The branch stays
   because a caller that does set a reasoning parser still needs it.) ``_responses_input_to_messages``
   (line 181) has no ``reasoning`` branch, so a reasoning item falls to the
   final ``else`` and is appended as a **user** message containing the flattened
   JSON of the model's own thinking. On the next turn the model sees its private
   reasoning replayed as if the user had said it.

3. **No tool-output truncation.** The standalone sweep proxy caps tool text with
   ``MAX_TOOL_OUTPUT`` (``openai_responses_proxy.py:188``); the adapter has no
   equivalent. Measured KTX observations reach p99=4,076 and max=10,169 tokens
   (``qwen_traj_findings.md``), so an untruncated wide result set can consume a
   large share of the context in a single turn.

4. **The SSE stream never emits ``response.output_item.done``**, so codex
   receives a syntactically valid stream carrying zero items. See the block
   above ``stream_response`` -- this makes every affected rollout score
   ``no_sql turns=0``.

5. **Tool-call arguments are rendered as a JSON string.** Qwen3.5's chat
   template iterates them (``chat_template.jinja:120``:
   ``for args_name, args_value in tool_call.arguments|items``), so a string
   raises ``TypeError: Can only get item pairs from a mapping``. Only reachable
   once the model actually calls a tool, which is why fixing (4) is what
   exposed it repeatedly under concurrent tool-call load. The same replay path
   also has to re-flatten the tool name, since codex echoes back the
   ``namespace`` we restored and the model was offered the flattened names.

Because the aiohttp routes bind the module-level handlers directly
(``self.app.router.add_post("/v1/responses", _handle_responses)``), there is no
instance method to override -- subclassing alone cannot reach these functions.
We therefore patch the module's own globals once, at import time, wrapping the
originals rather than copying their bodies so upstream fixes are inherited.
"""

from __future__ import annotations

import contextvars
import secrets
import json
from collections import OrderedDict
import logging
import os
from typing import Any

import asyncio

from aiohttp import web

import slime.agent.adapters.openai as _oa

logger = logging.getLogger(__name__)

# Char budget for a single tool result fed back to the model. ~4 chars/token in
# these transcripts, so 6000 ~= 1500 tokens, matching the agreed 1024-1536
# target. Overridable per run.
MAX_TOOL_OUTPUT_CHARS = int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "6000"))

_PATCHED = False

# Flat function name -> (namespace, real tool name).
#
# Codex advertises MCP tools as a namespace spec and expects the namespace back
# on the function_call item: ``build_tool_call``
# (``codex-rs/core/src/tools/router.rs:155``) rebuilds
# ``ToolName::new(namespace, name)`` and looks *that* up in its registry, so
# handing back the flattened ``mcp__ktx__sql_execution`` with no namespace
# yields a ToolName the registry does not contain and the call fails. The sweep
# proxy keeps the same reverse map (``openai_responses_proxy.py:220``).
#
# The KTX toolset is identical for every task, so one process-wide map is
# enough; entries are only ever added, never rewritten with a different target.
_TOOL_NAME_MAP: dict[str, tuple[str, str]] = {}


def _flatten_namespace_tools(tools: list[dict] | None) -> list[dict] | None:
    """Expand ``{"type": "namespace", "tools": [...]}`` into flat function tools.

    Codex names the resulting functions ``mcp__<namespace>__<tool>``; the same
    convention the sweep's proxy used, and what the model sees in the
    ``tools_offered`` field of every captured COT record. The reverse mapping is
    recorded in ``_TOOL_NAME_MAP`` so the function_call we send back can carry
    the namespace codex needs to resolve the tool.
    """
    if not tools:
        return tools
    flat: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            ns = tool.get("name") or "mcp"
            for sub in tool.get("tools") or []:
                if not isinstance(sub, dict):
                    continue
                name = sub.get("name")
                if not name:
                    continue
                flat_name = f"{ns}__{name}"
                _TOOL_NAME_MAP[flat_name] = (ns, name)
                flat.append({
                    "type": "function",
                    "function": {
                        "name": flat_name,
                        "description": sub.get("description", ""),
                        "parameters": (sub.get("parameters")
                                       or sub.get("input_schema")
                                       or {"type": "object", "properties": {}}),
                    },
                })
        else:
            flat.append(tool)
    return flat or None


def _split_tool_name(flat: str) -> tuple[str | None, str]:
    """Flat function name -> (namespace, real name); identity if not namespaced."""
    entry = _TOOL_NAME_MAP.get(flat)
    if entry is None:
        return None, flat
    return entry


def _truncate_tool_output(value: Any, limit: int) -> Any:
    """Cap a tool result, leaving an explicit marker so the model knows."""
    if not isinstance(value, str) or len(value) <= limit:
        return value
    dropped = len(value) - limit
    return value[:limit] + f"\n... [truncated {dropped} chars by RL harness]"


def _hoist_system_messages(messages: list[dict]) -> list[dict]:
    """Merge every system/developer message into ONE leading system message.

    Qwen3.5's chat template raises ``System message must be at the beginning.``
    if a system role appears anywhere but index 0. Codex trips this two ways:
    it sends ``instructions`` (which the adapter renders as a leading system
    message) AND a ``developer`` role item inside ``input``, and on continuation
    turns the replayed history can carry a second system message after tool
    output. Concatenating them preserves both instruction sets -- dropping the
    later one would silently discard Codex's operating instructions.
    """
    systems: list[str] = []
    rest: list[dict] = []
    for m in messages:
        if m.get("role") in ("system", "developer"):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                systems.append(content)
            elif content:
                systems.append(str(content))
        else:
            rest.append(m)
    if not systems:
        return rest
    return [{"role": "system", "content": "\n\n".join(systems)}] + rest


def _reflatten_tool_name(item: dict[str, Any]) -> dict[str, Any]:
    """``{namespace: mcp__ktx, name: sql_execution}`` -> ``mcp__ktx__sql_execution``.

    Mirrors ``_flatten_namespace_tools``. Leaves plain (non-namespaced) tools
    and already-flattened names untouched, so it is safe to apply blindly.
    """
    ns = item.get("namespace")
    name = item.get("name") or ""
    if not ns or name.startswith(f"{ns}__"):
        return item
    item = dict(item)
    item["name"] = f"{ns}__{name}"
    return item


def _as_mapping(arguments: Any) -> dict[str, Any]:
    """Tool-call arguments as a dict, whatever shape they arrive in.

    Qwen3.5's chat template does ``tool_call.arguments|items``
    (``chat_template.jinja:120``), which raises
    ``TypeError: Can only get item pairs from a mapping`` on the JSON *string*
    that the OpenAI chat-completions convention (and therefore slime's
    ``_normalize_tool_call``) produces. The failure occurs once per replayed
    tool call as soon as the model starts calling tools.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments or "{}")
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _sse(payload: dict[str, Any]) -> bytes:
    return (
        f"event: {payload['type']}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    ).encode()


def _item_text(item: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for part in (item.get("content") or [])
        if isinstance(part, dict)
    )


# Raw generated text per assistant turn, so it can be replayed verbatim.
# Keyed by every item id the turn produced (message id and each call_id), since
# codex echoes those back. The value carries a turn key as well, so the several
# items of one turn collapse to a single assistant message instead of repeating
# the text once per tool call.
_RAW_TURN_TEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "slime_raw_turn_text", default=None)
_RAW_BY_ID: "OrderedDict[str, tuple[str, str]]" = OrderedDict()
_RAW_BY_ID_MAX = 20000

# SGLang may or may not trim the stop token; the template appends its own, so a
# trailing one here would double up and break the very alignment this restores.
_TURN_ENDERS = ("<|im_end|>", "<|endoftext|>")


def _remember_raw(item_ids: list[str], turn_key: str, raw: str) -> None:
    for ident in item_ids:
        if not ident:
            continue
        _RAW_BY_ID[ident] = (turn_key, raw)
        _RAW_BY_ID.move_to_end(ident)
    while len(_RAW_BY_ID) > _RAW_BY_ID_MAX:
        _RAW_BY_ID.popitem(last=False)


def _strip_enders(text: str) -> str:
    out = text
    changed = True
    while changed:
        changed = False
        for end in _TURN_ENDERS:
            if out.endswith(end):
                out = out[: -len(end)]
                changed = True
    return out


def _item_text(item: dict) -> str:
    """Visible text of a response `message` item, if it has any."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("output_text", "text", "input_text"):
            parts.append(part.get("text") or "")
    return "".join(parts)


def _verbatim_assistant_items(input_value: list) -> list:
    """Collapse each assistant turn back into the text the model really wrote.

    Identity cannot come from item ids: codex echoes `call_id` back but not the
    `msg_...` id we mint, so keying on ids alone let the `message` item through
    and the turn was rendered twice. A turn's visible text is always a
    substring of its raw output, though -- `parsed.text` is derived from it --
    so that is what identifies a message item as belonging to a turn we have
    raw text for.

    Anything we cannot vouch for is passed through untouched, so an unfamiliar
    replay renders the old way rather than silently losing content.
    """
    # Raw texts in play for this request, in the order their turns appear.
    known: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for item in input_value:
        if not isinstance(item, dict):
            continue
        ident = item.get("call_id") or item.get("id")
        found = _RAW_BY_ID.get(ident) if ident else None
        if found and found[0] not in seen_keys:
            seen_keys.add(found[0])
            known.append(found)

    def match_raw(item: dict) -> tuple[str, str] | None:
        ident = item.get("call_id") or item.get("id")
        found = _RAW_BY_ID.get(ident) if ident else None
        if found is not None:
            return found
        if item.get("type") != "message" or item.get("role") != "assistant":
            return None
        text = _item_text(item).strip()
        if not text:
            return None
        for key, raw in known:
            if text in raw:
                return key, raw
        return None

    out: list = []
    emitted: set[str] = set()
    for item in input_value:
        if not isinstance(item, dict) or item.get("type") not in ("message", "function_call"):
            out.append(item)
            continue
        found = match_raw(item)
        if found is None:
            out.append(item)
            continue
        turn_key, raw = found
        if turn_key in emitted:
            continue
        emitted.add(turn_key)
        out.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": raw}],
        })
    return out


def _shutdown_session_tasks_loopsafe(
    sid: str,
    closed: set,
    inflight: dict,
    *,
    wait_timeout: float = 5.0,
) -> Any:
    """Loop-aware replacement for ``slime.agent.adapters.common``'s version.

    The adapter registers each codex HTTP request as a task on the aiohttp
    server's event loop. ``finish_session`` is awaited from the rollout loop.
    When a trajectory is killed mid-request the task set is non-empty and
    upstream's ``asyncio.wait``/``gather`` raises ``got Future ... attached to
    a different loop`` -- from a ``finally`` block, terminating the run after
    a clean training step. A task belonging to another loop cannot be awaited
    from here, but it can be cancelled on its own loop.
    """

    async def _run() -> None:
        closed.add(sid)
        tasks = [t for t in inflight.pop(sid, ()) if not t.done()]
        if not tasks:
            return
        running = asyncio.get_running_loop()
        mine, theirs = [], []
        for task in tasks:
            (mine if task.get_loop() is running else theirs).append(task)

        for task in theirs:
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                # Its loop is already closed; the task is dead either way.
                pass

        if mine:
            _, pending = await asyncio.wait(mine, timeout=wait_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    return _run()


def apply_patches(max_tool_output_chars: int | None = None) -> None:
    """Patch the openai adapter module in place. Idempotent."""
    global _PATCHED
    limit = MAX_TOOL_OUTPUT_CHARS if max_tool_output_chars is None else max_tool_output_chars
    if _PATCHED:
        return

    orig_normalize_tools = _oa._normalize_tools
    orig_input_to_messages = _oa._responses_input_to_messages
    orig_response_output = _oa._response_output
    orig_normalize_tool_call = _oa._normalize_tool_call
    orig_run_turn = _oa._run_turn

    async def run_turn(request, body, messages):
        # _handle_responses looks _run_turn up as a module global at call time,
        # so rebinding it here takes effect even though the aiohttp routes were
        # registered against the original handlers.
        turn, parsed, in_tok, out_tok = await orig_run_turn(request, body, messages)
        try:
            tok = request.app[_oa.TOKENIZER_KEY]
            _RAW_TURN_TEXT.set(_strip_enders(tok.decode(turn.output_ids,
                                                        skip_special_tokens=False)))
        except Exception:
            _RAW_TURN_TEXT.set(None)
        return turn, parsed, in_tok, out_tok

    def normalize_tools(tools: list[dict] | None) -> list[dict] | None:
        # Flatten namespaces BEFORE upstream's per-tool filter sees them.
        return orig_normalize_tools(_flatten_namespace_tools(tools))

    def input_to_messages(input_value: Any, instructions: Any = None) -> list[dict]:
        # Drop reasoning items and truncate tool outputs before upstream runs.
        # Reasoning is dropped rather than re-emitted as an assistant message:
        # the chat template already re-renders thinking for the turns we train
        # on, and re-injecting a *summary* would put tokens in the prompt that
        # the model never generated, breaking the prompt/response alignment
        # merge_turns() relies on.
        if isinstance(input_value, list):
            input_value = _verbatim_assistant_items(input_value)
            cleaned: list[Any] = []
            for item in input_value:
                if isinstance(item, dict):
                    if item.get("type") == "reasoning":
                        continue
                    if item.get("type") == "function_call_output":
                        item = dict(item)
                        item["output"] = _truncate_tool_output(item.get("output", ""), limit)
                    elif item.get("type") == "function_call":
                        # Codex replays the namespace we handed it, but the
                        # model was offered the FLATTENED names, so history has
                        # to be re-flattened or the transcript disagrees with
                        # the tool list on every continuation turn.
                        item = _reflatten_tool_name(item)
                cleaned.append(item)
            input_value = cleaned
        msgs = orig_input_to_messages(input_value, instructions)
        return _hoist_system_messages(msgs)

    def response_output(parsed) -> list[dict[str, Any]]:
        """Upstream's items, plus the two fields codex needs to accept them.

        ``encrypted_content`` is set explicitly on reasoning items to match the
        shape the sweep proxy sends; the namespace split is what makes a KTX
        tool call resolvable (see ``_TOOL_NAME_MAP``).
        """
        items = []
        for item in orig_response_output(parsed):
            item = dict(item)
            if item.get("type") == "reasoning":
                item.setdefault("encrypted_content", None)
            elif item.get("type") == "function_call":
                ns, real = _split_tool_name(item.get("name") or "")
                if ns:
                    item["name"] = real
                    item["namespace"] = ns
            items.append(item)
        raw = _RAW_TURN_TEXT.get()
        if raw:
            _remember_raw([it.get("call_id") or it.get("id") for it in items],
                          secrets.token_hex(8), raw)
        return items

    async def stream_response(request, body, parsed, finish, in_tok, out_tok):
        """Emit the item events codex actually reads.

        Upstream sends only response.created / output_text.delta /
        response.completed. Codex builds its items exclusively from
        ``response.output_item.done``
        (``codex-rs/codex-api/src/sse/responses.rs:331``); ``response.completed``
        deserializes into ``ResponseCompleted { id, usage, end_turn }`` and its
        ``output`` array is never read. So codex saw a well-formed stream with
        zero items -- no agent message, no tool call, no FINAL ANSWER -- which
        surfaced downstream as ``reward=0.0 status=no_sql turns=0`` on every
        affected rollout (nonempty segments prove the model generated; the
        reply was simply discarded).

        The event order mirrors the sweep proxy, i.e. the configuration that
        measured 95-100% reward against this same codex build. Items come from
        ``_response_response`` so the streamed and non-streamed views cannot
        drift apart.
        """
        out = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await out.prepare(request)
        response = _oa._response_response(body, parsed, finish, in_tok, out_tok)

        await out.write(_sse({"type": "response.created", "response": response}))

        for index, item in enumerate(response.get("output") or []):
            await out.write(_sse({
                "type": "response.output_item.added",
                "output_index": index,
                "item": item,
            }))
            itype = item.get("type")
            if itype == "reasoning":
                summary = item.get("summary") or []
                text = summary[0].get("text", "") if summary else ""
                await out.write(_sse({
                    "type": "response.reasoning_summary_part.added",
                    "item_id": item.get("id"),
                    "output_index": index,
                    "summary_index": 0,
                }))
                await out.write(_sse({
                    "type": "response.reasoning_summary_text.done",
                    "item_id": item.get("id"),
                    "output_index": index,
                    "summary_index": 0,
                    "text": text,
                }))
            elif itype == "message":
                text = _item_text(item)
                if text:
                    await out.write(_sse({
                        "type": "response.output_text.delta",
                        "item_id": item.get("id"),
                        "output_index": index,
                        "content_index": 0,
                        "delta": text,
                    }))
            await out.write(_sse({
                "type": "response.output_item.done",
                "output_index": index,
                "item": item,
            }))

        await out.write(_sse({"type": "response.completed", "response": response}))
        return out

    def normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
        # Upstream serialises arguments to a JSON string; Qwen3.5's template
        # needs a mapping. See _as_mapping.
        out = orig_normalize_tool_call(call)
        function = out.get("function")
        if isinstance(function, dict):
            function["arguments"] = _as_mapping(function.get("arguments"))
        return out

    _oa._normalize_tools = normalize_tools
    _oa._responses_input_to_messages = input_to_messages
    _oa._response_output = response_output
    _oa._stream_response = stream_response
    _oa._normalize_tool_call = normalize_tool_call
    _oa._run_turn = run_turn

    import slime.agent.adapters.common as _common
    _common.shutdown_session_tasks = _shutdown_session_tasks_loopsafe

    _PATCHED = True
    logger.info("[text2sql] patched OpenAIAdapter: namespace tools, reasoning drop, "
                "system hoist, SSE output items, dict tool args, "
                "loop-safe session shutdown, tool output cap=%d chars", limit)
