#!/usr/bin/env python3
"""Expose a small Responses API facade over a Chat Completions server.

SGLang 0.5.10 supports function calling on ``/v1/chat/completions`` but its
Responses endpoint accepts only built-in web/code tools.  The Codex benchmark
harness speaks Responses exclusively, so this localhost-only adapter performs
the lossless subset conversion needed by the SQL/KTX trajectory loop.

Request and response bodies are never logged.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import time
import urllib.error
import urllib.request
import uuid
from typing import Any


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text"))
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )


def responses_to_chat(payload: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    instructions = str(payload.get("instructions") or "")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    for item in payload.get("input") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            role = str(item.get("role") or "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _text(item.get("content"))})
        elif item_type == "function_call":
            call = {
                "id": str(item.get("call_id") or item.get("id") or "call_" + uuid.uuid4().hex),
                "type": "function",
                "function": {
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or "{}"),
                },
            }
            if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
                messages[-1]["tool_calls"] = [call]
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or ""),
                    "content": str(item.get("output") or ""),
                }
            )

    tools = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )

    result: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "stream": False,
        "max_tokens": int(payload.get("max_output_tokens") or 8192),
    }
    if tools:
        result["tools"] = tools
        result["tool_choice"] = payload.get("tool_choice") or "auto"
    for key in ("temperature", "top_p"):
        if payload.get(key) is not None:
            result[key] = payload[key]
    return result


def chat_to_responses(
    chat: dict[str, Any], request_payload: dict[str, Any]
) -> dict[str, Any]:
    choice = (chat.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: list[dict[str, Any]] = []
    tool_calls = message.get("tool_calls") or []
    content = message.get("content")
    # Some Chat Completions servers return an assistant preamble alongside a
    # function call. Responses clients treat an output message as a final
    # assistant answer, so forwarding both can terminate the agent loop after
    # one tool round. While calls are pending, expose only those calls.
    if not tool_calls and isinstance(content, str) and content.strip():
        output.append(
            {
                "id": "msg_" + uuid.uuid4().hex,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                        "logprobs": None,
                    }
                ],
            }
        )
    for call in tool_calls:
        function = call.get("function") or {}
        output.append(
            {
                "id": "fc_" + uuid.uuid4().hex,
                "type": "function_call",
                "status": "completed",
                "call_id": str(call.get("id") or "call_" + uuid.uuid4().hex),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or "{}"),
            }
        )

    usage = chat.get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    reasoning_tokens = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    return {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "created_at": int(chat.get("created") or time.time()),
        "status": "completed",
        "incomplete_details": None,
        "instructions": request_payload.get("instructions"),
        "model": chat.get("model") or request_payload.get("model"),
        "output": output,
        "parallel_tool_calls": True,
        "temperature": request_payload.get("temperature", 1.0),
        "tool_choice": request_payload.get("tool_choice", "auto"),
        "tools": request_payload.get("tools") or [],
        "top_p": request_payload.get("top_p", 1.0),
        "max_output_tokens": request_payload.get("max_output_tokens"),
        "reasoning": request_payload.get("reasoning"),
        "status_details": None,
        "usage": {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
        },
    }


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args: Any) -> None:
        return

    @property
    def upstream(self) -> str:
        return os.environ.get("CHAT_UPSTREAM", "http://127.0.0.1:18007").rstrip("/")

    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return self.rfile.read(length) if length else b""

    def _upstream(self, path: str, body: bytes | None = None) -> tuple[int, bytes, str]:
        headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
        request = urllib.request.Request(
            self.upstream + path,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.status, response.read(), response.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers.get("Content-Type", "application/json")

    def do_GET(self) -> None:
        status, body, content_type = self._upstream(self.path)
        self._send(status, body, content_type)

    def do_POST(self) -> None:
        raw = self._body()
        if not self.path.rstrip("/").endswith("/responses"):
            status, body, content_type = self._upstream(self.path, raw)
            self._send(status, body, content_type)
            return
        try:
            payload = json.loads(raw)
            chat_request = responses_to_chat(payload)
        except Exception as exc:
            body = json.dumps({"error": {"message": f"invalid Responses request: {type(exc).__name__}"}}).encode()
            self._send(400, body)
            return
        status, body, _ = self._upstream(
            "/v1/chat/completions", json.dumps(chat_request).encode()
        )
        if status >= 400:
            self._send(status, body)
            return
        try:
            response = chat_to_responses(json.loads(body), payload)
            converted = json.dumps(response).encode()
        except Exception as exc:
            converted = json.dumps({"error": {"message": f"invalid Chat response: {type(exc).__name__}"}}).encode()
            self._send(502, converted)
            return
        self._send(200, converted)


def main() -> int:
    port = int(os.environ.get("RESPONSES_CHAT_ADAPTER_PORT", "8130"))
    server = _ThreadingServer(("127.0.0.1", port), Handler)
    print(f"Responses-to-Chat adapter on 127.0.0.1:{port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
