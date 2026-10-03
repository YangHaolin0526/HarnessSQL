"""Small, dependency-light model adapters for synthesis and trajectory rollout."""

from __future__ import annotations

import json
import http.client
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class EmptyResponseExhausted(RuntimeError):
    """The endpoint returned no usable assistant payload on every allowed attempt."""

    def __init__(self, model: str, attempts: int, usage: dict[str, int] | None = None) -> None:
        self.model = model
        self.attempts = attempts
        self.usage = usage or {}
        super().__init__(f"model {model!r} returned an empty response {attempts} times")


def _message_has_response(message: dict[str, Any]) -> bool:
    """Recognize assistant content/refusal text or a native tool call.

    Reasoning-only output is still empty from an agent loop's perspective: it
    gives the caller no message or action to append to the conversation.
    """

    def has_text(value: Any) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return any(has_text(item) for item in value)
        if isinstance(value, dict):
            return any(
                has_text(value.get(key))
                for key in ("text", "content", "thinking", "reasoning", "refusal")
            )
        return False

    if message.get("tool_calls"):
        return True
    for key in ("content", "refusal"):
        if has_text(message.get(key)):
            return True
    return False


class ModelBackend(Protocol):
    model: str
    supports_native_tools: bool

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> "ModelReply": ...


@dataclass
class ModelReply:
    content: str
    tool_calls: list[dict[str, Any]]
    raw: dict[str, Any]
    usage: dict[str, Any]
    reasoning_summary: str = ""

    def as_assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class OpenAICompatibleBackend:
    """Call any OpenAI-compatible ``/chat/completions`` endpoint.

    The adapter uses urllib rather than importing the OpenAI SDK, keeping catalog
    and validation commands usable in a minimal Python environment.
    """

    supports_native_tools = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 300.0,
        extra_headers: dict[str, str] | None = None,
        empty_response_retries: int = 3,
        json_response: bool = False,
        reasoning_effort: str = "",
        transport_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.extra_headers = extra_headers or {}
        self.json_response = json_response
        self.reasoning_effort = reasoning_effort
        if transport_retries < 1:
            raise ValueError("transport_retries must be at least 1")
        self.transport_retries = transport_retries
        if empty_response_retries < 1:
            raise ValueError("empty_response_retries must be at least 1")
        self.empty_response_retries = empty_response_retries

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelReply:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"API key environment variable {self.api_key_env!r} is unset; "
                "credentials are never accepted on the command line"
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.json_response:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        usage_totals: dict[str, int] = {}
        for empty_attempt in range(1, self.empty_response_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            for transport_attempt in range(1, self.transport_retries + 1):
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                        raw = json.loads(response.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode("utf-8", errors="replace")[:4000]
                    if exc.code in {408, 429, 500, 502, 503, 504} and transport_attempt < self.transport_retries:
                        time.sleep(2 ** (transport_attempt - 1))
                        continue
                    raise RuntimeError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
                except (
                    urllib.error.URLError,
                    TimeoutError,
                    ConnectionResetError,
                    http.client.IncompleteRead,
                    http.client.RemoteDisconnected,
                ) as exc:
                    if transport_attempt < self.transport_retries:
                        time.sleep(2 ** (transport_attempt - 1))
                        continue
                    reason = getattr(exc, "reason", str(exc))
                    raise RuntimeError(f"model endpoint request failed: {reason}") from exc
            try:
                message = raw["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"invalid chat-completions response shape: {str(raw)[:2000]}") from exc
            for key, value in (raw.get("usage") or {}).items():
                if isinstance(value, int):
                    usage_totals[key] = usage_totals.get(key, 0) + value
            if _message_has_response(message):
                raw["empty_response_attempts"] = empty_attempt
                raw["empty_response_retries_limit"] = self.empty_response_retries
                raw["usage_including_empty_retries"] = usage_totals
                return ModelReply(
                    content=message.get("content") or "",
                    tool_calls=message.get("tool_calls") or [],
                    raw=raw,
                    usage=usage_totals,
                    reasoning_summary=str(
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or ""
                    ),
                )
        raise EmptyResponseExhausted(self.model, self.empty_response_retries, usage_totals)


class AnthropicMessagesBackend:
    """Call an Anthropic-compatible ``/v1/messages`` endpoint.

    This is intentionally a direct adapter rather than routing synthesis
    through Codex: question generation needs one JSON object, while teacher
    trajectory collection continues to use the signature-capturing bridge.
    """

    supports_native_tools = True

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        timeout_seconds: float = 900.0,
        empty_response_retries: int = 3,
        reasoning_effort: str = "",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.reasoning_effort = reasoning_effort
        if empty_response_retries < 1:
            raise ValueError("empty_response_retries must be at least 1")
        self.empty_response_retries = empty_response_retries

    @property
    def endpoint(self) -> str:
        return self.base_url + "/v1/messages?beta=true"

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelReply:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"API key environment variable {self.api_key_env!r} is unset; "
                "credentials are never accepted on the command line"
            )
        system_parts = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "system"
        ]
        anthropic_messages = [
            {"role": message["role"], "content": message.get("content") or ""}
            for message in messages
            if message.get("role") in {"user", "assistant"}
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
            "stream": True,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self.reasoning_effort or "high"},
        }
        # The Opus-5 route rejects temperature entirely. Other compatible
        # deployments may still use it, so retain it only when supported.
        if "opus-5" not in self.model.casefold():
            payload["temperature"] = temperature
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description") or "",
                    "input_schema": tool["function"].get("parameters")
                    or {"type": "object", "properties": {}},
                }
                for tool in tools
                if tool.get("type") == "function" and tool.get("function", {}).get("name")
            ]
        headers = {
            "x-api-key": api_key,
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "interleaved-thinking-2025-05-14",
            "content-type": "application/json",
        }
        usage_totals: dict[str, int] = {}
        for empty_attempt in range(1, self.empty_response_retries + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_text = response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:4000]
                raise RuntimeError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"model endpoint request failed: {exc.reason}") from exc
            try:
                raw = json.loads(response_text)
            except ValueError:
                blocks: list[dict[str, Any]] = []
                current: dict[str, Any] | None = None
                stream_usage: dict[str, Any] = {}
                for line in response_text.splitlines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except ValueError:
                        continue
                    event_type = event.get("type")
                    if event_type == "message_start":
                        stream_usage.update((event.get("message") or {}).get("usage") or {})
                    elif event_type == "message_delta":
                        stream_usage.update(event.get("usage") or {})
                    elif event_type == "content_block_start":
                        block = event.get("content_block") or {}
                        if block.get("type") == "thinking":
                            current = {
                                "type": "thinking",
                                "thinking": block.get("thinking") or "",
                            }
                        elif block.get("type") == "text":
                            current = {"type": "text", "text": block.get("text") or ""}
                        elif block.get("type") == "tool_use":
                            current = {
                                "type": "tool_use",
                                "id": block.get("id"),
                                "name": block.get("name"),
                                "input_json": "",
                            }
                        else:
                            current = None
                    elif event_type == "content_block_delta" and current is not None:
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta":
                            current["text"] += delta.get("text") or ""
                        elif delta.get("type") == "thinking_delta":
                            current["thinking"] += delta.get("thinking") or ""
                        elif delta.get("type") == "input_json_delta":
                            current["input_json"] += delta.get("partial_json") or ""
                    elif event_type == "content_block_stop" and current is not None:
                        if current.get("type") == "tool_use":
                            try:
                                current["input"] = json.loads(current.pop("input_json") or "{}")
                            except ValueError:
                                current["input"] = {}
                        blocks.append(current)
                        current = None
                raw = {"content": blocks, "usage": stream_usage, "stream": True}
            for key, value in (raw.get("usage") or {}).items():
                if isinstance(value, int):
                    usage_totals[key] = usage_totals.get(key, 0) + value
            text = "".join(
                str(block.get("text") or "")
                for block in raw.get("content") or []
                if isinstance(block, dict) and block.get("type") == "text"
            )
            reasoning_summary = "\n".join(
                str(block.get("thinking") or "").strip()
                for block in raw.get("content") or []
                if isinstance(block, dict)
                and block.get("type") == "thinking"
                and str(block.get("thinking") or "").strip()
            )
            tool_calls = [
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                }
                for block in raw.get("content") or []
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            if text.strip() or tool_calls:
                raw["empty_response_attempts"] = empty_attempt
                raw["empty_response_retries_limit"] = self.empty_response_retries
                return ModelReply(
                    content=text,
                    tool_calls=tool_calls,
                    raw=raw,
                    usage=usage_totals,
                    reasoning_summary=reasoning_summary,
                )
        raise EmptyResponseExhausted(self.model, self.empty_response_retries, usage_totals)


class LocalTransformersBackend:
    """Run a local Hugging Face causal LM in-process.

    Tool use is represented with the textual protocol documented in
    ``trajectory.py``.  Loading is lazy, so importing the synthesis package does
    not require torch/transformers.
    """

    supports_native_tools = False

    def __init__(
        self,
        *,
        model: str,
        device_map: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self.model = model
        self.device_map = device_map
        self.trust_remote_code = trust_remote_code
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "local-transformers backend requires torch and transformers; "
                "use the sql_cli conda environment or install them explicitly"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map=self.device_map,
            torch_dtype="auto",
            trust_remote_code=self.trust_remote_code,
        )

    @staticmethod
    def _flatten_messages(messages: list[dict[str, Any]]) -> str:
        parts = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content") or ""
            if message.get("tool_calls"):
                content += "\n" + json.dumps(message["tool_calls"], ensure_ascii=False)
            parts.append(f"{role}:\n{content}")
        parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelReply:
        self._load()
        assert self._tokenizer is not None and self._model is not None
        local_messages = messages
        if tools:
            schema_text = json.dumps(tools, ensure_ascii=False)
            local_messages = [
                {
                    "role": "system",
                    "content": (
                        "Available tool schemas follow. For a tool call, emit exactly "
                        "<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>.\n"
                        + schema_text
                    ),
                },
                *messages,
            ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                prompt = self._tokenizer.apply_chat_template(
                    local_messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                prompt = self._flatten_messages(local_messages)
        else:
            prompt = self._flatten_messages(local_messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generate_args: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_args["temperature"] = temperature
        output = self._model.generate(**generate_args)
        generated = output[0, inputs["input_ids"].shape[1] :]
        content = self._tokenizer.decode(generated, skip_special_tokens=True)
        return ModelReply(
            content=content,
            tool_calls=[],
            raw={"backend": "local-transformers", "model": self.model},
            usage={"completion_tokens": int(generated.shape[0])},
        )


def add_backend_arguments(parser: Any) -> None:
    parser.add_argument(
        "--backend",
        choices=["openai-compatible", "anthropic-messages", "local-transformers"],
        required=True,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base; can also point to a local vLLM/SGLang server",
    )
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--json-response",
        action="store_true",
        help="request OpenAI-compatible JSON-object mode (useful for synthesis blueprints)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="",
        help="optional OpenAI-compatible reasoning effort forwarded to the endpoint",
    )
    parser.add_argument(
        "--empty-response-retries",
        type=int,
        default=3,
        help="maximum total attempts for one empty API response before the sample is skipped",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300.0,
        help="wall-clock timeout for one API transport attempt",
    )
    parser.add_argument(
        "--transport-retries",
        type=int,
        default=3,
        help="transport/HTTP retry attempts; separate from empty-response retries",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")


def backend_from_args(args: Any) -> ModelBackend:
    if args.backend == "openai-compatible":
        return OpenAICompatibleBackend(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            empty_response_retries=args.empty_response_retries,
            json_response=args.json_response,
            reasoning_effort=args.reasoning_effort,
            transport_retries=args.transport_retries,
        )
    if args.backend == "anthropic-messages":
        return AnthropicMessagesBackend(
            model=args.model,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            empty_response_retries=args.empty_response_retries,
            reasoning_effort=args.reasoning_effort,
        )
    return LocalTransformersBackend(
        model=args.model,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
