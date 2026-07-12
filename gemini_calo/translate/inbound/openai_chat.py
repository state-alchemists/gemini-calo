"""Inbound adapter for the OpenAI Chat Completions protocol (/v1/chat/completions)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from gemini_calo.translate.ir import (
    FINISH_CONTENT_FILTER,
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    ChatRequest,
    ChatResponse,
    ContentPart,
    Message,
    StreamEvent,
    ToolCall,
    ToolDef,
    ToolResult,
)
from gemini_calo.translate.util import (
    make_chat_completion_id,
    openai_chat_sse_chunk,
    openai_sse_done,
    sse_data,
)

_IR_TO_OPENAI_FINISH = {
    FINISH_STOP: "stop",
    FINISH_LENGTH: "length",
    FINISH_TOOL_CALLS: "tool_calls",
    FINISH_CONTENT_FILTER: "content_filter",
}
_OPENAI_TO_IR_FINISH = {v: k for k, v in _IR_TO_OPENAI_FINISH.items()}


def parse_openai_content(content: Any) -> list[ContentPart]:
    """Parse OpenAI message ``content`` (string or content-part array) to IR parts."""
    if isinstance(content, str):
        return [ContentPart(type="text", text=content)]
    parts: list[ContentPart] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                itype = item.get("type")
                if itype in ("text", "input_text", "output_text"):
                    parts.append(ContentPart(type="text", text=item.get("text", "")))
                elif itype == "image_url":
                    url = item.get("image_url", "")
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    parts.append(ContentPart(type="image", image_url=url))
            elif isinstance(item, str):
                parts.append(ContentPart(type="text", text=item))
    return parts


def parse_openai_tools(body: dict[str, Any]) -> list[ToolDef]:
    """Parse OpenAI Chat ``tools`` (nested under ``function``) to IR ToolDefs."""
    tools: list[ToolDef] = []
    for t in body.get("tools", []) or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t)
        tools.append(
            ToolDef(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}) or {},
            )
        )
    return tools


def copy_common_params(body: dict[str, Any], req: ChatRequest) -> None:
    """Copy the sampling params shared by OpenAI Chat and Responses into IR."""
    if "temperature" in body:
        req.temperature = body["temperature"]
    if "top_p" in body:
        req.top_p = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        req.stop = [stop]
    elif isinstance(stop, list):
        req.stop = stop
    req.stream = bool(body.get("stream", False))
    if "tool_choice" in body:
        req.tool_choice = body["tool_choice"]


# Top-level OpenAI Chat keys the IR models explicitly; everything else is
# preserved verbatim in ``ChatRequest.extra`` for OpenAI-compatible upstreams.
_MODELLED_KEYS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "top_p",
        "stop",
        "stream",
        "tools",
        "tool_choice",
        "max_tokens",
    }
)


def _collect_extra(body: dict[str, Any], req: ChatRequest) -> None:
    """Preserve un-modelled OpenAI params (response_format, seed, penalties, ...).

    Only meaningful when the upstream is itself OpenAI-compatible; the OpenAI
    Chat outbound is the sole adapter that replays ``extra`` onto the wire.
    """
    for key, value in body.items():
        if key not in _MODELLED_KEYS:
            req.extra[key] = value


class OpenAIChatInbound:
    """OpenAI Chat Completions <-> canonical IR."""

    def parse_request(self, body: dict[str, Any]) -> ChatRequest:
        req = ChatRequest(model=body.get("model", ""))

        for msg in body.get("messages", []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            if role in ("system", "developer"):
                # Fold consecutive system messages into the system prompt.
                text = "".join(p.text for p in parse_openai_content(msg.get("content", "")))
                req.system = f"{req.system}\n{text}".strip() if req.system else text
                continue
            if role == "tool":
                req.messages.append(
                    Message(
                        role="tool",
                        tool_results=[
                            ToolResult(
                                tool_call_id=msg.get("tool_call_id", ""),
                                content=_as_text(msg.get("content", "")),
                            )
                        ],
                    )
                )
                continue

            message = Message(role=role, content=parse_openai_content(msg.get("content", "")))
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                message.tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=_load_args(fn.get("arguments", "")),
                    )
                )
            req.messages.append(message)

        if body.get("max_tokens") is not None:
            req.max_tokens = body["max_tokens"]
        copy_common_params(body, req)
        req.tools = parse_openai_tools(body)
        _collect_extra(body, req)
        return req

    def render_response(self, resp: ChatResponse) -> bytes:
        message: dict[str, Any] = {"role": "assistant", "content": resp.content or None}
        if resp.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id or make_chat_completion_id(),
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in resp.tool_calls
            ]
        out = {
            "id": resp.response_id or make_chat_completion_id(),
            "object": "chat.completion",
            "model": resp.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": _IR_TO_OPENAI_FINISH.get(resp.finish_reason, "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.total_tokens,
            },
        }
        return json.dumps(out).encode()

    async def render_stream(
        self, events: AsyncIterator[StreamEvent], model: str = ""
    ) -> AsyncIterator[bytes]:
        cid = make_chat_completion_id()
        first = True
        tool_index = 0
        finish_reason = FINISH_STOP
        saw_tool = False
        saw_usage = False
        prompt_tokens = 0
        completion_tokens = 0

        def role_delta() -> bytes:
            return sse_data(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
            )

        async for ev in events:
            if ev.type == "text" and ev.text:
                if first:
                    yield role_delta()
                    first = False
                yield openai_chat_sse_chunk(model=model, content=ev.text, completion_id=cid)
            elif ev.type == "tool_call" and ev.tool_call:
                if first:
                    yield role_delta()
                    first = False
                saw_tool = True
                tc = ev.tool_call
                yield sse_data(
                    {
                        "id": cid,
                        "object": "chat.completion.chunk",
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tool_index,
                                            "id": tc.id or make_chat_completion_id(),
                                            "type": "function",
                                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                tool_index += 1
            elif ev.type == "usage":
                saw_usage = True
                prompt_tokens = ev.prompt_tokens or prompt_tokens
                completion_tokens = ev.completion_tokens or completion_tokens
            elif ev.type == "finish":
                finish_reason = ev.finish_reason or FINISH_STOP

        if saw_tool and finish_reason == FINISH_STOP:
            finish_reason = FINISH_TOOL_CALLS
        yield openai_chat_sse_chunk(
            model=model, finish_reason=_IR_TO_OPENAI_FINISH.get(finish_reason, "stop"), completion_id=cid
        )
        if saw_usage:
            # Final usage-only chunk (empty choices), as OpenAI emits when
            # stream_options.include_usage is set.
            yield sse_data(
                {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
        yield openai_sse_done()


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.text for p in parse_openai_content(content))


def _load_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
