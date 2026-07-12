"""Outbound adapter for upstreams that speak OpenAI Chat Completions.

Used for OpenAI-compatible providers that only expose ``/v1/chat/completions``
(DeepSeek, Together, Groq, ...). Rendering the IR here also produces clean
messages (no ``content: null`` on tool-only turns), which is what some of those
providers require.
"""

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
    StreamEvent,
    ToolCall,
)

_OPENAI_TO_IR_FINISH = {
    "stop": FINISH_STOP,
    "length": FINISH_LENGTH,
    "tool_calls": FINISH_TOOL_CALLS,
    "content_filter": FINISH_CONTENT_FILTER,
}


class OpenAIChatOutbound:
    """Canonical IR -> OpenAI Chat Completions upstream."""

    def render_request(self, req: ChatRequest) -> tuple[dict[str, Any], str]:
        messages: list[dict[str, Any]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            if m.role == "tool":
                for tr in m.tool_results:
                    messages.append(
                        {"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content}
                    )
                continue
            if m.role not in ("user", "assistant"):
                continue
            msg: dict[str, Any] = {"role": m.role}
            if m.content:
                msg["content"] = m.text
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            elif "content" not in msg:
                msg["content"] = ""
            messages.append(msg)

        body: dict[str, Any] = {"model": req.model, "messages": messages}
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.top_p is not None:
            body["top_p"] = req.top_p
        if req.stop:
            body["stop"] = req.stop
        if req.stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        if req.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in req.tools
            ]
        if req.tool_choice is not None:
            body["tool_choice"] = req.tool_choice
        body.update(req.extra)
        return body, "/v1/chat/completions"

    def parse_response(self, body: bytes, content_type: str) -> ChatResponse:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return ChatResponse()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {}) or {}
        resp = ChatResponse(
            model=data.get("model", ""),
            content=message.get("content") or "",
            finish_reason=_OPENAI_TO_IR_FINISH.get(choice.get("finish_reason", "stop"), FINISH_STOP),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            response_id=data.get("id", ""),
        )
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", "")
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except (json.JSONDecodeError, ValueError):
                args = {}
            resp.tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=args or {}))
        return resp

    async def parse_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[StreamEvent]:
        buffer = b""
        # Tool calls stream as incremental delta.tool_calls fragments keyed by
        # "index": the id/name arrive first, then arguments accumulate across
        # chunks. Assemble them and emit complete tool_call events at finish.
        pending: dict[int, dict[str, Any]] = {}

        def flush_tools() -> list[StreamEvent]:
            events: list[StreamEvent] = []
            for idx in sorted(pending):
                events.append(StreamEvent(type="tool_call", tool_call=_assemble(pending[idx])))
            pending.clear()
            return events

        async for chunk in chunks:
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or line.startswith(b":") or not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    for ev in flush_tools():
                        yield ev
                    return
                try:
                    data = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    continue
                usage = data.get("usage")
                if usage:
                    yield StreamEvent(
                        type="usage",
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                    )
                choice = (data.get("choices") or [{}])
                if not choice:
                    continue
                choice = choice[0]
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    yield StreamEvent(type="text", text=text)
                for tc in delta.get("tool_calls", []) or []:
                    idx = tc.get("index", 0)
                    slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
                fr = choice.get("finish_reason")
                if fr:
                    for ev in flush_tools():
                        yield ev
                    yield StreamEvent(
                        type="finish",
                        finish_reason=_OPENAI_TO_IR_FINISH.get(fr, FINISH_STOP),
                    )
        for ev in flush_tools():
            yield ev


def _assemble(slot: dict[str, Any]) -> ToolCall:
    """Turn an accumulated {id,name,args} slot into a ToolCall (args -> dict)."""
    raw = slot.get("args", "")
    try:
        args = json.loads(raw) if raw else {}
        if not isinstance(args, dict):
            args = {}
    except (json.JSONDecodeError, ValueError):
        args = {}
    return ToolCall(id=slot.get("id", ""), name=slot.get("name", ""), arguments=args)
