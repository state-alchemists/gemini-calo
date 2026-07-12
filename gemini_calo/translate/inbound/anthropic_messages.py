"""Inbound adapter for the Anthropic Messages protocol (/v1/messages).

This is what Claude Code and other Anthropic-SDK clients speak. Note it is the
*client* protocol here: calo accepts Anthropic Messages and can route it to any
upstream (Gemini, Bedrock Nova, an OpenAI-compatible endpoint) via the outbound
adapters.
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
    ContentPart,
    Message,
    StreamEvent,
    ToolCall,
    ToolDef,
    ToolResult,
)
from gemini_calo.translate.util import make_message_id, sse_event

_IR_TO_ANTHROPIC_STOP = {
    FINISH_STOP: "end_turn",
    FINISH_LENGTH: "max_tokens",
    FINISH_TOOL_CALLS: "tool_use",
    FINISH_CONTENT_FILTER: "end_turn",
}


class AnthropicMessagesInbound:
    """Anthropic Messages API <-> canonical IR."""

    def parse_request(self, body: dict[str, Any]) -> ChatRequest:
        req = ChatRequest(model=body.get("model", ""))

        system = body.get("system")
        if isinstance(system, str):
            req.system = system
        elif isinstance(system, list):
            req.system = "".join(
                b.get("text", "") for b in system if isinstance(b, dict)
            )

        for msg in body.get("messages", []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role in ("system", "developer"):
                # Some clients place system content in the messages array; the
                # canonical IR keeps it in `system` (providers like Bedrock Nova
                # reject a "system" role inside messages).
                text = _result_text(content) if not isinstance(content, str) else content
                req.system = f"{req.system}\n{text}".strip() if req.system else text
                continue

            if isinstance(content, str):
                req.messages.append(Message.of_text(role, content))
                continue

            parts: list[ContentPart] = []
            tool_calls: list[ToolCall] = []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append(ContentPart(type="text", text=block.get("text", "")))
                elif btype == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input", {}) or {},
                        )
                    )
                elif btype == "tool_result":
                    req.messages.append(
                        Message(
                            role="tool",
                            tool_results=[
                                ToolResult(
                                    tool_call_id=block.get("tool_use_id", ""),
                                    content=_result_text(block.get("content", "")),
                                )
                            ],
                        )
                    )
                elif btype == "image":
                    parts.append(ContentPart(type="image", image_url=""))

            if parts or tool_calls:
                req.messages.append(
                    Message(role=role, content=parts, tool_calls=tool_calls)
                )

        if body.get("max_tokens") is not None:
            req.max_tokens = body["max_tokens"]
        if "temperature" in body:
            req.temperature = body["temperature"]
        if "top_p" in body:
            req.top_p = body["top_p"]
        if "stop_sequences" in body:
            req.stop = body["stop_sequences"]
        req.stream = bool(body.get("stream", False))
        if "tool_choice" in body:
            req.tool_choice = body["tool_choice"]

        for t in body.get("tools", []) or []:
            if isinstance(t, dict):
                req.tools.append(
                    ToolDef(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        parameters=t.get("input_schema", {}) or {},
                    )
                )
        return req

    def render_response(self, resp: ChatResponse) -> bytes:
        content: list[dict[str, Any]] = []
        if resp.content:
            content.append({"type": "text", "text": resp.content})
        for tc in resp.tool_calls:
            content.append(
                {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
            )
        out = {
            "id": resp.response_id or make_message_id(),
            "type": "message",
            "role": "assistant",
            "model": resp.model,
            "content": content,
            "stop_reason": _IR_TO_ANTHROPIC_STOP.get(resp.finish_reason, "end_turn"),
            "stop_sequence": None,
            "usage": {
                "input_tokens": resp.prompt_tokens,
                "output_tokens": resp.completion_tokens,
            },
        }
        return json.dumps(out).encode()

    async def render_stream(
        self, events: AsyncIterator[StreamEvent], model: str = ""
    ) -> AsyncIterator[bytes]:
        msg_id = make_message_id()
        prompt_tokens = 0
        completion_tokens = 0
        stop_reason = FINISH_STOP
        saw_tool = False

        # Streaming state: whether message_start was emitted, the current open
        # content block index (-1 = none), and its type.
        state = {"started": False, "index": -1, "open": None}

        def start_message() -> bytes:
            state["started"] = True
            return sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                    },
                },
            )

        def close_block() -> list[bytes]:
            if state["open"] is None:
                return []
            idx = state["index"]
            state["open"] = None
            return [sse_event("content_block_stop", {"type": "content_block_stop", "index": idx})]

        async for ev in events:
            if ev.type == "text" and ev.text:
                if not state["started"]:
                    yield start_message()
                if state["open"] != "text":
                    for b in close_block():
                        yield b
                    state["index"] += 1
                    state["open"] = "text"
                    yield sse_event(
                        "content_block_start",
                        {"type": "content_block_start", "index": state["index"], "content_block": {"type": "text", "text": ""}},
                    )
                yield sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": state["index"], "delta": {"type": "text_delta", "text": ev.text}},
                )
            elif ev.type == "tool_call" and ev.tool_call:
                if not state["started"]:
                    yield start_message()
                for b in close_block():
                    yield b
                saw_tool = True
                tc = ev.tool_call
                state["index"] += 1
                idx = state["index"]
                state["open"] = "tool_use"
                yield sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "tool_use", "id": tc.id, "name": tc.name, "input": {}},
                    },
                )
                yield sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(tc.arguments)},
                    },
                )
                for b in close_block():
                    yield b
            elif ev.type == "usage":
                prompt_tokens = ev.prompt_tokens or prompt_tokens
                completion_tokens = ev.completion_tokens or completion_tokens
            elif ev.type == "finish":
                stop_reason = ev.finish_reason or FINISH_STOP

        if not state["started"]:
            yield start_message()
            state["index"] = 0
            state["open"] = "text"
            yield sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
        for b in close_block():
            yield b

        if saw_tool and stop_reason == FINISH_STOP:
            stop_reason = FINISH_TOOL_CALLS
        # Upstream usage typically arrives at the end of the stream, so
        # message_start could only report input_tokens=0. Surface the real
        # prompt-token count here alongside the output count.
        usage: dict[str, Any] = {"output_tokens": completion_tokens}
        if prompt_tokens:
            usage["input_tokens"] = prompt_tokens
        yield sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _IR_TO_ANTHROPIC_STOP.get(stop_reason, "end_turn"),
                    "stop_sequence": None,
                },
                "usage": usage,
            },
        )
        yield sse_event("message_stop", {"type": "message_stop"})


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") in (None, "text")
        )
    return str(content)
