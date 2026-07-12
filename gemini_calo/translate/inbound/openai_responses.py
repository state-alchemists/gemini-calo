"""Inbound adapter for the OpenAI Responses protocol (/v1/responses)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from gemini_calo.translate.inbound.openai_chat import (
    copy_common_params,
    parse_openai_content,
)
from gemini_calo.translate.ir import (
    FINISH_TOOL_CALLS,
    ChatRequest,
    ChatResponse,
    Message,
    StreamEvent,
    ToolCall,
    ToolDef,
    ToolResult,
)
from gemini_calo.translate.util import (
    make_message_id,
    make_response_id,
    sse_data,
)


class OpenAIResponsesInbound:
    """OpenAI Responses API <-> canonical IR."""

    def parse_request(self, body: dict[str, Any]) -> ChatRequest:
        req = ChatRequest(model=body.get("model", ""))

        instructions = body.get("instructions")
        if isinstance(instructions, str) and instructions:
            req.system = instructions

        input_data = body.get("input", [])
        if isinstance(input_data, str):
            req.messages.append(Message.of_text("user", input_data))
        elif isinstance(input_data, list):
            for item in input_data:
                if not isinstance(item, dict):
                    continue
                itype = item.get("type", "message")
                if itype == "message":
                    req.messages.append(
                        Message(
                            role=item.get("role", "user"),
                            content=parse_openai_content(item.get("content", [])),
                        )
                    )
                elif itype in ("input_text", "output_text"):
                    req.messages.append(Message.of_text("user", item.get("text", "")))
                elif itype == "function_call":
                    req.messages.append(
                        Message(
                            role="assistant",
                            tool_calls=[
                                ToolCall(
                                    id=item.get("call_id", item.get("id", "")),
                                    name=item.get("name", ""),
                                    arguments=_load_args(item.get("arguments", "")),
                                )
                            ],
                        )
                    )
                elif itype == "function_call_output":
                    req.messages.append(
                        Message(
                            role="tool",
                            tool_results=[
                                ToolResult(
                                    tool_call_id=item.get("call_id", ""),
                                    content=_as_text(item.get("output", "")),
                                )
                            ],
                        )
                    )

        if body.get("max_output_tokens") is not None:
            req.max_tokens = body["max_output_tokens"]
        elif body.get("max_tokens") is not None:
            req.max_tokens = body["max_tokens"]
        copy_common_params(body, req)
        req.tools = _parse_responses_tools(body)
        return req

    def render_response(self, resp: ChatResponse) -> bytes:
        output: list[dict[str, Any]] = []
        if resp.content:
            output.append(
                {
                    "type": "message",
                    "id": make_message_id(),
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": resp.content, "annotations": []}
                    ],
                }
            )
        for tc in resp.tool_calls:
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{make_response_id()}",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                    "status": "completed",
                }
            )

        out = {
            "id": resp.response_id or make_response_id(),
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": resp.model,
            "output": output,
            "usage": {
                "input_tokens": resp.prompt_tokens,
                "output_tokens": resp.completion_tokens,
                "total_tokens": resp.total_tokens,
            },
            "metadata": {},
        }
        return json.dumps(out).encode()

    async def render_stream(
        self, events: AsyncIterator[StreamEvent], model: str = ""
    ) -> AsyncIterator[bytes]:
        resp_id = make_response_id()
        item_id = make_message_id()
        seq = 0
        text_acc = ""
        text_started = False
        created = False
        tool_calls: list[ToolCall] = []
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        def nxt() -> int:
            nonlocal seq
            seq += 1
            return seq

        def base_response(status: str, output: list | None = None) -> dict[str, Any]:
            return {
                "id": resp_id,
                "object": "response",
                "created_at": 0,
                "status": status,
                "model": model,
                "output": output or [],
                "usage": usage,
                "metadata": {},
            }

        async for ev in events:
            if ev.type == "text" and ev.text:
                if not created:
                    yield sse_data({"type": "response.created", "sequence_number": nxt(), "response": base_response("in_progress")})
                    yield sse_data({"type": "response.in_progress", "sequence_number": nxt(), "response": base_response("in_progress")})
                    created = True
                if not text_started:
                    yield sse_data(
                        {
                            "type": "response.output_item.added",
                            "sequence_number": nxt(),
                            "output_index": 0,
                            "item": {"type": "message", "id": item_id, "status": "in_progress", "role": "assistant", "content": []},
                        }
                    )
                    yield sse_data(
                        {
                            "type": "response.content_part.added",
                            "sequence_number": nxt(),
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        }
                    )
                    text_started = True
                text_acc += ev.text
                yield sse_data(
                    {
                        "type": "response.output_text.delta",
                        "sequence_number": nxt(),
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": ev.text,
                    }
                )
            elif ev.type == "tool_call" and ev.tool_call:
                tool_calls.append(ev.tool_call)
            elif ev.type == "usage":
                usage = {
                    "input_tokens": ev.prompt_tokens,
                    "output_tokens": ev.completion_tokens,
                    "total_tokens": ev.prompt_tokens + ev.completion_tokens,
                }

        if not created:
            yield sse_data({"type": "response.created", "sequence_number": nxt(), "response": base_response("in_progress")})
            yield sse_data({"type": "response.in_progress", "sequence_number": nxt(), "response": base_response("in_progress")})

        output_items: list[dict[str, Any]] = []
        output_index = 0
        if text_started:
            yield sse_data(
                {"type": "response.output_text.done", "sequence_number": nxt(), "item_id": item_id, "output_index": 0, "content_index": 0, "text": text_acc}
            )
            yield sse_data(
                {"type": "response.content_part.done", "sequence_number": nxt(), "item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": text_acc, "annotations": []}}
            )
            message_item = {
                "type": "message",
                "id": item_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text_acc, "annotations": []}],
            }
            yield sse_data({"type": "response.output_item.done", "sequence_number": nxt(), "output_index": 0, "item": message_item})
            output_items.append(message_item)
            output_index = 1

        for tc in tool_calls:
            fc_id = f"fc_{make_response_id()}"
            args_str = json.dumps(tc.arguments)
            fc_item = {
                "type": "function_call",
                "id": fc_id,
                "call_id": tc.id,
                "name": tc.name,
                "arguments": args_str,
                "status": "completed",
            }
            yield sse_data(
                {"type": "response.output_item.added", "sequence_number": nxt(), "output_index": output_index, "item": {**fc_item, "status": "in_progress", "arguments": ""}}
            )
            yield sse_data(
                {"type": "response.function_call_arguments.delta", "sequence_number": nxt(), "item_id": fc_id, "output_index": output_index, "delta": args_str}
            )
            yield sse_data(
                {"type": "response.function_call_arguments.done", "sequence_number": nxt(), "item_id": fc_id, "output_index": output_index, "arguments": args_str}
            )
            yield sse_data(
                {"type": "response.output_item.done", "sequence_number": nxt(), "output_index": output_index, "item": fc_item}
            )
            output_items.append(fc_item)
            output_index += 1

        yield sse_data(
            {"type": "response.completed", "sequence_number": nxt(), "response": base_response("completed", output_items)}
        )


def _parse_responses_tools(body: dict[str, Any]) -> list[ToolDef]:
    """Responses tools are flat: ``{type:"function", name, description, parameters}``."""
    tools: list[ToolDef] = []
    for t in body.get("tools", []) or []:
        if not isinstance(t, dict) or t.get("type") not in (None, "function"):
            continue
        tools.append(
            ToolDef(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("parameters", {}) or {},
            )
        )
    return tools


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "".join(p.text for p in parse_openai_content(value))


def _load_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
