"""Outbound adapter for the Bedrock InvokeModel API.

Supports both Amazon Nova and Anthropic Claude body formats (provider detected
from the model id). Nova's InvokeModel body mirrors the Converse schema, so its
tool encoding matches :mod:`bedrock_converse`.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from gemini_calo.translate.ir import (
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    ChatRequest,
    ChatResponse,
    StreamEvent,
    ToolCall,
)
from gemini_calo.translate.outbound._bedrock_common import (
    BEDROCK_TO_IR_FINISH,
    bedrock_stream_feed,
    tool_calls_from_content,
)

_BEDROCK_TO_IR_FINISH = BEDROCK_TO_IR_FINISH


def detect_provider(model: str) -> str:
    m = model.lower()
    if "anthropic" in m or "claude" in m:
        return "anthropic"
    return "nova"


class BedrockInvokeOutbound:
    """Canonical IR <-> Bedrock InvokeModel API."""

    def render_request(self, req: ChatRequest) -> tuple[dict[str, Any], str]:
        provider = detect_provider(req.model)
        if provider == "anthropic":
            body = _anthropic_body(req)
        else:
            body = _nova_body(req)
        action = "invoke-with-response-stream" if req.stream else "invoke"
        return body, f"/model/{req.model}/{action}"

    def parse_response(self, body: bytes, content_type: str) -> ChatResponse:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return ChatResponse()
        return _bedrock_to_ir(data)

    async def parse_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[StreamEvent]:
        state: dict[str, Any] = {}
        try:
            from botocore.eventstream import EventStreamBuffer
        except ImportError:
            async for chunk in chunks:
                for body in _raw_json_bodies(chunk):
                    for ev in bedrock_stream_feed(body, state):
                        yield ev
            return

        buffer = EventStreamBuffer()
        async for chunk in chunks:
            buffer.add_data(chunk)
            for message in buffer:
                try:
                    body = json.loads(message.payload)
                except (json.JSONDecodeError, ValueError, AttributeError):
                    continue
                for ev in bedrock_stream_feed(body, state):
                    yield ev


# -- Anthropic-format body --

def _anthropic_body(req: ChatRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": req.max_tokens or 4096,
        "messages": _anthropic_messages(req),
    }
    if req.system:
        body["system"] = req.system
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop_sequences"] = req.stop
    if req.tools:
        body["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters or {"type": "object", "properties": {}}}
            for t in req.tools
        ]
        tc = _anthropic_tool_choice(req.tool_choice)
        if tc:
            body["tool_choice"] = tc
    return body


def _anthropic_messages(req: ChatRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": tr.tool_call_id, "content": tr.content}
                        for tr in m.tool_results
                    ],
                }
            )
            continue
        if m.role not in ("user", "assistant"):
            continue
        if m.tool_calls:
            content: list[dict[str, Any]] = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            for tc in m.tool_calls:
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
            messages.append({"role": m.role, "content": content})
        else:
            messages.append({"role": m.role, "content": m.text})
    return messages


def _anthropic_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice in ("required", "any"):
        return {"type": "any"}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("name") or tool_choice.get("function", {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


# -- Nova-format body (Converse schema) --

# Amazon Nova caps generation at 10240 output tokens; clients routinely ask for
# more (Claude Code/opencode send tens of thousands), which Nova rejects with
# "maxTokens must be between 1 and 10240".
NOVA_MAX_TOKENS = 10240


def _nova_body(req: ChatRequest) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": _nova_messages(req)}
    if req.system:
        body["system"] = [{"text": req.system}]
    inference: dict[str, Any] = {}
    if req.max_tokens is not None:
        inference["maxTokens"] = min(req.max_tokens, NOVA_MAX_TOKENS)
    if req.temperature is not None:
        inference["temperature"] = req.temperature
    if req.top_p is not None:
        inference["topP"] = req.top_p
    if inference:
        body["inferenceConfig"] = inference
    if req.tools:
        tool_config: dict[str, Any] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.parameters or {"type": "object", "properties": {}}},
                    }
                }
                for t in req.tools
            ]
        }
        tc = _nova_tool_choice(req.tool_choice)
        if tc:
            tool_config["toolChoice"] = tc
        body["toolConfig"] = tool_config
    return body


def _nova_messages(req: ChatRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": tr.tool_call_id, "content": [{"text": tr.content}]}}
                        for tr in m.tool_results
                    ],
                }
            )
            continue
        if m.role not in ("user", "assistant"):
            continue
        content: list[dict[str, Any]] = []
        if m.text:
            content.append({"text": m.text})
        for tc in m.tool_calls:
            content.append({"toolUse": {"toolUseId": tc.id, "name": tc.name, "input": tc.arguments}})
        if not content:
            content = [{"text": ""}]
        messages.append({"role": m.role, "content": content})
    return messages


def _nova_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"auto": {}}
    if tool_choice in ("required", "any"):
        return {"any": {}}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("name") or tool_choice.get("function", {}).get("name")
        if name:
            return {"tool": {"name": name}}
    return None


def _bedrock_to_ir(data: dict[str, Any]) -> ChatResponse:
    resp = ChatResponse()
    content_blocks = data.get("content", [])
    if not content_blocks:
        content_blocks = data.get("output", {}).get("message", {}).get("content", [])
    resp.content = " ".join(
        b.get("text", "") for b in content_blocks if isinstance(b, dict) and "text" in b
    ).strip()
    resp.tool_calls = tool_calls_from_content(content_blocks)

    stop_reason = data.get("stop_reason") or data.get("stopReason", "end_turn")
    resp.finish_reason = (
        FINISH_TOOL_CALLS if resp.tool_calls else _BEDROCK_TO_IR_FINISH.get(str(stop_reason), FINISH_STOP)
    )

    usage = data.get("usage", {}) or {}
    resp.prompt_tokens = usage.get("input_tokens") or usage.get("inputTokens", 0)
    resp.completion_tokens = usage.get("output_tokens") or usage.get("outputTokens", 0)
    return resp


def _raw_json_bodies(chunk: bytes) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []
    for line in chunk.split(b"\n"):
        line = line.strip()
        if line.startswith(b"data: "):
            line = line[6:]
        if not line:
            continue
        try:
            bodies.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return bodies
