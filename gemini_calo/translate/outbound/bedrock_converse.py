"""Outbound adapter for the Bedrock Converse API.

Converse is a unified interface across all Bedrock model families, so unlike
InvokeModel there is a single body format regardless of provider.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from gemini_calo.translate.ir import (
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    ChatRequest,
    ChatResponse,
    StreamEvent,
)
from gemini_calo.translate.outbound._bedrock_common import (
    BEDROCK_TO_IR_FINISH,
    NOVA_MAX_TOKENS,
    bedrock_stream_feed,
    bedrock_stream_flush,
    is_nova_model,
    raw_json_bodies,
    tool_calls_from_content,
)


class BedrockConverseOutbound:
    """Canonical IR <-> Bedrock Converse API."""

    def render_request(self, req: ChatRequest) -> tuple[dict[str, Any], str]:
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

        body: dict[str, Any] = {"messages": messages}
        if req.system:
            body["system"] = [{"text": req.system}]

        inference: dict[str, Any] = {}
        if req.max_tokens is not None:
            # Converse is model-agnostic, but Nova still enforces its 10240 cap
            # and rejects larger values; clamp only for Nova model ids.
            inference["maxTokens"] = (
                min(req.max_tokens, NOVA_MAX_TOKENS)
                if is_nova_model(req.model)
                else req.max_tokens
            )
        if req.temperature is not None:
            inference["temperature"] = req.temperature
        if req.top_p is not None:
            inference["topP"] = req.top_p
        if req.stop:
            inference["stopSequences"] = req.stop
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
            tc = _converse_tool_choice(req.tool_choice)
            if tc:
                tool_config["toolChoice"] = tc
            body["toolConfig"] = tool_config

        action = "converse-stream" if req.stream else "converse"
        return body, f"/model/{req.model}/{action}"

    def parse_response(self, body: bytes, content_type: str) -> ChatResponse:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return ChatResponse()
        resp = ChatResponse()
        content_blocks = data.get("output", {}).get("message", {}).get("content", [])
        resp.content = "".join(
            b.get("text", "") for b in content_blocks if isinstance(b, dict) and "text" in b
        )
        resp.tool_calls = tool_calls_from_content(content_blocks)
        resp.finish_reason = (
            FINISH_TOOL_CALLS
            if resp.tool_calls
            else BEDROCK_TO_IR_FINISH.get(str(data.get("stopReason", "end_turn")), FINISH_STOP)
        )
        usage = data.get("usage", {}) or {}
        resp.prompt_tokens = usage.get("inputTokens", 0)
        resp.completion_tokens = usage.get("outputTokens", 0)
        return resp

    async def parse_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[StreamEvent]:
        state: dict[str, Any] = {}
        try:
            from botocore.eventstream import EventStreamBuffer
        except ImportError:
            async for chunk in chunks:
                for body in raw_json_bodies(chunk):
                    for ev in bedrock_stream_feed(body, state):
                        yield ev
            for ev in bedrock_stream_flush(state):
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
        for ev in bedrock_stream_flush(state):
            yield ev


def _converse_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
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
