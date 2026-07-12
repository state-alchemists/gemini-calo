"""Shared Bedrock helpers for the invoke and converse outbound adapters.

Both Bedrock APIs share a stop-reason vocabulary, a ``toolUse`` content-block
shape, and streaming events. The only real differences are (a) InvokeModel with
Anthropic wraps each streamed event as ``{"bytes": <base64 json>}`` and uses the
Anthropic event names (``content_block_delta`` + ``input_json_delta``), while
Converse and Nova use ``contentBlockDelta`` + ``toolUse.input`` deltas.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any

from gemini_calo.translate.ir import (
    FINISH_LENGTH,
    FINISH_STOP,
    FINISH_TOOL_CALLS,
    StreamEvent,
    ToolCall,
)

BEDROCK_TO_IR_FINISH = {
    "end_turn": FINISH_STOP,
    "max_tokens": FINISH_LENGTH,
    "stop_sequence": FINISH_STOP,
    "tool_use": FINISH_TOOL_CALLS,
}


def tool_calls_from_content(content_blocks: list[Any]) -> list[ToolCall]:
    """Extract tool calls from an Anthropic or Converse/Nova content array."""
    calls: list[ToolCall] = []
    for b in content_blocks:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "tool_use":  # Anthropic
            calls.append(ToolCall(id=b.get("id", ""), name=b.get("name", ""), arguments=b.get("input", {}) or {}))
        tu = b.get("toolUse")  # Converse / Nova
        if isinstance(tu, dict):
            calls.append(
                ToolCall(id=tu.get("toolUseId", ""), name=tu.get("name", ""), arguments=tu.get("input", {}) or {})
            )
    return calls


def _finish_tool(t: dict[str, Any]) -> ToolCall:
    raw = t.get("args", "")
    try:
        args = json.loads(raw) if raw else {}
        if not isinstance(args, dict):
            args = {}
    except (json.JSONDecodeError, ValueError):
        args = {}
    return ToolCall(id=t.get("id", ""), name=t.get("name", ""), arguments=args)


def bedrock_stream_feed(body: dict[str, Any], state: dict[str, Any]) -> list[StreamEvent]:
    """Feed one Bedrock streaming event body; return canonical StreamEvents.

    ``state`` is a caller-owned dict that persists across events (used to
    accumulate a tool call's incrementally-streamed JSON arguments).
    """
    # InvokeModel wraps each Anthropic event as {"bytes": <base64 json>}.
    if "bytes" in body and isinstance(body["bytes"], str):
        try:
            body = json.loads(base64.b64decode(body["bytes"]))
        except (binascii.Error, json.JSONDecodeError, ValueError):
            return []

    events: list[StreamEvent] = []
    btype = body.get("type", "")

    # -- Tool-call block start --
    cb = body.get("content_block") or {}
    if btype == "content_block_start" and cb.get("type") == "tool_use":  # Anthropic
        state["tool"] = {"id": cb.get("id", ""), "name": cb.get("name", ""), "args": ""}
        return events
    start = (body.get("contentBlockStart") or body).get("start") or {}
    if isinstance(start, dict) and "toolUse" in start:  # Converse / Nova
        tu = start["toolUse"]
        state["tool"] = {"id": tu.get("toolUseId", ""), "name": tu.get("name", ""), "args": ""}
        return events

    # -- Deltas (text or tool-argument JSON) --
    delta = (body.get("contentBlockDelta") or body).get("delta") or {}
    if isinstance(delta, dict):
        if delta.get("type") == "input_json_delta":  # Anthropic tool args
            if state.get("tool") is not None:
                state["tool"]["args"] += delta.get("partial_json", "")
            return events
        tud = delta.get("toolUse") or {}
        if "input" in tud:  # Converse / Nova tool args
            if state.get("tool") is not None:
                state["tool"]["args"] += tud.get("input", "")
            return events
        text = delta.get("text")
        if text:
            events.append(StreamEvent(type="text", text=text))
            return events

    # -- Block stop: finalise any accumulated tool call --
    if btype == "content_block_stop" or "contentBlockStop" in body:
        if state.get("tool") is not None:
            events.append(StreamEvent(type="tool_call", tool_call=_finish_tool(state.pop("tool"))))
        return events

    # -- Usage --
    usage = body.get("usage") or (body.get("metadata") or {}).get("usage")
    if usage:
        events.append(
            StreamEvent(
                type="usage",
                prompt_tokens=usage.get("input_tokens") or usage.get("inputTokens", 0),
                completion_tokens=usage.get("output_tokens") or usage.get("outputTokens", 0),
            )
        )

    # -- Finish --
    stop_reason = (
        body.get("stop_reason")
        or (body.get("delta") or {}).get("stop_reason")
        or (body.get("messageStop") or {}).get("stopReason")
        or body.get("stopReason")
    )
    if stop_reason or btype in ("message_stop", "message_delta"):
        events.append(
            StreamEvent(
                type="finish",
                finish_reason=BEDROCK_TO_IR_FINISH.get(str(stop_reason), FINISH_STOP),
            )
        )
    return events
