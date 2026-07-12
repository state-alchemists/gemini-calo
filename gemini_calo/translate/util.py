"""Shared utilities for protocol translation."""

from __future__ import annotations

import json
import uuid
from typing import Any


def make_chat_completion_id() -> str:
    """Generate an OpenAI-style chat completion ID."""
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def make_response_id() -> str:
    """Generate an OpenAI Responses-style response ID."""
    return f"resp_{uuid.uuid4().hex[:24]}"


def make_message_id() -> str:
    """Generate an Anthropic Messages-style message ID."""
    return f"msg_{uuid.uuid4().hex[:24]}"


def sse_data(payload: Any) -> bytes:
    """Format a JSON payload as a plain SSE ``data:`` line."""
    return f"data: {json.dumps(payload)}\n\n".encode()


def sse_event(event: str, payload: Any) -> bytes:
    """Format a named SSE event (``event:`` + ``data:``), as Anthropic uses."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


# -- OpenAI Chat Completions SSE (used by the openai-chat inbound adapter) --


def openai_chat_sse_chunk(
    *,
    model: str = "",
    content: str | None = None,
    finish_reason: str | None = None,
    completion_id: str = "",
) -> bytes:
    """Build a single OpenAI chat.completion.chunk SSE line."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    chunk: dict[str, Any] = {
        "id": completion_id or make_chat_completion_id(),
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return sse_data(chunk)


def openai_sse_done() -> bytes:
    """Build the OpenAI SSE ``[DONE]`` terminator."""
    return b"data: [DONE]\n\n"
