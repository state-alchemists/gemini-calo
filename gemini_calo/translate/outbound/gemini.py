"""Outbound adapter for the Google Gemini native API."""

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
    Message,
    StreamEvent,
    ToolCall,
)

_GEMINI_TO_IR_FINISH = {
    "STOP": FINISH_STOP,
    "MAX_TOKENS": FINISH_LENGTH,
    "SAFETY": FINISH_CONTENT_FILTER,
    "RECITATION": FINISH_CONTENT_FILTER,
}


class GeminiOutbound:
    """Canonical IR <-> Gemini generateContent API."""

    def render_request(self, req: ChatRequest) -> tuple[dict[str, Any], str]:
        contents: list[dict[str, Any]] = []
        for m in req.messages:
            if m.role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    # Gemini matches results to calls by function
                                    # name; our IR uses name as the tool_call id
                                    # for Gemini so this round-trips.
                                    "name": tr.tool_call_id,
                                    "response": _wrap_result(tr.content),
                                }
                            }
                            for tr in m.tool_results
                        ],
                    }
                )
                continue
            if m.role not in ("user", "assistant"):
                continue
            parts = _parts_from_message(m)
            if not parts:
                continue
            contents.append({"role": "model" if m.role == "assistant" else "user", "parts": parts})

        body: dict[str, Any] = {"contents": contents}
        if req.system:
            body["system_instruction"] = {"parts": [{"text": req.system}]}

        gen: dict[str, Any] = {}
        if req.temperature is not None:
            gen["temperature"] = req.temperature
        if req.max_tokens is not None:
            gen["maxOutputTokens"] = req.max_tokens
        if req.top_p is not None:
            gen["topP"] = req.top_p
        if req.stop:
            gen["stopSequences"] = req.stop
        if gen:
            body["generationConfig"] = gen

        if req.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        _function_declaration(t) for t in req.tools
                    ]
                }
            ]
            fcc = _function_calling_config(req.tool_choice)
            if fcc:
                body["toolConfig"] = {"functionCallingConfig": fcc}

        action = "streamGenerateContent" if req.stream else "generateContent"
        path = f"/v1beta/models/{req.model}:{action}"
        if req.stream:
            path += "?alt=sse"
        return body, path

    def parse_response(self, body: bytes, content_type: str) -> ChatResponse:
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return ChatResponse()
        return _gemini_to_ir(data)

    async def parse_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[StreamEvent]:
        # With ?alt=sse Gemini emits SSE ``data:`` lines; without it, NDJSON /
        # a JSON array. Handle SSE lines and bare JSON objects line-by-line.
        buffer = b""
        finish_emitted = False
        async for chunk in chunks:
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip().lstrip(b"[,").rstrip(b",]")
                if not line:
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                for ev in _gemini_chunk_to_events(data):
                    if ev.type == "finish":
                        finish_emitted = True
                    yield ev
        if not finish_emitted:
            yield StreamEvent(type="finish", finish_reason=FINISH_STOP)


def _function_declaration(t: Any) -> dict[str, Any]:
    decl: dict[str, Any] = {"name": t.name, "description": t.description}
    # Gemini rejects an empty parameters object; only include when populated.
    params = _sanitize_gemini_schema(t.parameters)
    if params:
        decl["parameters"] = params
    return decl


# Gemini's functionDeclarations.parameters accept only a restricted OpenAPI 3.0
# subset (the Schema proto). Real clients (opencode, Claude Code) emit full
# JSON Schema with keywords Gemini rejects ($schema, additionalProperties,
# exclusiveMinimum, ...). We recursively keep only the supported keys.
_GEMINI_SCHEMA_KEYS = {
    "type", "format", "description", "nullable", "enum", "items", "properties",
    "required", "minimum", "maximum", "minItems", "maxItems", "minLength",
    "maxLength", "pattern", "anyOf", "default", "example", "propertyOrdering",
    "title",
}


def _sanitize_gemini_schema(schema: Any) -> Any:
    """Recursively strip JSON-Schema keywords Gemini does not accept."""
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _GEMINI_SCHEMA_KEYS:
            # Best-effort: preserve the constraint intent of exclusive bounds.
            if key == "exclusiveMinimum" and isinstance(value, (int, float)):
                out.setdefault("minimum", value)
            elif key == "exclusiveMaximum" and isinstance(value, (int, float)):
                out.setdefault("maximum", value)
            continue
        if key == "properties" and isinstance(value, dict):
            out["properties"] = {k: _sanitize_gemini_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _sanitize_gemini_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            out["anyOf"] = [_sanitize_gemini_schema(v) for v in value]
        else:
            out[key] = value
    # Gemini requires OBJECT schemas to declare a (possibly empty) type.
    return out


def _function_calling_config(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"mode": "AUTO"}
    if tool_choice == "none":
        return {"mode": "NONE"}
    if tool_choice in ("required", "any"):
        return {"mode": "ANY"}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("name") or tool_choice.get("function", {}).get("name")
        if name:
            return {"mode": "ANY", "allowedFunctionNames": [name]}
    return {"mode": "AUTO"}


def _wrap_result(content: str) -> dict[str, Any]:
    # functionResponse.response must be an object; use the tool output as JSON
    # when it parses, else wrap the raw text.
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return {"result": content}


def _parts_from_message(m: Message) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for p in m.content:
        if p.type == "text" and p.text:
            parts.append({"text": p.text})
    for tc in m.tool_calls:
        parts.append({"functionCall": {"name": tc.name, "args": tc.arguments}})
    return parts


def _gemini_to_ir(data: dict[str, Any]) -> ChatResponse:
    resp = ChatResponse()
    meta = data.get("usageMetadata", {}) or {}
    resp.prompt_tokens = meta.get("promptTokenCount", 0)
    resp.completion_tokens = meta.get("candidatesTokenCount", 0)

    candidates = data.get("candidates", [])
    if not candidates:
        return resp
    candidate = candidates[0]
    parts = candidate.get("content", {}).get("parts", [])
    resp.content = "".join(p.get("text", "") for p in parts if "text" in p)
    for p in parts:
        fc = p.get("functionCall")
        if fc:
            resp.tool_calls.append(
                ToolCall(id=fc.get("name", ""), name=fc.get("name", ""), arguments=fc.get("args", {}) or {})
            )
    resp.finish_reason = (
        FINISH_TOOL_CALLS
        if resp.tool_calls
        else _GEMINI_TO_IR_FINISH.get(candidate.get("finishReason", "STOP"), FINISH_STOP)
    )
    return resp


def _gemini_chunk_to_events(data: dict[str, Any]) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    candidates = data.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        for part in candidate.get("content", {}).get("parts", []):
            text = part.get("text")
            if text:
                events.append(StreamEvent(type="text", text=text))
            fc = part.get("functionCall")
            if fc:
                # Gemini streams each functionCall whole.
                events.append(
                    StreamEvent(
                        type="tool_call",
                        tool_call=ToolCall(
                            id=fc.get("name", ""),
                            name=fc.get("name", ""),
                            arguments=fc.get("args", {}) or {},
                        ),
                    )
                )
        fr = candidate.get("finishReason")
        if fr:
            events.append(
                StreamEvent(type="finish", finish_reason=_GEMINI_TO_IR_FINISH.get(fr, FINISH_STOP))
            )
    meta = data.get("usageMetadata")
    if meta:
        events.append(
            StreamEvent(
                type="usage",
                prompt_tokens=meta.get("promptTokenCount", 0),
                completion_tokens=meta.get("candidatesTokenCount", 0),
            )
        )
    return events
