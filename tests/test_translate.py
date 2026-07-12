"""Tests for the IR-based protocol translation layer.

Covers the canonical adapters directly, plus end-to-end proxy flows that
regression-guard the two bugs the IR refactor fixed:

* Bug 1: /v1/chat/completions to an openai-chat upstream must come back as
  chat.completion (not Responses shape).
* Bug 2: /v1/responses to a Gemini upstream must carry the prompt.

...and the new capability: /v1/messages (Anthropic) routed to Bedrock Nova.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_calo.proxy import GeminiProxyService, RouteConfig
from gemini_calo.translate import (
    get_inbound_adapter,
    get_outbound_adapter,
    register_outbound_adapter,
)
from gemini_calo.translate.ir import ChatRequest, ChatResponse, Message

ALT = "https://upstream.example.com"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_inbound_lookup(self):
        assert get_inbound_adapter("openai-chat").__class__.__name__ == "OpenAIChatInbound"
        assert get_inbound_adapter("openai-responses").__class__.__name__ == "OpenAIResponsesInbound"
        assert get_inbound_adapter("anthropic-messages").__class__.__name__ == "AnthropicMessagesInbound"

    def test_outbound_lookup(self):
        assert get_outbound_adapter("gemini").__class__.__name__ == "GeminiOutbound"
        assert get_outbound_adapter("bedrock-invoke").__class__.__name__ == "BedrockInvokeOutbound"
        assert get_outbound_adapter("bedrock-converse").__class__.__name__ == "BedrockConverseOutbound"
        assert get_outbound_adapter("openai-chat").__class__.__name__ == "OpenAIChatOutbound"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_inbound_adapter("nope")
        with pytest.raises(ValueError, match="Unknown"):
            get_outbound_adapter("nope")

    def test_register_custom_outbound(self):
        class Dummy:
            pass

        register_outbound_adapter("custom", Dummy)
        assert isinstance(get_outbound_adapter("custom"), Dummy)


# ---------------------------------------------------------------------------
# Inbound adapters -> IR
# ---------------------------------------------------------------------------


class TestInboundParse:
    def test_openai_chat_system_and_messages(self):
        req = get_inbound_adapter("openai-chat").parse_request(
            {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "be nice"},
                    {"role": "user", "content": "hi"},
                ],
                "temperature": 0.5,
                "max_tokens": 10,
            }
        )
        assert req.system == "be nice"
        assert [m.role for m in req.messages] == ["user"]
        assert req.messages[0].text == "hi"
        assert req.temperature == 0.5
        assert req.max_tokens == 10

    def test_openai_responses_string_input(self):
        req = get_inbound_adapter("openai-responses").parse_request(
            {"model": "m", "input": "Hello", "instructions": "sys"}
        )
        assert req.system == "sys"
        assert req.messages[0].role == "user"
        assert req.messages[0].text == "Hello"

    def test_openai_responses_message_array(self):
        req = get_inbound_adapter("openai-responses").parse_request(
            {
                "model": "m",
                "input": [
                    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                ],
                "max_output_tokens": 42,
            }
        )
        assert req.messages[0].text == "hi"
        assert req.max_tokens == 42

    def test_anthropic_messages_parse(self):
        req = get_inbound_adapter("anthropic-messages").parse_request(
            {
                "model": "m",
                "max_tokens": 100,
                "system": "sys",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        )
        assert req.system == "sys"
        assert req.max_tokens == 100
        assert [m.role for m in req.messages] == ["user", "assistant"]
        assert req.messages[1].text == "hello"

    def test_anthropic_system_role_message_folded(self):
        # Regression: a "system"-role message must not survive into IR.messages
        # (Bedrock Nova rejects role=system inside messages).
        req = get_inbound_adapter("anthropic-messages").parse_request(
            {
                "model": "m",
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": "extra sys"},
                ],
            }
        )
        assert "extra sys" in req.system
        assert all(msg.role != "system" for msg in req.messages)

    def test_openai_developer_role_folded(self):
        req = get_inbound_adapter("openai-chat").parse_request(
            {"model": "m", "messages": [{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}]}
        )
        assert req.system == "rules"
        assert [m.role for m in req.messages] == ["user"]

    def test_anthropic_system_blocks(self):
        req = get_inbound_adapter("anthropic-messages").parse_request(
            {"model": "m", "max_tokens": 1, "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "messages": []}
        )
        assert req.system == "ab"


# ---------------------------------------------------------------------------
# Outbound adapters: IR -> upstream and back
# ---------------------------------------------------------------------------


class TestOutboundRender:
    def test_gemini_request(self):
        req = ChatRequest(
            model="gemini-2.5-pro",
            system="sys",
            messages=[Message.of_text("user", "hi"), Message.of_text("assistant", "yo")],
            temperature=0.3,
            max_tokens=64,
        )
        body, path = get_outbound_adapter("gemini").render_request(req)
        assert path == "/v1beta/models/gemini-2.5-pro:generateContent"
        assert body["system_instruction"] == {"parts": [{"text": "sys"}]}
        assert body["contents"][0] == {"role": "user", "parts": [{"text": "hi"}]}
        assert body["contents"][1]["role"] == "model"
        assert body["generationConfig"]["maxOutputTokens"] == 64

    def test_gemini_stream_path(self):
        req = ChatRequest(model="g", messages=[Message.of_text("user", "hi")], stream=True)
        _, path = get_outbound_adapter("gemini").render_request(req)
        assert "streamGenerateContent" in path and "alt=sse" in path

    def test_gemini_response_parse(self):
        resp = get_outbound_adapter("gemini").parse_response(
            json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": "Hello"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
                }
            ).encode(),
            "application/json",
        )
        assert resp.content == "Hello"
        assert resp.finish_reason == "stop"
        assert resp.prompt_tokens == 3

    def test_bedrock_invoke_nova_request(self):
        req = ChatRequest(model="amazon.nova-pro-v1:0", messages=[Message.of_text("user", "hi")], max_tokens=256)
        body, path = get_outbound_adapter("bedrock-invoke").render_request(req)
        assert path == "/model/amazon.nova-pro-v1:0/invoke"
        assert body["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
        assert body["inferenceConfig"]["maxTokens"] == 256
        assert "anthropic_version" not in body

    def test_bedrock_invoke_nova_stop_sequences(self):
        req = ChatRequest(model="amazon.nova-pro-v1:0", messages=[Message.of_text("user", "hi")], stop=["STOP"])
        body, _ = get_outbound_adapter("bedrock-invoke").render_request(req)
        assert body["inferenceConfig"]["stopSequences"] == ["STOP"]

    def test_bedrock_converse_request(self):
        req = ChatRequest(model="m", system="sys", messages=[Message.of_text("user", "hi")], max_tokens=10)
        body, path = get_outbound_adapter("bedrock-converse").render_request(req)
        assert path == "/model/m/converse"
        assert body["system"] == [{"text": "sys"}]
        assert body["inferenceConfig"]["maxTokens"] == 10

    def test_bedrock_converse_nova_max_tokens_capped(self):
        """Nova enforces a 10240 output-token ceiling even over Converse."""
        req = ChatRequest(model="amazon.nova-pro-v1:0", messages=[Message.of_text("user", "hi")], max_tokens=60000)
        body, _ = get_outbound_adapter("bedrock-converse").render_request(req)
        assert body["inferenceConfig"]["maxTokens"] == 10240

    def test_bedrock_converse_non_nova_max_tokens_uncapped(self):
        """Non-Nova models keep their requested max_tokens on the Converse path."""
        req = ChatRequest(model="anthropic.claude-3-5-sonnet", messages=[Message.of_text("user", "hi")], max_tokens=60000)
        body, _ = get_outbound_adapter("bedrock-converse").render_request(req)
        assert body["inferenceConfig"]["maxTokens"] == 60000

    def test_openai_chat_normalizes_anthropic_tool_choice(self):
        """An Anthropic-style tool_choice must be mapped to OpenAI form, not
        forwarded verbatim (which OpenAI-compatible upstreams reject)."""
        adapter = get_outbound_adapter("openai-chat")
        req = ChatRequest(model="m", messages=[Message.of_text("user", "hi")])

        req.tool_choice = {"type": "auto"}
        assert adapter.render_request(req)[0]["tool_choice"] == "auto"

        req.tool_choice = {"type": "any"}
        assert adapter.render_request(req)[0]["tool_choice"] == "required"

        req.tool_choice = {"type": "tool", "name": "get_weather"}
        assert adapter.render_request(req)[0]["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_openai_chat_passes_through_native_tool_choice(self):
        """OpenAI-native tool_choice values are preserved unchanged."""
        adapter = get_outbound_adapter("openai-chat")
        req = ChatRequest(model="m", messages=[Message.of_text("user", "hi")])

        req.tool_choice = "required"
        assert adapter.render_request(req)[0]["tool_choice"] == "required"

        native = {"type": "function", "function": {"name": "f"}}
        req.tool_choice = native
        assert adapter.render_request(req)[0]["tool_choice"] == native

        # Responses-style flat object -> OpenAI nested function form.
        req.tool_choice = {"type": "function", "name": "f"}
        assert adapter.render_request(req)[0]["tool_choice"] == {
            "type": "function",
            "function": {"name": "f"},
        }

    def test_openai_chat_passes_through_unmodelled_params(self):
        """Params the IR doesn't model (JSON mode, seed, penalties, ...) must
        survive an OpenAI-chat -> OpenAI-chat translation via ChatRequest.extra."""
        inbound = get_inbound_adapter("openai-chat")
        outbound = get_outbound_adapter("openai-chat")
        req = inbound.parse_request(
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
                "response_format": {"type": "json_object"},
                "seed": 42,
                "frequency_penalty": 0.3,
            }
        )
        # Modelled param captured normally, not duplicated into extra.
        assert req.temperature == 0.5
        assert "temperature" not in req.extra
        assert req.extra["seed"] == 42

        body = outbound.render_request(req)[0]
        assert body["response_format"] == {"type": "json_object"}
        assert body["seed"] == 42
        assert body["frequency_penalty"] == 0.3
        # Modelled fields still win over anything replayed from extra.
        assert body["temperature"] == 0.5
        assert body["model"] == "deepseek-chat"

    def test_openai_chat_roundtrip_no_null_content(self):
        # A tool-only assistant turn must not serialise content: null.
        from gemini_calo.translate.ir import ToolCall

        req = ChatRequest(
            model="m",
            messages=[Message(role="assistant", tool_calls=[ToolCall(id="t1", name="f", arguments={"x": 1})])],
        )
        body, _ = get_outbound_adapter("openai-chat").render_request(req)
        assert "content" not in body["messages"][0]
        assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "f"


# ---------------------------------------------------------------------------
# End-to-end proxy flows
# ---------------------------------------------------------------------------


def _client(**routes: RouteConfig) -> TestClient:
    proxy = GeminiProxyService(api_keys=["default-key"], model_routes=routes)
    app = FastAPI()
    app.include_router(proxy.openai_router)
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.bedrock_router)
    app.include_router(proxy.anthropic_router)
    return TestClient(app)


class TestEndToEnd:
    def test_bug1_chat_to_openai_chat_stays_chat(self, httpx_mock):
        """Regression: chat completions to an openai-chat upstream stays chat."""
        httpx_mock.add_response(
            url=f"{ALT}/v1/chat/completions",
            content=json.dumps(
                {
                    "id": "chatcmpl-x",
                    "object": "chat.completion",
                    "model": "deepseek-chat",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                }
            ),
        )
        c = _client(**{"deepseek-*": RouteConfig(url=ALT, api_keys=["k"], protocol="openai-chat")})
        r = c.post("/v1/chat/completions", json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hi!"

    def test_bug2_responses_to_gemini_carries_prompt(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-pro:generateContent",
            content=json.dumps(
                {
                    "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
                }
            ),
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post("/v1/responses", json={"model": "gemini-2.5-pro", "input": "Hello Gemini"})
        assert r.status_code == 200
        sent = json.loads(httpx_mock.get_requests()[0].content)
        assert sent["contents"][0]["parts"][0]["text"] == "Hello Gemini"
        data = r.json()
        assert data["object"] == "response"
        assert data["output"][0]["content"][0]["text"] == "hello"

    def test_responses_to_openai_chat(self, httpx_mock):
        """/v1/responses to a chat-only upstream: sent as chat, returned as responses."""
        httpx_mock.add_response(
            url=f"{ALT}/v1/chat/completions",
            content=json.dumps(
                {
                    "id": "chatcmpl-y",
                    "object": "chat.completion",
                    "model": "deepseek-chat",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Yo"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ),
        )
        c = _client(**{"deepseek-*": RouteConfig(url=ALT, api_keys=["k"], protocol="openai-chat")})
        r = c.post("/v1/responses", json={"model": "deepseek-chat", "input": "hi"})
        sent = json.loads(httpx_mock.get_requests()[0].content)
        assert "/v1/chat/completions" in str(httpx_mock.get_requests()[0].url)
        assert sent["messages"][0]["content"] == "hi"
        assert r.json()["object"] == "response"
        assert r.json()["output"][0]["content"][0]["text"] == "Yo"

    def test_messages_to_bedrock_nova(self, httpx_mock):
        """New: Anthropic /v1/messages routed to Bedrock Nova (Claude Code path)."""
        httpx_mock.add_response(
            url=f"{ALT}/model/amazon.nova-pro-v1:0/invoke",
            content=json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "Hi from Nova"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 8, "outputTokens": 4},
                }
            ),
        )
        c = _client(**{"amazon.*": RouteConfig(url=ALT, api_keys=["k"], protocol="bedrock-invoke")})
        r = c.post(
            "/v1/messages",
            json={"model": "amazon.nova-pro-v1:0", "max_tokens": 256, "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"][0]["text"] == "Hi from Nova"
        assert data["stop_reason"] == "end_turn"
        assert data["usage"]["input_tokens"] == 8

    def test_chat_to_gemini_streaming(self, httpx_mock):
        sse = (
            b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}\n\n'
            b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}]}\n\n'
        )
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
            content=sse,
            headers={"Content-Type": "text/event-stream"},
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post(
            "/v1/chat/completions",
            json={"model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 200
        body = r.content.decode()
        texts = [
            json.loads(line[6:])["choices"][0]["delta"].get("content", "")
            for line in body.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        assert "".join(texts) == "Hello"
        assert "chat.completion.chunk" in body
        assert body.strip().endswith("[DONE]")

    def test_messages_to_gemini_streaming(self, httpx_mock):
        """Anthropic streaming out (/v1/messages, stream) over a Gemini upstream:
        emits the message_start ... content_block_delta ... message_stop sequence."""
        sse = (
            b'data: {"candidates":[{"content":{"parts":[{"text":"Hi "}]}}]}\n\n'
            b'data: {"candidates":[{"content":{"parts":[{"text":"there"}]},"finishReason":"STOP"}]}\n\n'
        )
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
            content=sse,
            headers={"Content-Type": "text/event-stream"},
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post(
            "/v1/messages",
            json={"model": "gemini-2.5-flash", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 200
        body = r.content.decode()
        assert "event: message_start" in body
        assert "event: content_block_delta" in body
        assert "event: message_stop" in body
        deltas = [
            json.loads(line[6:])["delta"]["text"]
            for line in body.splitlines()
            if line.startswith("data: ") and '"text_delta"' in line
        ]
        assert "".join(deltas) == "Hi there"

    def test_upstream_model_alias(self, httpx_mock):
        """A colon-free alias route rewrites the model id sent upstream."""
        httpx_mock.add_response(
            url=f"{ALT}/model/amazon.nova-pro-v1:0/invoke",
            content=json.dumps(
                {"output": {"message": {"content": [{"text": "hi"}]}}, "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1}}
            ),
        )
        c = _client(**{"nova": RouteConfig(url=ALT, api_keys=["k"], protocol="bedrock-invoke", upstream_model="amazon.nova-pro-v1:0")})
        r = c.post("/v1/chat/completions", json={"model": "nova", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
        assert "/model/amazon.nova-pro-v1:0/invoke" in str(httpx_mock.get_requests()[0].url)

    def test_upstream_error_passthrough(self, httpx_mock):
        """A 4xx from upstream is returned raw, not swallowed by translation."""
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-pro:generateContent",
            status_code=429,
            content=json.dumps({"error": {"message": "rate limited"}}),
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post("/v1/responses", json={"model": "gemini-2.5-pro", "input": "hi"})
        assert r.status_code == 429
        assert "rate limited" in r.text


class TestRealClientQuirks:
    """Regressions for constraints that only real client payloads exposed."""

    def test_gemini_schema_sanitized(self):
        # opencode/Claude Code emit draft-07 JSON Schema; Gemini rejects
        # $schema / additionalProperties / exclusiveMinimum.
        from gemini_calo.translate.outbound.gemini import _sanitize_gemini_schema

        raw = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "exclusiveMinimum": 0},
                "items": {"type": "array", "items": {"type": "string", "additionalProperties": False}},
            },
            "required": ["path"],
        }
        clean = _sanitize_gemini_schema(raw)
        flat = json.dumps(clean)
        assert "$schema" not in flat
        assert "additionalProperties" not in flat
        assert "exclusiveMinimum" not in flat
        assert clean["type"] == "object"
        assert clean["required"] == ["path"]
        # exclusiveMinimum intent preserved as minimum
        assert clean["properties"]["limit"]["minimum"] == 0
        assert clean["properties"]["items"]["items"]["type"] == "string"

    def test_nova_max_tokens_clamped(self):
        # Clients send huge max_tokens; Nova caps at 10240.
        from gemini_calo.translate.ir import ChatRequest, Message

        req = ChatRequest(model="amazon.nova-pro-v1:0", messages=[Message.of_text("user", "hi")], max_tokens=32000)
        body, _ = get_outbound_adapter("bedrock-invoke").render_request(req)
        assert body["inferenceConfig"]["maxTokens"] == 10240

    def test_gemini_tools_sanitized_in_request(self):
        from gemini_calo.translate.ir import ChatRequest, Message, ToolDef

        req = ChatRequest(
            model="gemini-2.5-flash",
            messages=[Message.of_text("user", "hi")],
            tools=[ToolDef(name="f", description="d", parameters={"$schema": "x", "type": "object", "additionalProperties": False, "properties": {}})],
        )
        body, _ = get_outbound_adapter("gemini").render_request(req)
        decl = body["tools"][0]["functionDeclarations"][0]
        assert "$schema" not in json.dumps(decl)
        assert "additionalProperties" not in json.dumps(decl)


class TestBedrockStreamParsing:
    """Lock in the real Bedrock event-stream shapes (captured from live Nova),
    so the parser can't regress without needing binary AWS frames in tests."""

    def _feed(self, bodies):
        from gemini_calo.translate.outbound._bedrock_common import bedrock_stream_feed

        state, events = {}, []
        for b in bodies:
            events.extend(bedrock_stream_feed(b, state))
        return events

    def test_nova_text_stream(self):
        events = self._feed(
            [
                {"messageStart": {"role": "assistant"}},
                {"contentBlockDelta": {"delta": {"text": "Hel"}, "contentBlockIndex": 0}},
                {"contentBlockDelta": {"delta": {"text": "lo"}, "contentBlockIndex": 0}},
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": "end_turn"}},
            ]
        )
        assert "".join(e.text for e in events if e.type == "text") == "Hello"
        assert any(e.type == "finish" and e.finish_reason == "stop" for e in events)

    def test_nova_tool_stream(self):
        events = self._feed(
            [
                {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "tu1", "name": "get_weather"}}, "contentBlockIndex": 1}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"city":'}}, "contentBlockIndex": 1}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": '"SF"}'}}, "contentBlockIndex": 1}},
                {"contentBlockStop": {"contentBlockIndex": 1}},
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        )
        tcs = [e.tool_call for e in events if e.type == "tool_call"]
        assert tcs and tcs[0].name == "get_weather" and tcs[0].arguments == {"city": "SF"}

    def test_invoke_bytes_wrapper_unwrapped(self):
        import base64

        inner = {"contentBlockDelta": {"delta": {"text": "hi"}, "contentBlockIndex": 0}}
        wrapped = {"bytes": base64.b64encode(json.dumps(inner).encode()).decode(), "p": "abc"}
        events = self._feed([wrapped])
        assert [e.text for e in events if e.type == "text"] == ["hi"]

    def test_converse_tool_stream_unwrapped(self):
        """Regression: over the botocore EventStreamBuffer path the Converse
        payload carries the block index *without* the enclosing event-type key,
        so ``contentBlockStop`` is ``{"contentBlockIndex": N}``. The tool call
        must still be emitted (flushed at the following messageStop)."""
        events = self._feed(
            [
                {"role": "assistant"},  # messageStart payload
                {"start": {"toolUse": {"toolUseId": "tu1", "name": "get_weather"}}, "contentBlockIndex": 0},
                {"delta": {"toolUse": {"input": '{"city":'}}, "contentBlockIndex": 0},
                {"delta": {"toolUse": {"input": '"SF"}'}}, "contentBlockIndex": 0},
                {"contentBlockIndex": 0},  # contentBlockStop, no wrapper key
                {"stopReason": "tool_use"},  # messageStop payload
            ]
        )
        tcs = [e.tool_call for e in events if e.type == "tool_call"]
        assert tcs and tcs[0].name == "get_weather" and tcs[0].arguments == {"city": "SF"}
        # Tool call is emitted before the finish event.
        types = [e.type for e in events]
        assert types.index("tool_call") < types.index("finish")

    def test_converse_two_tools_unwrapped(self):
        """Two sequential tool blocks (unwrapped): the first flushes when the
        second one starts, so both calls survive."""
        events = self._feed(
            [
                {"start": {"toolUse": {"toolUseId": "a", "name": "f1"}}, "contentBlockIndex": 0},
                {"delta": {"toolUse": {"input": '{"x":1}'}}, "contentBlockIndex": 0},
                {"contentBlockIndex": 0},
                {"start": {"toolUse": {"toolUseId": "b", "name": "f2"}}, "contentBlockIndex": 1},
                {"delta": {"toolUse": {"input": '{"y":2}'}}, "contentBlockIndex": 1},
                {"contentBlockIndex": 1},
                {"stopReason": "tool_use"},
            ]
        )
        tcs = [e.tool_call for e in events if e.type == "tool_call"]
        assert [t.name for t in tcs] == ["f1", "f2"]
        assert [t.arguments for t in tcs] == [{"x": 1}, {"y": 2}]

    def test_converse_tool_stream_end_of_stream_flush(self):
        """A tool block with no trailing messageStop is flushed at stream end."""
        from gemini_calo.translate.outbound._bedrock_common import bedrock_stream_flush

        state, events = {}, []
        from gemini_calo.translate.outbound._bedrock_common import bedrock_stream_feed

        for b in [
            {"start": {"toolUse": {"toolUseId": "tu1", "name": "f"}}, "contentBlockIndex": 0},
            {"delta": {"toolUse": {"input": '{"a":1}'}}, "contentBlockIndex": 0},
            {"contentBlockIndex": 0},
        ]:
            events.extend(bedrock_stream_feed(b, state))
        events.extend(bedrock_stream_flush(state))
        tcs = [e.tool_call for e in events if e.type == "tool_call"]
        assert tcs and tcs[0].name == "f" and tcs[0].arguments == {"a": 1}


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


class TestToolCalls:
    def test_chat_tools_to_gemini_and_back(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-pro:generateContent",
            content=json.dumps(
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
                }
            ),
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post(
            "/v1/chat/completions",
            json={"model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "weather in SF?"}], "tools": [WEATHER_TOOL]},
        )
        assert r.status_code == 200
        # Request carried the tool declaration to Gemini.
        sent = json.loads(httpx_mock.get_requests()[0].content)
        decls = sent["tools"][0]["functionDeclarations"]
        assert decls[0]["name"] == "get_weather"
        # Response surfaced the tool call in OpenAI shape.
        data = r.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        call = data["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "SF"}

    def test_tool_result_roundtrip_to_gemini(self, httpx_mock):
        """A follow-up request with an assistant tool_call + tool result encodes
        to Gemini functionCall + functionResponse parts.

        Real clients key tool results by an opaque call id (``call_...``), not
        the function name. Gemini correlates results to calls *by name*, so the
        adapter must resolve the id back to the originating function name.
        """
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-pro:generateContent",
            content=json.dumps({"candidates": [{"content": {"parts": [{"text": "It's sunny"}]}, "finishReason": "STOP"}]}),
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "gemini-2.5-pro",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_abc123", "content": '{"temp":72}'},
                ],
            },
        )
        assert r.status_code == 200
        sent = json.loads(httpx_mock.get_requests()[0].content)
        parts = [p for c in sent["contents"] for p in c["parts"]]
        assert any("functionCall" in p and p["functionCall"]["name"] == "get_weather" for p in parts)
        fr = [p["functionResponse"] for p in parts if "functionResponse" in p]
        # The opaque id must be resolved back to the function name Gemini expects.
        assert fr and fr[0]["name"] == "get_weather"
        assert fr[0]["response"] == {"temp": 72}

    def test_anthropic_tools_to_nova_and_back(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{ALT}/model/amazon.nova-pro-v1:0/invoke",
            content=json.dumps(
                {
                    "output": {"message": {"role": "assistant", "content": [{"toolUse": {"toolUseId": "tu1", "name": "get_weather", "input": {"city": "SF"}}}]}},
                    "stopReason": "tool_use",
                    "usage": {"inputTokens": 5, "outputTokens": 3},
                }
            ),
        )
        c = _client(**{"amazon.*": RouteConfig(url=ALT, api_keys=["k"], protocol="bedrock-invoke")})
        r = c.post(
            "/v1/messages",
            json={
                "model": "amazon.nova-pro-v1:0",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "weather in SF?"}],
                "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
            },
        )
        assert r.status_code == 200
        # Tool spec reached Nova via toolConfig.
        sent = json.loads(httpx_mock.get_requests()[0].content)
        assert sent["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"
        # Anthropic tool_use block came back.
        data = r.json()
        assert data["stop_reason"] == "tool_use"
        block = [b for b in data["content"] if b["type"] == "tool_use"][0]
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "SF"}

    def test_responses_tools_to_gemini(self, httpx_mock):
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-pro:generateContent",
            content=json.dumps(
                {"candidates": [{"content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "SF"}}}]}, "finishReason": "STOP"}]}
            ),
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post(
            "/v1/responses",
            json={"model": "gemini-2.5-pro", "input": "weather?", "tools": [{"type": "function", "name": "get_weather", "parameters": {"type": "object"}}]},
        )
        assert r.status_code == 200
        fc = [o for o in r.json()["output"] if o["type"] == "function_call"][0]
        assert fc["name"] == "get_weather"
        assert json.loads(fc["arguments"]) == {"city": "SF"}

    def test_chat_tool_call_streaming_from_gemini(self, httpx_mock):
        sse = b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]},"finishReason":"STOP"}]}\n\n'
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-pro:streamGenerateContent?alt=sse",
            content=sse,
            headers={"Content-Type": "text/event-stream"},
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post(
            "/v1/chat/completions",
            json={"model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "hi"}], "tools": [WEATHER_TOOL], "stream": True},
        )
        body = r.content.decode()
        tool_deltas = []
        for line in body.splitlines():
            if not line.startswith("data: ") or "[DONE]" in line:
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"]
            if "tool_calls" in delta:
                tool_deltas.append(delta["tool_calls"][0])
        assert tool_deltas and tool_deltas[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_deltas[0]["function"]["arguments"]) == {"city": "SF"}
        assert '"finish_reason": "tool_calls"' in body

    def test_openai_chat_streamed_tool_call_from_deepseek(self, httpx_mock):
        """Regression: DeepSeek streams tool calls as incremental delta.tool_calls
        fragments; the openai-chat outbound must assemble them (opencode hang)."""
        sse = (
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":"}}]}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"SF\\"}"}}]}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        httpx_mock.add_response(
            url=f"{ALT}/v1/chat/completions",
            content=sse,
            headers={"Content-Type": "text/event-stream"},
        )
        c = _client(**{"deepseek-*": RouteConfig(url=ALT, api_keys=["k"], protocol="openai-chat")})
        r = c.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "weather?"}], "tools": [WEATHER_TOOL], "stream": True},
        )
        body = r.content.decode()
        assembled = {}
        for line in body.splitlines():
            if not line.startswith("data: ") or "[DONE]" in line:
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"]
            for tc in delta.get("tool_calls", []) or []:
                slot = assembled.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                slot["name"] = tc["function"].get("name") or slot["name"]
                slot["args"] += tc["function"].get("arguments", "")
        assert assembled and assembled[0]["name"] == "get_weather"
        assert json.loads(assembled[0]["args"]) == {"city": "SF"}
        assert '"finish_reason": "tool_calls"' in body

    def test_anthropic_tool_call_streaming_from_gemini(self, httpx_mock):
        sse = b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"SF"}}}]},"finishReason":"STOP"}]}\n\n'
        httpx_mock.add_response(
            url=f"{ALT}/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
            content=sse,
            headers={"Content-Type": "text/event-stream"},
        )
        c = _client(**{"gemini-*": RouteConfig(url=ALT, api_keys=["k"], protocol="gemini")})
        r = c.post(
            "/v1/messages",
            json={"model": "gemini-2.5-flash", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        body = r.content.decode()
        # tool_use content block with input_json_delta and tool_use stop reason.
        starts = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ") and '"tool_use"' in line]
        assert any(s.get("content_block", {}).get("name") == "get_weather" for s in starts)
        assert '"input_json_delta"' in body
        assert '"stop_reason": "tool_use"' in body
