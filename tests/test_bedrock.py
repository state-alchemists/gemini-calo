"""Tests for AWS Bedrock-compatible endpoint."""

import json
from unittest.mock import MagicMock

import pytest
from cachetools import LRUCache
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from gemini_calo.auth import (
    AWSSigV4Auth,
    BearerAuth,
    NoAuth,
    create_passthrough_bedrock_provider,
)
from gemini_calo.middlewares.model_override import create_model_override_middleware
from gemini_calo.middlewares.rollup import (
    _extract_bedrock_messages,
    _inject_bedrock_system_prompt,
    create_rollup_middleware,
)
from gemini_calo.proxy import REQUEST_TYPE, GeminiProxyService

BEDROCK_BASE = "https://bedrock-runtime.us-east-1.amazonaws.com"
MODEL_ID = "anthropic.claude-3-5-sonnet-20241022-v1:0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(proxy=None):
    if proxy is None:
        proxy = GeminiProxyService(api_keys=["dummy-gemini-key"])
    app = FastAPI()
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    app.include_router(proxy.bedrock_router)
    return app, proxy


# ---------------------------------------------------------------------------
# REQUEST_TYPE detection
# ---------------------------------------------------------------------------


def test_get_request_type_bedrock_invoke():
    from starlette.testclient import TestClient as SC

    app = FastAPI()

    @app.post("/model/{model_id:path}/invoke")
    async def dummy(request: Request):
        return {"type": GeminiProxyService.get_request_type(request).value}

    client = SC(app)
    r = client.post(f"/model/{MODEL_ID}/invoke", json={})
    assert r.json()["type"] == REQUEST_TYPE.BEDROCK_INVOKE.value


def test_get_request_type_bedrock_streaming():
    from starlette.testclient import TestClient as SC

    app = FastAPI()

    @app.post("/model/{model_id:path}/invoke-with-response-stream")
    async def dummy(request: Request):
        return {"type": GeminiProxyService.get_request_type(request).value}

    client = SC(app)
    r = client.post(f"/model/{MODEL_ID}/invoke-with-response-stream", json={})
    assert r.json()["type"] == REQUEST_TYPE.BEDROCK_STREAMING_INVOKE.value


def test_get_request_type_bedrock_converse():
    from starlette.testclient import TestClient as SC

    app = FastAPI()

    @app.post("/model/{model_id:path}/converse")
    async def dummy(request: Request):
        return {"type": GeminiProxyService.get_request_type(request).value}

    client = SC(app)
    r = client.post(f"/model/{MODEL_ID}/converse", json={})
    assert r.json()["type"] == REQUEST_TYPE.BEDROCK_CONVERSE.value


def test_get_request_type_bedrock_converse_stream():
    from starlette.testclient import TestClient as SC

    app = FastAPI()

    @app.post("/model/{model_id:path}/converse-stream")
    async def dummy(request: Request):
        return {"type": GeminiProxyService.get_request_type(request).value}

    client = SC(app)
    r = client.post(f"/model/{MODEL_ID}/converse-stream", json={})
    assert r.json()["type"] == REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE.value


def test_get_request_type_non_bedrock_not_misdetected():
    from starlette.testclient import TestClient as SC

    app = FastAPI()

    @app.get("/v1beta/models")
    async def dummy(request: Request):
        return {"type": GeminiProxyService.get_request_type(request).value}

    client = SC(app)
    r = client.get("/v1beta/models")
    assert r.json()["type"] == REQUEST_TYPE.OTHER.value


# ---------------------------------------------------------------------------
# forward_bedrock_request — bearer token auth
# ---------------------------------------------------------------------------


def test_bedrock_invoke_with_bearer_token(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    mock_body = json.dumps(
        {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
        }
    )
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/invoke",
        content=mock_body,
        status_code=200,
    )

    response = client.post(
        f"/model/{MODEL_ID}/invoke",
        json={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers={"X-AWS-Bearer-Token": "my-bedrock-api-key"},
    )

    assert response.status_code == 200
    sent = httpx_mock.get_requests()[0]
    assert sent.headers["authorization"] == "Bearer my-bedrock-api-key"


def test_bedrock_invoke_uses_region_header(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    region = "ap-southeast-1"
    httpx_mock.add_response(
        url=f"https://bedrock-runtime.{region}.amazonaws.com/model/{MODEL_ID}/invoke",
        content=json.dumps({"content": [{"type": "text", "text": "Hi"}]}),
        status_code=200,
    )

    response = client.post(
        f"/model/{MODEL_ID}/invoke",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={
            "X-AWS-Bearer-Token": "my-bedrock-api-key",
            "X-AWS-Region": region,
        },
    )

    assert response.status_code == 200


def test_bedrock_streaming_invoke(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    mock_chunks = b'{"bytes": "chunk1"}\n{"bytes": "chunk2"}'
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/invoke-with-response-stream",
        content=mock_chunks,
        status_code=200,
    )

    response = client.post(
        f"/model/{MODEL_ID}/invoke-with-response-stream",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-AWS-Bearer-Token": "my-bedrock-api-key"},
    )

    assert response.status_code == 200
    assert b"".join(response.iter_bytes()) == mock_chunks


def test_bedrock_converse_request(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    mock_body = json.dumps(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello from Converse!"}],
                }
            },
            "stopReason": "end_turn",
        }
    )
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/converse",
        content=mock_body,
        status_code=200,
    )

    response = client.post(
        f"/model/{MODEL_ID}/converse",
        json={"messages": [{"role": "user", "content": [{"text": "Hello"}]}]},
        headers={"X-AWS-Bearer-Token": "my-bedrock-api-key"},
    )

    assert response.status_code == 200
    sent = httpx_mock.get_requests()[0]
    assert sent.headers["authorization"] == "Bearer my-bedrock-api-key"


def test_bedrock_converse_stream_request(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    mock_chunks = b'{"contentBlockDelta": {"delta": {"text": "Hi"}}}'
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/converse-stream",
        content=mock_chunks,
        status_code=200,
    )

    response = client.post(
        f"/model/{MODEL_ID}/converse-stream",
        json={"messages": [{"role": "user", "content": [{"text": "Hi"}]}]},
        headers={"X-AWS-Bearer-Token": "my-bedrock-api-key"},
    )

    assert response.status_code == 200
    assert b"".join(response.iter_bytes()) == mock_chunks


def test_bedrock_invoke_forwards_optional_headers(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/invoke",
        content=json.dumps({"content": [{"type": "text", "text": "Hi"}]}),
        status_code=200,
    )

    client.post(
        f"/model/{MODEL_ID}/invoke",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={
            "X-AWS-Bearer-Token": "tok",
            "X-Amzn-Bedrock-Trace": "ENABLED",
            "X-Amzn-Bedrock-GuardrailIdentifier": "my-guardrail",
            "X-Amzn-Bedrock-GuardrailVersion": "1",
            "X-Amzn-Bedrock-PerformanceConfig-Latency": "optimized",
            "X-Amzn-Bedrock-Service-Tier": "priority",
            "Accept": "application/json",
        },
    )

    sent = httpx_mock.get_requests()[0]
    assert sent.headers.get("x-amzn-bedrock-trace") == "ENABLED"
    assert sent.headers.get("x-amzn-bedrock-guardrailidentifier") == "my-guardrail"
    assert sent.headers.get("x-amzn-bedrock-guardrailversion") == "1"
    assert sent.headers.get("x-amzn-bedrock-performanceconfig-latency") == "optimized"
    assert sent.headers.get("x-amzn-bedrock-service-tier") == "priority"
    assert sent.headers.get("accept") == "application/json"


def test_bedrock_invoke_does_not_forward_absent_optional_headers(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/invoke",
        content=json.dumps({"content": [{"type": "text", "text": "Hi"}]}),
        status_code=200,
    )

    client.post(
        f"/model/{MODEL_ID}/invoke",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    sent = httpx_mock.get_requests()[0]
    assert "x-amzn-bedrock-trace" not in sent.headers
    assert "x-amzn-bedrock-guardrailidentifier" not in sent.headers


def test_bedrock_converse_forwards_optional_headers(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/converse",
        content=json.dumps(
            {"output": {"message": {"role": "assistant", "content": []}}}
        ),
        status_code=200,
    )

    client.post(
        f"/model/{MODEL_ID}/converse",
        json={"messages": [{"role": "user", "content": [{"text": "Hi"}]}]},
        headers={
            "X-AWS-Bearer-Token": "tok",
            "X-Amzn-Bedrock-Trace": "ENABLED_FULL",
        },
    )

    sent = httpx_mock.get_requests()[0]
    assert sent.headers.get("x-amzn-bedrock-trace") == "ENABLED_FULL"


def test_bedrock_converse_preserves_content_type(httpx_mock):
    app, _ = _make_app()
    client = TestClient(app)

    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/converse",
        content=json.dumps(
            {"output": {"message": {"role": "assistant", "content": []}}}
        ),
        status_code=200,
    )

    client.post(
        f"/model/{MODEL_ID}/converse",
        content=json.dumps(
            {"messages": [{"role": "user", "content": [{"text": "Hi"}]}]}
        ),
        headers={
            "X-AWS-Bearer-Token": "tok",
            "Content-Type": "application/x-amz-json-1.1",
        },
    )

    sent = httpx_mock.get_requests()[0]
    assert sent.headers["content-type"] == "application/x-amz-json-1.1"


# ---------------------------------------------------------------------------
# create_passthrough_bedrock_provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passthrough_bedrock_provider_prefers_bearer_token():
    provider = create_passthrough_bedrock_provider()
    request = MagicMock(spec=Request)
    request.headers = {
        "X-AWS-Bearer-Token": "my-bedrock-key",
        "X-AWS-Access-Key": "AKIA...",
        "X-AWS-Secret-Key": "secret",
    }

    auth = await provider(request)

    assert isinstance(auth, BearerAuth)
    assert auth.token == "my-bedrock-key"


@pytest.mark.asyncio
async def test_passthrough_bedrock_provider_falls_back_to_sigv4():
    provider = create_passthrough_bedrock_provider()
    request = MagicMock(spec=Request)
    request.headers = {
        "X-AWS-Access-Key": "AKIAIOSFODNN7EXAMPLE",
        "X-AWS-Secret-Key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "X-AWS-Region": "eu-west-1",
    }

    auth = await provider(request)

    assert isinstance(auth, AWSSigV4Auth)
    assert auth.access_key == "AKIAIOSFODNN7EXAMPLE"
    assert auth.region == "eu-west-1"
    assert auth.service == "bedrock"


@pytest.mark.asyncio
async def test_passthrough_bedrock_provider_no_creds_returns_no_auth():
    provider = create_passthrough_bedrock_provider()
    request = MagicMock(spec=Request)
    request.headers = {}

    auth = await provider(request)

    assert isinstance(auth, NoAuth)


@pytest.mark.asyncio
async def test_passthrough_bedrock_provider_custom_bearer_header():
    provider = create_passthrough_bedrock_provider(bearer_token_header="X-Custom-Token")
    request = MagicMock(spec=Request)
    request.headers = {"X-Custom-Token": "custom-tok"}

    auth = await provider(request)

    assert isinstance(auth, BearerAuth)
    assert auth.token == "custom-tok"


# ---------------------------------------------------------------------------
# Model override middleware — Bedrock paths
# ---------------------------------------------------------------------------


@pytest.fixture
def bedrock_override_client(request):
    transformer = getattr(request, "param", None)
    app = FastAPI()
    proxy = GeminiProxyService(api_keys=["dummy-gemini-key"])
    if transformer:
        app.middleware("http")(
            create_model_override_middleware(model_transformer=transformer)
        )
    app.include_router(proxy.bedrock_router)
    return TestClient(app)


@pytest.mark.parametrize(
    "bedrock_override_client", ["anthropic.claude-3-haiku-20240307-v1:0"], indirect=True
)
def test_model_override_bedrock_converse(bedrock_override_client, httpx_mock):
    new_model = "anthropic.claude-3-haiku-20240307-v1:0"
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{new_model}/converse",
        content=json.dumps(
            {"output": {"message": {"role": "assistant", "content": [{"text": "Hi"}]}}}
        ),
        status_code=200,
    )

    response = bedrock_override_client.post(
        f"/model/{MODEL_ID}/converse",
        json={"messages": [{"role": "user", "content": [{"text": "Hi"}]}]},
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    assert response.status_code == 200
    assert httpx_mock.get_requests()[0].url.path == f"/model/{new_model}/converse"


@pytest.mark.parametrize(
    "bedrock_override_client", ["anthropic.claude-3-haiku-20240307-v1:0"], indirect=True
)
def test_model_override_bedrock_converse_stream(bedrock_override_client, httpx_mock):
    new_model = "anthropic.claude-3-haiku-20240307-v1:0"
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{new_model}/converse-stream",
        content=b"chunk",
        status_code=200,
    )

    response = bedrock_override_client.post(
        f"/model/{MODEL_ID}/converse-stream",
        json={"messages": [{"role": "user", "content": [{"text": "Hi"}]}]},
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    assert response.status_code == 200
    assert (
        httpx_mock.get_requests()[0].url.path == f"/model/{new_model}/converse-stream"
    )


@pytest.mark.parametrize(
    "bedrock_override_client", ["anthropic.claude-3-haiku-20240307-v1:0"], indirect=True
)
def test_model_override_bedrock_invoke(bedrock_override_client, httpx_mock):
    new_model = "anthropic.claude-3-haiku-20240307-v1:0"
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{new_model}/invoke",
        content=json.dumps({"content": [{"type": "text", "text": "Hi"}]}),
        status_code=200,
    )

    response = bedrock_override_client.post(
        f"/model/{MODEL_ID}/invoke",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    assert response.status_code == 200
    assert httpx_mock.get_requests()[0].url.path == f"/model/{new_model}/invoke"


@pytest.mark.parametrize(
    "bedrock_override_client", ["anthropic.claude-3-haiku-20240307-v1:0"], indirect=True
)
def test_model_override_bedrock_streaming(bedrock_override_client, httpx_mock):
    new_model = "anthropic.claude-3-haiku-20240307-v1:0"
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{new_model}/invoke-with-response-stream",
        content=b"chunk",
        status_code=200,
    )

    response = bedrock_override_client.post(
        f"/model/{MODEL_ID}/invoke-with-response-stream",
        json={"messages": [{"role": "user", "content": "Hi"}]},
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    assert response.status_code == 200
    assert (
        httpx_mock.get_requests()[0].url.path
        == f"/model/{new_model}/invoke-with-response-stream"
    )


# ---------------------------------------------------------------------------
# Rollup middleware helpers — Bedrock format
# ---------------------------------------------------------------------------


def test_extract_bedrock_messages_excludes_system():
    body = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    msgs = _extract_bedrock_messages(body)
    assert len(msgs) == 2
    assert all(m["role"] != "system" for m in msgs)


def test_extract_bedrock_messages_empty():
    assert _extract_bedrock_messages({}) == []


def test_inject_bedrock_system_prompt_string_format():
    # Anthropic-style: string content in messages → string system
    body = {"messages": [{"role": "user", "content": "Hi"}]}
    result = _inject_bedrock_system_prompt(body, "Injected context")
    assert result["system"] == "Injected context"


def test_inject_bedrock_system_prompt_array_format():
    # Nova-style: array content in messages → array system
    body = {"messages": [{"role": "user", "content": [{"text": "Hi"}]}]}
    result = _inject_bedrock_system_prompt(body, "Injected context")
    assert isinstance(result["system"], list)
    assert result["system"][0]["text"] == "Injected context"


def test_inject_bedrock_system_prompt_prepends_to_existing_string():
    body = {"system": "Original system."}
    result = _inject_bedrock_system_prompt(body, "Context")
    assert result["system"].startswith("Context\n")
    assert "Original system." in result["system"]


def test_inject_bedrock_system_prompt_prepends_to_existing_list():
    body = {"system": [{"text": "Original."}]}
    result = _inject_bedrock_system_prompt(body, "Context")
    assert isinstance(result["system"], list)
    assert result["system"][0]["text"] == "Context"
    assert result["system"][1]["text"] == "Original."


# ---------------------------------------------------------------------------
# Rollup middleware — integration with Bedrock endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def bedrock_rollup_client():
    proxy = GeminiProxyService(api_keys=["dummy-gemini-key"])
    lru_cache = LRUCache(maxsize=128)
    app = FastAPI()
    app.middleware("http")(
        create_rollup_middleware(gemini_proxy=proxy, lru_cache=lru_cache)
    )
    app.include_router(proxy.bedrock_router)
    return TestClient(app), lru_cache


def test_bedrock_rollup_caches_assistant_response(bedrock_rollup_client, httpx_mock):
    client, cache = bedrock_rollup_client
    mock_response = json.dumps(
        {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "I am Claude."}],
            "stop_reason": "end_turn",
        }
    )
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/invoke",
        content=mock_response,
        status_code=200,
    )

    client.post(
        f"/model/{MODEL_ID}/invoke",
        json={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": "Who are you?"}],
        },
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    assert len(cache) == 1


def test_bedrock_rollup_injects_cached_context(bedrock_rollup_client, httpx_mock):
    client, cache = bedrock_rollup_client

    # Turn 1: populate cache (Nova-style response)
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/invoke",
        content=json.dumps(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "I am Nova."}],
                    }
                },
                "stopReason": "end_turn",
            }
        ),
        status_code=200,
    )
    client.post(
        f"/model/{MODEL_ID}/invoke",
        json={"messages": [{"role": "user", "content": [{"text": "Who are you?"}]}]},
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    assert len(cache) == 1

    # Turn 2: send full history — rollup should hit the cache, inject system prompt,
    # and drop the already-cached messages.
    # The assistant content must use the same array format the rollup stored.
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/invoke",
        content=json.dumps(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Nice to meet you."}],
                    }
                },
            }
        ),
        status_code=200,
    )
    client.post(
        f"/model/{MODEL_ID}/invoke",
        json={
            "messages": [
                {"role": "user", "content": [{"text": "Who are you?"}]},
                {"role": "assistant", "content": [{"text": "I am Nova."}]},
                {"role": "user", "content": [{"text": "Nice!"}]},
            ]
        },
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    second_req = httpx_mock.get_requests()[1]
    sent_body = json.loads(second_req.content)
    # system prompt injected as array (Nova format auto-detected from array content)
    assert "system" in sent_body
    assert isinstance(sent_body["system"], list)
    # matched messages dropped
    assert len(sent_body["messages"]) < 3


# ---------------------------------------------------------------------------
# Bedrock Converse API — rollup middleware
# ---------------------------------------------------------------------------


def test_bedrock_converse_rollup_caches_response(bedrock_rollup_client, httpx_mock):
    client, cache = bedrock_rollup_client
    mock_response = json.dumps(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "I am Claude via Converse."}],
                }
            },
            "stopReason": "end_turn",
        }
    )
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/converse",
        content=mock_response,
        status_code=200,
    )

    client.post(
        f"/model/{MODEL_ID}/converse",
        json={
            "messages": [{"role": "user", "content": [{"text": "Who are you?"}]}],
        },
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    assert len(cache) == 1


def test_bedrock_converse_rollup_injects_cached_context(
    bedrock_rollup_client, httpx_mock
):
    client, cache = bedrock_rollup_client

    # Turn 1: populate cache via converse endpoint
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/converse",
        content=json.dumps(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "I am the Converse bot."}],
                    }
                },
                "stopReason": "end_turn",
            }
        ),
        status_code=200,
    )
    client.post(
        f"/model/{MODEL_ID}/converse",
        json={"messages": [{"role": "user", "content": [{"text": "Who are you?"}]}]},
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    assert len(cache) == 1

    # Turn 2: send full history — rollup injects system prompt and trims old messages
    httpx_mock.add_response(
        url=f"{BEDROCK_BASE}/model/{MODEL_ID}/converse",
        content=json.dumps(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Nice to meet you too."}],
                    }
                },
            }
        ),
        status_code=200,
    )
    client.post(
        f"/model/{MODEL_ID}/converse",
        json={
            "messages": [
                {"role": "user", "content": [{"text": "Who are you?"}]},
                {"role": "assistant", "content": [{"text": "I am the Converse bot."}]},
                {"role": "user", "content": [{"text": "Nice!"}]},
            ]
        },
        headers={"X-AWS-Bearer-Token": "tok"},
    )

    second_req = httpx_mock.get_requests()[1]
    sent_body = json.loads(second_req.content)
    assert "system" in sent_body
    assert isinstance(sent_body["system"], list)
    assert len(sent_body["messages"]) < 3
