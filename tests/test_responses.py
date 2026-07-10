"""Tests for the /v1/responses passthrough endpoint."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from gemini_calo.proxy import REQUEST_TYPE, GeminiProxyService, RouteConfig

ALT_BASE_URL = "https://openai.example.com"


def _make_client(**routes: RouteConfig) -> TestClient:
    proxy = GeminiProxyService(api_keys=["default-key"], model_routes=routes)
    app = FastAPI()
    app.include_router(proxy.openai_router)
    return TestClient(app)


def _make_request(body: bytes, path: str = "/v1/responses") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }

    async def receive():
        return {"type": "http.request", "body": body}

    return Request(scope, receive)


def test_get_request_type_responses():
    request = _make_request(b"{}")
    assert GeminiProxyService.get_request_type(request) == REQUEST_TYPE.OPENAI_RESPONSES


async def test_extract_model_name_from_responses_body():
    proxy = GeminiProxyService(api_keys=["k"])
    request = _make_request(json.dumps({"model": "gpt-5", "input": "hi"}).encode())
    assert await proxy._extract_model_name(request) == "gpt-5"


async def test_body_requests_streaming():
    assert await GeminiProxyService._body_requests_streaming(
        _make_request(b'{"stream": true}')
    )
    assert not await GeminiProxyService._body_requests_streaming(
        _make_request(b'{"stream": false}')
    )
    assert not await GeminiProxyService._body_requests_streaming(_make_request(b"{}"))
    assert not await GeminiProxyService._body_requests_streaming(
        _make_request(b"not json")
    )


def test_responses_routed_to_custom_upstream(httpx_mock):
    route = RouteConfig(url=ALT_BASE_URL, api_keys=["alt-key"], auth="bearer")
    client = _make_client(**{"gpt-5*": route})

    httpx_mock.add_response(
        url=f"{ALT_BASE_URL}/v1/responses",
        content=json.dumps({"id": "resp_1", "output": []}),
        status_code=200,
    )

    response = client.post("/v1/responses", json={"model": "gpt-5", "input": "hi"})
    assert response.status_code == 200
    assert response.json()["id"] == "resp_1"

    sent = httpx_mock.get_requests()[0]
    assert str(sent.url).startswith(f"{ALT_BASE_URL}/v1/responses")
    assert sent.headers["authorization"] == "Bearer alt-key"


def test_responses_streaming_passthrough(httpx_mock):
    route = RouteConfig(url=ALT_BASE_URL, api_keys=["alt-key"], auth="bearer")
    client = _make_client(**{"gpt-5*": route})

    sse = b'event: response.completed\ndata: {"type": "response.completed"}\n\n'
    httpx_mock.add_response(
        url=f"{ALT_BASE_URL}/v1/responses",
        content=sse,
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
    )

    response = client.post(
        "/v1/responses", json={"model": "gpt-5", "input": "hi", "stream": True}
    )
    assert response.status_code == 200
    assert b"".join(response.iter_bytes()) == sse
