"""Tests for model_routes routing, RouteConfig, and related helpers."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from gemini_calo.proxy import GeminiProxyService, RouteConfig

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"
ALT_BASE_URL = "https://openai.example.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gemini_request(path: str) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": [],
    }

    async def receive():
        return {"type": "http.request", "body": b"{}"}

    return Request(scope, receive)


def _make_openai_request(model: str | None = "gpt-4o") -> Request:
    body = json.dumps({"model": model, "messages": []}).encode() if model is not None else b"{}"
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1beta/openai/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }

    async def receive():
        return {"type": "http.request", "body": body}

    return Request(scope, receive)


def _make_proxy_with_routes(**routes: RouteConfig) -> GeminiProxyService:
    return GeminiProxyService(
        base_url=GEMINI_BASE_URL,
        api_keys=["default-key"],
        model_routes=routes,
    )


def _make_test_client(proxy: GeminiProxyService) -> TestClient:
    app = FastAPI()
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# RouteConfig round-robin
# ---------------------------------------------------------------------------

def test_route_config_round_robin_cycles_through_keys():
    route = RouteConfig(url="https://example.com", api_keys=["key-a", "key-b", "key-c"])
    assert route.get_api_key() == "key-a"
    assert route.get_api_key() == "key-b"
    assert route.get_api_key() == "key-c"
    assert route.get_api_key() == "key-a"  # wraps around


def test_route_config_single_key_always_returns_same():
    route = RouteConfig(url="https://example.com", api_keys=["only-key"])
    for _ in range(5):
        assert route.get_api_key() == "only-key"


# ---------------------------------------------------------------------------
# _extract_model_name — Gemini paths
# ---------------------------------------------------------------------------

async def test_extract_model_name_gemini_generate_content():
    proxy = _make_proxy_with_routes()
    request = _make_gemini_request("/v1beta/models/gemini-1.5-flash:generateContent")
    assert await proxy._extract_model_name(request) == "gemini-1.5-flash"


async def test_extract_model_name_gemini_stream_generate_content():
    proxy = _make_proxy_with_routes()
    request = _make_gemini_request("/v1beta/models/gemini-2.0-pro:streamGenerateContent")
    assert await proxy._extract_model_name(request) == "gemini-2.0-pro"


async def test_extract_model_name_gemini_embed_content():
    proxy = _make_proxy_with_routes()
    request = _make_gemini_request("/v1beta/models/text-embedding-004:embedContent")
    assert await proxy._extract_model_name(request) == "text-embedding-004"


async def test_extract_model_name_gemini_with_slash_in_model():
    """Model names that include a slash (path param uses :path converter)."""
    proxy = _make_proxy_with_routes()
    request = _make_gemini_request("/v1beta/models/publishers/google/models/gemini-pro:generateContent")
    assert await proxy._extract_model_name(request) == "publishers/google/models/gemini-pro"


# ---------------------------------------------------------------------------
# _extract_model_name — OpenAI body
# ---------------------------------------------------------------------------

async def test_extract_model_name_openai_from_body():
    proxy = _make_proxy_with_routes()
    request = _make_openai_request("gpt-4o")
    assert await proxy._extract_model_name(request) == "gpt-4o"


async def test_extract_model_name_openai_missing_model_field():
    proxy = _make_proxy_with_routes()
    request = _make_openai_request(model=None)
    # body is `{}`, no "model" key
    assert await proxy._extract_model_name(request) is None


async def test_extract_model_name_other_endpoint_returns_none():
    proxy = _make_proxy_with_routes()
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": [],
    }
    request = Request(scope, lambda: None)
    assert await proxy._extract_model_name(request) is None


# ---------------------------------------------------------------------------
# _find_route — glob matching
# ---------------------------------------------------------------------------

def test_find_route_exact_match():
    route = RouteConfig(url="https://example.com", api_keys=["k"])
    proxy = _make_proxy_with_routes(**{"gemini-1.5-flash": route})
    assert proxy._find_route("gemini-1.5-flash") is route


def test_find_route_glob_prefix():
    route = RouteConfig(url="https://example.com", api_keys=["k"])
    proxy = _make_proxy_with_routes(**{"gpt-4*": route})
    assert proxy._find_route("gpt-4o") is route
    assert proxy._find_route("gpt-4-turbo") is route
    assert proxy._find_route("gpt-3.5-turbo") is None


def test_find_route_first_match_wins():
    route_a = RouteConfig(url="https://a.example.com", api_keys=["a"])
    route_b = RouteConfig(url="https://b.example.com", api_keys=["b"])
    # Python dicts preserve insertion order
    proxy = _make_proxy_with_routes(**{"gpt-*": route_a, "gpt-4*": route_b})
    assert proxy._find_route("gpt-4o") is route_a


def test_find_route_no_match_returns_none():
    route = RouteConfig(url="https://example.com", api_keys=["k"])
    proxy = _make_proxy_with_routes(**{"gpt-4*": route})
    assert proxy._find_route("gemini-pro") is None


def test_find_route_none_model_returns_none():
    route = RouteConfig(url="https://example.com", api_keys=["k"])
    proxy = _make_proxy_with_routes(**{"*": route})
    assert proxy._find_route(None) is None


# ---------------------------------------------------------------------------
# Integration: request goes to the correct upstream URL
# ---------------------------------------------------------------------------

def test_gemini_request_routed_to_custom_url(httpx_mock):
    route = RouteConfig(url=ALT_BASE_URL, api_keys=["alt-key"], auth_type="bearer")
    proxy = _make_proxy_with_routes(**{"gemini-2.5-pro*": route})
    client = _make_test_client(proxy)

    httpx_mock.add_response(
        url=f"{ALT_BASE_URL}/v1beta/models/gemini-2.5-pro:generateContent",
        content=json.dumps({"candidates": []}),
        status_code=200,
    )

    response = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"parts": [{"text": "hi"}]}]},
    )
    assert response.status_code == 200

    sent_request = httpx_mock.get_requests()[0]
    assert str(sent_request.url).startswith(ALT_BASE_URL)


def test_openai_request_routed_to_custom_url(httpx_mock):
    route = RouteConfig(url=ALT_BASE_URL, api_keys=["alt-key"], auth_type="bearer")
    proxy = _make_proxy_with_routes(**{"gpt-4*": route})
    client = _make_test_client(proxy)

    httpx_mock.add_response(
        url=f"{ALT_BASE_URL}/v1beta/openai/chat/completions",
        content=json.dumps({"choices": []}),
        status_code=200,
    )

    response = client.post(
        "/v1beta/openai/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200

    sent_request = httpx_mock.get_requests()[0]
    assert str(sent_request.url).startswith(ALT_BASE_URL)


def test_unmatched_model_falls_back_to_base_url(httpx_mock):
    route = RouteConfig(url=ALT_BASE_URL, api_keys=["alt-key"])
    proxy = _make_proxy_with_routes(**{"gpt-4*": route})
    client = _make_test_client(proxy)

    httpx_mock.add_response(
        url=f"{GEMINI_BASE_URL}/v1beta/models/gemini-1.5-flash:generateContent",
        content=json.dumps({"candidates": []}),
        status_code=200,
    )

    response = client.post(
        "/v1beta/models/gemini-1.5-flash:generateContent",
        json={"contents": []},
    )
    assert response.status_code == 200

    sent_request = httpx_mock.get_requests()[0]
    assert str(sent_request.url).startswith(GEMINI_BASE_URL)


# ---------------------------------------------------------------------------
# Integration: auth headers
# ---------------------------------------------------------------------------

def test_route_bearer_auth_sends_authorization_header(httpx_mock):
    route = RouteConfig(
        url=ALT_BASE_URL,
        api_keys=["secret-key"],
        auth_type="bearer",
    )
    proxy = _make_proxy_with_routes(**{"gpt-4*": route})
    client = _make_test_client(proxy)

    httpx_mock.add_response(
        url=f"{ALT_BASE_URL}/v1beta/openai/chat/completions",
        content=json.dumps({"choices": []}),
        status_code=200,
    )

    client.post(
        "/v1beta/openai/chat/completions",
        json={"model": "gpt-4o", "messages": []},
    )

    sent_request = httpx_mock.get_requests()[0]
    assert sent_request.headers["authorization"] == "Bearer secret-key"
    assert "x-goog-api-key" not in sent_request.headers


def test_route_goog_auth_sends_x_goog_api_key_header(httpx_mock):
    route = RouteConfig(
        url=ALT_BASE_URL,
        api_keys=["goog-key"],
        auth_type="x-goog-api-key",
    )
    proxy = _make_proxy_with_routes(**{"gemini-alt*": route})
    client = _make_test_client(proxy)

    httpx_mock.add_response(
        url=f"{ALT_BASE_URL}/v1beta/models/gemini-alt-flash:generateContent",
        content=json.dumps({"candidates": []}),
        status_code=200,
    )

    client.post(
        "/v1beta/models/gemini-alt-flash:generateContent",
        json={"contents": []},
    )

    sent_request = httpx_mock.get_requests()[0]
    assert sent_request.headers["x-goog-api-key"] == "goog-key"
    assert "authorization" not in sent_request.headers


# ---------------------------------------------------------------------------
# Integration: round-robin across RouteConfig api_keys
# ---------------------------------------------------------------------------

def test_route_round_robin_across_requests(httpx_mock):
    route = RouteConfig(
        url=ALT_BASE_URL,
        api_keys=["key-1", "key-2"],
        auth_type="bearer",
    )
    proxy = _make_proxy_with_routes(**{"gpt-4*": route})
    client = _make_test_client(proxy)

    for _ in range(2):
        httpx_mock.add_response(
            url=f"{ALT_BASE_URL}/v1beta/openai/chat/completions",
            content=json.dumps({"choices": []}),
            status_code=200,
        )

    client.post("/v1beta/openai/chat/completions", json={"model": "gpt-4o", "messages": []})
    client.post("/v1beta/openai/chat/completions", json={"model": "gpt-4o", "messages": []})

    requests = httpx_mock.get_requests()
    assert requests[0].headers["authorization"] == "Bearer key-1"
    assert requests[1].headers["authorization"] == "Bearer key-2"
