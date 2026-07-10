import json
from functools import partial

import pytest
from cachetools import LRUCache
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_calo.middlewares.rollup import create_rollup_middleware
from gemini_calo.proxy import GeminiProxyService

BASE_URL = "https://generativelanguage.googleapis.com"


@pytest.fixture
def client(request):
    """
    Fixture to create a TestClient with the middleware.
    """
    app = FastAPI()
    proxy = GeminiProxyService(api_keys=["dummy-gemini-key"])

    lru_cache = LRUCache(maxsize=128)
    app.middleware("http")(
        create_rollup_middleware(gemini_proxy=proxy, lru_cache=lru_cache)
    )

    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    return TestClient(app), lru_cache


def test_non_completion_request(client, httpx_mock):
    client, _ = client
    """
    Tests that a request that is not a completion request is not affected.
    """
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models", status_code=200, json={"models": []}
    )
    response = client.get("/v1beta/models")
    assert response.status_code == 200
    assert len(httpx_mock.get_requests()) == 1


def test_gemini_completion_not_in_cache(client, httpx_mock):
    client, _ = client
    mock_response_content = json.dumps(
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "World"}]}}]}
    )
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content)
    assert len(httpx_mock.get_requests()) == 1


def test_gemini_completion_in_cache(client, httpx_mock):
    client, cache = client
    mock_response_content_1 = json.dumps(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "I am doing well."}]}}
            ]
        }
    )
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        content=mock_response_content_1,
        status_code=200,
    )

    # First request to populate the cache
    client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )

    mock_response_content_2 = json.dumps(
        {
            "candidates": [
                {"content": {"parts": [{"text": "I am a large language model."}]}}
            ]
        }
    )
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        content=mock_response_content_2,
        status_code=200,
    )

    # Second request with the same history plus a new message
    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={
            "contents": [
                {"role": "user", "parts": [{"text": "Hello"}]},
                {"role": "model", "parts": [{"text": "I am doing well."}]},
                {"role": "user", "parts": [{"text": "How are you?"}]},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content_2)
    assert len(httpx_mock.get_requests()) == 2
    request = httpx_mock.get_requests()[1]
    request_body = json.loads(request.content)
    assert "system_instruction" in request_body


def test_openai_completion_not_in_cache(client, httpx_mock):
    client, _ = client
    mock_response_content = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "World"}}]}
    )
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/openai/chat/completions",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content)
    assert len(httpx_mock.get_requests()) == 1


def test_openai_completion_in_cache(client, httpx_mock):
    client, cache = client
    mock_response_content_1 = json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": "I am doing well."}}]}
    )
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/openai/chat/completions",
        content=mock_response_content_1,
        status_code=200,
    )

    # First request to populate the cache
    client.post(
        "/v1beta/openai/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )

    mock_response_content_2 = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I am a large language model.",
                    }
                }
            ]
        }
    )
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/openai/chat/completions",
        content=mock_response_content_2,
        status_code=200,
    )

    # Second request with the same history plus a new message
    response = client.post(
        "/v1beta/openai/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "I am doing well."},
                {"role": "user", "content": "How are you?"},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content_2)
    assert len(httpx_mock.get_requests()) == 2
    request = httpx_mock.get_requests()[1]
    request_body = json.loads(request.content)
    assert request_body["messages"][0]["role"] == "system"
