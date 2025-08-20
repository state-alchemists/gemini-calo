import json
from functools import partial

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_calo.middlewares.auth import auth_middleware
from gemini_calo.proxy import GeminiProxyService

BASE_URL = "https://generativelanguage.googleapis.com"
VALID_API_KEY = "test-proxy-key"


@pytest.fixture
def client():
    api_keys = ["dummy-gemini-key"]
    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=api_keys)

    proxy_api_keys = [VALID_API_KEY]
    auth_middleware_with_key = partial(
        auth_middleware, user_api_key_checker=proxy_api_keys
    )
    app.middleware("http")(auth_middleware_with_key)

    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)

    @app.get("/")
    def read_root():
        return {"Hello": "Proxy"}

    return TestClient(app)


def test_gemini_generate_content_with_auth(client, httpx_mock):
    mock_response_content = json.dumps({"candidates": [{"finishReason": "STOP"}]})
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-1.5-flash:generateContent",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/models/gemini-1.5-flash:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
        headers={"Authorization": f"Bearer {VALID_API_KEY}"},
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content)


def test_gemini_generate_content_with_invalid_auth(client):
    response = client.post(
        "/v1beta/models/gemini-1.5-flash:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
        headers={"Authorization": "Bearer INVALID_KEY"},
    )
    assert response.status_code == 401


def test_gemini_generate_content_without_auth(client):
    response = client.post(
        "/v1beta/models/gemini-1.5-flash:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert response.status_code == 401


def test_gemini_function_calling(client, httpx_mock):
    mock_response_content = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "find_weather",
                                    "args": {"location": "Boston"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-1.5-flash:generateContent",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/models/gemini-1.5-flash:generateContent",
        json={
            "contents": {"parts": {"text": "What is the weather in Boston?"}},
            "tools": [{"function_declarations": [{"name": "find_weather"}]}],
        },
        headers={"Authorization": f"Bearer {VALID_API_KEY}"},
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content)


def test_gemini_stream_generate_content(client, httpx_mock):
    mock_response_content = b"{\"candidates\": [{\"content\": {\"parts\": [{\"text\": \"Hello\"}]}}]}\n{\"candidates\": [{\"content\": {\"parts\": [{\"text\": \" world\"}]}}]}"

    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-1.5-flash:streamGenerateContent",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/models/gemini-1.5-flash:streamGenerateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
        headers={"Authorization": f"Bearer {VALID_API_KEY}"},
    )

    assert response.status_code == 200
    streaming_content = response.iter_bytes()
    chunks = [chunk for chunk in streaming_content]
    assert b"".join(chunks) == mock_response_content


def test_openai_chat_completions(client, httpx_mock):
    mock_response_content = json.dumps({"choices": [{"message": {"content": "Hello"}}]})
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/openai/chat/completions",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/openai/chat/completions",
        json={
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers={"Authorization": f"Bearer {VALID_API_KEY}"},
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content)


def test_openai_function_calling(client, httpx_mock):
    mock_response_content = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "find_weather",
                                    "arguments": '{\n  "location": "Boston"\n}'
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/openai/chat/completions",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/openai/chat/completions",
        json={
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": "What is the weather in Boston?"}],
            "tools": [{"type": "function", "function": {"name": "find_weather"}}],
        },
        headers={"Authorization": f"Bearer {VALID_API_KEY}"},
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content)


def test_openai_chat_completions_streaming(client, httpx_mock):
    mock_response_content = b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\ndata: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n\ndata: [DONE]\n\n"

    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/openai/chat/completions",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/openai/chat/completions",
        json={
            "model": "gemini-1.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
        headers={"Authorization": f"Bearer {VALID_API_KEY}"},
    )

    assert response.status_code == 200
    streaming_content = response.iter_bytes()
    chunks = [chunk for chunk in streaming_content]
    assert b"".join(chunks) == mock_response_content

def test_openai_to_gemini_conversion(client, httpx_mock):
    mock_response_content = json.dumps({"candidates": [{"finishReason": "STOP"}]})
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/openai/chat/completions",
        content=mock_response_content,
        status_code=200,
    )

    response = client.post(
        "/v1beta/openai/chat/completions",
        json={
            "model": "gemini-pro",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers={"Authorization": f"Bearer {VALID_API_KEY}"},
    )

    assert response.status_code == 200
    assert response.json() == json.loads(mock_response_content)