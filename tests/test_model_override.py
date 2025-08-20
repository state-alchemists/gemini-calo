import json
from functools import partial

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_calo.middlewares.model_override import model_override_middleware
from gemini_calo.proxy import GeminiProxyService

BASE_URL = "https://generativelanguage.googleapis.com"
VALID_API_KEY = "test-proxy-key"


def create_test_client(model_transformer=None):
    """Helper function to create a TestClient with the middleware."""
    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=["dummy-gemini-key"])

    # Apply the model override middleware using partial
    middleware = partial(model_override_middleware, model_transformer=model_transformer)
    app.middleware("http")(middleware)

    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    return TestClient(app)


# --- Test Cases ---

def test_no_override(httpx_mock):
    """
    Tests that the model is NOT changed when no transformer is provided.
    """
    client = create_test_client()
    mock_url = f"{BASE_URL}/v1beta/models/gemini-pro:generateContent"
    httpx_mock.add_response(url=mock_url, status_code=200, json={"ok": True})

    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert response.status_code == 200
    assert len(httpx_mock.get_requests()) == 1
    assert httpx_mock.get_requests()[0].url.path == "/v1beta/models/gemini-pro:generateContent"


def test_string_override_for_gemini(httpx_mock):
    """
    Tests that a simple string override works for Gemini requests.
    """
    client = create_test_client(model_transformer="gemini-1.5-flash")
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-1.5-flash:generateContent",
        status_code=200,
        json={"ok": True},
    )

    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert response.status_code == 200
    assert len(httpx_mock.get_requests()) == 1


def test_string_override_for_openai(httpx_mock):
    """
    Tests that a simple string override works for OpenAI requests.
    """
    client = create_test_client(model_transformer="gemini-1.5-flash")
    # This test now mocks the URL that the proxy actually calls for OpenAI requests.
    mock_url = f"{BASE_URL}/v1beta/openai/chat/completions"
    httpx_mock.add_response(
        url=mock_url,
        status_code=200,
        json={"candidates": [{"content": {"parts": [{"text": "response"}]}}]},
    )

    response = client.post(
        "/v1beta/openai/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200
    # Check that the request was actually sent to the correctly transformed URL
    assert len(httpx_mock.get_requests()) == 1
    assert httpx_mock.get_requests()[0].url.path == "/v1beta/openai/chat/completions"


def test_env_var_override(monkeypatch, httpx_mock):
    """
    Tests that the middleware correctly uses the environment variable for override.
    """
    monkeypatch.setenv("GEMINI_CALO_MODEL_OVERRIDE", "gemini-ultra")
    # We need to re-import config for the change to take effect
    from gemini_calo import config
    import importlib
    importlib.reload(config)

    client = create_test_client()  # No transformer passed directly
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-ultra:generateContent",
        status_code=200,
        json={"ok": True},
    )

    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert response.status_code == 200
    assert len(httpx_mock.get_requests()) == 1


def test_function_override(httpx_mock):
    """
    Tests that a callable function can be used to transform the model name.
    """
    def model_switcher(model: str):
        if "pro" in model:
            return "gemini-1.5-flash"
        return model

    client = create_test_client(model_transformer=model_switcher)
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-1.5-flash:generateContent",
        status_code=200,
        json={"ok": True},
    )

    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert response.status_code == 200
    assert len(httpx_mock.get_requests()) == 1


async def async_model_switcher(model: str):
    """Async version of the model switcher."""
    if "pro" in model:
        return "gemini-1.5-flash"
    return model

@pytest.mark.asyncio
async def test_async_function_override(httpx_mock):
    """
    Tests that an async callable function can be used to transform the model name.
    """
    client = create_test_client(model_transformer=async_model_switcher)
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-1.5-flash:generateContent",
        status_code=200,
        json={"ok": True},
    )

    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert response.status_code == 200
    assert len(httpx_mock.get_requests()) == 1