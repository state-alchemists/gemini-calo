import json
from functools import partial

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_calo.middlewares.model_override import create_model_override_middleware
from gemini_calo.proxy import GeminiProxyService

BASE_URL = "https://generativelanguage.googleapis.com"


@pytest.fixture
def client(request):
    """
    Fixture to create a TestClient with the middleware.
    The model_transformer is passed via pytest.mark.parametrize.
    """
    model_transformer = getattr(request, "param", None)
    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=["dummy-gemini-key"])

    if model_transformer:
        app.middleware("http")(
            create_model_override_middleware(model_transformer=model_transformer)
        )

    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    return TestClient(app)


# --- Test Cases ---

@pytest.mark.parametrize("client", [None], indirect=True)
def test_no_override(client, httpx_mock):
    """
    Tests that the model is NOT changed when no transformer is provided.
    """
    mock_url = f"{BASE_URL}/v1beta/models/gemini-pro:generateContent"
    httpx_mock.add_response(url=mock_url, status_code=200, json={"ok": True})

    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"parts": [{"text": "Hello"}]}]},
    )
    assert response.status_code == 200
    assert len(httpx_mock.get_requests()) == 1
    assert (
        httpx_mock.get_requests()[0].url.path
        == "/v1beta/models/gemini-pro:generateContent"
    )


@pytest.mark.parametrize("client", ["gemini-1.5-flash"], indirect=True)
def test_string_override_for_gemini(client, httpx_mock):
    """
    Tests that a simple string override works for Gemini requests.
    """
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


@pytest.mark.parametrize("client", ["gemini-1.5-flash"], indirect=True)
def test_string_override_for_openai(client, httpx_mock):
    """
    Tests that a simple string override works for OpenAI requests.
    """
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
    assert len(httpx_mock.get_requests()) == 1
    assert httpx_mock.get_requests()[0].url.path == "/v1beta/openai/chat/completions"


def test_env_var_override(monkeypatch, httpx_mock):
    """
    Tests that the middleware correctly uses the environment variable for override.
    """
    monkeypatch.setenv("GEMINI_CALO_MODEL_OVERRIDE", "gemini-ultra")
    from gemini_calo import config
    import importlib
    importlib.reload(config)

    client = TestClient(FastAPI())
    proxy = GeminiProxyService(gemini_api_keys=["dummy-gemini-key"])
    client.app.middleware("http")(create_model_override_middleware())
    client.app.include_router(proxy.gemini_router)
    client.app.include_router(proxy.openai_router)

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


def model_switcher(model: str):
    if "pro" in model:
        return "gemini-1.5-flash"
    return model

@pytest.mark.parametrize("client", [model_switcher], indirect=True)
def test_function_override(client, httpx_mock):
    """
    Tests that a callable function can be used to transform the model name.
    """
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
@pytest.mark.parametrize("client", [async_model_switcher], indirect=True)
async def test_async_function_override(client, httpx_mock):
    """
    Tests that an async callable function can be used to transform the model name.
    """
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