import json
from functools import partial

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_calo.middlewares.auth import auth_middleware
from gemini_calo.proxy import GeminiProxyService

VALID_API_KEY = "test-proxy-key"


@pytest.fixture
def client():
    api_keys = ["dummy-gemini-key"]
    app = FastAPI()
    proxy = GeminiProxyService(api_keys=api_keys)

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
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
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
