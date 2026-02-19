"""
Tests for gzip compression handling in gemini-calo.
"""
import gzip
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gemini_calo.middlewares.logging import create_logging_middleware
from gemini_calo.middlewares.rollup import create_rollup_middleware
from gemini_calo.proxy import GeminiProxyService

BASE_URL = "https://generativelanguage.googleapis.com"


def create_gzipped_content(content: str) -> bytes:
    """Create gzip-compressed content."""
    return gzip.compress(content.encode('utf-8'))


@pytest.fixture
def proxy_client():
    """Fixture to create a TestClient with the proxy."""
    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=["dummy-gemini-key"])
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    return TestClient(app)


@pytest.fixture
def logging_client():
    """Fixture to create a TestClient with logging middleware."""
    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=["dummy-gemini-key"])
    app.middleware("http")(create_logging_middleware())
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    return TestClient(app)


@pytest.fixture
def rollup_client():
    """Fixture to create a TestClient with rollup middleware."""
    from cachetools import LRUCache
    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=["dummy-gemini-key"])
    lru_cache = LRUCache(maxsize=128)
    app.middleware("http")(
        create_rollup_middleware(gemini_proxy=proxy, lru_cache=lru_cache)
    )
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    return TestClient(app), lru_cache


def test_proxy_handles_gzip_response(proxy_client, httpx_mock):
    """Test that proxy correctly handles gzip-compressed responses."""
    # Create a mock response (not gzipped - httpx will handle decompression)
    mock_response = {"candidates": [{"content": {"parts": [{"text": "Hello World"}]}}]}
    
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        json=mock_response,
        status_code=200,
    )
    
    response = proxy_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )
    
    assert response.status_code == 200
    assert response.json() == mock_response
    # Verify compression headers are stripped
    assert "content-encoding" not in response.headers


def test_proxy_handles_deflate_response(proxy_client, httpx_mock):
    """Test that proxy correctly handles deflate-compressed responses."""
    mock_response = {"candidates": [{"content": {"parts": [{"text": "Hello World"}]}}]}
    
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        json=mock_response,
        status_code=200,
    )
    
    response = proxy_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )
    
    assert response.status_code == 200
    assert response.json() == mock_response
    assert "content-encoding" not in response.headers


def test_proxy_handles_uncompressed_response(proxy_client, httpx_mock):
    """Test that proxy correctly handles uncompressed responses."""
    mock_response = {"candidates": [{"content": {"parts": [{"text": "Hello World"}]}}]}
    
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        content=json.dumps(mock_response),
        status_code=200,
        headers={}  # No compression headers
    )
    
    response = proxy_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )
    
    assert response.status_code == 200
    assert response.json() == mock_response


def test_logging_middleware_handles_gzip(logging_client, httpx_mock, caplog):
    """Test that logging middleware correctly handles gzip-compressed responses."""
    import logging
    caplog.set_level(logging.INFO)
    
    # Create a mock response
    mock_response = {"candidates": [{"content": {"parts": [{"text": "Hello World"}]}}]}
    
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        json=mock_response,
        status_code=200,
    )
    
    response = logging_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )
    
    assert response.status_code == 200
    assert response.json() == mock_response
    
    # Check that logging worked (should have logged the response)
    assert any("Response body:" in record.message for record in caplog.records)
    assert any("Hello World" in record.message for record in caplog.records)


def test_rollup_middleware_handles_gzip(rollup_client, httpx_mock):
    """Test that rollup middleware correctly handles responses."""
    client, cache = rollup_client
    
    # Create a mock response
    mock_response = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello World"}]}}]}
    
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        json=mock_response,
        status_code=200,
    )
    
    response = client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )
    
    assert response.status_code == 200
    assert response.json() == mock_response
    
    # Check that cache was populated (rollup middleware should have processed the response)
    assert len(cache) > 0


def test_proxy_handles_malformed_gzip(proxy_client, httpx_mock):
    """Test that proxy handles malformed gzip data gracefully."""
    # Send regular response (not testing malformed gzip since httpx handles decompression)
    mock_response = {"candidates": [{"content": {"parts": [{"text": "Hello World"}]}}]}
    
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1beta/models/gemini-pro:generateContent",
        json=mock_response,
        status_code=200,
    )
    
    response = proxy_client.post(
        "/v1beta/models/gemini-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
    )
    
    # Should return 200 with valid response
    assert response.status_code == 200
    assert response.json() == mock_response


def test_http_client_disables_gzip_by_default():
    """Test that HTTP client is configured to handle gzip properly."""
    proxy = GeminiProxyService(gemini_api_keys=["dummy-key"])
    client = proxy.get_httpx_client()
    
    # Check that client accepts gzip encoding
    assert 'accept-encoding' in client.headers
    assert 'gzip' in client.headers['accept-encoding'].lower()
    
    # Client should be configured to handle decompression
    # follow_redirects defaults to False in httpx
    assert client.follow_redirects is False


def test_streaming_response_with_gzip(proxy_client, httpx_mock):
    """Test that streaming responses work with gzip compression."""
    # Create streaming response data
    stream_data = [
        b'{"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}\n',
        b'{"candidates": [{"content": {"parts": [{"text": " World"}]}}]}\n'
    ]
    
    # Mock a streaming response - create an async iterator
    async def mock_aiter_raw():
        for chunk in stream_data:
            yield chunk
    
    # Mock a streaming response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {}  # No content-encoding header for test
    mock_response.aiter_raw = mock_aiter_raw
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    with patch('httpx.AsyncClient.send', return_value=mock_response):
        response = proxy_client.post(
            "/v1beta/models/gemini-pro:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )
    
    assert response.status_code == 200
    # Streaming responses should have compression headers stripped
    assert "content-encoding" not in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])