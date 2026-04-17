"""Tests for the new auth module and auth providers."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import httpx

from gemini_calo.auth import (
    BearerAuth,
    XGoogApiKeyAuth,
    NoAuth,
    create_bearer_provider,
    create_xgoog_provider,
    AWSCredentials,
    AWSSigV4Auth,
    create_aws_sigv4_provider,
    ExtractedAWSCreds,
    extract_aws_creds_from_headers,
    create_passthrough_aws_provider,
)
from gemini_calo.proxy import GeminiProxyService, RouteConfig


# ---------------------------------------------------------------------------
# BearerAuth and XGoogApiKeyAuth tests
# ---------------------------------------------------------------------------

def test_bearer_auth_adds_header():
    """BearerAuth should add Authorization header."""
    auth = BearerAuth(token="my-secret-token")
    request = httpx.Request("POST", "https://example.com/api", content=b"{}")
    
    # Run the auth flow
    gen = auth.auth_flow(request)
    signed_request = next(gen)
    
    assert signed_request.headers["Authorization"] == "Bearer my-secret-token"


def test_xgoog_auth_adds_header():
    """XGoogApiKeyAuth should add x-goog-api-key header."""
    auth = XGoogApiKeyAuth(api_key="my-google-key")
    request = httpx.Request("POST", "https://example.com/api", content=b"{}")
    
    gen = auth.auth_flow(request)
    signed_request = next(gen)
    
    assert signed_request.headers["x-goog-api-key"] == "my-google-key"


def test_no_auth_passes_through():
    """NoAuth should not modify the request."""
    auth = NoAuth()
    request = httpx.Request("GET", "https://example.com/api")
    
    gen = auth.auth_flow(request)
    signed_request = next(gen)
    
    # Headers should be unchanged
    assert "Authorization" not in signed_request.headers
    assert "x-goog-api-key" not in signed_request.headers


# ---------------------------------------------------------------------------
# Provider factory tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_bearer_provider_rotates_keys():
    """Bearer provider should rotate through keys."""
    provider = create_bearer_provider(["key1", "key2", "key3"])
    request = MagicMock(spec=Request)
    
    auth1 = await provider(request)
    assert isinstance(auth1, BearerAuth)
    assert auth1.token == "key1"
    
    auth2 = await provider(request)
    assert auth2.token == "key2"
    
    auth3 = await provider(request)
    assert auth3.token == "key3"
    
    # Wrap around
    auth4 = await provider(request)
    assert auth4.token == "key1"


@pytest.mark.asyncio
async def test_create_xgoog_provider_rotates_keys():
    """XGoog provider should rotate through keys."""
    provider = create_xgoog_provider(["goog1", "goog2"])
    request = MagicMock(spec=Request)
    
    auth1 = await provider(request)
    assert isinstance(auth1, XGoogApiKeyAuth)
    assert auth1.api_key == "goog1"
    
    auth2 = await provider(request)
    assert auth2.api_key == "goog2"
    
    # Wrap around
    auth3 = await provider(request)
    assert auth3.api_key == "goog1"


# ---------------------------------------------------------------------------
# RouteConfig with new auth field tests
# ---------------------------------------------------------------------------

def test_route_config_with_auth_bearer_preset():
    """RouteConfig with auth='bearer' should create bearer provider."""
    route = RouteConfig(
        url="https://api.example.com",
        api_keys=["sk-test"],
        auth="bearer",
    )
    assert route._auth_provider is not None


def test_route_config_with_auth_xgoog_preset():
    """RouteConfig with auth='x-goog-api-key' should create xgoog provider."""
    route = RouteConfig(
        url="https://api.example.com",
        api_keys=["google-key"],
        auth="x-goog-api-key",
    )
    assert route._auth_provider is not None


def test_route_config_with_auth_none():
    """RouteConfig with auth=None should create no-auth provider."""
    route = RouteConfig(
        url="https://api.example.com",
        api_keys=[],
        auth=None,
    )
    assert route._auth_provider is not None


def test_route_config_with_auth_none_string():
    """RouteConfig with auth='none' should create no-auth provider."""
    route = RouteConfig(
        url="https://api.example.com",
        api_keys=["unused"],
        auth="none",
    )
    assert route._auth_provider is not None


def test_route_config_with_custom_provider():
    """RouteConfig should accept custom auth provider."""
    
    async def custom_provider(request: Request) -> httpx.Auth:
        return BearerAuth(token="custom-token")
    
    route = RouteConfig(
        url="https://api.example.com",
        api_keys=[],
        auth=custom_provider,
    )
    assert route._auth_provider is custom_provider


def test_route_config_auth_type_deprecated():
    """auth_type should trigger deprecation warning and map to auth."""
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        route = RouteConfig(
            url="https://api.example.com",
            api_keys=["test-key"],
            auth_type="x-goog-api-key",
        )
        
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "auth_type is deprecated" in str(w[0].message)


@pytest.mark.asyncio
async def test_route_config_get_auth_returns_bearer():
    """RouteConfig.get_auth() should return httpx.Auth for bearer."""
    route = RouteConfig(
        url="https://api.example.com",
        api_keys=["my-token"],
        auth="bearer",
    )
    request = MagicMock(spec=Request)
    
    auth = await route.get_auth(request)
    assert isinstance(auth, BearerAuth)
    assert auth.token == "my-token"


@pytest.mark.asyncio
async def test_route_config_get_auth_returns_xgoog():
    """RouteConfig.get_auth() should return httpx.Auth for x-goog-api-key."""
    route = RouteConfig(
        url="https://api.example.com",
        api_keys=["my-goog-key"],
        auth="x-goog-api-key",
    )
    request = MagicMock(spec=Request)
    
    auth = await route.get_auth(request)
    assert isinstance(auth, XGoogApiKeyAuth)
    assert auth.api_key == "my-goog-key"


# ---------------------------------------------------------------------------
# AWS Credential extraction tests
# ---------------------------------------------------------------------------

def test_extract_aws_creds_from_headers_success():
    """Should extract AWS credentials from request headers."""
    request = MagicMock(spec=Request)
    request.headers = {
        "X-AWS-Access-Key": "AKIAIOSFODNN7EXAMPLE",
        "X-AWS-Secret-Key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "X-AWS-Session-Token": "session-token-123",
        "X-AWS-Region": "ap-southeast-1",
    }
    
    creds = extract_aws_creds_from_headers(request)
    
    assert creds.access_key == "AKIAIOSFODNN7EXAMPLE"
    assert creds.secret_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert creds.session_token == "session-token-123"
    assert creds.region == "ap-southeast-1"
    assert creds.is_complete()


def test_extract_aws_creds_with_custom_headers():
    """Should use custom header names when provided."""
    request = MagicMock(spec=Request)
    request.headers = {
        "Custom-Access": "custom-key",
        "Custom-Secret": "custom-secret",
    }
    
    creds = extract_aws_creds_from_headers(
        request,
        access_key_header="Custom-Access",
        secret_key_header="Custom-Secret",
    )
    
    assert creds.access_key == "custom-key"
    assert creds.secret_key == "custom-secret"


def test_extract_aws_creds_missing_credentials():
    """Should handle missing credentials gracefully."""
    request = MagicMock(spec=Request)
    request.headers = {}
    
    creds = extract_aws_creds_from_headers(request)
    
    assert creds.access_key is None
    assert creds.secret_key is None
    assert not creds.is_complete()


def test_extracted_aws_creds_is_complete():
    """is_complete() should return True only when both keys present."""
    complete = ExtractedAWSCreds(
        access_key="key",
        secret_key="secret",
    )
    assert complete.is_complete()
    
    missing_secret = ExtractedAWSCreds(
        access_key="key",
        secret_key=None,
    )
    assert not missing_secret.is_complete()
    
    missing_access = ExtractedAWSCreds(
        access_key=None,
        secret_key="secret",
    )
    assert not missing_access.is_complete()


# ---------------------------------------------------------------------------
# Pass-through AWS provider tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_passthrough_aws_provider_extracts_creds():
    """Pass-through provider should extract creds from request."""
    provider = create_passthrough_aws_provider(
        required=True,
        default_region="us-east-1",
        service="bedrock",
    )
    
    request = MagicMock(spec=Request)
    request.headers = {
        "X-AWS-Access-Key": "AKIAIOSFODNN7EXAMPLE",
        "X-AWS-Secret-Key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "X-AWS-Region": "eu-west-1",
    }
    
    auth = await provider(request)
    
    assert isinstance(auth, AWSSigV4Auth)
    assert auth.access_key == "AKIAIOSFODNN7EXAMPLE"
    assert auth.secret_key == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    assert auth.region == "eu-west-1"
    assert auth.service == "bedrock"


@pytest.mark.asyncio
async def test_passthrough_aws_provider_raises_when_required():
    """Pass-through provider should raise when credentials missing and required=True."""
    provider = create_passthrough_aws_provider(required=True)
    
    request = MagicMock(spec=Request)
    request.headers = {}
    
    with pytest.raises(ValueError, match="AWS credentials required"):
        await provider(request)


@pytest.mark.asyncio
async def test_passthrough_aws_provider_fallback_when_not_required():
    """Pass-through provider should return NoAuth when credentials missing and required=False."""
    provider = create_passthrough_aws_provider(required=False)
    
    request = MagicMock(spec=Request)
    request.headers = {}
    
    auth = await provider(request)
    
    assert isinstance(auth, NoAuth)


# ---------------------------------------------------------------------------
# Static AWS SigV4 provider tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_static_aws_sigv4_provider():
    """Static AWS provider should return AWSSigV4Auth with static creds."""
    creds = AWSCredentials(
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        session_token="session-token",
        region="us-west-2",
        service="bedrock",
    )
    
    provider = create_aws_sigv4_provider(creds)
    request = MagicMock(spec=Request)
    
    auth = await provider(request)
    
    assert isinstance(auth, AWSSigV4Auth)
    assert auth.access_key == creds.access_key
    assert auth.secret_key == creds.secret_key
    assert auth.session_token == creds.session_token
    assert auth.region == creds.region
    assert auth.service == creds.service


# ---------------------------------------------------------------------------
# AWS SigV4 with streaming request tests
# ---------------------------------------------------------------------------

import sys

# Skip AWS SigV4 auth flow tests if botocore is not installed
try:
    import botocore  # noqa: F401
    HAS_BOTOCORE = True
except ImportError:
    HAS_BOTOCORE = False


@pytest.mark.skipif(not HAS_BOTOCORE, reason="botocore is required for AWS SigV4 tests (optional dependency)")
def test_aws_sigv4_auth_handles_streaming_request():
    """AWSSigV4Auth should handle streaming requests by buffering content."""
    # Create a streaming request (generator as content)
    def content_generator():
        yield b'{"input": "'
        yield b'hello'
        yield b'"}'
    
    auth = AWSSigV4Auth(
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        service="bedrock",
    )
    
    # Create streaming request - this will raise RequestNotRead on .content
    request = httpx.Request(
        "POST",
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/invoke",
        content=content_generator(),
    )
    
    # Before auth_flow, accessing content raises RequestNotRead
    with pytest.raises(httpx.RequestNotRead):
        _ = request.content
    
    # Run auth_flow - this should buffer the content
    gen = auth.auth_flow(request)
    signed_request = next(gen)
    
    # After auth_flow, content should be buffered
    assert request.content == b'{"input": "hello"}'
    
    # Signed headers should be present
    assert "Authorization" in signed_request.headers
    assert "X-Amz-Date" in signed_request.headers


@pytest.mark.skipif(not HAS_BOTOCORE, reason="botocore is required for AWS SigV4 tests (optional dependency)")
def test_aws_sigv4_auth_handles_regular_request():
    """AWSSigV4Auth should handle regular (non-streaming) requests."""
    auth = AWSSigV4Auth(
        access_key="AKIAIOSFODNN7EXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        service="bedrock",
    )
    
    request = httpx.Request(
        "POST",
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/invoke",
        content=b'{"input": "hello"}',
    )
    
    # Content should be immediately available
    assert request.content == b'{"input": "hello"}'
    
    # Run auth_flow
    gen = auth.auth_flow(request)
    signed_request = next(gen)
    
    # Signed headers should be present
    assert "Authorization" in signed_request.headers
    assert "X-Amz-Date" in signed_request.headers


# ---------------------------------------------------------------------------
# Integration: RouteConfig with custom auth in proxy
# ---------------------------------------------------------------------------

def test_proxy_with_custom_auth_provider(httpx_mock):
    """GeminiProxyService should use custom auth provider from RouteConfig."""
    
    async def custom_provider(request: Request) -> httpx.Auth:
        return BearerAuth(token="custom-route-token")
    
    route = RouteConfig(
        url="https://custom.example.com",
        api_keys=[],
        auth=custom_provider,
    )
    
    proxy = GeminiProxyService(
        base_url="https://generativelanguage.googleapis.com",
        api_keys=["default-key"],
        model_routes={"custom-*": route},
    )
    
    app = FastAPI()
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    client = TestClient(app)
    
    httpx_mock.add_response(
        url="https://custom.example.com/v1beta/openai/chat/completions",
        content=json.dumps({"choices": []}),
        status_code=200,
    )
    
    response = client.post(
        "/v1beta/openai/chat/completions",
        json={"model": "custom-model", "messages": []},
    )
    
    assert response.status_code == 200
    
    sent_request = httpx_mock.get_requests()[0]
    assert sent_request.headers["authorization"] == "Bearer custom-route-token"


def test_proxy_route_with_no_auth(httpx_mock):
    """RouteConfig with auth=None should send requests without auth headers."""
    route = RouteConfig(
        url="https://public.example.com",
        api_keys=[],
        auth=None,
    )
    
    proxy = GeminiProxyService(
        base_url="https://generativelanguage.googleapis.com",
        api_keys=["default-key"],
        model_routes={"public-*": route},
    )
    
    app = FastAPI()
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    client = TestClient(app)
    
    httpx_mock.add_response(
        url="https://public.example.com/v1beta/openai/chat/completions",
        content=json.dumps({"choices": []}),
        status_code=200,
    )
    
    response = client.post(
        "/v1beta/openai/chat/completions",
        json={"model": "public-model", "messages": []},
    )
    
    assert response.status_code == 200
    
    sent_request = httpx_mock.get_requests()[0]
    assert "authorization" not in sent_request.headers
    assert "x-goog-api-key" not in sent_request.headers