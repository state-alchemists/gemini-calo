"""Built-in authentication providers for common auth schemes."""

import threading
from dataclasses import dataclass
from typing import Generator

import httpx
from fastapi import Request

from gemini_calo.auth.providers import AuthProviderFunc


@dataclass
class BearerAuth(httpx.Auth):
    """Simple bearer token authentication.

    Adds Authorization: Bearer <token> header to requests.
    """

    token: str

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


@dataclass
class XGoogApiKeyAuth(httpx.Auth):
    """Google API key authentication.

    Adds x-goog-api-key header to requests.
    """

    api_key: str

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["x-goog-api-key"] = self.api_key
        yield request


@dataclass
class NoAuth(httpx.Auth):
    """No authentication - passes requests through unchanged.

    Useful for public endpoints or when auth is handled elsewhere.
    """

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        yield request


def create_bearer_provider(api_keys: list[str]) -> AuthProviderFunc:
    """Factory for round-robin bearer token authentication.

    Args:
        api_keys: List of bearer tokens to rotate through.

    Returns:
        An AuthProviderFunc that rotates through the tokens.

    Example:
        >>> provider = create_bearer_provider(["token1", "token2"])
        >>> auth = await provider(request)  # Returns BearerAuth("token1")
        >>> auth = await provider(request)  # Returns BearerAuth("token2")
        >>> auth = await provider(request)  # Returns BearerAuth("token1") - wraps around
    """
    state = {"index": 0}
    lock = threading.Lock()

    async def provider(request: Request) -> httpx.Auth:
        with lock:
            index = state["index"]
            token = api_keys[index]
            state["index"] = (index + 1) % len(api_keys)
        return BearerAuth(token=token)

    return provider


def create_xgoog_provider(api_keys: list[str]) -> AuthProviderFunc:
    """Factory for round-robin Google API key authentication.

    Args:
        api_keys: List of Google API keys to rotate through.

    Returns:
        An AuthProviderFunc that rotates through the keys.
    """
    state = {"index": 0}
    lock = threading.Lock()

    async def provider(request: Request) -> httpx.Auth:
        with lock:
            index = state["index"]
            api_key = api_keys[index]
            state["index"] = (index + 1) % len(api_keys)
        return XGoogApiKeyAuth(api_key=api_key)

    return provider
