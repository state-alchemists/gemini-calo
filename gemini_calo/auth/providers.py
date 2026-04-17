"""Core auth provider types and protocols."""

from typing import Literal, Callable, Awaitable, Union
from fastapi import Request
import httpx


# Functional form: a callable that receives the incoming request
# and returns an httpx.Auth for the outgoing request
AuthProviderFunc = Callable[[Request], Awaitable[httpx.Auth]]


# AuthConfig can be:
# - A preset string ("bearer", "x-goog-api-key", "none")
# - A custom callable (AuthProviderFunc)
# - None (no auth)
AuthConfig = Union[
    Literal["bearer", "x-goog-api-key", "none"],
    AuthProviderFunc,
    None,
]


# Preset auth types for backward compatibility
AuthPreset = Literal["bearer", "x-goog-api-key"]