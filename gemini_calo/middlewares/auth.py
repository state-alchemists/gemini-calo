import inspect
from functools import partial
from typing import Any, Callable, Coroutine

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from gemini_calo.proxy import REQUEST_TYPE, GeminiProxyService


def create_auth_middleware(
    user_api_key_checker: (
        str | list[str] | Callable[[str], bool | Coroutine[Any, Any, bool]] | None
    ) = None,
) -> Callable[
    [Request, Callable[[Request], Coroutine[Any, Any, Response]]],
    Coroutine[Any, Any, Response],
]:
    return partial(auth_middleware, user_api_key_checker=user_api_key_checker)


async def auth_middleware(
    request: Request,
    call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    user_api_key_checker: (
        str | list[str] | Callable[[str], bool | Coroutine[Any, Any, bool]] | None
    ) = None,
) -> Response:
    # If no checker is set, skip this
    if user_api_key_checker is None or (isinstance(user_api_key_checker, list) and len(user_api_key_checker) == 0):
        return await call_next(request)
    # Only apply check for specifics URLs
    request_type = GeminiProxyService.get_request_type(request)
    if request_type == REQUEST_TYPE.OTHER:
        return await call_next(request)
    # Get User API Key from request header
    api_key = _get_request_api_key(request)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={
                "error": (
                    "API key is missing. Provide the API key in "
                    "'Authorization: Bearer <key>' or 'x-goog-api-key' header."
                )
            },
        )
    # Check whether User API Key is valid
    is_authorized = await _check_is_authorized(user_api_key_checker, api_key)
    if not is_authorized:
        return JSONResponse(
            status_code=401,
            content={"error": "The provided API Key is not valid."},
        )
    return await call_next(request)


def _get_request_api_key(request: Request) -> str | None:
    api_key = None
    auth_header = request.headers.get("Authorization")
    if auth_header:
        try:
            scheme, _, api_key = auth_header.partition(" ")
            if scheme.lower() == "bearer":
                return api_key
            return None
        except ValueError:
            return None
    return request.headers.get("x-goog-api-key")


async def _check_is_authorized(
    user_api_key_checker: (
        str | list[str] | Callable[[str], bool | Coroutine[Any, Any, bool]] | None
    ),
    api_key: str,
) -> bool:
    if inspect.iscoroutinefunction(user_api_key_checker):
        return await user_api_key_checker(api_key)
    if callable(user_api_key_checker):
        return user_api_key_checker(api_key)
    if isinstance(user_api_key_checker, str):
        return api_key == user_api_key_checker
    return api_key in user_api_key_checker
