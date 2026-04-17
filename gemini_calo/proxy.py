import fnmatch
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union, Callable, Awaitable

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from gemini_calo.logger import logger
from gemini_calo.util.request import create_http_client, strip_compression_headers
from gemini_calo.auth import (
    AuthConfig,
    AuthProviderFunc,
    BearerAuth,
    XGoogApiKeyAuth,
    NoAuth,
    create_bearer_provider,
    create_xgoog_provider,
)


@dataclass
class RouteConfig:
    """Configuration for routing requests to a specific upstream provider.
    
    Attributes:
        url: The base URL for the upstream provider.
        api_keys: List of API keys to rotate through (used by preset auth types).
        auth: Authentication configuration. Can be:
            - "bearer": Use Bearer token auth with api_keys (default)
            - "x-goog-api-key": Use Google API key auth with api_keys
            - "none": No authentication
            - A callable (AuthProviderFunc) for custom auth logic
            - None: No authentication
        timeout: Request timeout in seconds.
        auth_type: DEPRECATED - Use auth instead. Kept for backward compatibility.
    """
    url: str
    api_keys: list[str] = field(default_factory=list)
    auth: AuthConfig = "bearer"
    timeout: float = 300.0
    
    # Deprecated field for backward compatibility
    auth_type: Literal["bearer", "x-goog-api-key"] | None = field(default=None, repr=False)
    
    # Internal state
    _current_index: int = field(default=0, init=False, repr=False)
    _auth_provider: AuthProviderFunc | None = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        # Handle deprecated auth_type
        if self.auth_type is not None:
            warnings.warn(
                "auth_type is deprecated, use auth instead. "
                "auth_type will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Only override auth if auth is still default and auth_type was explicitly set
            if self.auth == "bearer":
                self.auth = self.auth_type
        
        # Convert preset strings to provider callables
        self._auth_provider = self._create_auth_provider()
    
    def _create_auth_provider(self) -> AuthProviderFunc:
        """Create the internal auth provider based on auth config."""
        if self.auth is None or self.auth == "none":
            return self._no_auth_provider
        elif self.auth == "bearer":
            if not self.api_keys:
                warnings.warn(
                    "auth='bearer' but no api_keys provided. Requests will have no auth.",
                    UserWarning,
                )
                return self._no_auth_provider
            return create_bearer_provider(self.api_keys)
        elif self.auth == "x-goog-api-key":
            if not self.api_keys:
                warnings.warn(
                    "auth='x-goog-api-key' but no api_keys provided. Requests will have no auth.",
                    UserWarning,
                )
                return self._no_auth_provider
            return create_xgoog_provider(self.api_keys)
        elif callable(self.auth):
            return self.auth
        else:
            raise ValueError(f"Invalid auth config: {self.auth}")
    
    @staticmethod
    async def _no_auth_provider(request: Request) -> httpx.Auth:
        return NoAuth()
    
    def get_api_key(self) -> str:
        """Get the next API key in round-robin order.
        
        Kept for backward compatibility. For new code, use get_auth() instead.
        """
        if not self.api_keys:
            raise ValueError("No api_keys configured")
        key = self.api_keys[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.api_keys)
        return key
    
    async def get_auth(self, request: Request) -> httpx.Auth:
        """Get the httpx.Auth for authenticating the outgoing request.
        
        Args:
            request: The incoming FastAPI request (used for pass-through auth).
            
        Returns:
            An httpx.Auth instance to sign the outgoing request.
        """
        return await self._auth_provider(request)


class REQUEST_TYPE(Enum):
    OPENAI_COMPLETION: str = "openai-completion"
    OPENAI_EMBEDDING: str = "openai-embedding"
    GEMINI_COMPLETION: str = "gemini-completion"
    GEMINI_STREAMING_COMPLETION: str = "gemini-streaming-completion"
    GEMINI_EMBEDDING: str = "gemini-embedding"
    OTHER: str = "other"


class GeminiProxyService:
    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com",
        api_keys: list[str] = [],
        model_routes: dict[str, RouteConfig] = {},
    ):
        self._api_keys = api_keys
        self._current_api_key_index = 0
        self._base_url = base_url
        self._model_routes = model_routes
        self.openai_router = APIRouter()
        self.gemini_router = APIRouter()
        self._add_routes()

    def _add_routes(self):
        self.openai_router.add_api_route(
            "/v1/chat/completions",
            self.forward_openai_request,
            methods=["POST"],
            response_model=Any,
        )
        self.openai_router.add_api_route(
            "/v1/embeddings",
            self.forward_openai_request,
            methods=["POST"],
            response_model=Any,
        )
        self.openai_router.add_api_route(
            "/v1beta/openai/chat/completions",
            self.forward_openai_request,
            methods=["POST"],
            response_model=Any,
        )
        self.openai_router.add_api_route(
            "/v1beta/openai/embeddings",
            self.forward_openai_request,
            methods=["POST"],
            response_model=Any,
        )
        self.gemini_router.add_api_route(
            "/v1beta/models/{model_name:path}:generateContent",
            self.forward_gemini_request,
            methods=["POST"],
            response_model=Any,
        )
        self.gemini_router.add_api_route(
            "/v1beta/models/{model_name:path}:streamGenerateContent",
            self.forward_gemini_request,
            methods=["POST"],
            response_model=Any,
        )
        self.gemini_router.add_api_route(
            "/v1beta/models/{model_name:path}:embedContent",
            self.forward_gemini_request,
            methods=["POST"],
            response_model=Any,
        )
        self.gemini_router.add_api_route(
            "/v1beta/models",
            self.forward_gemini_request,
            methods=["GET"],
            response_model=Any,
        )

    @classmethod
    def get_request_type(cls, request: Request) -> str:
        if request.url.path in ("/v1/chat/completions", "/v1beta/openai/chat/completions"):
            return REQUEST_TYPE.OPENAI_COMPLETION
        if request.url.path in ("/v1/embeddings", "/v1beta/openai/embeddings"):
            return REQUEST_TYPE.OPENAI_EMBEDDING
        if request.url.path.startswith("/v1beta/models/"):
            if request.url.path.endswith(":generateContent"):
                return REQUEST_TYPE.GEMINI_COMPLETION
            if request.url.path.endswith(":streamGenerateContent"):
                return REQUEST_TYPE.GEMINI_STREAMING_COMPLETION
            if request.url.path.endswith(":embedContent"):
                return REQUEST_TYPE.GEMINI_EMBEDDING
        return REQUEST_TYPE.OTHER

    def get_api_key(self) -> str:
        key = self._api_keys[self._current_api_key_index]
        self._current_api_key_index += 1
        self._current_api_key_index %= len(self._api_keys)
        return key

    def get_gemini_api_key(self) -> str:
        """Backward-compatible alias for get_api_key."""
        return self.get_api_key()

    def get_httpx_client(self) -> httpx.AsyncClient:
        return create_http_client(
            base_url=self._base_url,
            accept_compression=True,
            follow_redirects=False,
            timeout=300.0,
        )

    async def _extract_model_name(self, request: Request) -> str | None:
        request_type = self.get_request_type(request)
        if request_type in (
            REQUEST_TYPE.GEMINI_COMPLETION,
            REQUEST_TYPE.GEMINI_STREAMING_COMPLETION,
            REQUEST_TYPE.GEMINI_EMBEDDING,
        ):
            path = request.url.path
            try:
                # /v1beta/models/{model_name}:action  →  strip prefix and suffix action
                after_models = path.split("/v1beta/models/", 1)[1]
                return after_models.rsplit(":", 1)[0]
            except (IndexError, ValueError):
                return None
        elif request_type in (REQUEST_TYPE.OPENAI_COMPLETION, REQUEST_TYPE.OPENAI_EMBEDDING):
            try:
                body = await request.body()
                return json.loads(body).get("model")
            except (json.JSONDecodeError, Exception):
                return None
        return None

    def _find_route(self, model_name: str | None) -> RouteConfig | None:
        if not model_name:
            return None
        for pattern, route in self._model_routes.items():
            if fnmatch.fnmatch(model_name, pattern):
                return route
        return None

    async def _resolve_upstream(
        self,
        request: Request,
        default_auth: Literal["bearer", "x-goog-api-key"] = "bearer",
    ) -> tuple[httpx.AsyncClient, httpx.Auth | None, float]:
        """
        Returns (client, auth, timeout) for the upstream request.
        
        Checks model_routes first (glob-matched), falls back to base_url + api_keys.
        Auth is now an httpx.Auth instance that handles request signing.
        """
        model_name = await self._extract_model_name(request)
        route = self._find_route(model_name)
        
        if route:
            logger.info(f"Routing model '{model_name}' to {route.url}")
            client = create_http_client(
                base_url=route.url,
                accept_compression=True,
                follow_redirects=False,
                timeout=route.timeout,
            )
            auth = await route.get_auth(request)
            return client, auth, route.timeout

        # Default route - use legacy auth for backward compatibility
        client = create_http_client(
            base_url=self._base_url,
            accept_compression=True,
            follow_redirects=False,
            timeout=300.0,
        )
        
        # Create auth based on default_auth preset
        if self._api_keys:
            if default_auth == "bearer":
                auth: httpx.Auth = BearerAuth(token=self.get_api_key())
            else:
                auth = XGoogApiKeyAuth(api_key=self.get_api_key())
        else:
            auth = None
        
        return client, auth, 300.0

    async def forward_openai_request(self, request: Request) -> Response:
        """Forward openai request"""
        client, auth, timeout = await self._resolve_upstream(request, "bearer")
        path = request.url.path
        
        # Map standard OpenAI paths to Gemini-OpenAI paths only if target is Google
        is_google = str(client.base_url).startswith("https://generativelanguage.googleapis.com")
        if is_google:
            if path == "/v1/chat/completions":
                path = "/v1beta/openai/chat/completions"
            elif path == "/v1/embeddings":
                path = "/v1beta/openai/embeddings"

        url = httpx.URL(path=path, query=request.url.query.encode("utf-8"))
        
        # Start with content-type header; auth will be added by httpx.Auth
        headers = {"Content-Type": "application/json"}
        
        # Add any other headers from the original request that should be forwarded
        # (excluding auth headers which are handled by the auth provider)
        
        logger.info(f"Forwarding OpenAI request to: {url}")
        
        # Get request body
        content = (
            request.state.modified_body 
            if hasattr(request.state, "modified_body") 
            else request.stream()
        )
        
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=content,
            timeout=timeout,
        )
        
        # Determine if streaming
        request_type = self.get_request_type(request)
        is_streaming = request.url.path.endswith(":streamGenerateContent") or (
            request_type == REQUEST_TYPE.OPENAI_COMPLETION
            and hasattr(request.state, "stream")
            and request.state.stream
        )
        
        # Send with auth
        if auth:
            response = await client.send(req, auth=auth, stream=is_streaming)
        else:
            response = await client.send(req, stream=is_streaming)
        
        if is_streaming:
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=strip_compression_headers(dict(response.headers)),
            )
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=strip_compression_headers(dict(response.headers)),
        )

    async def forward_gemini_request(self, request: Request) -> Response:
        """Forward gemini request"""
        client, auth, timeout = await self._resolve_upstream(request, "x-goog-api-key")
        url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
        
        # Start with content-type header; auth will be added by httpx.Auth
        headers = {"Content-Type": "application/json"}
        
        logger.info(f"Forwarding Gemini request to: {url}")
        
        # Get request body
        content = (
            request.state.modified_body 
            if hasattr(request.state, "modified_body") 
            else request.stream()
        )
        
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=content,
            timeout=timeout,
        )
        
        # Determine if streaming
        request_type = self.get_request_type(request)
        is_streaming = request.url.path.endswith(":streamGenerateContent") or (
            request_type == REQUEST_TYPE.OPENAI_COMPLETION
            and hasattr(request.state, "stream")
            and request.state.stream
        )
        
        # Send with auth
        if auth:
            response = await client.send(req, auth=auth, stream=is_streaming)
        else:
            response = await client.send(req, stream=is_streaming)
        
        if is_streaming:
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=strip_compression_headers(dict(response.headers)),
            )
        
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=strip_compression_headers(dict(response.headers)),
        )