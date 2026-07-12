import fnmatch
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Literal, Union

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from gemini_calo.auth import (
    AuthConfig,
    AuthProviderFunc,
    BearerAuth,
    NoAuth,
    XGoogApiKeyAuth,
    create_bearer_provider,
    create_passthrough_bedrock_provider,
    create_xgoog_provider,
)
from gemini_calo.logger import logger
from gemini_calo.util.request import create_http_client, strip_compression_headers

_default_bedrock_passthrough = create_passthrough_bedrock_provider()

# Optional Bedrock-specific request headers that should be forwarded upstream when present.
# Content-Type and Authorization are handled separately and are not in this list.
_BEDROCK_PASSTHROUGH_HEADERS = (
    "Accept",
    "X-Amzn-Bedrock-Trace",
    "X-Amzn-Bedrock-GuardrailIdentifier",
    "X-Amzn-Bedrock-GuardrailVersion",
    "X-Amzn-Bedrock-PerformanceConfig-Latency",
    "X-Amzn-Bedrock-Service-Tier",
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
        protocol: The upstream's native protocol format. Determines whether
            request/response translation is applied.  Valid values:
            - "openai" (default): OpenAI-compatible, no translation (pure
              passthrough — use for real OpenAI, incl. its native /responses).
            - "openai-chat": upstream speaks only /v1/chat/completions
              (DeepSeek, Together, ...). Any client protocol is translated
              through the canonical IR into chat completions.
            - "gemini": Google Gemini native API.
            - "bedrock-invoke": Bedrock InvokeModel API.
            - "bedrock-converse": Bedrock Converse API.
        upstream_model: Optional model id to send upstream instead of the one
            the client requested. Lets you expose a friendly alias (e.g. route
            key "nova" -> upstream_model "amazon.nova-pro-v1:0"), which also
            works around clients that mangle model ids containing ":" (zrb).
            Only applied on the translated path (non-"openai" protocol).
        outbound: An OutboundAdapter for IR->upstream conversion.
            Auto-resolved from *protocol* if not provided. ``None`` means
            passthrough (protocol == "openai").
    """

    url: str
    api_keys: list[str] = field(default_factory=list)
    auth: AuthConfig = "bearer"
    timeout: float = 300.0
    protocol: str = "openai"
    upstream_model: str = ""  # if set, overrides the client's model id upstream
    outbound: Any | None = None  # OutboundAdapter | None – Any avoids circular import

    # Deprecated field for backward compatibility
    auth_type: Literal["bearer", "x-goog-api-key"] | None = field(
        default=None, repr=False
    )

    # Internal state
    _current_index: int = field(default=0, init=False, repr=False)
    _auth_provider: AuthProviderFunc | None = field(
        default=None, init=False, repr=False
    )

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

        # Auto-resolve the outbound adapter from protocol if not explicitly
        # provided. protocol == "openai" means pure passthrough (no adapter).
        if self.outbound is None and self.protocol != "openai":
            from gemini_calo.translate import get_outbound_adapter

            self.outbound = get_outbound_adapter(self.protocol)

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
    OPENAI_RESPONSES: str = "openai-responses"
    OPENAI_EMBEDDING: str = "openai-embedding"
    GEMINI_COMPLETION: str = "gemini-completion"
    GEMINI_STREAMING_COMPLETION: str = "gemini-streaming-completion"
    GEMINI_EMBEDDING: str = "gemini-embedding"
    BEDROCK_INVOKE: str = "bedrock-invoke"
    BEDROCK_STREAMING_INVOKE: str = "bedrock-streaming-invoke"
    BEDROCK_CONVERSE: str = "bedrock-converse"
    BEDROCK_STREAMING_CONVERSE: str = "bedrock-streaming-converse"
    ANTHROPIC_MESSAGES: str = "anthropic-messages"
    OTHER: str = "other"


# Maps a client request type to the inbound-adapter protocol name that parses it.
_INBOUND_PROTOCOL: dict["REQUEST_TYPE", str] = {
    REQUEST_TYPE.OPENAI_COMPLETION: "openai-chat",
    REQUEST_TYPE.OPENAI_RESPONSES: "openai-responses",
    REQUEST_TYPE.ANTHROPIC_MESSAGES: "anthropic-messages",
}


class GeminiProxyService:
    def __init__(
        self,
        base_url: str = "https://generativelanguage.googleapis.com",
        api_keys: list[str] | None = None,
        model_routes: dict[str, RouteConfig] | None = None,
    ):
        self._api_keys = api_keys if api_keys is not None else []
        self._current_api_key_index = 0
        self._base_url = base_url
        self._model_routes = model_routes if model_routes is not None else {}
        self.openai_router = APIRouter()
        self.gemini_router = APIRouter()
        self.bedrock_router = APIRouter()
        self.anthropic_router = APIRouter()
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
            "/v1/responses",
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
        self.bedrock_router.add_api_route(
            "/model/{model_id:path}/invoke",
            self.forward_bedrock_request,
            methods=["POST"],
            response_model=Any,
        )
        self.bedrock_router.add_api_route(
            "/model/{model_id:path}/invoke-with-response-stream",
            self.forward_bedrock_request,
            methods=["POST"],
            response_model=Any,
        )
        self.bedrock_router.add_api_route(
            "/model/{model_id:path}/converse",
            self.forward_bedrock_request,
            methods=["POST"],
            response_model=Any,
        )
        self.bedrock_router.add_api_route(
            "/model/{model_id:path}/converse-stream",
            self.forward_bedrock_request,
            methods=["POST"],
            response_model=Any,
        )
        self.anthropic_router.add_api_route(
            "/v1/messages",
            self.forward_anthropic_request,
            methods=["POST"],
            response_model=Any,
        )

    @classmethod
    def get_request_type(cls, request: Request) -> str:
        if request.url.path == "/v1/messages":
            return REQUEST_TYPE.ANTHROPIC_MESSAGES
        if request.url.path in (
            "/v1/chat/completions",
            "/v1beta/openai/chat/completions",
        ):
            return REQUEST_TYPE.OPENAI_COMPLETION
        if request.url.path == "/v1/responses":
            return REQUEST_TYPE.OPENAI_RESPONSES
        if request.url.path in ("/v1/embeddings", "/v1beta/openai/embeddings"):
            return REQUEST_TYPE.OPENAI_EMBEDDING
        if request.url.path.startswith("/v1beta/models/"):
            if request.url.path.endswith(":generateContent"):
                return REQUEST_TYPE.GEMINI_COMPLETION
            if request.url.path.endswith(":streamGenerateContent"):
                return REQUEST_TYPE.GEMINI_STREAMING_COMPLETION
            if request.url.path.endswith(":embedContent"):
                return REQUEST_TYPE.GEMINI_EMBEDDING
        if request.url.path.startswith("/model/"):
            if request.url.path.endswith("/invoke-with-response-stream"):
                return REQUEST_TYPE.BEDROCK_STREAMING_INVOKE
            if request.url.path.endswith("/invoke"):
                return REQUEST_TYPE.BEDROCK_INVOKE
            if request.url.path.endswith("/converse-stream"):
                return REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE
            if request.url.path.endswith("/converse"):
                return REQUEST_TYPE.BEDROCK_CONVERSE
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
        elif request_type in (
            REQUEST_TYPE.OPENAI_COMPLETION,
            REQUEST_TYPE.OPENAI_RESPONSES,
            REQUEST_TYPE.OPENAI_EMBEDDING,
            REQUEST_TYPE.ANTHROPIC_MESSAGES,
        ):
            try:
                body = await request.body()
                return json.loads(body).get("model")
            except (json.JSONDecodeError, Exception):
                return None
        elif request_type in (
            REQUEST_TYPE.BEDROCK_INVOKE,
            REQUEST_TYPE.BEDROCK_STREAMING_INVOKE,
            REQUEST_TYPE.BEDROCK_CONVERSE,
            REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE,
        ):
            path = request.url.path
            try:
                # /model/{model_id}/invoke|converse[|...]
                after_model = path.split("/model/", 1)[1]
                return after_model.rsplit("/", 1)[0]
            except (IndexError, ValueError):
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
    ) -> tuple[httpx.AsyncClient, httpx.Auth | None, float, RouteConfig | None]:
        """
        Returns (client, auth, timeout, route) for the upstream request.

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
            return client, auth, route.timeout, route

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

        return client, auth, 300.0, None

    async def _send_upstream(
        self,
        request: Request,
        client: httpx.AsyncClient,
        auth: httpx.Auth | None,
        headers: dict[str, str],
        timeout: float,
        is_streaming: bool,
        path: str | None = None,
        body_override: bytes | None = None,
    ) -> Response:
        """Build, send, and wrap the upstream request. Shared by all forwarders."""
        # A translated path may carry its own query string (e.g. Gemini's
        # ?alt=sse); split it out and prefer it over the client's query.
        target_path = path or request.url.path
        target_path, _, target_query = target_path.partition("?")
        query = (
            target_query.encode("utf-8")
            if target_query
            else request.url.query.encode("utf-8")
        )
        url = httpx.URL(path=target_path, query=query)
        if body_override is not None:
            content = body_override
        elif hasattr(request.state, "modified_body"):
            content = request.state.modified_body
        else:
            content = request.stream()
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=content,
            timeout=timeout,
        )
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

    @staticmethod
    async def _body_requests_streaming(request: Request) -> bool:
        """True if the JSON request body sets "stream": true (OpenAI-style)."""
        raw = (
            request.state.modified_body
            if hasattr(request.state, "modified_body")
            else await request.body()
        )
        try:
            return json.loads(raw).get("stream") is True
        except (json.JSONDecodeError, ValueError, AttributeError):
            return False

    async def forward_openai_request(self, request: Request) -> Response:
        """Forward an OpenAI Chat Completions / Responses / Embeddings request.

        Embeddings are always passed through. Chat/Responses are translated
        through the canonical IR when the matched route declares a non-OpenAI
        upstream protocol; otherwise they pass through (with Google path
        mapping) exactly as before.
        """
        request_type = self.get_request_type(request)
        client, auth, timeout, route = await self._resolve_upstream(request, "bearer")

        inbound_protocol = _INBOUND_PROTOCOL.get(request_type)
        if route and route.outbound is not None and inbound_protocol is not None:
            return await self._forward_translated(
                request, client, auth, timeout, route, inbound_protocol
            )
        return await self._forward_openai_passthrough(request, client, auth, timeout)

    async def forward_anthropic_request(self, request: Request) -> Response:
        """Forward an Anthropic Messages (/v1/messages) request.

        Requires a matched route with a translating protocol, because no
        upstream here speaks Anthropic Messages natively.
        """
        client, auth, timeout, route = await self._resolve_upstream(request, "bearer")
        if route and route.outbound is not None:
            return await self._forward_translated(
                request, client, auth, timeout, route, "anthropic-messages"
            )
        # No translating route matched — pass through so a misconfiguration
        # surfaces as the upstream's own error rather than a silent hang.
        return await self._forward_openai_passthrough(request, client, auth, timeout)

    async def _forward_openai_passthrough(
        self,
        request: Request,
        client: httpx.AsyncClient,
        auth: httpx.Auth | None,
        timeout: float,
    ) -> Response:
        """Legacy passthrough: forward the OpenAI body verbatim (no IR)."""
        path = request.url.path
        # Map standard OpenAI paths to Gemini-OpenAI paths only if target is Google.
        if str(client.base_url).startswith(
            "https://generativelanguage.googleapis.com"
        ):
            if path == "/v1/chat/completions":
                path = "/v1beta/openai/chat/completions"
            elif path == "/v1/embeddings":
                path = "/v1beta/openai/embeddings"

        logger.info(f"Forwarding OpenAI request to: {path}")
        is_streaming = await self._body_requests_streaming(request)
        return await self._send_upstream(
            request,
            client,
            auth,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            is_streaming=is_streaming,
            path=path,
        )

    async def _forward_translated(
        self,
        request: Request,
        client: httpx.AsyncClient,
        auth: httpx.Auth | None,
        timeout: float,
        route: RouteConfig,
        inbound_protocol: str,
    ) -> Response:
        """Client protocol -> IR -> upstream protocol, then reverse the response."""
        from gemini_calo.translate import get_inbound_adapter

        raw_body = (
            request.state.modified_body
            if hasattr(request.state, "modified_body")
            else await request.body()
        )
        try:
            body = json.loads(raw_body)
        except (json.JSONDecodeError, ValueError):
            body = {}

        inbound = get_inbound_adapter(inbound_protocol)
        outbound = route.outbound

        ir_req = inbound.parse_request(body)
        if route.upstream_model:
            ir_req.model = route.upstream_model
        up_body, up_path = outbound.render_request(ir_req)
        is_streaming = ir_req.stream

        logger.info(
            f"Translating {inbound_protocol} -> {route.protocol}: {up_path} "
            f"(stream={is_streaming})"
        )

        response = await self._send_upstream(
            request,
            client,
            auth,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            is_streaming=is_streaming,
            path=up_path,
            body_override=json.dumps(up_body).encode(),
        )

        # On upstream error, return the raw error body untouched so the real
        # status/message reaches the client instead of an empty translation.
        if response.status_code >= 400:
            return response

        if is_streaming:
            events = outbound.parse_stream(response.body_iterator)
            return StreamingResponse(
                inbound.render_stream(events, ir_req.model),
                status_code=response.status_code,
                media_type="text/event-stream",
            )

        ir_resp = outbound.parse_response(
            response.body, response.headers.get("content-type", "")
        )
        if not ir_resp.model:
            ir_resp.model = ir_req.model
        return Response(
            content=inbound.render_response(ir_resp),
            status_code=response.status_code,
            media_type="application/json",
        )

    async def _resolve_bedrock_upstream(
        self, request: Request
    ) -> tuple[httpx.AsyncClient, httpx.Auth | None, float]:
        model_name = await self._extract_model_name(request)
        route = self._find_route(model_name)

        if route:
            logger.info(f"Routing Bedrock model '{model_name}' to {route.url}")
            client = create_http_client(
                base_url=route.url,
                accept_compression=True,
                follow_redirects=False,
                timeout=route.timeout,
            )
            auth = await route.get_auth(request)
            return client, auth, route.timeout

        region = request.headers.get("x-aws-region", "us-east-1")
        bedrock_url = f"https://bedrock-runtime.{region}.amazonaws.com"
        client = create_http_client(
            base_url=bedrock_url,
            accept_compression=True,
            follow_redirects=False,
            timeout=300.0,
        )
        auth = await _default_bedrock_passthrough(request)
        return client, auth, 300.0

    async def forward_gemini_request(self, request: Request) -> Response:
        """Forward gemini request"""
        client, auth, timeout, _route = await self._resolve_upstream(
            request, "x-goog-api-key"
        )

        logger.info(f"Forwarding Gemini request to: {request.url.path}")

        return await self._send_upstream(
            request,
            client,
            auth,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
            is_streaming=request.url.path.endswith(":streamGenerateContent"),
        )

    async def forward_bedrock_request(self, request: Request) -> Response:
        """Forward AWS Bedrock runtime request."""
        client, auth, timeout = await self._resolve_bedrock_upstream(request)

        # Bedrock Runtime uses application/json (REST-JSON protocol).
        # Pass through whatever the client sends so we don't second-guess it.
        headers = {
            "Content-Type": request.headers.get("Content-Type", "application/json"),
        }
        for header in _BEDROCK_PASSTHROUGH_HEADERS:
            value = request.headers.get(header)
            if value is not None:
                headers[header] = value

        logger.info(f"Forwarding Bedrock request to: {request.url.path}")

        is_streaming = self.get_request_type(request) in (
            REQUEST_TYPE.BEDROCK_STREAMING_INVOKE,
            REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE,
        )

        return await self._send_upstream(
            request,
            client,
            auth,
            headers=headers,
            timeout=timeout,
            is_streaming=is_streaming,
        )
