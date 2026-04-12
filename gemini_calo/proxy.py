import fnmatch
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from gemini_calo.logger import logger
from gemini_calo.util.request import create_http_client, strip_compression_headers


@dataclass
class RouteConfig:
    url: str
    api_keys: list[str]
    auth_type: Literal["bearer", "x-goog-api-key"] = "bearer"
    timeout: float = 300.0
    _current_index: int = field(default=0, init=False, repr=False)

    def get_api_key(self) -> str:
        key = self.api_keys[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.api_keys)
        return key


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
        if request.url.path == "/v1beta/openai/chat/completions":
            return REQUEST_TYPE.OPENAI_COMPLETION
        if request.url.path == "/v1beta/openai/embeddings":
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
        default_auth_type: Literal["bearer", "x-goog-api-key"],
    ) -> tuple[httpx.AsyncClient, dict[str, str], float]:
        """
        Returns (client, auth_headers, timeout) for the upstream request.
        Checks model_routes first (glob-matched), falls back to base_url + api_keys.
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
            api_key = route.get_api_key()
            auth_headers = (
                {"Authorization": f"Bearer {api_key}"}
                if route.auth_type == "bearer"
                else {"x-goog-api-key": api_key}
            )
            return client, auth_headers, route.timeout

        client = create_http_client(
            base_url=self._base_url,
            accept_compression=True,
            follow_redirects=False,
            timeout=300.0,
        )
        api_key = self.get_api_key()
        auth_headers = (
            {"Authorization": f"Bearer {api_key}"}
            if default_auth_type == "bearer"
            else {"x-goog-api-key": api_key}
        )
        return client, auth_headers, 300.0

    async def forward_openai_request(self, request: Request) -> Response:
        """Forward openai request"""
        client, auth_headers, timeout = await self._resolve_upstream(request, "bearer")
        url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
        headers = {"Content-Type": "application/json", **auth_headers}
        logger.info(f"Forwarding OpenAI request to: {url}")
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.state.modified_body if hasattr(request.state, "modified_body") else request.stream(),
            timeout=timeout,
        )
        # Check if this is a streaming endpoint
        request_type = self.get_request_type(request)
        is_streaming = request.url.path.endswith(":streamGenerateContent") or (
            request_type == REQUEST_TYPE.OPENAI_COMPLETION
            and hasattr(request.state, "stream")
            and request.state.stream
        )
        if is_streaming:
            response = await client.send(req, stream=True)
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=strip_compression_headers(dict(response.headers)),
            )
        response = await client.send(req, stream=False)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=strip_compression_headers(dict(response.headers)),
        )

    async def forward_gemini_request(self, request: Request) -> Response:
        """Forward gemini request"""
        client, auth_headers, timeout = await self._resolve_upstream(request, "x-goog-api-key")
        url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
        headers = {"Content-Type": "application/json", **auth_headers}
        logger.info(f"Forwarding Gemini request to: {url}")
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.state.modified_body if hasattr(request.state, "modified_body") else request.stream(),
            timeout=timeout,
        )
        # Check if this is a streaming endpoint
        request_type = self.get_request_type(request)
        is_streaming = request.url.path.endswith(":streamGenerateContent") or (
            request_type == REQUEST_TYPE.OPENAI_COMPLETION
            and hasattr(request.state, "stream")
            and request.state.stream
        )
        if is_streaming:
            response = await client.send(req, stream=True)
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=strip_compression_headers(dict(response.headers)),
            )
        response = await client.send(req, stream=False)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=strip_compression_headers(dict(response.headers)),
        )
