from enum import Enum
from typing import Any

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from gemini_calo.logger import logger


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
        gemini_api_keys: list[str] = [],
    ):
        self._gemini_api_keys = gemini_api_keys
        self._current_gemini_api_key_index = 0
        self._base_url = base_url
        self.openai_router = APIRouter()
        self.gemini_router = APIRouter()
        self._add_routes()

    def _add_routes(self):
        self.openai_router.add_api_route(
            "/v1beta/openai/chat/completions",
            self.forward_openai_request,
            methods=["POST"],
            response_model=Any,
            # response_class=Response,
        )
        self.openai_router.add_api_route(
            "/v1beta/openai/embeddings",
            self.forward_openai_request,
            methods=["POST"],
            response_model=Any,
            # response_class=Response,
        )
        self.gemini_router.add_api_route(
            "/v1beta/models/{model_name:path}:generateContent",
            self.forward_gemini_request,
            methods=["POST"],
            response_model=Any,
            # response_class=Response,
        )
        self.gemini_router.add_api_route(
            "/v1beta/models/{model_name:path}:streamGenerateContent",
            self.forward_gemini_request,
            methods=["POST"],
            response_model=Any,
            # response_class=Response,
        )
        self.gemini_router.add_api_route(
            "/v1beta/models/{model_name:path}:embedContent",
            self.forward_gemini_request,
            methods=["POST"],
            response_model=Any,
            # response_class=Response,
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

    def get_gemini_api_key(self):
        key = self._gemini_api_keys[self._current_gemini_api_key_index]
        self._current_gemini_api_key_index += 1
        self._current_gemini_api_key_index %= len(self._gemini_api_keys)
        return key

    def get_httpx_client(self):
        return httpx.AsyncClient(base_url=self._base_url)

    async def forward_openai_request(self, request: Request) -> Response:
        """Forward openai request"""
        api_key = self.get_gemini_api_key()
        client = self.get_httpx_client()
        url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        logger.info(f"Forwarding OpenAI request to: {url}")
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.state.modified_body if hasattr(request.state, 'modified_body') else request.stream(),
            timeout=300.0,
        )
        response = await client.send(req, stream=True)
        if hasattr(response, "aiter_raw"):
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        response_body = await response.aread()
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
        )

    async def forward_gemini_request(self, request: Request) -> Response:
        """Forward gemini request"""
        api_key = self.get_gemini_api_key()
        client = self.get_httpx_client()
        url = httpx.URL(path=request.url.path, query=request.url.query.encode("utf-8"))
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        logger.info(f"Forwarding request to: {url}")
        req = client.build_request(
            request.method,
            url,
            headers=headers,
            content=request.state.modified_body if hasattr(request.state, 'modified_body') else request.stream(),
            timeout=300.0,
        )
        response = await client.send(req, stream=True)
        if hasattr(response, "aiter_raw"):
            return StreamingResponse(
                response.aiter_raw(),
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        response_body = await response.aread()
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
        )
