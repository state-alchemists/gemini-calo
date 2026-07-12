"""Adapter protocols for the IR-based translation layer.

Translation is split into two independent halves around the canonical
:mod:`gemini_calo.translate.ir`:

* An :class:`InboundAdapter` speaks the *client's* protocol. It parses an
  incoming request into a :class:`~gemini_calo.translate.ir.ChatRequest` and
  renders a :class:`~gemini_calo.translate.ir.ChatResponse` (or a stream of
  :class:`~gemini_calo.translate.ir.StreamEvent`) back into the client's wire
  format.
* An :class:`OutboundAdapter` speaks the *upstream's* protocol. It renders a
  ``ChatRequest`` into the upstream's wire format and parses the upstream
  response back into IR.

The proxy picks an inbound adapter from the request path and an outbound
adapter from ``RouteConfig.protocol``; neither half knows about the other.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from gemini_calo.translate.ir import ChatRequest, ChatResponse, StreamEvent


@runtime_checkable
class InboundAdapter(Protocol):
    """Translates the client-facing protocol to/from the canonical IR."""

    def parse_request(self, body: dict[str, Any]) -> ChatRequest:
        """Parse a client request body into a canonical ``ChatRequest``."""
        ...

    def render_response(self, resp: ChatResponse) -> bytes:
        """Render a canonical ``ChatResponse`` into the client's wire format."""
        ...

    async def render_stream(
        self, events: AsyncIterator[StreamEvent], model: str = ""
    ) -> AsyncIterator[bytes]:
        """Render canonical stream events into the client's SSE wire format."""
        ...
        if False:  # pragma: no cover - typing aid for async generator
            yield b""


@runtime_checkable
class OutboundAdapter(Protocol):
    """Translates the canonical IR to/from an upstream provider's protocol."""

    def render_request(self, req: ChatRequest) -> tuple[dict[str, Any], str]:
        """Render a ``ChatRequest`` into ``(upstream_body, upstream_path)``."""
        ...

    def parse_response(self, body: bytes, content_type: str) -> ChatResponse:
        """Parse a non-streaming upstream response body into IR."""
        ...

    async def parse_stream(
        self, chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[StreamEvent]:
        """Parse raw upstream streaming chunks into canonical stream events."""
        ...
        if False:  # pragma: no cover - typing aid for async generator
            yield StreamEvent(type="finish")
