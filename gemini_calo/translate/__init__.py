"""IR-based protocol translation layer.

Two independent halves around a canonical IR (:mod:`gemini_calo.translate.ir`):
inbound adapters speak the client's protocol, outbound adapters speak the
upstream's. The proxy picks an inbound adapter from the request path and an
outbound adapter from ``RouteConfig.protocol``.
"""

from gemini_calo.translate.base import InboundAdapter, OutboundAdapter
from gemini_calo.translate.registry import (
    get_inbound_adapter,
    get_outbound_adapter,
    register_inbound_adapter,
    register_outbound_adapter,
)

__all__ = [
    "InboundAdapter",
    "OutboundAdapter",
    "get_inbound_adapter",
    "get_outbound_adapter",
    "register_inbound_adapter",
    "register_outbound_adapter",
]
