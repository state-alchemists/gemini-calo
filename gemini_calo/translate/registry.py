"""Registries mapping client request types and upstream protocols to adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemini_calo.translate.base import InboundAdapter, OutboundAdapter

# Lazily populated to avoid import cycles.
_inbound: dict[str, type[InboundAdapter]] = {}
_outbound: dict[str, type[OutboundAdapter]] = {}


def _ensure_registries() -> None:
    if _inbound and _outbound:
        return

    from gemini_calo.translate.inbound.anthropic_messages import (
        AnthropicMessagesInbound,
    )
    from gemini_calo.translate.inbound.openai_chat import OpenAIChatInbound
    from gemini_calo.translate.inbound.openai_responses import OpenAIResponsesInbound
    from gemini_calo.translate.outbound.bedrock_converse import BedrockConverseOutbound
    from gemini_calo.translate.outbound.bedrock_invoke import BedrockInvokeOutbound
    from gemini_calo.translate.outbound.gemini import GeminiOutbound
    from gemini_calo.translate.outbound.openai_chat import OpenAIChatOutbound

    _inbound.update(
        {
            "openai-chat": OpenAIChatInbound,
            "openai-responses": OpenAIResponsesInbound,
            "anthropic-messages": AnthropicMessagesInbound,
        }
    )
    _outbound.update(
        {
            "openai-chat": OpenAIChatOutbound,
            "gemini": GeminiOutbound,
            "bedrock-invoke": BedrockInvokeOutbound,
            "bedrock-converse": BedrockConverseOutbound,
        }
    )


def get_inbound_adapter(client_protocol: str) -> "InboundAdapter":
    """Get the inbound adapter for a client protocol name.

    Args:
        client_protocol: "openai-chat", "openai-responses", or "anthropic-messages".
    """
    _ensure_registries()
    cls = _inbound.get(client_protocol)
    if cls is None:
        raise ValueError(
            f"Unknown client protocol '{client_protocol}'. "
            f"Available: {', '.join(sorted(_inbound))}"
        )
    return cls()


def get_outbound_adapter(protocol: str) -> "OutboundAdapter":
    """Get the outbound adapter for an upstream protocol name.

    Args:
        protocol: "openai-chat", "gemini", "bedrock-invoke", or "bedrock-converse".
    """
    _ensure_registries()
    cls = _outbound.get(protocol)
    if cls is None:
        raise ValueError(
            f"Unknown upstream protocol '{protocol}'. "
            f"Available: {', '.join(sorted(_outbound))}"
        )
    return cls()


def register_inbound_adapter(client_protocol: str, cls: type) -> None:
    """Register a custom inbound adapter."""
    _ensure_registries()
    _inbound[client_protocol] = cls


def register_outbound_adapter(protocol: str, cls: type) -> None:
    """Register a custom outbound adapter."""
    _ensure_registries()
    _outbound[protocol] = cls
