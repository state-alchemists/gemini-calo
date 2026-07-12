"""Canonical intermediate representation (IR) for cross-protocol translation.

Every inbound client protocol (OpenAI Chat, OpenAI Responses, Anthropic
Messages, ...) is parsed *into* these dataclasses, and every outbound upstream
protocol (OpenAI Chat, Gemini, Bedrock Invoke/Converse, ...) is rendered *from*
them. Adding a protocol on either side is a single adapter that speaks IR — the
translation matrix stays N+M instead of N*M.

The IR is deliberately provider-neutral: text content is normalised to a list
of :class:`ContentPart`, tool calls are parsed into :class:`ToolCall` with
already-decoded arguments, and finish reasons use a small canonical vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical finish reasons. Adapters map their provider-specific values onto
# these on the way in, and off these on the way out.
FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL_CALLS = "tool_calls"
FINISH_CONTENT_FILTER = "content_filter"


@dataclass
class ContentPart:
    """A single piece of message content.

    ``type`` is ``"text"`` or ``"image"``. Image parts carry a data URL or
    remote URL in ``image_url`` (best-effort; not all providers support it).
    """

    type: str = "text"
    text: str = ""
    image_url: str = ""


@dataclass
class ToolCall:
    """An assistant's request to call a tool, with decoded arguments."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The result of a tool call, supplied back by the client."""

    tool_call_id: str
    content: str = ""


@dataclass
class ToolDef:
    """A tool the model is allowed to call (JSON-Schema parameters)."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A single conversation turn in canonical form.

    ``role`` is one of ``"system"``, ``"user"``, ``"assistant"``, ``"tool"``.
    Assistant turns may carry ``tool_calls``; ``tool`` turns carry
    ``tool_results``.
    """

    role: str
    content: list[ContentPart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    @property
    def text(self) -> str:
        """All text parts concatenated — the common case."""
        return "".join(p.text for p in self.content if p.type == "text")

    @classmethod
    def of_text(cls, role: str, text: str) -> "Message":
        return cls(role=role, content=[ContentPart(type="text", text=text)])


@dataclass
class ChatRequest:
    """Canonical request IR — the output of every inbound adapter."""

    model: str = ""
    messages: list[Message] = field(default_factory=list)
    system: str = ""
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    tools: list[ToolDef] = field(default_factory=list)
    tool_choice: Any | None = None
    # Passthrough for provider-specific params we don't model explicitly.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Canonical response IR — the output of every outbound adapter's parse."""

    model: str = ""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = FINISH_STOP
    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class StreamEvent:
    """A single normalised streaming event.

    ``type`` is:
      - ``"text"``       — a text delta in ``text``
      - ``"tool_call"``  — a (possibly partial) tool call in ``tool_call``
      - ``"finish"``     — end of turn, ``finish_reason`` set
      - ``"usage"``      — token counts (``prompt_tokens``/``completion_tokens``)
    """

    type: str
    text: str = ""
    finish_reason: str | None = None
    tool_call: ToolCall | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
