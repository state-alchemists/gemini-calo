import hashlib
import json
from functools import partial
from typing import Any, Callable, Coroutine, cast

from cachetools import LRUCache
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from gemini_calo.config import (
    CONVERSATION_SUMMARIZATION_LRU_SIZE,
    DEFAULT_SUMMARIZER_PROMPT,
    SUMMARIZATION_SIZE_THRESHOLD,
)
from gemini_calo.logger import logger
from gemini_calo.proxy import REQUEST_TYPE, GeminiProxyService
from gemini_calo.util.request import create_http_client, decompress_content

_lru_cache = LRUCache(maxsize=CONVERSATION_SUMMARIZATION_LRU_SIZE)


def create_rollup_middleware(
    gemini_proxy: GeminiProxyService,
    lru_cache: LRUCache = _lru_cache,
    conversation_size_threshold: int = SUMMARIZATION_SIZE_THRESHOLD,
) -> Callable[
    [Request, Callable[[Request], Coroutine[Any, Any, Response]]],
    Coroutine[Any, Any, Response],
]:
    """
    Creates a middleware to handle conversation roll-ups.
    """
    return partial(
        rollup_middleware,
        cache=lru_cache,
        conversation_size_threshold=conversation_size_threshold,
        gemini_proxy=gemini_proxy,
    )


async def rollup_middleware(
    request: Request,
    call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    gemini_proxy: GeminiProxyService,
    cache: LRUCache,
    conversation_size_threshold: int,
) -> Response:
    request_type = GeminiProxyService.get_request_type(request)
    is_completion = request_type in [
        REQUEST_TYPE.OPENAI_COMPLETION,
        REQUEST_TYPE.OPENAI_RESPONSES,
        REQUEST_TYPE.ANTHROPIC_MESSAGES,
        REQUEST_TYPE.GEMINI_COMPLETION,
        REQUEST_TYPE.GEMINI_STREAMING_COMPLETION,
        REQUEST_TYPE.BEDROCK_INVOKE,
        REQUEST_TYPE.BEDROCK_STREAMING_INVOKE,
        REQUEST_TYPE.BEDROCK_CONVERSE,
        REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE,
    ]

    if not is_completion:
        return await call_next(request)

    body = await request.body()
    try:
        json_body = json.loads(body)
    except json.JSONDecodeError:
        json_body = {}

    messages: list[dict] = []
    if request_type == REQUEST_TYPE.OPENAI_COMPLETION:
        messages = _extract_openai_messages(json_body)
    elif request_type == REQUEST_TYPE.OPENAI_RESPONSES:
        messages = _extract_responses_messages(json_body)
    elif request_type == REQUEST_TYPE.ANTHROPIC_MESSAGES:
        messages = _extract_anthropic_messages(json_body)
    elif request_type in (
        REQUEST_TYPE.BEDROCK_INVOKE,
        REQUEST_TYPE.BEDROCK_STREAMING_INVOKE,
        REQUEST_TYPE.BEDROCK_CONVERSE,
        REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE,
    ):
        messages = _extract_bedrock_messages(json_body)
    else:
        messages = _extract_gemini_messages(json_body)

    if not messages:
        return await call_next(request)

    found_key = None
    num_matched_messages = 0
    for i in range(len(messages), 0, -1):
        sub_messages = messages[:i]
        key = _get_message_key(sub_messages)
        if key in cache:
            found_key = key
            num_matched_messages = i
            break

    if found_key:
        logger.debug(
            f"Rollup cache hit: key={found_key}, "
            f"matched_messages={num_matched_messages}"
        )
        context = cast(str, cache[found_key])
        if request_type == REQUEST_TYPE.OPENAI_COMPLETION:
            json_body = _inject_openai_system_prompt(_copy_json(json_body), context)
            original_messages = json_body.get("messages", [])
            system_messages = [
                m for m in original_messages if m.get("role") == "system"
            ]
            user_messages = [m for m in original_messages if m.get("role") != "system"]
            json_body["messages"] = (
                system_messages + user_messages[num_matched_messages:]
            )
        elif request_type == REQUEST_TYPE.OPENAI_RESPONSES:
            json_body = _inject_responses_system_prompt(_copy_json(json_body), context)
            json_body["input"] = messages[num_matched_messages:]
        elif request_type == REQUEST_TYPE.ANTHROPIC_MESSAGES:
            json_body = _inject_anthropic_system_prompt(_copy_json(json_body), context)
            json_body["messages"] = messages[num_matched_messages:]
        elif request_type in (
            REQUEST_TYPE.BEDROCK_INVOKE,
            REQUEST_TYPE.BEDROCK_STREAMING_INVOKE,
            REQUEST_TYPE.BEDROCK_CONVERSE,
            REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE,
        ):
            json_body = _inject_bedrock_system_prompt(_copy_json(json_body), context)
            original_messages = json_body.get("messages", [])
            json_body["messages"] = original_messages[num_matched_messages:]
        else:
            json_body = _inject_gemini_system_prompt(_copy_json(json_body), context)
            json_body["contents"] = messages[num_matched_messages:]

        new_body = json.dumps(json_body).encode()
        request.state.modified_body = new_body  # Store modified body in request.state

        async def receive():
            return {"type": "http.request", "body": new_body}

        request = Request(request.scope, receive)

    response = await call_next(request)

    async def update_cache(response_body: bytes) -> None:
        # Add gzip detection and decompression before JSON parsing
        content_encoding = response.headers.get("content-encoding")
        if content_encoding:
            response_body = decompress_content(response_body, content_encoding)

        # Bedrock streaming responses use binary AWS Event Stream format, not JSON
        is_binary_streaming = request_type in (
            REQUEST_TYPE.BEDROCK_STREAMING_INVOKE,
            REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE,
        )
        try:
            response_json = {} if is_binary_streaming else json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            response_json = {}

        # Handle case where response_json might be a list (from streaming responses)
        if not isinstance(response_json, dict):
            response_json = {}

        # The assistant turn can be more than one item (e.g. the OpenAI
        # Responses "output" array carries a message plus function calls), so
        # collect a list and append it verbatim — future requests replay these
        # same items, which keeps the conversation hashes aligned.
        assistant_items: list[dict] = []
        if request_type == REQUEST_TYPE.OPENAI_COMPLETION:
            choice = response_json.get("choices", [{}])[0]
            assistant_message = choice.get("message", {})
            if assistant_message:
                assistant_items = [assistant_message]
        elif request_type == REQUEST_TYPE.OPENAI_RESPONSES:
            output = response_json.get("output", [])
            if isinstance(output, list):
                assistant_items = [it for it in output if isinstance(it, dict)]
        elif request_type == REQUEST_TYPE.ANTHROPIC_MESSAGES:
            # Anthropic responses carry a top-level "content" block array; store
            # it as an assistant message so it round-trips into future requests.
            content_blocks = response_json.get("content")
            if isinstance(content_blocks, list) and content_blocks:
                assistant_items = [{"role": "assistant", "content": content_blocks}]
        elif request_type in (
            REQUEST_TYPE.BEDROCK_INVOKE,
            REQUEST_TYPE.BEDROCK_STREAMING_INVOKE,
            REQUEST_TYPE.BEDROCK_CONVERSE,
            REQUEST_TYPE.BEDROCK_STREAMING_CONVERSE,
        ):
            # Anthropic Bedrock: top-level "content" array
            # Amazon Nova / Converse API: nested "output.message.content" array
            content_blocks = response_json.get("content") or (
                response_json.get("output", {}).get("message", {}).get("content", [])
            )
            if content_blocks:
                text = "".join(
                    b.get("text", "")
                    for b in content_blocks
                    if isinstance(b, dict) and "text" in b
                )
                # Store in array format so it round-trips into future requests
                assistant_items = [{"role": "assistant", "content": [{"text": text}]}]
        else:  # Gemini
            candidate = response_json.get("candidates", [{}])[0]
            assistant_message = candidate.get("content", {})
            if assistant_message:
                assistant_items = [assistant_message]

        if assistant_items:
            new_history = messages + assistant_items
            new_key = _get_message_key(new_history)

            conversation_text = json.dumps(new_history)
            if len(conversation_text) > conversation_size_threshold:
                summary = await _summarize_conversation(
                    conversation_text, gemini_proxy.get_gemini_api_key()
                )
                cache[new_key] = summary
            else:
                cache[new_key] = conversation_text

    if hasattr(response, "body_iterator"):  # Streaming response
        # Pass chunks through as they arrive; update the cache once complete
        async def stream_and_capture():
            chunks: list[bytes] = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
                yield chunk
            await update_cache(b"".join(chunks))

        return StreamingResponse(
            stream_and_capture(),
            status_code=response.status_code,
            headers=response.headers,
        )

    await update_cache(response.body)
    return response


def _extract_openai_messages(body: dict) -> list[dict]:
    messages = body.get("messages", [])
    return [m for m in messages if m.get("role") != "system"]


def _extract_gemini_messages(body: dict) -> list[dict]:
    return body.get("contents", [])


def _extract_bedrock_messages(body: dict) -> list[dict]:
    messages = body.get("messages", [])
    return [m for m in messages if m.get("role") != "system"]


def _extract_anthropic_messages(body: dict) -> list[dict]:
    # Anthropic keeps the system prompt in a top-level "system" field, but be
    # defensive against clients that put a system turn inside "messages".
    messages = body.get("messages", [])
    return [
        m for m in messages if isinstance(m, dict) and m.get("role") != "system"
    ]


def _extract_responses_messages(body: dict) -> list[dict]:
    # The Responses API takes "input" as either a bare string (a single-shot
    # prompt with no prior turns to roll up) or an array of items.
    input_data = body.get("input")
    if not isinstance(input_data, list):
        return []
    return [item for item in input_data if isinstance(item, dict)]


def _get_message_key(messages: list[dict]) -> str:
    if not messages:
        return ""
    message_str = json.dumps(messages, sort_keys=True)
    return hashlib.md5(message_str.encode()).hexdigest()


def _inject_openai_system_prompt(body: dict, context: str) -> dict:
    messages = body.get("messages", [])
    for message in messages:
        if message.get("role") == "system":
            message["content"] = f"{context}\n{message.get('content', '')}"
            return body
    messages.insert(0, {"role": "system", "content": context})
    body["messages"] = messages
    return body


def _inject_bedrock_system_prompt(body: dict, context: str) -> dict:
    existing = body.get("system")
    if isinstance(existing, list):
        # Nova-style: system is an array of content objects
        body["system"] = [{"text": context}] + existing
    elif isinstance(existing, str) and existing:
        # Anthropic-style: system is a plain string
        body["system"] = f"{context}\n{existing}"
    else:
        # No existing system — infer format from messages content
        messages = body.get("messages", [])
        first_content = messages[0].get("content", "") if messages else ""
        if isinstance(first_content, list):
            body["system"] = [{"text": context}]
        else:
            body["system"] = context
    return body


def _inject_anthropic_system_prompt(body: dict, context: str) -> dict:
    existing = body.get("system")
    if isinstance(existing, list):
        # Block-array form: prepend a text block.
        body["system"] = [{"type": "text", "text": context}] + existing
    elif isinstance(existing, str) and existing:
        body["system"] = f"{context}\n{existing}"
    else:
        body["system"] = context
    return body


def _inject_responses_system_prompt(body: dict, context: str) -> dict:
    existing = body.get("instructions")
    if isinstance(existing, str) and existing:
        body["instructions"] = f"{context}\n{existing}"
    else:
        body["instructions"] = context
    return body


def _inject_gemini_system_prompt(body: dict, context: str) -> dict:
    if "system_instruction" in body:
        existing_instruction = body["system_instruction"]
        if isinstance(existing_instruction, dict):
            existing_text = existing_instruction.get("parts", [{}])[0].get("text", "")
            new_text = f"{context}\n{existing_text}"
            existing_instruction["parts"][0]["text"] = new_text
        else:
            body["system_instruction"] = f"{context}\n{existing_instruction}"
    else:
        body["system_instruction"] = {"parts": [{"text": context}]}
    return body


async def _summarize_conversation(
    conversation: str,
    api_key: str,
) -> str:
    """Calls Gemini API to summarize the conversation."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"  # noqa
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{DEFAULT_SUMMARIZER_PROMPT}\n\n{conversation}"}],
            }
        ]
    }
    async with create_http_client() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            summary = data["candidates"][0]["content"]["parts"][0]["text"]
            return f"{summary}\n\nVerbatim Transcript:\n{conversation}"
        except Exception as e:
            # In case of summarization failure,
            # return the original conversation
            return f"Summarization failed: {e}. Original conversation: {conversation}"


def _copy_json(obj: dict) -> dict:
    """
    Creates a deep copy of a JSON-serializable dictionary.
    This is a workaround to avoid issues with copy.deepcopy.
    """
    return json.loads(json.dumps(obj))
