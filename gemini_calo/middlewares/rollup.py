import json
import hashlib
import httpx
from cachetools import LRUCache
from functools import partial
from typing import Any, Callable, Coroutine, cast

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from gemini_calo.config import (
    CONVERSATION_SUMMARIZATION_LRU_SIZE,
    DEFAULT_SUMMARIZER_PROMPT,
    SUMMARIZATION_SIZE_THRESHOLD,
)
from gemini_calo.proxy import REQUEST_TYPE, GeminiProxyService

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
        REQUEST_TYPE.GEMINI_COMPLETION,
        REQUEST_TYPE.GEMINI_STREAMING_COMPLETION,
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

    original_request = request
    if found_key:
        print(f"found_key: {found_key}")
        print(f"num_matched_messages: {num_matched_messages}")
        context = cast(str, cache[found_key])
        if request_type == REQUEST_TYPE.OPENAI_COMPLETION:
            json_body = _inject_openai_system_prompt(_copy_json(json_body), context)
            original_messages = json_body.get("messages", [])
            system_messages = [m for m in original_messages if m.get("role") == "system"]
            user_messages = [
                m for m in original_messages if m.get("role") != "system"
            ]
            json_body["messages"] = system_messages + user_messages[num_matched_messages:]
        else:
            json_body = _inject_gemini_system_prompt(_copy_json(json_body), context)
            json_body["contents"] = messages[num_matched_messages:]

        new_body = json.dumps(json_body).encode()
        request.state.modified_body = new_body  # Store modified body in request.state

        async def receive():
            return {"type": "http.request", "body": new_body}

        request = Request(request.scope, receive)

    response = await call_next(request)

    # Response handling
    response_body = b""
    if hasattr(response, 'body_iterator'): # Check if it's a streaming response
        # Read the entire streaming response into response_body
        async for chunk in response.body_iterator:
            response_body += chunk
        # Create a new StreamingResponse from the captured body for the client
        return_response = StreamingResponse(
            iter([response_body]),
            status_code=response.status_code,
            headers=response.headers
        )
    else:
        response_body = response.body  # For regular responses, use .body
        # Create a new response from the captured body for the client
        return_response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=response.headers
        )

    try:
        response_json = json.loads(response_body)
    except json.JSONDecodeError:
        response_json = {}

    assistant_message = {}
    if request_type == REQUEST_TYPE.OPENAI_COMPLETION:
        choice = response_json.get("choices", [{}])[0]
        assistant_message = choice.get("message", {})
    else: # Gemini
        candidate = response_json.get("candidates", [{}])[0]
        assistant_message = candidate.get("content", {})

    if assistant_message:
        new_history = messages + [assistant_message]
        new_key = _get_message_key(new_history)
        
        conversation_text = json.dumps(new_history)
        if len(conversation_text) > conversation_size_threshold:
            summary = await _summarize_conversation(
                conversation_text, gemini_proxy.get_gemini_api_key()
            )
            cache[new_key] = summary
        else:
            cache[new_key] = conversation_text

    return return_response if 'return_response' in locals() else response


def _extract_openai_messages(body: dict) -> list[dict]:
    messages = body.get("messages", [])
    return [m for m in messages if m.get("role") != "system"]


def _extract_gemini_messages(body: dict) -> list[dict]:
    return body.get("contents", [])


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


def _inject_gemini_system_prompt(body: dict, context: str) -> dict:
    if "system_instruction" in body:
        existing_instruction = body["system_instruction"]
        if isinstance(existing_instruction, dict):
            existing_text = existing_instruction.get(
                "parts", [{}]
            )[0].get("text", "")
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
            {"role": "user", "parts": [{"text": f"{DEFAULT_SUMMARIZER_PROMPT}\n\n{conversation}"}]}
        ]
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url, json=payload, headers=headers, timeout=60
            )
            response.raise_for_status()
            data = response.json()
            summary = data["candidates"][0]["content"]["parts"][0]["text"]
            return f"{summary}\n\nVerbatim Transcript:\n{conversation}"
        except (httpx.HTTPStatusError, KeyError, IndexError) as e:
            # In case of summarization failure,
            # return the original conversation
            return f"Summarization failed: {e}. Original conversation: {conversation}"


def _copy_json(obj: dict) -> dict:
    """
    Creates a deep copy of a JSON-serializable dictionary.
    This is a workaround to avoid issues with copy.deepcopy.
    """
    return json.loads(json.dumps(obj))
