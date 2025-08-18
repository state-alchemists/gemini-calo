import json
from functools import partial
from logging import Logger
from typing import Any, Callable, Coroutine

from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from gemini_calo.logger import logger as default_logger


def create_logging_middleware(
    logger: Logger | None = None,
) -> Callable[
    [Request, Callable[[Request], Coroutine[Any, Any, Response]]],
    Coroutine[Any, Any, Response],
]:
    return partial(logging_middleware, logger=logger)


async def logging_middleware(request: Request, call_next, logger: Logger | None):
    """
    Logs incoming request and outgoing response details.
    Handles both regular and streaming responses.
    """
    if logger is None:
        logger = default_logger
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Request headers: {request.headers}")

    # Read the body and re-create the request to make the body available again
    body = await request.body()
    if body:
        try:
            logger.info(f"Request body: {json.loads(body)}")
        except json.JSONDecodeError:
            logger.info(f"Request body (non-JSON): {body.decode(errors='ignore')}")

    async def receive():
        return {"type": "http.request", "body": body}

    request = Request(request.scope, receive)

    response = await call_next(request)

    logger.info(f"Outgoing response: Status {response.status_code}")
    logger.info(f"Response headers: {response.headers}")

    if isinstance(response, StreamingResponse):
        # For streaming responses, we need to wrap the iterator to log chunks
        original_iterator = response.body_iterator

        async def logging_iterator():
            full_body = b""
            async for chunk in original_iterator:
                full_body += chunk
                yield chunk
            try:
                logger.info(f"Response body (stream): {json.loads(full_body)}")
            except json.JSONDecodeError:
                logger.info(
                    f"Response body (stream, non-JSON): {full_body.decode(errors='ignore')}"
                )

        return StreamingResponse(
            logging_iterator(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    # For non-streaming responses
    response_body = b""
    if hasattr(response, "body"):
        response_body = response.body

    if response_body:
        try:
            logger.info(f"Response body: {json.loads(response_body)}")
        except json.JSONDecodeError:
            logger.info(
                f"Response body (non-JSON): {response_body.decode(errors='ignore')}"
            )

    return response
