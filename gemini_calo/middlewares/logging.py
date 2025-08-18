from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from gemini_calo.logger import logger
import json


async def logging_middleware(request: Request, call_next):
    """
    Logs incoming request and outgoing response details.
    Handles both regular and streaming responses.
    """
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.debug(f"Request headers: {request.headers}")

    # Read the body and re-create the request to make the body available again
    body = await request.body()
    if body:
        try:
            logger.debug(f"Request body: {json.loads(body)}")
        except json.JSONDecodeError:
            logger.debug(f"Request body (non-JSON): {body.decode(errors='ignore')}")

    async def receive():
        return {"type": "http.request", "body": body}

    request = Request(request.scope, receive)

    response = await call_next(request)

    logger.info(f"Outgoing response: Status {response.status_code}")
    logger.debug(f"Response headers: {response.headers}")

    if isinstance(response, StreamingResponse):
        # For streaming responses, we need to wrap the iterator to log chunks
        original_iterator = response.body_iterator

        async def logging_iterator():
            full_body = b""
            async for chunk in original_iterator:
                full_body += chunk
                yield chunk
            try:
                logger.debug(f"Response body (stream): {json.loads(full_body)}")
            except json.JSONDecodeError:
                logger.debug(f"Response body (stream, non-JSON): {full_body.decode(errors='ignore')}")

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
            logger.debug(f"Response body: {json.loads(response_body)}")
        except json.JSONDecodeError:
            logger.debug(f"Response body (non-JSON): {response_body.decode(errors='ignore')}")

    return response