import json
import inspect
from functools import partial
from typing import Any, Callable, Coroutine

from fastapi import Request, Response

from gemini_calo.proxy import REQUEST_TYPE, GeminiProxyService


def create_model_override_middleware(
    model_transformer: Callable[[str], str | Coroutine[Any, Any, str]] | None = None,
) -> Callable[
    [Request, Callable[[Request], Coroutine[Any, Any, Response]]],
    Coroutine[Any, Any, Response],
]:
    """
    Creates a middleware to override model names in requests using a transformer function.
    """
    return partial(model_override_middleware, model_transformer=model_transformer)


async def model_override_middleware(
    request: Request,
    call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    model_transformer: Callable[[str], str | Coroutine[Any, Any, str]] | None = None,
) -> Response:
    """
    Overrides the model in the request if a transformer function is provided.
    Handles both OpenAI (body) and Gemini (URL path) style requests.
    """
    if not callable(model_transformer):
        return await call_next(request)

    request_type = GeminiProxyService.get_request_type(request)

    # Handle OpenAI completion requests by modifying the request body
    if request_type == REQUEST_TYPE.OPENAI_COMPLETION:
        body = await request.body()
        try:
            json_body = json.loads(body)
            original_model = json_body.get("model")
            if original_model:
                if inspect.iscoroutinefunction(model_transformer):
                    new_model = await model_transformer(original_model)
                else:
                    new_model = model_transformer(original_model)

                if new_model != original_model:
                    json_body["model"] = new_model
                    # Re-serialize the body and create a new request
                    new_body = json.dumps(json_body).encode()

                    async def receive():
                        return {"type": "http.request", "body": new_body}

                    request = Request(request.scope, receive)
        except (json.JSONDecodeError, AttributeError):
            # If body is not valid JSON or not present, proceed without modification
            pass

    # Handle Gemini completion requests by modifying the URL path
    elif request_type in [
        REQUEST_TYPE.GEMINI_COMPLETION,
        REQUEST_TYPE.GEMINI_STREAMING_COMPLETION,
    ]:
        # Extract model from path, e.g., /v1beta/models/gemini-pro:generateContent
        path = request.scope.get("path", "")
        try:
            # The model is between the 3rd and last part of the path
            parts = path.split("/")
            model_part = parts[3]
            action_part = model_part.split(":")[-1]
            original_model = model_part.removesuffix(f":{action_part}")

            if original_model:
                if inspect.iscoroutinefunction(model_transformer):
                    new_model = await model_transformer(original_model)
                else:
                    new_model = model_transformer(original_model)

                if new_model != original_model:
                    new_path = path.replace(original_model, new_model, 1)
                    request.scope["path"] = new_path
        except IndexError:
            # If path structure is unexpected, proceed without modification
            pass

    return await call_next(request)
