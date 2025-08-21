import inspect
import json
from typing import Any, Callable, Coroutine, Union
from functools import partial
from fastapi import Request, Response

from gemini_calo import config
from gemini_calo.proxy import REQUEST_TYPE, GeminiProxyService

# Define type hints for clarity
ModelTransformerCallable = Callable[[str], Union[str, Coroutine[Any, Any, str]]]
ModelTransformerArg = Union[ModelTransformerCallable, str, None]


def create_model_override_middleware(
    model_transformer: ModelTransformerArg = None,
) -> Callable[
    [Request, Callable[[Request], Coroutine[Any, Any, Response]]],
    Coroutine[Any, Any, Response],
]:
    return partial(model_override_middleware, model_transformer=model_transformer)


async def model_override_middleware(
    request: Request,
    call_next: Callable[[Request], Coroutine[Any, Any, Response]],
    model_transformer: ModelTransformerArg = None,
) -> Response:
    """
    Core middleware logic that resolves the transformer on each request and
    applies the model transformation.
    """
    # Determine the transformer to use, falling back to the environment variable
    transformer = model_transformer
    if transformer is None and config.MODEL_OVERRIDE:
        transformer = config.MODEL_OVERRIDE

    # If no transformer is defined, proceed without modification
    if transformer is None:
        return await call_next(request)

    request_type = GeminiProxyService.get_request_type(request)

    if request_type == REQUEST_TYPE.OPENAI_COMPLETION:
        request = await _transform_model_in_openai_request(request, transformer)
    elif request_type in [
        REQUEST_TYPE.GEMINI_COMPLETION,
        REQUEST_TYPE.GEMINI_STREAMING_COMPLETION,
    ]:
        request = await _transform_model_in_gemini_request(request, transformer)

    return await call_next(request)


async def _resolve_new_model_name(
    transformer: ModelTransformerArg, original_model: str
) -> str:
    """
    Resolves the new model name from a transformer argument, which can be a
    callable, a coroutine, a string, or None.
    """
    if callable(transformer):
        if inspect.iscoroutinefunction(transformer):
            return await transformer(original_model)
        return transformer(original_model)
    if isinstance(transformer, str):
        return transformer
    return original_model


async def _transform_model_in_openai_request(
    request: Request, transformer: ModelTransformerArg
) -> Request:
    """Transforms the model in an OpenAI completion request."""
    body = await request.body()
    try:
        json_body = json.loads(body)
        original_model = json_body.get("model")
        if not original_model:
            return request

        new_model = await _resolve_new_model_name(transformer, original_model)

        if new_model and new_model != original_model:
            json_body["model"] = new_model
            new_body = json.dumps(json_body).encode()

            async def receive():
                return {"type": "http.request", "body": new_body}

            return Request(request.scope, receive)
    except (json.JSONDecodeError, AttributeError):
        pass
    return request


async def _transform_model_in_gemini_request(
    request: Request, transformer: ModelTransformerArg
) -> Request:
    """Transforms the model in a Gemini completion request."""
    path = request.scope.get("path", "")
    try:
        parts = path.split("/")
        model_part = parts[3]
        action_part = model_part.split(":")[-1]
        original_model = model_part.removesuffix(f":{action_part}")

        if not original_model:
            return request

        new_model = await _resolve_new_model_name(transformer, original_model)

        if new_model and new_model != original_model:
            new_path = path.replace(original_model, new_model, 1)
            request.scope["path"] = new_path
    except IndexError:
        pass
    return request

