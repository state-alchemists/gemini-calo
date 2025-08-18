from gemini_calo.middlewares.auth import auth_middleware, create_auth_middleware
from gemini_calo.middlewares.logging import (
    create_logging_middleware,
    logging_middleware,
)
from gemini_calo.proxy import GeminiProxyService

assert GeminiProxyService
assert auth_middleware
assert create_auth_middleware
assert create_logging_middleware
assert logging_middleware
