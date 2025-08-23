import uvicorn
from fastapi import FastAPI

from gemini_calo.config import (
    GEMINI_API_KEYS, MODEL_OVERRIDE, PROXY_API_KEYS, HTTP_PORT
)
from gemini_calo.middlewares.auth import create_auth_middleware
from gemini_calo.middlewares.logging import create_logging_middleware
from gemini_calo.middlewares.model_override import (
    create_model_override_middleware
)
from gemini_calo.middlewares.rollup import create_rollup_middleware
from gemini_calo.proxy import GeminiProxyService


def start_server():
    if len(GEMINI_API_KEYS) == 0:
        raise ValueError("GEMINI_CALO_API_KEYS is empty")

    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=GEMINI_API_KEYS)

    app.middleware("http")(create_logging_middleware())
    app.middleware("http")(create_auth_middleware(PROXY_API_KEYS))
    app.middleware("http")(create_model_override_middleware(MODEL_OVERRIDE))
    app.middleware("http")(create_rollup_middleware(proxy))
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)

    @app.get("/")
    def read_root():
        return {"Status": "Okay"}

    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT)
