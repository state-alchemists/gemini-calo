import os

import uvicorn
from fastapi import FastAPI

from gemini_calo.middlewares.auth import create_auth_middleware
from gemini_calo.middlewares.logging import create_logging_middleware
from gemini_calo.proxy import GeminiProxyService


def start_server():
    api_keys = os.getenv("GEMINI_CALO_API_KEYS", "").split(",")
    if not api_keys or api_keys == [""]:
        raise ValueError("GEMINI_CALO_API_KEYS is empty")

    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=api_keys)

    proxy_api_key = os.getenv("GEMINI_CALO_PROXY_API_KEYS", "")
    if proxy_api_key != "":
        proxy_api_keys = proxy_api_key.split(",")
        app.middleware("http")(create_auth_middleware(proxy_api_keys))

    app.middleware("http")(create_logging_middleware())
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)

    @app.get("/")
    def read_root():
        return {"Status": "Okay"}

    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("GEMINI_CALO_HTTP_PORT", "8000"))
    )
