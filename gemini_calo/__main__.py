import os
import uvicorn
from fastapi import FastAPI
from gemini_calo.proxy import GeminiProxyService
from gemini_calo.middlewares.logging import logging_middleware
from gemini_calo.middlewares.auth import auth_middleware
from functools import partial


def start_server():
    api_keys = os.getenv("GEMINI_CALO_API_KEYS", "").split(",")
    if not api_keys or api_keys == [""]:
        raise ValueError("GEMINI_CALO_API_KEYS not found or empty in .env file")

    app = FastAPI()
    proxy = GeminiProxyService(gemini_api_keys=api_keys)

    proxy_api_key = os.getenv("GEMINI_CALO_PROXY_API_KEYS", "")
    if proxy_api_key != "":
        proxy_api_keys = proxy_api_key.split(",")
        auth_middleware_with_key = partial(auth_middleware, user_api_key_checker=proxy_api_keys)
        app.middleware("http")(auth_middleware_with_key)

    app.middleware("http")(logging_middleware)
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)

    @app.get("/")
    def read_root():
        return {"Hello": "Proxy"}

    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("GEMINI_CALO_HTTP_PORT", "8000"))
    )