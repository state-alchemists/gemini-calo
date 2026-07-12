"""
Gemini Calo — Universal LLM Hub Example

Starts a proxy that routes to DeepSeek, Gemini, and Bedrock,
each with its own API key and protocol translation.

Usage:
    cd example
    source .env
    python app.py

Then test with:
    curl http://localhost:8000/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "deepseek-chat", "messages": [{"role": "user", "content": "Hello"}]}'
"""

import os

import uvicorn
from fastapi import FastAPI

from gemini_calo.auth.aws import AWSCredentials, create_aws_sigv4_provider
from gemini_calo.middlewares.rollup import create_rollup_middleware
from gemini_calo.proxy import GeminiProxyService, RouteConfig


def build_model_routes() -> dict[str, RouteConfig]:
    routes: dict[str, RouteConfig] = {}

    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    if deepseek_key:
        routes["deepseek-*"] = RouteConfig(
            url="https://api.deepseek.com",
            api_keys=[deepseek_key],
            auth="bearer",
            protocol="openai-chat",
        )

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        # Route Gemini through the native-API translator so every client
        # endpoint works (/v1/chat/completions, /v1/responses, /v1/messages).
        # Without this, Gemini would fall back to Google's OpenAI-compat layer,
        # which only handles chat completions.
        routes["gemini-*"] = RouteConfig(
            url="https://generativelanguage.googleapis.com",
            api_keys=[gemini_key],
            auth="x-goog-api-key",
            protocol="gemini",
        )

    bedrock_token = os.getenv("BEDROCK_BEARER_TOKEN", "")
    bedrock_region = os.getenv("BEDROCK_REGION", "us-east-1")
    if bedrock_token:
        bedrock_url = f"https://bedrock-runtime.{bedrock_region}.amazonaws.com"
        # Amazon Nova (non-Anthropic) Bedrock models.
        routes["amazon.*"] = RouteConfig(
            url=bedrock_url,
            api_keys=[bedrock_token],
            auth="bearer",
            protocol="bedrock-invoke",
        )
        # Colon-free alias: Bedrock model ids contain ":" (amazon.nova-pro-v1:0),
        # which some clients (e.g. zrb) parse as a "provider:model" separator and
        # mangle. Point the alias at the real id via upstream_model so those
        # clients can use a simple name.
        routes["nova"] = RouteConfig(
            url=bedrock_url,
            api_keys=[bedrock_token],
            auth="bearer",
            protocol="bedrock-invoke",
            upstream_model="amazon.nova-pro-v1:0",
        )

    return routes


def create_app() -> FastAPI:
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is required")

    model_routes = build_model_routes()

    proxy = GeminiProxyService(
        api_keys=[gemini_key],
        model_routes=model_routes,
    )

    app = FastAPI(title="Gemini Calo Hub")
    # Conversation roll-up / summarization (works across every client protocol,
    # including /v1/messages and /v1/responses).
    app.middleware("http")(create_rollup_middleware(proxy))
    app.include_router(proxy.gemini_router)
    app.include_router(proxy.openai_router)
    app.include_router(proxy.bedrock_router)
    app.include_router(proxy.anthropic_router)

    @app.get("/")
    def health():
        providers = ["gemini"]
        if "deepseek-*" in model_routes:
            providers.append("deepseek")
        if "amazon.*" in model_routes:
            providers.append("bedrock")
        return {"status": "ok", "providers": providers}

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("GEMINI_CALO_HTTP_PORT", "8000"))
    print(f"Gemini Calo Hub starting on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
