# Gemini Calo

**Gemini Calo** is a powerful, yet simple, FastAPI-based proxy that puts a single, authenticated, observable front door in front of **many LLM providers**. It began as a proxy for Google's Gemini API and is now a small **universal LLM hub**: point your client at Calo and it handles authentication, request logging, and conversation rollup — and, crucially, **protocol translation**.

A client that speaks **OpenAI Chat Completions**, the **OpenAI Responses API**, or **Anthropic Messages** can be served by **Google Gemini**, **AWS Bedrock** (InvokeModel *or* Converse), **real OpenAI**, or any **OpenAI-compatible** upstream — chosen per model, with no change to the client. Calo also exposes the **native Gemini** and **native Bedrock** wire APIs as authenticated passthroughs for clients that already speak those formats. See [Supported Routes & Provider Interchangeability](#supported-routes--provider-interchangeability) for the full picture.

It's designed to run as a standalone server or mount into an existing FastAPI application.

## Key Features

*   **Universal, interchangeable providers:** Serve one client protocol from any upstream provider. An OpenAI, Responses, or Anthropic-Messages client can talk to Gemini, Bedrock, or OpenAI-compatible backends, selected per model via glob-matched routes — the client never knows which provider actually answered.
*   **Authentication:** Secure access with an additional layer of proxy API-key authentication in front of every provider.
*   **Request Logging:** Detailed logging of all incoming requests and outgoing responses.
*   **OpenAI Compatibility:** OpenAI Chat Completions (`/v1/chat/completions`) and Responses (`/v1/responses`) endpoints. Requests are translated to the routed upstream (Gemini/Bedrock/OpenAI-compatible), or passed through verbatim to a native OpenAI/Responses upstream.
*   **Anthropic Compatibility:** An Anthropic Messages endpoint (`/v1/messages`) that translates to whichever provider the model routes to.
*   **AWS Bedrock Compatibility:** Native Bedrock endpoints covering both the InvokeModel API (`/model/{modelId}/invoke`, `/model/{modelId}/invoke-with-response-stream`) and the Converse API (`/model/{modelId}/converse`, `/model/{modelId}/converse-stream`) — supports both Bedrock API key (bearer token) and SigV4 signing.
*   **Round-Robin API Keys:** Distribute your requests across multiple API keys, both globally and per model route.
*   **Multi-Provider Routing:** Route specific models (via glob patterns) to different upstream providers and protocols, each with its own credentials and timeout.
*   **Extensible Authentication:** Support for complex auth schemes like AWS SigV4, OAuth, or custom providers via pluggable auth modules.
*   **Easy Integration:** Use it as a standalone server or mount it into your existing FastAPI project.
*   **Extensible:** Easily add your own custom middleware to suit your needs.

## How It's Useful

-   **Centralized API Key Management:** Instead of hardcoding your Gemini API keys in various clients, you can manage them in one place.
-   **Security:** Protect your expensive Gemini API keys by exposing only a proxy key to your users or client applications.
-   **Monitoring & Observability:** The logging middleware gives you insight into how your API is being used, helping you debug issues and monitor usage patterns.
-   **Seamless Migration:** Existing OpenAI, Responses, or Anthropic-Messages tooling can switch to Gemini, Bedrock, or any OpenAI-compatible backend without significant code changes — usually just a base URL and a model name.

## Supported Routes & Provider Interchangeability

Calo separates **what your client speaks** (the inbound route) from **what actually serves the request** (the upstream provider). Any *translatable* client protocol can be served by any supported provider, chosen per model through [`model_routes`](#routing-models-to-different-providers).

### Client-facing routes (what your client calls)

| Client protocol | Routes | Translatable? |
|---|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions`, `POST /v1beta/openai/chat/completions` | ✅ |
| OpenAI Responses | `POST /v1/responses` | ✅ |
| Anthropic Messages | `POST /v1/messages` | ✅ |
| OpenAI Embeddings | `POST /v1/embeddings`, `POST /v1beta/openai/embeddings` | passthrough |
| Gemini (native) | `POST /v1beta/models/{model}:generateContent`, `:streamGenerateContent`, `:embedContent`; `GET /v1beta/models` | passthrough |
| Bedrock InvokeModel (native) | `POST /model/{modelId}/invoke`, `/invoke-with-response-stream` | passthrough |
| Bedrock Converse (native) | `POST /model/{modelId}/converse`, `/converse-stream` | passthrough |

**Translatable** routes are converted through a canonical intermediate representation (IR) to whichever upstream the model routes to. The **native** Gemini and Bedrock routes are authenticated passthroughs — they forward to the matching native upstream unchanged, with logging, auth, model-override, and rollup still applied.

### Upstream providers (what serves the request)

Set per route via `RouteConfig.protocol` (see [Routing Models to Different Providers](#routing-models-to-different-providers)):

| `protocol` | Upstream it targets | Translation |
|---|---|---|
| `"openai"` (default) | Real OpenAI, or any upstream that already speaks the client's exact protocol | none — verbatim passthrough |
| `"openai-chat"` | Chat-Completions-only backends (DeepSeek, Together, …) | client → `/v1/chat/completions` |
| `"gemini"` | Google Gemini native API | client → Gemini |
| `"bedrock-invoke"` | AWS Bedrock InvokeModel API | client → Bedrock InvokeModel |
| `"bedrock-converse"` | AWS Bedrock Converse API | client → Bedrock Converse |

> With `protocol="openai"` (the default), no translation happens — so the upstream must already speak whatever the client sent (e.g. a Responses client can only passthrough to a Responses-native upstream). To reach Gemini/Bedrock, or to serve an OpenAI-only backend from a Responses/Anthropic client, set an explicit translating `protocol`.

### The interchangeability matrix

Because translation is split into an inbound half (client → IR) and an outbound half (IR → upstream), **any translatable client can be served by any provider**:

| Client ↓ / Upstream → | Gemini | Bedrock Invoke | Bedrock Converse | OpenAI-chat | OpenAI (passthrough) |
|---|:---:|:---:|:---:|:---:|:---:|
| OpenAI Chat | ✅ | ✅ | ✅ | ✅ | ✅ |
| OpenAI Responses | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic Messages | ✅ | ✅ | ✅ | ✅ | ✅ |

The client sends the request it always would; you decide which provider answers by matching the model name to a route. Moving a model from Gemini to Bedrock is a **one-line route change** — no client edits. Calo also normalizes provider quirks along the way (e.g. stripping JSON-Schema keywords Gemini rejects, clamping Nova's `max_tokens`), so the same tool-using client works everywhere.

### Known limitation: Gemini thought signatures on the translated path

When a client protocol (OpenAI Chat, OpenAI Responses, or Anthropic Messages) is **translated to a Gemini upstream** (`protocol="gemini"`), Calo does **not** currently round-trip Gemini's `thoughtSignature`. Gemini returns an opaque `thoughtSignature` on each `functionCall`; on the next turn it expects that signature echoed back. Calo's IR drops it during translation, so multi-turn tool-calling conversations resend function calls without their signature. The impact depends on the Gemini model:

| Model family | Effect of the missing signature |
|---|---|
| Gemini 2.5 | Non-fatal **warning** — *"missing thought_signature … may lead to degraded model performance."* Tool calls still work. |
| Gemini 3 (`gemini-3-*`) | **Hard `400 INVALID_ARGUMENT`** — the API strictly requires the signature, so multi-turn tool loops fail. |

**Not affected:** the **native Gemini passthrough** (`gemini_router`, e.g. the google-genai SDK), which forwards requests unmodified so the SDK preserves the signature itself; non-Gemini upstreams (Bedrock, OpenAI-compatible), which don't use thought signatures; and any single-turn or non-tool request.

**Workarounds until this is fixed:**

- For tool-heavy agents on **Gemini 3**, use the **native Gemini endpoint** rather than the OpenAI/Anthropic-translated path, or route those models to a **non-Gemini** provider (e.g. Bedrock Nova/Claude).
- **Gemini 2.5** models tolerate the missing signature (degraded, not broken).

The tractable fix is to carry the signature through the round-tripping `tool_call.id`; see the [Gemini thought-signatures docs](https://ai.google.dev/gemini-api/docs/thinking#signatures) for the requirement.

## Running the Built-in Server

You can quickly get the proxy server up and running with just a few steps.

### 1. Installation

Install the package using pip:

```bash
pip install gemini-calo
```

### 2. Configuration

The server is configured through environment variables. You can create a `.env` file in your working directory to store them.

*   `GEMINI_CALO_API_KEYS`: A comma-separated list of your Google Gemini API keys. The proxy will rotate through these keys for outgoing requests. Required when using the built-in server.
*   `GEMINI_CALO_PROXY_API_KEYS`: (Optional) A comma-separated list of API keys that clients must provide to use the proxy. If not set, the proxy accepts all requests without authentication.
*   `GEMINI_CALO_HTTP_PORT`: The port on which the server will run. Defaults to `8000`.
*   `GEMINI_CALO_LOG_LEVEL`: Sets the logging level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Defaults to `CRITICAL`.
*   `GEMINI_CALO_LOG_FILE`: Specifies the file where logs will be written. Defaults to `app.log`.
*   `GEMINI_CALO_CONVERSATION_SUMMARIZATION_LRU_CACHE`: Size of the LRU cache for conversation summarization. Defaults to `20`.
*   `GEMINI_CALO_MODEL_OVERRIDE`: Forces all requests to use a specific model name, overriding whatever the client sends.

**Example `.env` file:**

```bash
# Your gemini API Keys
export GEMINI_CALO_API_KEYS=AIaYourGeminiKey1,AIaYourGeminiKey2
# API Keys for your internal user
export GEMINI_CALO_PROXY_API_KEYS=my_secret_proxy_key_1,my_secret_proxy_key_2
# Gemini Calo HTTP Port
export GEMINI_CALO_HTTP_PORT=8080
# Logging level
export GEMINI_CALO_LOG_LEVEL=DEBUG
# Log file
export GEMINI_CALO_LOG_FILE=gemini_calo.log
```

### 3. Running the Server

Once configured, you can start the server with the `gemini-calo` command:

```bash
gemini-calo
```

The server will start on the configured port (e.g., `http://0.0.0.0:8080`).

## Integrating with an Existing FastAPI Application

If you have an existing FastAPI application, you can easily integrate Gemini Calo's proxy functionality into it.

```python
from fastapi import FastAPI
from gemini_calo.proxy import GeminiProxyService
from gemini_calo.middlewares.auth import auth_middleware
from gemini_calo.middlewares.logging import logging_middleware
from functools import partial
import os

# Your existing FastAPI app
app = FastAPI()

# 1. Initialize the GeminiProxyService
api_keys = os.getenv("GEMINI_CALO_API_KEYS", "").split(",")
proxy_service = GeminiProxyService(api_keys=api_keys)

# 2. (Optional) Add Authentication Middleware
proxy_api_keys = os.getenv("GEMINI_CALO_PROXY_API_KEYS", "").split(",")
if proxy_api_keys:
    auth_middleware_with_keys = partial(auth_middleware, user_api_key_checker=proxy_api_keys)
    app.middleware("http")(auth_middleware_with_keys)

# 3. (Optional) Add Logging Middleware
app.middleware("http")(logging_middleware)

# 4. Mount the routers you want to expose.
#    - openai_router:    /v1/chat/completions, /v1/responses, /v1/embeddings
#    - anthropic_router: /v1/messages (Anthropic Messages, e.g. Claude Code)
#    - gemini_router:    native Gemini passthrough
#    - bedrock_router:   native Bedrock Invoke + Converse passthrough
app.include_router(proxy_service.openai_router)
app.include_router(proxy_service.anthropic_router)
app.include_router(proxy_service.gemini_router)
app.include_router(proxy_service.bedrock_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Now you can run your app as usual with uvicorn
# uvicorn your_app_file:app --reload
```

## Routing Models to Different Providers

`GeminiProxyService` supports a `model_routes` parameter — a `dict` that maps glob patterns to a `RouteConfig`. When a request arrives, the proxy extracts the model name (from the URL path for Gemini/Bedrock-format requests, or from the JSON body for OpenAI/Anthropic-format requests) and checks it against each pattern in order. The first match wins; unmatched models fall back to `base_url` + `api_keys`.

The route's `protocol` is what makes providers interchangeable: it selects the outbound adapter that translates the incoming request to the upstream's native API. See [Supported Routes & Provider Interchangeability](#supported-routes--provider-interchangeability) for the full matrix.

### `RouteConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | — | Upstream base URL for this route |
| `api_keys` | `list[str]` | `[]` | Keys rotated round-robin for preset auth types |
| `auth` | `str` \| `callable` \| `None` | `"bearer"` | Authentication configuration (see below) |
| `protocol` | `str` | `"openai"` | Upstream protocol / outbound adapter: `"openai"` (passthrough), `"openai-chat"`, `"gemini"`, `"bedrock-invoke"`, `"bedrock-converse"` |
| `upstream_model` | `str` | `""` | If set, replaces the client's model id when calling upstream. Handy for friendly aliases (`nova` → `amazon.nova-pro-v1:0`) and clients that mangle ids containing `:` (zrb). Only applied on the translated path |
| `timeout` | `float` | `300.0` | Per-request timeout in seconds |
| `outbound` | `OutboundAdapter` \| `None` | auto | Custom outbound adapter; auto-resolved from `protocol` when omitted |
| `auth_type` | `"bearer"` \| `"x-goog-api-key"` | — | **Deprecated:** Use `auth` instead |

### Authentication Configuration

The `auth` field supports multiple authentication modes:

| Value | Description |
|-------|-------------|
| `"bearer"` | Uses `api_keys` with `Authorization: Bearer <key>` header (round-robin) |
| `"x-goog-api-key"` | Uses `api_keys` with `x-goog-api-key` header (round-robin) |
| `"none"` or `None` | No authentication headers added |
| `callable` | Custom auth provider function for advanced scenarios |

### Example: a universal hub over Gemini, Bedrock, and DeepSeek

This mirrors [`example/app.py`](example/app.py). One proxy fronts three providers; clients pick the provider by model name, regardless of which client protocol they speak.

```python
import os
from fastapi import FastAPI
from gemini_calo.proxy import GeminiProxyService, RouteConfig

app = FastAPI()

proxy = GeminiProxyService(
    api_keys=[os.environ["GEMINI_API_KEY"]],  # fallback for unmatched models
    model_routes={
        # DeepSeek — a Chat-Completions-only OpenAI-compatible backend
        "deepseek-*": RouteConfig(
            url="https://api.deepseek.com",
            api_keys=[os.environ["DEEPSEEK_API_KEY"]],
            auth="bearer",
            protocol="openai-chat",
        ),
        # Gemini — translate every client protocol to the native Gemini API
        "gemini-*": RouteConfig(
            url="https://generativelanguage.googleapis.com",
            api_keys=[os.environ["GEMINI_API_KEY"]],
            auth="x-goog-api-key",
            protocol="gemini",
        ),
        # Bedrock — Amazon Nova via the InvokeModel API (Bedrock API key)
        "amazon.*": RouteConfig(
            url="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_keys=[os.environ["BEDROCK_BEARER_TOKEN"]],
            auth="bearer",
            protocol="bedrock-invoke",
        ),
        # Colon-free alias for clients (e.g. zrb) that choke on ":" in model ids
        "nova": RouteConfig(
            url="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_keys=[os.environ["BEDROCK_BEARER_TOKEN"]],
            auth="bearer",
            protocol="bedrock-invoke",
            upstream_model="amazon.nova-pro-v1:0",
        ),
    },
)

# Expose every client protocol; each is served by whichever provider the model matches.
app.include_router(proxy.openai_router)     # /v1/chat/completions, /v1/responses
app.include_router(proxy.anthropic_router)  # /v1/messages
app.include_router(proxy.gemini_router)     # native Gemini passthrough
app.include_router(proxy.bedrock_router)    # native Bedrock passthrough
```

With this running, an OpenAI client asking for `gemini-2.5-flash`, an Anthropic-Messages client (Claude Code) asking for `amazon.nova-pro-v1:0`, and a Responses client asking for `deepseek-chat` are all served correctly — each translated to its provider's native API. See [Client Integrations](#client-integrations) for the client side.

Pattern matching uses Python's `fnmatch`, so `*` matches any substring within a segment and `?` matches a single character. Patterns are checked in insertion order — the first match wins.

## AWS Bedrock Endpoint

Gemini Calo exposes a native Bedrock-compatible endpoint. Any client targeting `bedrock-runtime` can point at the proxy instead with minimal or no code changes.

### Supported routes

#### InvokeModel API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/model/{modelId}/invoke` | Synchronous invocation |
| `POST` | `/model/{modelId}/invoke-with-response-stream` | Streaming invocation |

#### Converse API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/model/{modelId}/converse` | Synchronous Converse invocation |
| `POST` | `/model/{modelId}/converse-stream` | Streaming Converse invocation |

The Converse API routes support the same authentication options as the InvokeModel routes. The proxy preserves the incoming `Content-Type` header and forwards the following optional Bedrock-specific request headers when present:

| Header | Purpose |
|--------|---------|
| `Accept` | Desired MIME type for the response body |
| `X-Amzn-Bedrock-Trace` | Enable tracing (`ENABLED` / `DISABLED` / `ENABLED_FULL`) |
| `X-Amzn-Bedrock-GuardrailIdentifier` | ID of a guardrail to apply (InvokeModel only) |
| `X-Amzn-Bedrock-GuardrailVersion` | Guardrail version (InvokeModel only) |
| `X-Amzn-Bedrock-PerformanceConfig-Latency` | `standard` or `optimized` (InvokeModel only) |
| `X-Amzn-Bedrock-Service-Tier` | `priority` / `default` / `flex` / `reserved` (InvokeModel only) |

For the Converse API, guardrail config, inference config, and service tier are passed in the **JSON body** (not as headers) and are forwarded via normal body passthrough.

> **Streaming note:** `converse-stream` returns a binary AWS Event Stream (`application/vnd.amazon.eventstream`), the same binary framing as `invoke-with-response-stream`. The proxy streams the raw bytes through intact, so any boto3 or SDK client that reads the event stream will work correctly.

### Authentication options

Clients authenticate with the upstream Bedrock service by sending one of the following sets of headers:

| Scenario | Headers to send | How the proxy signs the upstream request |
|----------|-----------------|------------------------------------------|
| Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`) | `X-AWS-Bearer-Token: <token>` | `Authorization: Bearer <token>` (no signing) |
| IAM credentials (SigV4) | `X-AWS-Access-Key`, `X-AWS-Secret-Key` (+ optional `X-AWS-Session-Token`, `X-AWS-Region`) | AWSSig4-signed request |

The proxy auto-detects which path to use: bearer token takes priority over SigV4. If neither is present, the request is forwarded unsigned.

The upstream URL is built dynamically from the `X-AWS-Region` header (default: `us-east-1`) unless the model is matched by a `model_routes` entry that provides a fixed URL.

### Example: proxy as a Bedrock passthrough

```python
from fastapi import FastAPI
from gemini_calo.proxy import GeminiProxyService

proxy = GeminiProxyService(api_keys=["gemini-key"])

app = FastAPI()
app.include_router(proxy.gemini_router)
app.include_router(proxy.openai_router)
app.include_router(proxy.bedrock_router)
```

Clients then call the proxy exactly like they would call `bedrock-runtime`:

```bash
# InvokeModel — using a Bedrock API key (AWS_BEARER_TOKEN_BEDROCK)
curl -X POST http://localhost:8000/model/anthropic.claude-3-5-sonnet-20241022-v1:0/invoke \
  -H "Content-Type: application/json" \
  -H "X-AWS-Bearer-Token: $AWS_BEARER_TOKEN_BEDROCK" \
  -d '{"anthropic_version":"bedrock-2023-05-31","max_tokens":256,"messages":[{"role":"user","content":"Hello"}]}'

# Converse API — using a Bedrock API key
curl -X POST http://localhost:8000/model/anthropic.claude-3-5-sonnet-20241022-v1:0/converse \
  -H "Content-Type: application/x-amz-json-1.1" \
  -H "X-AWS-Bearer-Token: $AWS_BEARER_TOKEN_BEDROCK" \
  -d '{"messages":[{"role":"user","content":[{"text":"Hello"}]}]}'

# InvokeModel — using IAM credentials (SigV4)
curl -X POST http://localhost:8000/model/anthropic.claude-3-5-sonnet-20241022-v1:0/invoke \
  -H "Content-Type: application/json" \
  -H "X-AWS-Access-Key: $AWS_ACCESS_KEY_ID" \
  -H "X-AWS-Secret-Key: $AWS_SECRET_ACCESS_KEY" \
  -H "X-AWS-Region: us-east-1" \
  -d '{"anthropic_version":"bedrock-2023-05-31","max_tokens":256,"messages":[{"role":"user","content":"Hello"}]}'
```

### Example: route specific Bedrock models to a fixed region with static SigV4 credentials

```python
from fastapi import FastAPI
from gemini_calo.proxy import GeminiProxyService, RouteConfig
from gemini_calo.auth import AWSCredentials, create_aws_sigv4_provider

bedrock_creds = AWSCredentials(
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    region="us-east-1",
)

proxy = GeminiProxyService(
    base_url="https://generativelanguage.googleapis.com",
    api_keys=["gemini-key"],
    model_routes={
        "anthropic.*": RouteConfig(
            url="https://bedrock-runtime.us-east-1.amazonaws.com",
            api_keys=[],
            auth=create_aws_sigv4_provider(bedrock_creds),
        ),
    },
)

app = FastAPI()
app.include_router(proxy.gemini_router)
app.include_router(proxy.openai_router)
app.include_router(proxy.bedrock_router)
```

### Example: per-client credentials with auto-detect (bearer or SigV4)

Use `create_passthrough_bedrock_provider` when each client provides its own credentials and you want the proxy to accept either a Bedrock API key or IAM credentials without additional configuration:

```python
from fastapi import FastAPI
from gemini_calo.proxy import GeminiProxyService, RouteConfig
from gemini_calo.auth import create_passthrough_bedrock_provider

proxy = GeminiProxyService(
    base_url="https://generativelanguage.googleapis.com",
    api_keys=["gemini-key"],
    model_routes={
        "anthropic.*": RouteConfig(
            url="https://bedrock-runtime.ap-southeast-3.amazonaws.com",
            api_keys=[],
            auth=create_passthrough_bedrock_provider(),
        ),
    },
)

app = FastAPI()
app.include_router(proxy.bedrock_router)
```

The provider checks headers in this order: `X-AWS-Bearer-Token` (bearer, no signing) → `X-AWS-Access-Key` + `X-AWS-Secret-Key` (SigV4) → no auth.

## Advanced Authentication: Custom Auth Providers

For providers that require more complex authentication, you can provide a custom auth provider function. This function receives the incoming request and returns an `httpx.Auth` instance.

### Custom Auth Provider Example

You can create your own auth provider for any authentication scheme:

```python
import httpx
from fastapi import Request
from gemini_calo.auth import BearerAuth

async def custom_auth_provider(request: Request) -> httpx.Auth:
    """Example: Extract token from custom header and use as bearer token."""
    custom_token = request.headers.get("X-My-Custom-Token", "default-token")
    return BearerAuth(token=custom_token)

# Use in RouteConfig
route = RouteConfig(
    url="https://api.example.com",
    api_keys=[],
    auth=custom_auth_provider,
)
```

## How the Middleware Works

Middleware in FastAPI are functions that process every request before it reaches the specific path operation and every response before it is sent back to the client. Gemini Calo includes four built-in middlewares, applied in this order by the built-in server:

```
request → logging → auth → model_override → rollup → handler → upstream
```

### Logging Middleware (`logging_middleware`)

Logs every incoming request and outgoing response, including headers and body. Handles both standard and streaming responses. Controlled by `GEMINI_CALO_LOG_LEVEL` and `GEMINI_CALO_LOG_FILE`.

### Authentication Middleware (`auth_middleware`)

Validates the proxy API key on all OpenAI, Anthropic, Gemini, and Bedrock requests. Accepts the key via `Authorization: Bearer <key>` or `x-goog-api-key` header. Returns `401` if the key is missing or invalid. Configured via `GEMINI_CALO_PROXY_API_KEYS`.

### Model Override Middleware (`model_override_middleware`)

Rewrites the model name before the request is forwarded upstream. Works on all three endpoint types:

- **Gemini:** rewrites the model in the URL path (`/v1beta/models/{model}:generateContent`)
- **OpenAI:** rewrites the `model` field in the JSON body
- **Bedrock (InvokeModel & Converse):** rewrites the model ID in the URL path (`/model/{modelId}/invoke`, `/model/{modelId}/converse`, etc.)

Configured via `GEMINI_CALO_MODEL_OVERRIDE` or the `model_transformer` argument.

### Rollup Middleware (`rollup_middleware`)

Caches conversation history in an LRU cache and injects a summary as a system prompt once the conversation exceeds a size threshold, replacing the earlier messages. This keeps context windows manageable for long conversations without losing information.

Supports all three request formats:

| Format | Messages field | System prompt field |
|--------|---------------|---------------------|
| OpenAI | `messages[]` (excludes `role: system`) | `messages[0]` with `role: system` |
| Gemini | `contents[]` | `system_instruction` |
| Bedrock — Anthropic InvokeModel | `messages[]` | `system` (string) |
| Bedrock — Amazon Nova / Converse API | `messages[]` | `system` (array of `{"text": "..."}`) |

The system prompt format for Bedrock is auto-detected from the `messages[].content` shape: array content → Nova/Converse-style array system; string content → Anthropic-style string system. This applies uniformly to InvokeModel (`/invoke`) and Converse API (`/converse`, `/converse-stream`) routes.

Configured via `GEMINI_CALO_CONVERSATION_SUMMARIZATION_LRU_CACHE` and `GEMINI_CALO_CONVERSATION_SIZE_SUMMARIZATION_THRESHOLD`.

### Adding Your Own Middleware

Because Gemini Calo is built on FastAPI, you can easily add your own custom middleware. For example, you could add a middleware for rate limiting, CORS, or custom header injection.

#### Advanced Middleware: Modifying Request Body and Headers

Here is a more advanced example that intercepts a request, modifies its JSON body, adds a new header, and then forwards it to the actual endpoint. This can be useful for injecting default values, adding metadata, or transforming request payloads.

**Important:** Reading the request body consumes it. To allow the endpoint to read the body again, we must reconstruct the request with the modified body.

```python
from fastapi import FastAPI, Request
from starlette.datastructures import MutableHeaders
import json

app = FastAPI()

# This middleware will add a 'user_id' to the request body
# and a 'X-Request-ID' to the headers.
async def modify_request_middleware(request: Request, call_next):
    # Get the original request body
    body = await request.body()
    
    # Modify headers
    request_headers = MutableHeaders(request.headers)
    request_headers["X-Request-ID"] = "some-unique-id"
    
    # Modify body (if it's JSON)
    new_body = body
    if body and request.headers.get("content-type") == "application/json":
        try:
            json_body = json.loads(body)
            # Add or modify a key
            json_body["user_id"] = "injected-user-123"
            new_body = json.dumps(json_body).encode()
        except json.JSONDecodeError:
            # Body is not valid JSON, pass it through
            pass

    # To pass the modified body and headers, we need to create a new Request object.
    # We do this by defining a new 'receive' channel.
    async def receive():
        return {"type": "http.request", "body": new_body, "more_body": False}

    # We replace the original request's scope with the modified headers
    request.scope["headers"] = request_headers.raw

    # Create the new request object and pass it to the next middleware/endpoint
    new_request = Request(request.scope, receive)
    response = await call_next(new_request)
    
    return response

app.middleware("http")(modify_request_middleware)

# ... then add the Gemini Calo proxy and routers as shown above
```

## Client Integrations

Every integration is the same shape: **point the client's base URL at Calo, give it any proxy key, and pick a model that a route matches.** Calo translates and forwards.

The examples below assume the [universal hub](#example-a-universal-hub-over-gemini-bedrock-and-deepseek) running on `http://localhost:8000` with routes for `deepseek-*` (→ DeepSeek), `gemini-*` (→ Gemini), and `amazon.*` / `nova` (→ Bedrock). If `GEMINI_CALO_PROXY_API_KEYS` is unset, Calo accepts any key, so the tokens below can be any non-empty string. Ready-to-run configs for each client live in [`example/`](example/).

### OpenAI SDK (and any OpenAI-compatible client)

Point the OpenAI client at Calo's `/v1` base. The **same** client reaches every provider — just change the model:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my_secret_proxy_key_1",  # any proxy key
)

for model in ["gemini-2.5-flash", "deepseek-chat", "amazon.nova-pro-v1:0"]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one word."}],
    )
    print(model, "→", resp.choices[0].message.content)
```

Tool/function calling and streaming work across providers; Calo normalizes provider-specific quirks so your client code stays unchanged.

### Claude Code (Anthropic Messages)

Claude Code speaks the Anthropic Messages API. Calo's `/v1/messages` endpoint (the `anthropic_router`) translates it to the routed provider, so you can run Claude Code against Gemini, Bedrock, or DeepSeek:

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_AUTH_TOKEN="my_secret_proxy_key_1"      # any proxy key
export ANTHROPIC_MODEL="amazon.nova-pro-v1:0"            # or gemini-2.5-pro, deepseek-chat
export ANTHROPIC_SMALL_FAST_MODEL="amazon.nova-lite-v1:0"

claude              # interactive
claude -p "list the files in this repo"   # headless
```

Make sure the proxy mounts `proxy.anthropic_router`. See [`example/claude-code.sh`](example/claude-code.sh) for a ready-to-source setup that leaves your global Claude Code config untouched.

### zrb

zrb's LLM uses an OpenAI-compatible client, so point it at Calo's `/v1` base and pick any routed model:

```bash
export ZRB_LLM_BASE_URL="http://localhost:8000/v1"
export ZRB_LLM_API_KEY="my_secret_proxy_key_1"   # any proxy key
export ZRB_LLM_MODEL="gemini-2.5-flash"          # or deepseek-chat, nova

zrb llm ask "What is the current weather at my current location?"
# interactive:  zrb llm chat
```

> **Note:** zrb parses a `:` in a model id as `provider:model`, so a Bedrock id like `amazon.nova-pro-v1:0` gets mangled. Use a colon-free route alias — the hub example maps `nova` → `amazon.nova-pro-v1:0` via `upstream_model`.

See [`example/zrb-env.sh`](example/zrb-env.sh).

### opencode

Configured by a **file, no env vars**. Provide an `opencode.json` (in the cwd or
`~/.config/opencode/opencode.json`) registering Calo as an OpenAI-compatible provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "calo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Gemini Calo",
      "options": {
        "baseURL": "http://localhost:8000/v1",
        "apiKey": "my_secret_proxy_key_1"
      },
      "models": {
        "gemini-2.5-pro": {},
        "deepseek-chat": {},
        "amazon.nova-pro-v1:0": {}
      }
    }
  }
}
```

```bash
opencode run --model calo/gemini-2.5-pro "hi"
```

Ready-made: [`example/opencode.json`](example/opencode.json).

### codex

codex speaks the OpenAI **Responses API**. It needs **one config file** plus env
pointing at it. Put a `config.toml` in a `CODEX_HOME` dir so your global
`~/.codex` (config + ChatGPT login) is untouched:

`$CODEX_HOME/config.toml`:

```toml
model = "gemini-2.5-pro"
model_provider = "calo"

[model_providers.calo]
name = "Gemini Calo"
base_url = "http://localhost:8000/v1"
wire_api = "responses"
env_key = "CALO_API_KEY"   # codex reads the proxy key from this env var
```

```bash
export CODEX_HOME="$PWD/codex-home"        # dir holding the config.toml above
export CALO_API_KEY="my_secret_proxy_key_1"
codex exec "hi"
codex exec -m deepseek-chat "hi"
```

Ready-made: [`example/codex-env.sh`](example/codex-env.sh) +
[`example/codex-home/config.toml`](example/codex-home/config.toml).

### Native Gemini / Bedrock SDKs

If a client already speaks a provider's native wire format, use the passthrough routes directly — set the SDK's base URL to Calo and it forwards with auth/logging/rollup applied:

- **google-genai:** `genai.Client(api_key="my_secret_proxy_key_1", http_options=HttpOptions(base_url="http://localhost:8000"))` → hits `/v1beta/models/…`.
- **boto3 `bedrock-runtime`:** point the endpoint at Calo → hits `/model/{modelId}/invoke` or `/converse`. See [AWS Bedrock Endpoint](#aws-bedrock-endpoint).

## Development & Testing

Calo translates between many client protocols and many upstream providers, so
testing happens in **three layers**. Each catches what the layer below cannot:

| Layer | What it runs | Catches | Needs |
|-------|--------------|---------|-------|
| 1. Unit / integration | `pytest` (mocked upstreams) | translation logic, routing, middleware | dev deps |
| 2. Live smoke | `example/smoke_test.py` (real providers) | token caps, tool-schema keywords, streaming framing | real API keys |
| 3. End-to-end clients | `claude` / `opencode` / `codex` / `zrb` headless | the *actual* payloads each client sends, full agentic tool loops | keys + clients installed |

> ⚠️ **Layer 1 alone is not enough.** Mocked tests pass with toy inputs but miss
> real constraints (e.g. Nova's 10240 `max_tokens` cap, Gemini rejecting
> `$schema`/`exclusiveMinimum`, codex requiring `wire_api = "responses"`). After
> any change to the `translate/` layer, run layers 2 and 3.

### Run everything at once

```bash
cd example
cp ../template.env .env   # then fill in real keys (or reuse .env)
./test-all.sh             # layers 1–3, all clients × all providers
./test-all.sh gemini-2.5-flash   # limit to specific model(s)
```

`test-all.sh` runs pytest, starts the hub, runs the live smoke test, then drives
every installed client against each provider with a real tool-using task, and
exits non-zero if anything fails. It touches **no global client config**
(see the per-client setup below).

### Layer 1 — Unit & integration tests

```bash
git clone https://github.com/state-alchemists/gemini-calo.git
cd gemini-calo
pip install -e ".[dev]"

python -m pytest                                   # all tests
python -m pytest tests/test_translate.py -v        # the IR translation layer
python -m pytest --cov=gemini_calo --cov-report=term-missing
```

### Layer 2 — Live smoke test (real providers)

Start the hub, then hit every endpoint/model with **realistic** payloads
(large `max_tokens`, rich JSON-Schema tools, streaming):

```bash
cd example && source .env && python app.py    # terminal 1
python example/smoke_test.py                  # terminal 2 (all models)
python example/smoke_test.py amazon.nova-pro-v1:0   # a single model
```

### Layer 3 — End-to-end with the real clients (ground truth)

Each client is pointed at Calo using **local/env config only — your global
config is never modified**:

| Client | How it's configured | Headless command |
|--------|---------------------|------------------|
| Claude Code | env vars (`source example/claude-code.sh`) | `claude -p "…"` |
| opencode | cwd `opencode.json` with an inlined key | `opencode run --model calo/<model> "…"` |
| codex | local `CODEX_HOME` (`source example/codex-env.sh`) | `codex exec -m <model> "…"` |
| zrb | env vars (`source example/zrb-env.sh`) | `zrb llm chat --interactive false --message "…"` |

Choosing a model per client: set `ANTHROPIC_MODEL` (Claude Code), `--model
calo/<model>` (opencode), `-m <model>` (codex), or `ZRB_LLM_MODEL` (zrb) to any
of `deepseek-chat`, `gemini-2.5-pro`/`gemini-2.5-flash`, `amazon.nova-pro-v1:0`.

### Provider constraints the tests guard

If you add a provider or model, verify these — they are the things that broke in
practice and are now covered:

- **Nova** caps output at **10240** `max_tokens` (Calo clamps it).
- **Gemini** function schemas accept only a Schema subset — `$schema`,
  `additionalProperties`, `exclusiveMinimum`, … are stripped by Calo.
- **codex** ≥ 0.144 requires `wire_api = "responses"` (not `"chat"`).
- **zrb** treats `:` in a model id as `provider:model`; use a colon-free alias
  (the example maps `nova` → `amazon.nova-pro-v1:0` via `upstream_model`).
- **Bedrock streaming** uses AWS binary event-stream framing — requires
  `botocore` (a hard dependency, installed with the package).

### Test Structure

| File | Purpose |
|------|---------|
| `test_translate.py` | IR translation: inbound/outbound adapters, tool calls, streaming, real-client quirks |
| `test_responses.py` | `/v1/responses` handling |
| `test_auth.py` | Proxy authentication middleware tests |
| `test_auth_providers.py` | Auth module tests (Bearer, XGoog, AWS SigV4) |
| `test_bedrock.py` | Bedrock endpoint, auth providers, model override, and rollup tests |
| `test_gzip_handling.py` | Gzip compression handling tests |
| `test_logging.py` | Logging middleware tests |
| `test_main.py` | Main proxy functionality tests |
| `test_model_override.py` | Model override middleware tests |
| `test_model_routes.py` | Model routing and RouteConfig tests |
| `test_rollup.py` | Conversation rollup tests |
