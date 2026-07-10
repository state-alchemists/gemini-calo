# Gemini Calo

**Gemini Calo** is a powerful, yet simple, FastAPI-based proxy server for Google's Gemini API. It provides a seamless way to add a layer of authentication, logging, and monitoring to your Gemini API requests. It's designed to be run as a standalone server or integrated into your existing FastAPI applications.

One of its key features is providing an OpenAI-compatible endpoint, allowing you to use Gemini models with tools and libraries that are built for the OpenAI API. It also exposes native **AWS Bedrock-compatible endpoints** (InvokeModel and Converse APIs), so any client that targets `bedrock-runtime` works without changes.

## Key Features

*   **Authentication:** Secure your Gemini API access with an additional layer of API key authentication.
*   **Request Logging:** Detailed logging of all incoming requests and outgoing responses.
*   **OpenAI Compatibility:** Use Gemini models through an OpenAI-compatible `/v1/chat/completions` endpoint.
*   **AWS Bedrock Compatibility:** Native Bedrock endpoints covering both the InvokeModel API (`/model/{modelId}/invoke`, `/model/{modelId}/invoke-with-response-stream`) and the Converse API (`/model/{modelId}/converse`, `/model/{modelId}/converse-stream`) — supports both Bedrock API key (bearer token) and SigV4 signing.
*   **Round-Robin API Keys:** Distribute your requests across multiple API keys, both globally and per model route.
*   **Multi-Provider Routing:** Route specific models (via glob patterns) to different upstream providers — use OpenAI, Anthropic, AWS Bedrock, or any OpenAI-compatible endpoint alongside Gemini.
*   **Extensible Authentication:** Support for complex auth schemes like AWS SigV4, OAuth, or custom providers via pluggable auth modules.
*   **Easy Integration:** Use it as a standalone server or mount it into your existing FastAPI project.
*   **Extensible:** Easily add your own custom middleware to suit your needs.

## How It's Useful

-   **Centralized API Key Management:** Instead of hardcoding your Gemini API keys in various clients, you can manage them in one place.
-   **Security:** Protect your expensive Gemini API keys by exposing only a proxy key to your users or client applications.
-   **Monitoring & Observability:** The logging middleware gives you insight into how your API is being used, helping you debug issues and monitor usage patterns.
-   **Seamless Migration:** If you have existing tools that use the OpenAI API, you can switch to using Google's Gemini models without significant code changes.

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

# 4. Mount the Gemini, OpenAI, and Bedrock routers
app.include_router(proxy_service.gemini_router)
app.include_router(proxy_service.openai_router)
app.include_router(proxy_service.bedrock_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Now you can run your app as usual with uvicorn
# uvicorn your_app_file:app --reload
```

## Routing Models to Different Providers

`GeminiProxyService` supports a `model_routes` parameter — a `dict` that maps glob patterns to a `RouteConfig`. When a request arrives, the proxy extracts the model name (from the URL path for Gemini-format requests, or from the JSON body for OpenAI-format requests) and checks it against each pattern in order. The first match wins; unmatched models fall back to `base_url` + `api_keys`.

### `RouteConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | — | Upstream base URL for this route |
| `api_keys` | `list[str]` | `[]` | Keys rotated round-robin for preset auth types |
| `auth` | `str` \| `callable` \| `None` | `"bearer"` | Authentication configuration (see below) |
| `timeout` | `float` | `300.0` | Per-request timeout in seconds |
| `auth_type` | `"bearer"` \| `"x-goog-api-key"` | — | **Deprecated:** Use `auth` instead |

### Authentication Configuration

The `auth` field supports multiple authentication modes:

| Value | Description |
|-------|-------------|
| `"bearer"` | Uses `api_keys` with `Authorization: Bearer <key>` header (round-robin) |
| `"x-goog-api-key"` | Uses `api_keys` with `x-goog-api-key` header (round-robin) |
| `"none"` or `None` | No authentication headers added |
| `callable` | Custom auth provider function for advanced scenarios |

### Example: mixing Gemini and OpenAI

```python
import os
from fastapi import FastAPI
from gemini_calo.proxy import GeminiProxyService, RouteConfig

app = FastAPI()

proxy = GeminiProxyService(
    base_url="https://generativelanguage.googleapis.com",
    api_keys=["gemini-key-1", "gemini-key-2"],  # default: round-robined for unmatched models
    model_routes={
        # Glob pattern → RouteConfig
        "gpt-4*": RouteConfig(
            url="https://api.openai.com",
            api_keys=["openai-key-1", "openai-key-2"],
            auth="bearer",
        ),
        "claude-*": RouteConfig(
            url="https://api.anthropic.com",
            api_keys=["anthropic-key-1"],
            auth="bearer",
            timeout=600.0,
        ),
        # Gemini requests not matched above use base_url + api_keys
    },
)

app.include_router(proxy.gemini_router)
app.include_router(proxy.openai_router)
app.include_router(proxy.bedrock_router)
```

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

Validates the proxy API key on all Gemini, OpenAI, and Bedrock requests. Accepts the key via `Authorization: Bearer <key>` or `x-goog-api-key` header. Returns `401` if the key is missing or invalid. Configured via `GEMINI_CALO_PROXY_API_KEYS`.

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

## Integration with Zrb

Suppose you run Gemini Calo with the following configuration, then you will have Gemini Calo run on `http://localhost:8080`.

```bash
# Your gemini API Keys
export GEMINI_CALO_API_KEYS=AIaYourGeminiKey1,AIaYourGeminiKey2
# API Keys for your internal user
export GEMINI_CALO_PROXY_API_KEYS=my_secret_proxy_key_1,my_secret_proxy_key_2
# Gemini Calo HTTP Port
export GEMINI_CALO_HTTP_PORT=8080

# Start Gemini Calo
gemini-calo
```

### Integration Using OpenAI Compatibility Layer

To use OpenAI compatibility layer with Zrb, you need to set some environment variables.

```bash
# OpenAI compatibility URL
export ZRB_LLM_BASE_URL=http://localhost:8080/v1beta/openai/
# One of your valid API Key for internal user
export ZRB_LLM_API_KEY=my_secret_proxy_key_1
# The model you want to use
export ZRB_LLM_MODEL=gemini-2.5-flash

# Run `zrb llm ask` or `zrb llm chat`
zrb llm ask "What is the current weather at my current location?"
```

### Integration Using Gemini Endpoint

To use Gemini Endpoint, you will need to edit or create `zrb_init.py`

```python
from google import genai
from google.genai.types import HttpOptions
from pydantic_ai.models.gemini import GeminiModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from zrb import llm_config

client = genai.Client(
    api_key="my_secret_proxy_key_1",  # One of your valid API Key for internal user
    http_options=HttpOptions(
        base_url="http://localhost:8080",
    ),
)

provider = GoogleProvider(client=client)

model = GoogleModel(
    model_name="gemini-2.5-flash",
    provider=provider,
    settings=GeminiModelSettings(
        temperature=0.0,
        gemini_safety_settings=[
            # Let's become evil 😈😈😈
            # https://ai.google.dev/gemini-api/docs/safety-settings
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                "threshold": "BLOCK_NONE",
            },
        ]
    )
)
llm_config.set_default_model(model)
```

Once you set up everything, you can start interacting with Zrb.

```bash
# Run `zrb llm ask` or `zrb llm chat`
zrb chat "What is the current weather at my current location?"
```

## Development & Testing

### Running Tests

Clone the repository and install development dependencies:

```bash
git clone https://github.com/state-alchemists/gemini-calo.git
cd gemini-calo
pip install -e ".[dev]"
```

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_auth_providers.py -v

# Run specific test
python -m pytest tests/test_auth_providers.py::test_create_bearer_provider_rotates_keys -v
```

### Code Coverage

Run tests with coverage report:

```bash
# Run with coverage
python -m pytest tests/ --cov=gemini_calo --cov-report=term-missing

# Generate HTML coverage report
python -m pytest tests/ --cov=gemini_calo --cov-report=html
open htmlcov/index.html
```

Current coverage: **~82%**

### Optional Dependencies

For AWS SigV4 authentication tests, install `botocore`:

```bash
pip install botocore
```

Tests that require `botocore` are automatically skipped if it's not installed.

### Test Structure

| File | Purpose |
|------|---------|
| `test_auth.py` | Proxy authentication middleware tests |
| `test_auth_providers.py` | Auth module tests (Bearer, XGoog, AWS SigV4) |
| `test_bedrock.py` | Bedrock endpoint, auth providers, model override, and rollup tests |
| `test_gzip_handling.py` | Gzip compression handling tests |
| `test_logging.py` | Logging middleware tests |
| `test_main.py` | Main proxy functionality tests |
| `test_model_override.py` | Model override middleware tests |
| `test_model_routes.py` | Model routing and RouteConfig tests |
| `test_rollup.py` | Conversation rollup tests |
