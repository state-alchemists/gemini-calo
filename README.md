# Gemini Calo

**Gemini Calo** is a powerful, yet simple, FastAPI-based proxy server for Google's Gemini API. It provides a seamless way to add a layer of authentication, logging, and monitoring to your Gemini API requests. It's designed to be run as a standalone server or integrated into your existing FastAPI applications.

One of its key features is providing an OpenAI-compatible endpoint, allowing you to use Gemini models with tools and libraries that are built for the OpenAI API.

## Key Features

*   **Authentication:** Secure your Gemini API access with an additional layer of API key authentication.
*   **Request Logging:** Detailed logging of all incoming requests and outgoing responses.
*   **OpenAI Compatibility:** Use Gemini models through an OpenAI-compatible `/v1/chat/completions` endpoint.
*   **Round-Robin API Keys:** Distribute your requests across multiple Gemini API keys.
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

*   `GEMINI_CALO_API_KEYS`: A comma-separated list of your Google Gemini API keys. The proxy will rotate through these keys for outgoing requests. If this is not set, the proxy will not be able to make requests to the Gemini API.
*   `GEMINI_CALO_PROXY_API_KEYS`: (Optional) A comma-separated list of API keys that clients must provide to use the proxy. If not set, the proxy will be open to anyone, meaning no API key is required for access.
*   `GEMINI_CALO_HTTP_PORT`: The port on which the server will run. Defaults to `8000`.
*   `GEMINI_CALO_LOG_LEVEL`: Sets the logging level for the application. Options include `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. If not set, it defaults to `CRITICAL`.
*   `GEMINI_CALO_LOG_FILE`: Specifies the file where logs will be written. Defaults to `app.log`.
*   `GEMINI_CALO_CONVERSATION_SUMMARIZATION_LRU_CACHE`: Sets the size of the LRU cache for conversation summarization. Defaults to `20`.
*   `GEMINI_CALO_MODEL_OVERRIDE`: Allows you to specify a model to override the default Gemini model.
*   `PY_IGNORE_IMPORTMISMATCH`: (Used by `pydantic-ai`) If set, ignores import mismatches.
*   `PYTEST_THEME`: (Used by `pytest-sugar`) Sets the theme for pytest output.
*   `PYTEST_THEME_MODE`: (Used by `pytest-sugar`) Sets the theme mode (e.g., `dark`) for pytest output.
*   `XDG_DATA_HOME`, `XDG_CONFIG_HOME`, `XDG_CONFIG_DIRS`, `XDG_CACHE_HOME`, `XDG_STATE_HOME`: Standard XDG base directory environment variables.
*   `DISTUTILS_USE_SDK`: (Used by `setuptools`) If set, indicates whether to use the SDK.
*   `SETUPTOOLS_EXT_SUFFIX`: (Used by `setuptools`) Specifies the suffix for extension modules.
*   `PYDANTIC_PRIVATE_ALLOW_UNHANDLED_SCHEMA_TYPES`: (Used by `pydantic`) Allows unhandled schema types.
*   `PYDANTIC_DISABLE_PLUGINS`: (Used by `pydantic`) Disables pydantic plugins.
*   `PYDANTIC_VALIDATE_CORE_SCHEMAS`: (Used by `pydantic`) Enables validation of core schemas.
*   `EXCEPTIONGROUP_NO_PATCH`: (Used by `exceptiongroup`) If set, prevents patching of the `ExceptionGroup` class.

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
gemini_api_keys = os.getenv("GEMINI_CALO_API_KEYS", "").split(",")
proxy_service = GeminiProxyService(gemini_api_keys=gemini_api_keys)

# 2. (Optional) Add Authentication Middleware
proxy_api_keys = os.getenv("GEMINI_CALO_PROXY_API_KEYS", "").split(",")
if proxy_api_keys:
    auth_middleware_with_keys = partial(auth_middleware, user_api_key_checker=proxy_api_keys)
    app.middleware("http")(auth_middleware_with_keys)

# 3. (Optional) Add Logging Middleware
app.middleware("http")(logging_middleware)

# 4. Mount the Gemini and OpenAI routers
app.include_router(proxy_service.gemini_router)
app.include_router(proxy_service.openai_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Now you can run your app as usual with uvicorn
# uvicorn your_app_file:app --reload
```

## How the Middleware Works

Middleware in FastAPI are functions that process every request before it reaches the specific path operation and every response before it is sent back to the client. Gemini Calo includes two useful middlewares by default.

### Logging Middleware (`logging_middleware`)

This middleware logs the details of every incoming request and outgoing response, including headers and body content. This is invaluable for debugging and monitoring. It's designed to handle both standard and streaming responses correctly.

### Authentication Middleware (`auth_middleware`)

This middleware protects your proxy endpoints. It checks for an API key in the `Authorization` header (as a Bearer token) or the `x-goog-api-key` header. It then validates this key against the list of keys you provided in the `GEMINI_CALO_PROXY_API_KEYS` environment variable. If the key is missing or invalid, it returns a `401 Unauthorized` error.

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
zrb llm ask "What is the current weather at my current location?"
```