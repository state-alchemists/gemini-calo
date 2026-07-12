"""Live smoke test against a running Calo hub (real providers).

Unlike the pytest suite (which mocks upstreams), this drives the *real*
DeepSeek / Gemini / Bedrock backends with REALISTIC payloads — big max_tokens
and rich JSON-Schema tools — the kinds of inputs that actually break
translation. Run it after any change to the translate layer.

Usage:
    # 1. start the hub
    cd example && source .env && python app.py
    # 2. in another shell
    python example/smoke_test.py                 # all models, all endpoints
    python example/smoke_test.py deepseek-chat    # a single model

Exit code is non-zero if any check fails.
"""

import sys
import json
import httpx

BASE = "http://localhost:8000"
ALL_MODELS = ["deepseek-chat", "gemini-2.5-flash", "amazon.nova-pro-v1:0"]

# Rich tool schema like opencode / Claude Code emit (draft-07 keywords that
# some providers reject unless the proxy sanitises them).
_PARAMS = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "city": {"type": "string", "description": "City name"},
        "days": {"type": "integer", "exclusiveMinimum": 0, "maximum": 14},
    },
    "required": ["city"],
}
OA_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": _PARAMS}}]
ANTH_TOOLS = [{"name": "get_weather", "description": "Weather", "input_schema": _PARAMS}]
RESP_TOOLS = [{"type": "function", "name": "get_weather", "description": "Weather", "parameters": _PARAMS}]

BIG = 32000  # deliberately large; providers with lower caps must be clamped
_failures = 0


def check(label, ok, detail=""):
    global _failures
    if not ok:
        _failures += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{': ' + detail if detail else ''}")


def post(path, body):
    with httpx.Client(timeout=90) as c:
        return c.post(f"{BASE}{path}", json=body)


def sse_texts(body, extract):
    out = []
    for ln in body.splitlines():
        if ln.startswith("data: ") and "[DONE]" not in ln:
            try:
                out.append(extract(json.loads(ln[6:])))
            except Exception:
                pass
    return [x for x in out if x]


def check_model(m):
    print(f"\n=== {m} ===")
    # 1. chat completions + big max_tokens + rich tools (opencode path)
    r = post("/v1/chat/completions", {"model": m, "max_tokens": BIG, "tools": OA_TOOLS, "tool_choice": "auto",
             "messages": [{"role": "user", "content": "What's the weather in SF? Use the tool."}]})
    tc = r.json()["choices"][0]["message"].get("tool_calls") if r.status_code == 200 else None
    check("chat + tools", bool(tc), tc[0]["function"]["name"] if tc else f"{r.status_code} {r.text[:120]}")

    # 2. chat streaming + tools (the opencode 'hang' regression)
    r = post("/v1/chat/completions", {"model": m, "max_tokens": BIG, "tools": OA_TOOLS, "stream": True,
             "messages": [{"role": "user", "content": "Weather in SF? Use the tool."}]})
    names = sse_texts(r.text, lambda d: (d["choices"][0]["delta"].get("tool_calls") or [{}])[0].get("function", {}).get("name"))
    check("chat + tools (stream)", r.status_code == 200 and "get_weather" in names, f"{r.status_code}")

    # 3. /v1/messages + tools (Claude Code path)
    r = post("/v1/messages", {"model": m, "max_tokens": BIG, "tools": ANTH_TOOLS, "system": [{"type": "text", "text": "be brief"}],
             "messages": [{"role": "user", "content": "Weather in SF? Use the tool."}]})
    tu = [b for b in r.json().get("content", [])] if r.status_code == 200 else []
    check("messages + tools", any(b.get("type") == "tool_use" for b in tu), f"{r.status_code} {r.text[:120]}")

    # 4. /v1/responses + tools (codex path)
    r = post("/v1/responses", {"model": m, "max_output_tokens": BIG, "tools": RESP_TOOLS,
             "input": "Weather in SF? Use the tool."})
    fc = [o for o in r.json().get("output", [])] if r.status_code == 200 else []
    check("responses + tools", any(o.get("type") == "function_call" for o in fc), f"{r.status_code} {r.text[:120]}")


def main():
    models = sys.argv[1:] or ALL_MODELS
    try:
        httpx.get(f"{BASE}/", timeout=5)
    except Exception:
        print(f"Hub not reachable at {BASE}. Start it: cd example && source .env && python app.py")
        sys.exit(2)
    for m in models:
        check_model(m)
    print(f"\n{'ALL PASSED' if not _failures else f'{_failures} FAILED'}")
    sys.exit(1 if _failures else 0)


if __name__ == "__main__":
    main()
