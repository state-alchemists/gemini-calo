#!/bin/bash
# -----------------------------------------------------------
# Zrb + Gemini Calo Hub  (env only — no global config touched)
#
# Zrb's LLM uses the OpenAI-compatible client, so point it at Calo's /v1 base
# and pick any routed model. Calo translates to the backend's native API.
#
# NOTE: zrb parses a ":" in the model name as "provider:model", so a Bedrock id
# like amazon.nova-pro-v1:0 gets mangled. Use the colon-free alias "nova"
# (defined in app.py, mapped to the real id via upstream_model) instead.
#
# Usage:
#   source zrb-env.sh
#   zrb llm chat --interactive false --message "hello"
#   ZRB_LLM_MODEL=gemini-2.5-flash  zrb llm chat --interactive false --message "hello"
#   ZRB_LLM_MODEL=nova              zrb llm chat --interactive false --message "hello"
# -----------------------------------------------------------

# ZRB_INIT_SCRIPTS="": ignore any ambient user init script (e.g. a personal
# ~/zrb_init.py pulling in unrelated deps) so zrb starts clean against Calo.
export ZRB_INIT_SCRIPTS=""
export ZRB_LLM_BASE_URL="http://localhost:8000/v1"
export ZRB_LLM_API_KEY="not-needed"
export ZRB_LLM_MODEL="${ZRB_LLM_MODEL:-deepseek-chat}"

echo "ZRB_LLM_BASE_URL=$ZRB_LLM_BASE_URL  ZRB_LLM_MODEL=$ZRB_LLM_MODEL"
echo "Start the hub first:  source .env && python app.py"
