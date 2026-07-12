#!/bin/bash
# -----------------------------------------------------------
# Claude Code + Gemini Calo Hub
#
# Claude Code speaks the Anthropic Messages API (/v1/messages).
# Calo now exposes that endpoint and translates it to the routed
# model's native protocol (e.g. Bedrock Nova via bedrock-invoke).
#
#   source claude-code.sh
#   claude
# -----------------------------------------------------------

# Point Claude Code at the local Calo proxy.
export ANTHROPIC_BASE_URL="http://localhost:8000"

# Calo doesn't validate proxy-level keys by default, so any
# non-empty string works here.
export ANTHROPIC_AUTH_TOKEN="calo-proxy"

# Choose which model Calo should route to. Claude Code only speaks the
# Anthropic protocol; Calo translates it to the routed model's native API.
# Any of the three providers work — set ANTHROPIC_MODEL to one of:
#   amazon.nova-pro-v1:0   (Bedrock, via amazon.* -> bedrock-invoke)
#   gemini-2.5-pro         (Gemini, via gemini-* -> gemini)
#   deepseek-chat          (DeepSeek, via deepseek-* -> openai-chat)
export ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-amazon.nova-pro-v1:0}"
export ANTHROPIC_SMALL_FAST_MODEL="${ANTHROPIC_SMALL_FAST_MODEL:-amazon.nova-lite-v1:0}"

echo "Claude Code env vars set. Start the Calo hub first:"
echo "  cd $(dirname "$0") && source .env && python app.py"
