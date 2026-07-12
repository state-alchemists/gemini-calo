#!/bin/bash
# -----------------------------------------------------------
# Codex CLI + Gemini Calo Hub  (no global config touched)
#
# Codex speaks the OpenAI Responses API. We point CODEX_HOME at a local dir
# (example/codex-home) so your real ~/.codex config and login are untouched,
# and codex authenticates to Calo with an API key instead of a ChatGPT account.
#
# Usage:
#   source codex-env.sh
#   codex exec "hello"                         # deepseek-chat (default)
#   codex exec -m gemini-2.5-pro "hello"
#   codex exec -m amazon.nova-pro-v1:0 "hello"
# -----------------------------------------------------------

_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export CODEX_HOME="$_here/codex-home"
export CALO_API_KEY="not-needed"

echo "CODEX_HOME=$CODEX_HOME (global ~/.codex untouched)"
echo "Start the hub first:  source .env && python app.py"
echo "Then:  codex exec \"hello\"   |   codex exec -m gemini-2.5-pro \"hello\""
