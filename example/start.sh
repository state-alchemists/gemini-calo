#!/bin/bash
# -----------------------------------------------------------
# Quick-start: boot Gemini Calo Hub and print test commands
# -----------------------------------------------------------
set -e
cd "$(dirname "$0")"

if [ ! -f .env ]; then
  echo "Missing example/.env — copy from template.env and fill in your keys"
  exit 1
fi

source .env
echo "Starting Gemini Calo Hub on port ${GEMINI_CALO_HTTP_PORT:-8000}..."
echo ""
echo "Providers:"
[ -n "$DEEPSEEK_API_KEY" ]       && echo "  deepseek  -> api.deepseek.com"
[ -n "$GEMINI_API_KEY" ]         && echo "  gemini    -> generativelanguage.googleapis.com"
[ -n "$BEDROCK_BEARER_TOKEN" ]   && echo "  bedrock   -> bedrock-runtime.${BEDROCK_REGION:-us-east-1}.amazonaws.com"
echo ""
echo "Test it:"
echo "  curl http://localhost:${GEMINI_CALO_HTTP_PORT:-8000}/"
echo "  curl http://localhost:${GEMINI_CALO_HTTP_PORT:-8000}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"deepseek-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo ""
echo "Clients (all local/env config — nothing global is touched):"
echo "  opencode:  run from a dir containing opencode.json:  opencode run --model calo/deepseek-chat \"hi\""
echo "  codex:     source codex-env.sh   (local CODEX_HOME)   then: codex exec \"hi\""
echo "  claude:    source claude-code.sh (env vars)           then: claude \"hi\""
echo "  zrb:       source zrb-env.sh      (env vars)           then: zrb llm chat --interactive false --message \"hi\""
echo ""
echo "Test everything (all clients x providers):  ./test-all.sh"
echo ""

python app.py
