#!/usr/bin/env bash
# ===========================================================================
# Gemini Calo — end-to-end test harness
#
# Runs, in order:
#   1. pytest (unit/integration, mocked upstreams)
#   2. starts the Calo hub against your real provider keys (example/.env)
#   3. live smoke test (real providers, realistic tool/token payloads)
#   4. every installed client (claude, opencode, codex, zrb) driving a real
#      agentic tool loop against each of the three providers
#   5. tears the hub down
#
# No global client config is touched: claude & zrb use env vars, codex uses a
# local CODEX_HOME (example/codex-home), opencode uses a cwd opencode.json with
# an inlined key.
#
# Usage:
#   ./test-all.sh                         # all models, all clients
#   ./test-all.sh gemini-2.5-flash        # restrict to one or more models
# ===========================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
[ -x "$PY" ] || PY="python3"
PORT="${GEMINI_CALO_HTTP_PORT:-8000}"
BASE="http://localhost:$PORT"
TASK="How many files are in the current directory? Use your tools, then answer in one short sentence."
# A good answer mentions files / a directory / a number, and isn't an error.
GOOD='(file|director|[0-9])'
# Error markers. Status codes are matched only in an explicit error context
# (e.g. "status_code: 404", 'HTTP/1.1 404') so a random id like "soft-face-4403"
# is not treated as a 403.
BAD='(error|exception|traceback|not supported|unknown model|unknown provider|malformed|invalid_request|bad request|status_code: [45][0-9][0-9]|HTTP/[0-9.]+" [45][0-9][0-9]| [45][0-9][0-9] (not found|bad request|too many|internal server))'

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(deepseek-chat gemini-2.5-flash amazon.nova-pro-v1:0)

PASS=0; FAIL=0; SKIP=0
pass(){ printf '  \033[32m✅ PASS\033[0m %s\n' "$1"; PASS=$((PASS+1)); }
fail(){ printf '  \033[31m❌ FAIL\033[0m %s%s\n' "$1" "${2:+ — $2}"; FAIL=$((FAIL+1)); }
skip(){ printf '  \033[33m⏭  SKIP\033[0m %s%s\n' "$1" "${2:+ — $2}"; SKIP=$((SKIP+1)); }
section(){ printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
have(){ command -v "$1" >/dev/null 2>&1; }
clean(){ sed 's/\x1b\[[0-9;]*m//g; s/\x1b\][0-9;]*//g' | tr -d '\000'; }

# Assert client output looks like a successful agentic answer.
judge(){ # label, output
  local label="$1" out; out="$(printf '%s' "$2" | clean)"
  if printf '%s' "$out" | grep -qiE "$BAD"; then
    fail "$label" "$(printf '%s' "$out" | grep -iE "$BAD" | head -1 | cut -c1-100)"
  elif printf '%s' "$out" | grep -qiE "$GOOD"; then
    pass "$label — $(printf '%s' "$out" | grep -iE "$GOOD" | tail -1 | cut -c1-70)"
  else
    fail "$label" "no answer (got: $(printf '%s' "$out" | tail -1 | cut -c1-80))"
  fi
}

# Scratch workspace with a few files for "list files" tasks.
WS="$(mktemp -d)"
printf 'hello\n' > "$WS/a.txt"; printf 'x\ny\n' > "$WS/b.txt"
mkdir -p "$WS/sub"; printf 'z\n' > "$WS/sub/c.txt"
cp "$HERE/opencode.json" "$WS/opencode.json"

HUB_PID=""
cleanup(){ [ -n "$HUB_PID" ] && kill "$HUB_PID" 2>/dev/null; rm -rf "$WS"; }
trap cleanup EXIT

printf '\033[1mGemini Calo test-all\033[0m — models: %s\n' "${MODELS[*]}"

# --- 1. pytest -------------------------------------------------------------
section "1. pytest (mocked)"
if "$PY" -m pytest "$ROOT/tests" -q >/tmp/calo-pytest.log 2>&1; then
  pass "pytest ($(grep -oE '[0-9]+ passed' /tmp/calo-pytest.log | tail -1))"
else
  fail "pytest" "see /tmp/calo-pytest.log"; tail -5 /tmp/calo-pytest.log
fi

# --- 2. start hub ----------------------------------------------------------
section "2. Start Calo hub"
if [ ! -f "$HERE/.env" ]; then
  fail "hub" "missing example/.env — copy template.env and add your keys"
  printf '\nSUMMARY: %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"; exit 1
fi
set -a; . "$HERE/.env"; set +a
lsof -ti:"$PORT" 2>/dev/null | xargs kill -9 2>/dev/null
( cd "$HERE" && exec "$PY" app.py ) >/tmp/calo-hub.log 2>&1 &
HUB_PID=$!
for _ in $(seq 1 20); do sleep 1; curl -sf "$BASE/" >/dev/null 2>&1 && break; done
if curl -sf "$BASE/" >/dev/null 2>&1; then
  pass "hub up on $BASE ($(curl -s "$BASE/"))"
else
  fail "hub" "did not start; see /tmp/calo-hub.log"; tail -15 /tmp/calo-hub.log
  printf '\nSUMMARY: %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"; exit 1
fi

# --- 3. live smoke ---------------------------------------------------------
section "3. Live smoke (real providers, realistic payloads)"
if "$PY" "$HERE/smoke_test.py" "${MODELS[@]}" >/tmp/calo-smoke.log 2>&1; then
  pass "smoke_test.py (tools + streaming, all endpoints)"
else
  fail "smoke_test.py"; grep -E '\[FAIL\]' /tmp/calo-smoke.log | head -20
fi

# --- 4. headless clients ---------------------------------------------------
section "4. Clients × providers (real agentic tool loop)"

if have claude; then
  for M in "${MODELS[@]}"; do
    out="$(cd "$WS" && ANTHROPIC_BASE_URL="$BASE" ANTHROPIC_AUTH_TOKEN=calo \
      ANTHROPIC_MODEL="$M" ANTHROPIC_SMALL_FAST_MODEL="$M" \
      claude -p --dangerously-skip-permissions "$TASK" 2>&1)"
    judge "claude · $M" "$out"
  done
else skip "claude (all models)" "claude not installed"; fi

if have opencode; then
  for M in "${MODELS[@]}"; do
    out="$(cd "$WS" && opencode run --model "calo/$M" "$TASK" 2>&1)"
    judge "opencode · $M" "$out"
  done
else skip "opencode (all models)" "opencode not installed"; fi

if have codex; then
  # Use a throwaway copy so codex's per-run [projects.*] trust writes don't
  # dirty the committed example/codex-home/config.toml.
  cp -R "$HERE/codex-home" "$WS/codex-home"
  export CODEX_HOME="$WS/codex-home" CALO_API_KEY="not-needed"
  for M in "${MODELS[@]}"; do
    # --dangerously-bypass-approvals-and-sandbox: run tools non-interactively.
    # Safe here — the task runs in a throwaway temp workspace.
    out="$(cd "$WS" && codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox -m "$M" "$TASK" 2>&1)"
    judge "codex · $M" "$out"
  done
else skip "codex (all models)" "codex not installed"; fi

if have zrb; then
  # ZRB_INIT_SCRIPTS="": ignore any ambient user init script (e.g. a personal
  # ~/zrb_init.py pulling in unrelated deps) so zrb starts clean against Calo.
  export ZRB_LLM_BASE_URL="$BASE/v1" ZRB_LLM_API_KEY="not-needed" ZRB_INIT_SCRIPTS=""
  for M in "${MODELS[@]}"; do
    # zrb treats ":" in a model id as provider:model, so use the colon-free
    # "nova" alias for Bedrock ids (see app.py / zrb-env.sh).
    ZM="$M"; case "$M" in amazon.nova*) ZM="nova";; esac
    out="$(cd "$WS" && ZRB_LLM_MODEL="$ZM" zrb llm chat --interactive false --yolo true --message "$TASK" 2>&1)"
    judge "zrb · $M (as '$ZM')" "$out"
  done
else skip "zrb (all models)" "zrb not installed"; fi

# --- summary ---------------------------------------------------------------
printf '\n\033[1mSUMMARY:\033[0m \033[32m%d passed\033[0m, \033[31m%d failed\033[0m, \033[33m%d skipped\033[0m\n' "$PASS" "$FAIL" "$SKIP"
[ "$FAIL" -eq 0 ]
