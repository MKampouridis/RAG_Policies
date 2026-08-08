#!/bin/zsh
# Run an eval end-to-end with the correct server topology.
#
#   ./eval_session.sh <output_name> [questions.json ...]
#   ./eval_session.sh postfix eval/questions.json eval/questions_set2.json
#
# WHY THIS EXISTS
# The eval drives a server over HTTP, so one server = one configuration. Two
# invariants have to hold together and each was violated at least once before
# this script existed (2026-08-07/08):
#
#   1. PRODUCTION MAY USE CLOUD MODELS; THE EVAL MUST NOT. Cloud calls cannot be
#      temperature-pinned (Sonnet 5 rejects the parameter), and every number in
#      eval/report.md was measured on the local models. Pointing the eval at
#      production's port silently measures a different system - and an eval run
#      against a non-deterministic server had to be thrown away entirely.
#   2. PRODUCTION MUST BE STOPPED WHILE THE EVAL RUNS. Each server instance
#      costs ~5GB (its own ColBERT reranker + transformers + Chroma client) on a
#      16GB machine. Running both drove swap from 7GB to 12GB.
#
# It also handles two operational traps: production is launchd-managed with
# KeepAlive, so a plain `kill` respawns it (must `launchctl unload`); and long
# runs must be detached, because harness-tracked background jobs get killed
# around 60-80 minutes. run_eval.py resumes from a partial results file, so an
# interrupted run is cheap to finish - re-run the same command.
#
# Env passthrough: JUDGE_PROVIDER=anthropic ./eval_session.sh <name>  (frontier
# judge; never set it globally - eval/rejudge.py pins a judge deliberately).

set -e
cd "$(dirname "$0")"

NAME="${1:?usage: ./eval_session.sh <output_name> [questions.json ...]}"
shift
QUESTION_SETS=("${@:-eval/questions.json}")

PLIST="$HOME/Library/LaunchAgents/com.mkampo.ragpolicies.plist"
EVAL_PORT="${EVAL_PORT:-8001}"
PIDFILE="/tmp/rag_eval_server_${EVAL_PORT}.pid"

restore_production() {
  [ -f "$PIDFILE" ] && { kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; }
  if [ -f "$PLIST" ]; then
    echo ">> restarting production"
    launchctl load "$PLIST" 2>/dev/null || true
  fi
}
# restore production however we exit - including failure or Ctrl-C, so a crashed
# eval never leaves the assistant down
trap restore_production EXIT INT TERM

echo ">> stopping production (frees ~5GB; launchd KeepAlive means unload, not kill)"
[ -f "$PLIST" ] && launchctl unload "$PLIST" 2>/dev/null || true
sleep 3

# A stale server on this port is the dangerous case, not a harmless one: the new
# process fails to bind and dies, but the health check below is satisfied by the
# OLD server, so the eval runs against whatever that was configured with and
# writes results labelled as this run. Caught exactly this in testing - a
# leftover GENERATOR_PROVIDER=anthropic server answered a "local" smoke test.
# Refuse to continue rather than silently measure the wrong system.
if lsof -nP -iTCP:"$EVAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "!! something is already listening on :$EVAL_PORT - refusing to run."
  echo "   A stale server would silently serve this eval with ITS configuration."
  lsof -nP -iTCP:"$EVAL_PORT" -sTCP:LISTEN | tail -n +2 | sed 's/^/   /'
  echo "   Stop it first:  kill \$(lsof -t -iTCP:$EVAL_PORT -sTCP:LISTEN)"
  exit 1
fi

# Local by default; cloud components only when EXPLICITLY requested, e.g.
#   EVAL_GENERATOR=anthropic EVAL_GENERATOR_MODEL=claude-haiku-4-5 ./eval_session.sh haiku_gen ...
# The distinction that matters is deliberate-vs-accidental: an ambient
# GENERATOR_PROVIDER in the shell must NOT silently change what is measured, so
# only these EVAL_-prefixed variables are honoured and the effective
# configuration is printed before anything runs.
CFG="local generator, local contextualizer"
GEN_ENV=(); CTX_ENV=()
if [ -n "$EVAL_GENERATOR" ]; then
  GEN_ENV=(GENERATOR_PROVIDER="$EVAL_GENERATOR")
  [ -n "$EVAL_GENERATOR_MODEL" ] && GEN_ENV+=(GENERATOR_MODEL="$EVAL_GENERATOR_MODEL")
  CFG="generator=${EVAL_GENERATOR}${EVAL_GENERATOR_MODEL:+/$EVAL_GENERATOR_MODEL}, local contextualizer"
fi
if [ -n "$EVAL_CONTEXTUALIZER" ]; then
  CTX_ENV=(CONTEXTUALIZE_PROVIDER="$EVAL_CONTEXTUALIZER")
  [ -n "$EVAL_CONTEXTUALIZER_MODEL" ] && CTX_ENV+=(ANTHROPIC_CONTEXTUALIZE_MODEL="$EVAL_CONTEXTUALIZER_MODEL")
  CFG="${CFG%, local contextualizer}, contextualizer=${EVAL_CONTEXTUALIZER}${EVAL_CONTEXTUALIZER_MODEL:+/$EVAL_CONTEXTUALIZER_MODEL}"
fi
if [ -n "$EVAL_GENERATOR$EVAL_CONTEXTUALIZER" ]; then
  echo "!! NOTE: cloud components requested - this run is NOT reproducible"
  echo "   (cloud calls cannot be temperature-pinned) and its absolute numbers"
  echo "   are not comparable to the local ledger. Compare like-for-like only."
fi

echo ">> starting eval server on :$EVAL_PORT  [$CFG, deterministic]"
env PORT="$EVAL_PORT" HOST=127.0.0.1 RAG_DETERMINISTIC=1 "${GEN_ENV[@]}" "${CTX_ENV[@]}" \
    .venv/bin/python3 run_server.py > "data/server_eval${EVAL_PORT}.log" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PIDFILE"

for i in {1..30}; do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "!! eval server process died on startup:"; tail -5 "data/server_eval${EVAL_PORT}.log"; exit 1; }
  curl -sf -o /dev/null "http://127.0.0.1:${EVAL_PORT}/" && break
  sleep 2
done
curl -sf -o /dev/null "http://127.0.0.1:${EVAL_PORT}/" || { echo "!! eval server never came up"; exit 1; }

# confirm the server answering us is OURS and is configured local+deterministic
SERVED_BY=$(lsof -t -iTCP:"$EVAL_PORT" -sTCP:LISTEN 2>/dev/null | head -1)
[ "$SERVED_BY" = "$SERVER_PID" ] || { echo "!! :$EVAL_PORT is served by pid $SERVED_BY, not ours ($SERVER_PID)"; exit 1; }
# guard against LEAKED providers: the server must carry only what we asked for
for v in GENERATOR_PROVIDER CONTEXTUALIZE_PROVIDER; do
  set_on_server=$(ps eww "$SERVER_PID" 2>/dev/null | tr ' ' '\n' | grep "^${v}=" || true)
  case "$v" in
    GENERATOR_PROVIDER)      wanted="$EVAL_GENERATOR" ;;
    CONTEXTUALIZE_PROVIDER)  wanted="$EVAL_CONTEXTUALIZER" ;;
  esac
  if [ -n "$set_on_server" ] && [ -z "$wanted" ]; then
    echo "!! $v is set on the eval server but was not requested via EVAL_* - refusing"
    echo "   (an ambient provider would silently change what is measured)"; exit 1
  fi
done
echo ">> eval server ready (pid $SERVER_PID) [$CFG]"

for QS in "${QUESTION_SETS[@]}"; do
  SUFFIX=$(basename "$QS" .json | sed 's/^questions//; s/^_//')
  OUT="${NAME}${SUFFIX:+_$SUFFIX}"
  echo ">> running $QS -> eval/results_${OUT}.json"
  RAG_API_BASE="http://127.0.0.1:${EVAL_PORT}" RAG_DETERMINISTIC=1 PYTHONPATH=. \
    .venv/bin/python3 eval/run_eval.py "$OUT" "$QS"
done

echo ">> all question sets complete"
# production is restored by the EXIT trap
