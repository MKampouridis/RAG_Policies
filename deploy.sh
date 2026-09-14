#!/bin/zsh
# Put the saved code into production, and prove it landed.
#
# WHY THIS EXISTS
# Restarting after a change was a manual step done by hand, and it was forgotten
# repeatedly - including on 2026-09-13, when the drift detector's own commit went
# unrestarted and the detector caught its author within the hour. Catching a
# stale deployment is second best to not having one.
#
# The half-done case is what makes a script worth it: `launchctl unload` then
# forgetting `load` leaves production DOWN, and unload-load without waiting
# leaves you thinking it worked while the retrieval stack is still loading. So
# this waits for the server to answer and then checks the revision it reports
# against HEAD - the deploy is not "done" until production says it is running
# the code you meant.
#
# Usage:
#   ./deploy.sh            verify (static), restart, wait, confirm revision
#   ./deploy.sh --no-verify   skip the checks (a hotfix you have already run)
#   ./deploy.sh --full        also run the live-request check afterwards
set -e
cd "$(dirname "$0")"

PLIST="$HOME/Library/LaunchAgents/com.mkampo.ragpolicies.plist"
PORT=8000
VERIFY=1
FULL=0
for a in "$@"; do
  case "$a" in
    --no-verify) VERIFY=0 ;;
    --full) FULL=1 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

HEAD_REV=$(git rev-parse --short HEAD)
echo "==> deploying $HEAD_REV"

# Uncommitted changes are not an error - a deploy of the working tree is a
# legitimate thing to do - but they ARE worth saying out loud, because the
# revision production reports afterwards will be HEAD and will not describe
# whatever is uncommitted.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "    note: working tree has uncommitted changes; production will report $HEAD_REV"
fi

if [ "$VERIFY" = "1" ]; then
  echo "==> verifying (static)"
  .venv/bin/python3 verify.py --static
fi

echo "==> restarting"
# `launchctl unload` without `load` leaves production down, so the reload is
# unconditional even if unload reports an error (it does when not loaded).
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "==> waiting for the server"
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/login" 2>/dev/null || echo 000)
  [ "$code" = "200" ] && break
  sleep 2
done
if [ "$code" != "200" ]; then
  echo "    FAILED: server did not answer on :$PORT after 120s" >&2
  echo "    check data/server.log" >&2
  exit 1
fi

# The retrieval stack loads lazily in a background thread, so answering /login
# does not mean it is ready to answer a QUESTION. Give the warmup a moment
# before reading provenance, or the first read races it.
sleep 3

# The access password is sourced by the daemon launcher, not by this shell.
PW=""
[ -f "$HOME/.config/ragpolicies/env" ] && PW=$(sed -n 's/.*RAG_ACCESS_PASSWORD=//p' "$HOME/.config/ragpolicies/env" | tr -d "\"'")
# An ARRAY, not a string. zsh does not word-split an unquoted expansion the way
# bash does, so `${COOKIE:+-H "Cookie: $COOKIE"}` reached curl as ONE argument
# and the request came back 400 - which this script then reported as a failed
# deploy of a perfectly good server.
CURL_ARGS=()
if [ -n "$PW" ]; then
  TOKEN=$(printf 'rag-access:%s' "$PW" | shasum -a 256 | cut -d' ' -f1)
  CURL_ARGS=(-H "Cookie: rag_access=$TOKEN")
fi

RUNNING=$(curl -s "${CURL_ARGS[@]}" "http://127.0.0.1:$PORT/api/config" \
  | .venv/bin/python3 -c 'import json,sys; print((json.load(sys.stdin).get("provenance") or {}).get("code_revision") or "")' 2>/dev/null || echo "")

if [ "$RUNNING" = "$HEAD_REV" ]; then
  echo "==> live: $RUNNING (matches HEAD)"
elif [ -z "$RUNNING" ]; then
  # Distinguished from a mismatch: this is "I could not read what is running",
  # which usually means the access password or the warmup, not a bad deploy.
  echo "    FAILED: could not read the running revision from /api/config" >&2
  echo "    the server answered, so check the access password and data/server.log" >&2
  exit 1
else
  echo "    FAILED: production reports '$RUNNING', expected '$HEAD_REV'" >&2
  echo "    the restart did not take - check data/server.log" >&2
  exit 1
fi

if [ "$FULL" = "1" ]; then
  echo "==> live request check"
  .venv/bin/python3 verify.py | tail -4
fi

# Refresh the dashboard's alert state so a drift warning from before the deploy
# does not sit there looking current. Free - the hourly monitor asks no model.
.venv/bin/python3 run_monitor.py --no-probe --quiet >/dev/null 2>&1 || true
echo "==> done"
