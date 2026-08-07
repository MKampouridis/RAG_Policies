#!/bin/zsh
# Launcher for the LaunchAgent (com.mkampo.ragpolicies).
#
# Exists because launchd does NOT read ~/.zshenv, so the server would start
# without ANTHROPIC_API_KEY and every answer would fail. Sourcing the same
# chmod-600 env file keeps the key in exactly one place instead of copying it
# into the plist (which is world-readable plaintext).
#
# Manual equivalent:
#   GENERATOR_PROVIDER=anthropic HOST=0.0.0.0 .venv/bin/python3 run_server.py
set -e
cd "$(dirname "$0")"

[ -f "$HOME/.config/anthropic/env" ] && source "$HOME/.config/anthropic/env"

# Cloud answer generation; unset GENERATOR_PROVIDER to fall back to local gemma3.
# If the key is missing, fall back rather than serving errors for weeks.
if [ -n "$ANTHROPIC_API_KEY" ]; then
  export GENERATOR_PROVIDER=anthropic
else
  echo "$(date '+%F %T') WARNING: ANTHROPIC_API_KEY not found - falling back to local generator" >&2
fi

export HOST=0.0.0.0
exec .venv/bin/python3 run_server.py
