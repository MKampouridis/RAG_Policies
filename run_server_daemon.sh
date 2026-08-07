#!/bin/zsh
# Launcher for the LaunchAgent (com.mkampo.ragpolicies).
#
# DEFAULT: the LOCAL generator (gemma3:12b), matching what every committed
# Round 4-6 result was measured on. Because eval/run_eval.py drives this
# server over HTTP, one server can only have one generator - so keeping the
# default local means an eval run measures the same thing the ledger does,
# with no second instance and no risk of silently measuring the cloud model.
#
# To serve answers from the cloud generator instead (faster, and fixes the
# list-under-enumeration failure), set RAG_CLOUD_GENERATOR=1 in the plist's
# EnvironmentVariables, or run manually:
#   GENERATOR_PROVIDER=anthropic HOST=0.0.0.0 .venv/bin/python3 run_server.py
#
# The key is sourced either way (launchd does not read ~/.zshenv) so flipping
# is a one-line change, and the key stays in one chmod-600 file rather than
# being copied into the world-readable plist.
set -e
cd "$(dirname "$0")"

[ -f "$HOME/.config/anthropic/env" ] && source "$HOME/.config/anthropic/env"

if [ -n "$RAG_CLOUD_GENERATOR" ]; then
  if [ -n "$ANTHROPIC_API_KEY" ]; then
    export GENERATOR_PROVIDER=anthropic
  else
    echo "$(date '+%F %T') WARNING: RAG_CLOUD_GENERATOR set but ANTHROPIC_API_KEY missing - using local generator" >&2
  fi
fi

export HOST=0.0.0.0
exec .venv/bin/python3 run_server.py
