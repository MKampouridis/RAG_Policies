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

# Production serves from the cloud models by default (2026-08-08, user's call).
# Rationale is latency and RAM, not answer quality alone: with gemma3 (8GB) and
# qwen2.5:7b (5GB) both resident on a 16GB machine the stack swaps constantly,
# which is most of the 30-60s response time. Moving generation and query
# rewriting off-device leaves ~0.9GB (nomic-embed + ColBERT) and no swapping.
# Cost is ~$0.009/question (~$4.75/month at 20/day).
#
# The EVAL deliberately does NOT follow this - it runs local via its own server
# on :8001, so numbers stay comparable to the ledger and reproducible under
# RAG_DETERMINISTIC (cloud calls can't be temperature-pinned).
#
# Set RAG_LOCAL_ONLY=1 to serve fully locally (no API spend, no network need).
if [ -z "$RAG_LOCAL_ONLY" ]; then
  if [ -n "$ANTHROPIC_API_KEY" ]; then
    export GENERATOR_PROVIDER=anthropic
    export CONTEXTUALIZE_PROVIDER=anthropic
    # Haiku for the rewriter, not Sonnet. Sonnet measured NULL against the local
    # 7B (+1.2pts follow-up hit@6, +0.0 useful answer), so this task is not
    # capability-bound and the right pick is the fastest adequate model. Haiku
    # produced identical rewrites on the elliptical, pronoun and topic-switch
    # cases at 0.85s vs 2.17s - and the contextualizer sits SERIALLY in front of
    # retrieval, so that 1.3s comes straight off every follow-up (~11% of a ~12s
    # turn). Cost is a rounding error either way ($0.14 vs $0.28/month).
    export ANTHROPIC_CONTEXTUALIZE_MODEL=claude-haiku-4-5
  else
    echo "$(date '+%F %T') WARNING: ANTHROPIC_API_KEY missing - falling back to local models" >&2
  fi
fi

export HOST=0.0.0.0
exec .venv/bin/python3 run_server.py
