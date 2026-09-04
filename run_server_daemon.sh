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
[ -f "$HOME/.config/groq/env" ] && source "$HOME/.config/groq/env"

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
# TRIAL (2026-09-04, user's call): serve ANSWERS from Groq's gpt-oss-120b so
# the change can be judged in real use, not just in the bake-off. Measured on
# the 80-turn fixed-context bake-off against Sonnet 5 (same contexts, same
# local judge): groundedness 94% vs 84%, answer_score 4.38 vs 4.30 overall and
# 4.39 vs 4.03 on rules-of-assessment, true unthrottled latency ~0.8s vs 5.3s,
# ~$0.0005 vs $0.0129 per answer. Sonnet's one remaining edge is Policy-type
# completeness (4.58 vs 4.38).
#
# The contextualizer deliberately STAYS on Haiku - that choice was made on
# latency grounds (see the note below) and was not part of this comparison.
#
# CAVEAT while on the free tier: requests are rate-limited per minute AND per
# day, shared with eval runs, and an exhausted quota surfaces to the user as a
# 503 rather than a wrong answer. Groq's paid tier is $0.15/$0.60 per MTok if
# this trial sticks.
#
# TO REVERT: set RAG_GROQ_GENERATOR=0 (or delete this block's export lines);
# the Anthropic branch below then applies unchanged.
: "${RAG_GROQ_GENERATOR:=1}"

if [ -z "$RAG_LOCAL_ONLY" ]; then
  if [ "$RAG_GROQ_GENERATOR" = "1" ] && [ -n "$GROQ_API_KEY" ]; then
    export GENERATOR_PROVIDER=groq
    export GENERATOR_MODEL=openai/gpt-oss-120b
    # Without this, gpt-oss spends most of a small token budget on an invisible
    # reasoning field before any visible answer (48 of 50 tokens, measured).
    export GENERATOR_REASONING_EFFORT=low
    # rewriter stays on Haiku, unchanged by this trial
    export CONTEXTUALIZE_PROVIDER=anthropic
    export ANTHROPIC_CONTEXTUALIZE_MODEL=claude-haiku-4-5
  elif [ -n "$ANTHROPIC_API_KEY" ]; then
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
    # A log line nobody reads is not a warning. This reaches the UI, which
    # shows a banner: an expired key otherwise degrades every answer silently
    # and looks like the model simply got worse. Deliberate local serving
    # (RAG_LOCAL_ONLY=1) is NOT degraded and does not set this.
    export RAG_DEGRADED=1
  fi
fi

# Per-stage latency to data/latency.jsonl (gitignored). Costs one appended
# line per stage; enabled because out-of-process timing produced three
# inconsistent answers and the ~9s median has a long tail worth attributing
# from real traffic rather than estimating a fourth time.
export RAG_TIMING=1

export HOST=0.0.0.0
exec .venv/bin/python3 run_server.py
