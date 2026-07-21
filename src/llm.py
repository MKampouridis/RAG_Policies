"""Thin wrappers around the local Ollama models. Swap models by changing the
constants below — nothing else in the codebase needs to change."""

import os

import ollama

# Phase 1 determinism fix (external code-review round, 2026-07-21, see
# eval/report.md): no call site anywhere in this codebase set temperature,
# seed, or num_ctx - Ollama's sampling defaults (temperature ~0.8) meant the
# same code + same corpus could legitimately produce different retrieval
# queries (via the contextualizer), different generated answers, and
# different judge scores on repeat runs. This was the project's own
# documented ~1-2-turn "noise floor" and made every eval delta under that
# size unreadable. RAG_DETERMINISTIC=1 pins temperature=0/seed=42 (a fixed,
# arbitrary integer - any constant works, it just has to be the same one
# every run) and raises num_ctx to a size that comfortably covers a full
# generation prompt (system + history + context + question), ruling out
# silent truncation as a confound too. Off by default so normal production
# traffic keeps natural sampling variation; eval runs opt in by setting the
# env var before starting both the server and the eval script.
DETERMINISTIC = os.environ.get("RAG_DETERMINISTIC") == "1"
DETERMINISTIC_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192}

CHAT_MODEL = "qwen2.5:7b-instruct"
# generator bake-off (2026-07-20, deferred LLM-experiments phase) tested
# qwen2.5:14b and llama3.1:8b as replacements - both rejected (see
# eval/report.md): llama3.1:8b was cleanly, independently judged worse across
# the board; qwen2.5:14b looked best but only under self-judging (it was also
# JUDGE_MODEL), a bias already proven as large as +0.3 on RoA specifically -
# not trustworthy without an independent judge we don't have access to.
# qwen2.5:7b-instruct remains the best-supported choice.

# Query contextualizer (src/rag.py's _contextualize_query) pinned separately
# from CHAT_MODEL. First bake-off pass (qwen2.5:14b as CHAT_MODEL, no
# separate constant yet) showed CHAT_MODEL swaps silently changed BOTH answer
# generation and follow-up query rewriting - follow-up hit@6 regressed
# 82.5%->75.0% while primary hit@6 (contextualizer doesn't run without
# history) was unchanged, isolating the rewriter as the cause. Pinning it to
# the validated qwen2.5:7b-instruct (the model _is_faithful_rewrite's guard
# was tuned against) lets CHAT_MODEL vary for a clean generation-only test.
CONTEXTUALIZE_MODEL = "qwen2.5:7b-instruct"

# Embedding model + its required task prefixes (asymmetric embedding models
# need different prefix text for indexed documents vs search queries, and get
# it wrong silently - always set all three together when swapping EMBED_MODEL).
# Two alternatives tested and rejected (see eval/EXPERIMENTS.md): mxbai-embed-large
# won on policy but regressed RoA (likely its 512-token window truncating dense
# chunks); bge-m3 (8192-token context, so no truncation risk) was a wash-to-
# slight-regression on RoA anyway - the corpus's near-duplicate-boilerplate
# structure appears to be the dominant constraint, not embedding model choice.
EMBED_MODEL = "nomic-embed-text"
EMBED_DOCUMENT_PREFIX = "search_document: "
EMBED_QUERY_PREFIX = "search_query: "
# bge-m3 alternative (wash/slight regression on RoA, no prefix needed):
# EMBED_MODEL = "bge-m3"
# EMBED_DOCUMENT_PREFIX = ""
# EMBED_QUERY_PREFIX = ""
# mxbai-embed-large alternative (stronger on policy, weaker on RoA):
# EMBED_MODEL = "mxbai-embed-large"
# EMBED_DOCUMENT_PREFIX = ""
# EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def chat(
    messages: list[dict], format: str | None = None, model: str = CHAT_MODEL, options: dict | None = None
) -> str:
    if options is None and DETERMINISTIC:
        options = DETERMINISTIC_OPTIONS
    response = ollama.chat(model=model, messages=messages, format=format, options=options)
    return response["message"]["content"]


def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]
