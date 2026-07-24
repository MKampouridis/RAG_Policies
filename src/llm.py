"""Thin wrappers around the local Ollama models. Swap models by changing the
constants below — nothing else in the codebase needs to change."""

import os
import time

import ollama
import requests

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

# Eval-only answer/groundedness judge. Centralized here (was triplicated across
# eval/run_eval.py, hallucination_eval.py, rejudge.py). NOTE: this now equals
# LOCAL_GENERATOR_MODEL - generator == judge is self-judging (proven +0.3 RoA
# bias, see comment above); for headline claims judge cross-family instead
# (eval/rejudge.py takes a model arg).
JUDGE_MODEL = "qwen2.5:14b-instruct"

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
    if options is None:
        # num_ctx is a CAPACITY setting, not a determinism one - it MUST apply in
        # production too, or long real conversations (system + history + ~2k
        # tokens of retrieved context + question) silently truncate at Ollama's
        # default window, dropping the system prompt first (external review round
        # 5, Fable 5, verified: the old `and DETERMINISTIC` guard left production
        # at the default context). Only temperature/seed stay behind the flag.
        options = DETERMINISTIC_OPTIONS if DETERMINISTIC else {"num_ctx": 8192}
    response = ollama.chat(model=model, messages=messages, format=format, options=options)
    return response["message"]["content"]


# Item 3 (2026-07-23): optional CLOUD generator for the ANSWER-GENERATION call
# only (src/rag.py answer()'s final chat). Everything else - the query
# contextualizer, the 14B judge, the memory summarizer, relevance, and the
# (off) decomposition/CRAG calls - stays LOCAL: those were validated against
# the local 7B and don't need a stronger model. Motivation: the 78.8%
# groundedness baseline is limited by the 7B fabricating figures/provenance it
# can't support from context (round-4 item-2 finding); a genuinely stronger
# generator is the last untested lever (D2 proved a prompt rule can't close it
# on the 7B). Free tiers only, via OpenAI-compatible endpoints, gated by env so
# production stays fully local unless GENERATOR_PROVIDER is set. Under
# RAG_DETERMINISTIC the cloud temperature is pinned to 0 (+ seed) for a
# reproducible A/B against the local baseline.
GENERATOR_PROVIDER = os.environ.get("GENERATOR_PROVIDER", "").lower()  # "" -> local ollama (LOCAL_GENERATOR_MODEL)
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "")  # override: cloud model name, or a specific local model

# Round 5 (2026-07-24): production ANSWER generator switched qwen2.5:14b ->
# gemma3:12b, after a 10-model bake-off (eval/generator_bakeoff.py, report.md
# "Round 5"). gemma3 grounds far better AND is RAM-safer: RoA groundedness 92.5%
# vs the 14B's 85%; and critically, on retrieval-MISS turns gemma3 faithfully
# ABSTAINS (92% grounded) while the 14B guessed from parametric memory (69%
# grounded = ~31% hallucination on failed retrieval) - so hallucination-on-miss
# drops ~31%->~8%. On HIT turns completeness is comparable (answer_score 3.94 vs
# 4.12). gemma3 is 8.1GB (vs 9GB) and faster. gpt-oss:20b scored higher still but
# 13GB is impractical alongside the contextualizer+retrieval stack on 16GB. ONLY
# answer generation uses this; CONTEXTUALIZE_MODEL (7B) and the judge unchanged.
# History: 7B -> 14B (item 3) -> gemma3:12b (round 5). Override via GENERATOR_MODEL.
LOCAL_GENERATOR_MODEL = "gemma3:12b"

_CLOUD_GENERATORS = {
    # provider: (OpenAI-compatible chat-completions endpoint, api-key env var, default model)
    "groq": (
        "https://api.groq.com/openai/v1/chat/completions",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
    ),
}


def generate(messages: list[dict]) -> str:
    """Answer-generation call. Routes to a cloud generator when
    GENERATOR_PROVIDER is set (else the local CHAT_MODEL via chat()). Kept
    separate from chat() so ONLY answer generation moves to the cloud while the
    contextualizer/judge/etc. stay local and free."""
    if not GENERATOR_PROVIDER:
        # local generation: the 14B production generator (LOCAL_GENERATOR_MODEL),
        # or a GENERATOR_MODEL override. CHAT_MODEL (7B) is untouched so the
        # misc local calls that use it (summary, relevance) stay on the 7B.
        return chat(messages=messages, model=GENERATOR_MODEL or LOCAL_GENERATOR_MODEL)
    if GENERATOR_PROVIDER not in _CLOUD_GENERATORS:
        raise ValueError(
            f"unknown GENERATOR_PROVIDER {GENERATOR_PROVIDER!r}; known: {sorted(_CLOUD_GENERATORS)}"
        )
    url, key_env, default_model = _CLOUD_GENERATORS[GENERATOR_PROVIDER]
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"GENERATOR_PROVIDER={GENERATOR_PROVIDER!r} set but {key_env} is empty")
    payload = {
        "model": GENERATOR_MODEL or default_model,
        "messages": messages,
        "temperature": 0 if DETERMINISTIC else 0.7,
    }
    if DETERMINISTIC:
        payload["seed"] = 42  # honored by Groq; harmless if a provider ignores it
    headers = {"Authorization": f"Bearer {api_key}"}
    # Free tiers rate-limit by tokens-per-minute (Groq: 6k TPM), and a larger
    # RoA context prompt sitting near that ceiling gets a 429. Back off INSIDE
    # this call (honoring the Retry-After header) instead of letting the 429
    # bubble up - otherwise the eval's turn-level retry immediately re-sends the
    # same big prompt and spikes further over the limit, cascading. TPM windows
    # reset each minute, so a short wait clears it.
    for attempt in range(10):
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 429:
            raw = resp.headers.get("retry-after")
            try:
                wait = float(raw) if raw else min(2 ** attempt, 30)
            except ValueError:
                wait = min(2 ** attempt, 30)  # Retry-After can be an HTTP-date, not seconds
            time.sleep(min(wait + 0.5, 30))
            continue
        if not resp.ok:
            # surface the provider's error body (rate/quota/model messages) instead
            # of a bare status - the daily-token-limit diagnosis came from this body
            raise RuntimeError(f"{GENERATOR_PROVIDER} generator HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"{GENERATOR_PROVIDER} generator rate-limited (429) after retries")


def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]
