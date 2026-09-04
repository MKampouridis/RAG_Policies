"""Thin wrappers around the local Ollama models. Swap models by changing the
constants below — nothing else in the codebase needs to change."""

import json
import os
import sys
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

# GENERATOR_REASONING_EFFORT, if set, is passed through as `reasoning_effort`
# in the cloud payload. Needed for Groq's gpt-oss models (2026-09-04
# free-tier bake-off): unconstrained, gpt-oss-120b spent 48 of a 50-token
# budget on an invisible "reasoning" field before any visible answer,
# eating the free-tier token quota far faster than its raw TPD suggests.
# "low" cut that to 17 tokens with no loss of the visible answer. Left unset
# by default - only relevant to reasoning-capable cloud models, and other
# providers may reject an unrecognised field.
GENERATOR_REASONING_EFFORT = os.environ.get("GENERATOR_REASONING_EFFORT", "").strip() or None

# Providers that accept an OpenAI-style `seed` for reproducibility under
# RAG_DETERMINISTIC. NOT harmless-if-ignored, despite the old assumption:
# Gemini 400s outright on an unrecognised field ("Unknown name \"seed\":
# Cannot find field"), confirmed 2026-09-04 - it doesn't silently drop it.
_SEED_SUPPORTED = {"groq"}

_CLOUD_GENERATORS = {
    # provider: (OpenAI-compatible chat-completions endpoint, api-key env var, default model)
    "groq": (
        "https://api.groq.com/openai/v1/chat/completions",
        "GROQ_API_KEY",
        # llama-3.3-70b-versatile was retired from Groq's free tier at some
        # point after item 3 (2026-07-20) shipped this default - confirmed
        # gone 2026-09-04 (404 model_not_found; absent from /v1/models).
        # gpt-oss-120b is Groq's own fallback choice from that same round.
        "openai/gpt-oss-120b",
    ),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "GEMINI_API_KEY",
        "gemini-2.5-flash",
    ),
}


# Anthropic (2026-08-07): NOT in _CLOUD_GENERATORS because the Messages API is
# not OpenAI-shaped - different auth header (x-api-key + anthropic-version),
# `system` hoisted out of messages, a REQUIRED max_tokens, and content[0].text
# instead of choices[0].message.content. Three things that will 400 if copied
# from the OpenAI path: (1) `temperature` - Sonnet 5 rejects any non-default
# sampling parameter, so it is omitted entirely (steer via prompt, not
# temperature); (2) `seed` - no such parameter; (3) a `system` role inside
# messages[]. Unlike the Groq/Gemini free tiers this is PAID, so there is no
# daily token cap - the spend ceiling is the account balance, which is what
# made the round-4 "cloud caps unfit for standing prod" objection moot.
#
# Thinking is DISABLED by default here: adaptive thinking is on by default on
# Sonnet 5 and costs latency + output tokens, and answering from retrieved
# context is an extraction task that gains little from it. Set
# ANTHROPIC_THINKING=adaptive to turn it back on.
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-5"
ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "2048"))
ANTHROPIC_THINKING = os.environ.get("ANTHROPIC_THINKING", "disabled").lower()


USAGE_PATH = os.environ.get("RAG_USAGE_PATH", "data/usage.jsonl")

# Token counts + stop_reason from the most recent generate() call, for callers
# that need per-call attribution rather than the append-only USAGE_PATH log
# (which carries only a timestamp and so has to be correlated by time).
# Added 2026-09-04 for the generator bake-off: token counts are the one thing a
# paid comparison run cannot reconstruct afterwards, since the stored answer
# text supports re-judging any quality metric locally for free, but nothing
# recovers what the provider billed. Overwritten per call, single-threaded use
# only; production ignores it.
LAST_USAGE: dict = {}


def _log_usage(data: dict) -> None:
    """Token counts beside the generate timing. Without these, a slow turn
    cannot be told apart from a long answer - and `max_tokens` cannot be
    judged a latency lever without knowing whether it is ever REACHED
    (stop_reason 'max_tokens' vs 'end_turn'). Best-effort; never breaks a
    request."""
    if os.environ.get("RAG_TIMING") != "1":
        return
    try:
        u = data.get("usage") or {}
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "input_tokens": u.get("input_tokens"),
               "output_tokens": u.get("output_tokens"),
               "cache_read": u.get("cache_read_input_tokens"),
               "stop_reason": data.get("stop_reason")}
        with open(USAGE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


_SESSION = None


def _session():
    """One pooled HTTPS connection instead of a fresh TCP+TLS handshake per
    call. Every cloud request paid that handshake before; it is ~0.2-0.3s on
    the generator and again on the contextualizer."""
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def _anthropic_generate_stream(payload: dict, headers: dict, on_token) -> str:
    """Streaming variant. Emits text deltas through `on_token` as they arrive
    and still returns the complete text, so callers that also need the whole
    answer (to store it) are unchanged.

    Retries only BEFORE the first token: once text has reached the user,
    replaying the request would duplicate what they have already read.
    """
    payload = dict(payload, stream=True)
    emitted = False
    for attempt in range(6):
        with _session().post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=payload, stream=True, timeout=120,
        ) as resp:
            if resp.status_code in (429, 529) and not emitted:
                raw = resp.headers.get("retry-after")
                try:
                    wait = float(raw) if raw else min(2 ** attempt, 30)
                except ValueError:
                    wait = min(2 ** attempt, 30)
                time.sleep(min(wait + 0.5, 30))
                continue
            if not resp.ok:
                raise RuntimeError(
                    f"anthropic generator HTTP {resp.status_code}: {resp.text[:500]}"
                )
            parts: list[str] = []
            # A stream that ends WITHOUT message_stop was cut short - a dropped
            # connection, a proxy timeout. Returning what arrived stores a
            # silently truncated answer that is indistinguishable from a
            # complete one. On a policy assistant an answer ending at "students
            # may appeal if" is worse than an error, because the reader cannot
            # tell it is incomplete.
            saw_stop = False
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                try:
                    evt = json.loads(line[5:].strip())
                except ValueError:
                    continue
                kind = evt.get("type")
                if kind == "content_block_delta":
                    delta = evt.get("delta") or {}
                    # text_delta only - thinking_delta must never reach the user
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        parts.append(delta["text"])
                        emitted = True
                        on_token(delta["text"])
                elif kind == "error":
                    raise RuntimeError(f"anthropic stream error: {evt.get('error')}")
                elif kind == "message_stop":
                    saw_stop = True
                elif kind == "message_delta":
                    _log_usage({"usage": evt.get("usage") or {},
                                "stop_reason": (evt.get("delta") or {}).get("stop_reason")})
                    if (evt.get("delta") or {}).get("stop_reason") == "refusal":
                        return "I can't answer that from the provided documents."
            if not saw_stop:
                raise RuntimeError(
                    "the answer stream ended before it was complete "
                    f"({len(''.join(parts))} characters received)")
            return "".join(parts)
    raise RuntimeError("anthropic generator rate-limited/overloaded after retries")


def _anthropic_generate(messages: list[dict], model: str | None = None, on_token=None) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("anthropic provider selected but ANTHROPIC_API_KEY is empty")

    # the Messages API takes system prompts as a top-level field, not a role
    system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
    convo = [m for m in messages if m.get("role") != "system"]

    payload = {
        "model": model or GENERATOR_MODEL or ANTHROPIC_DEFAULT_MODEL,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": convo,
    }
    if system:
        payload["system"] = system
    if ANTHROPIC_THINKING != "disabled":
        payload["thinking"] = {"type": ANTHROPIC_THINKING}
    else:
        payload["thinking"] = {"type": "disabled"}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if on_token is not None:
        return _anthropic_generate_stream(payload, headers, on_token)
    for attempt in range(6):
        resp = _session().post(
            "https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=120
        )
        # 429 = rate limited, 529 = overloaded; both are retryable per Anthropic's docs
        if resp.status_code in (429, 529):
            raw = resp.headers.get("retry-after")
            try:
                wait = float(raw) if raw else min(2 ** attempt, 30)
            except ValueError:
                wait = min(2 ** attempt, 30)  # Retry-After may be an HTTP-date
            time.sleep(min(wait + 0.5, 30))
            continue
        if not resp.ok:
            raise RuntimeError(f"anthropic generator HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        # A safety refusal returns HTTP 200 with stop_reason 'refusal' and no text
        # block, so indexing content[0] blindly would IndexError on a live refusal.
        _log_usage(data)
        u = data.get("usage") or {}
        LAST_USAGE.clear()
        LAST_USAGE.update({
            "provider": "anthropic",
            "model": payload["model"],
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "cache_read": u.get("cache_read_input_tokens"),
            "stop_reason": data.get("stop_reason"),
        })
        if data.get("stop_reason") == "refusal":
            return "I can't answer that from the provided documents."
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    raise RuntimeError("anthropic generator rate-limited/overloaded after retries")


# Judge routing (2026-08-07). The judge is the project's weakest measurement
# link: a candidate model grading its own family swung synthetic-miss
# groundedness by 24 POINTS (round-6 Tier 0), which is why phi4 became the
# "neutral cross-family" judge - but phi4 is still a small local model making
# subtle groundedness calls, and the value-level metric flagged residual
# leniency on definitional claims. A frontier judge is the highest-value use
# of paid credits here (~$0.40 per 80-turn run on Sonnet via batching).
#
# Deliberately SEPARATE from GENERATOR_PROVIDER: the agreed configuration is
# local generation (comparable to the whole Round 4-6 ledger, and pinnable by
# RAG_DETERMINISTIC) judged by a cloud model. Setting both to anthropic would
# reintroduce exactly the self-preference bias phi4 exists to avoid.
#   JUDGE_PROVIDER=anthropic RAG_API_BASE=... python eval/run_eval.py <name>
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", "").lower()  # "" -> local JUDGE_MODEL
ANTHROPIC_JUDGE_MODEL = os.environ.get("ANTHROPIC_JUDGE_MODEL", "claude-sonnet-5")


def judge_chat(messages: list[dict], format: str | None = None, model: str | None = None) -> str:
    """Judging call. Routes to Anthropic when JUDGE_PROVIDER=anthropic, else
    the local JUDGE_MODEL. Kept separate from chat()/generate() so the judge
    can be swapped without touching what is being judged."""
    if JUDGE_PROVIDER == "anthropic":
        if GENERATOR_PROVIDER == "anthropic":
            # not fatal - a same-family judge is still usable - but it is the
            # exact confound that produced the 24-point round-6 swing.
            print(
                "WARNING: JUDGE_PROVIDER and GENERATOR_PROVIDER are both 'anthropic' - "
                "the judge shares a family with the generator it is grading (self-preference bias).",
                file=sys.stderr,
            )
        # Callers pass a LOCAL judge name explicitly (judge_answer defaults to
        # JUDGE_MODEL; the re-judge scripts pass "phi4"), which would 404 as an
        # Anthropic model id. Only honour an override that is actually an
        # Anthropic model, otherwise fall back to the configured cloud judge.
        # Substituting loudly, not silently: eval/synthmiss_rejudge.py,
        # value_sufficient.py and the bake-offs pin phi4 *deliberately* as the
        # neutral cross-family judge. Swapping that out without saying so would
        # change what "neutral re-judge" means in already-published numbers -
        # the same silent-mismeasurement class this project keeps getting bitten
        # by. Run those scripts WITHOUT JUDGE_PROVIDER to keep phi4.
        if model and not model.startswith("claude-"):
            print(
                f"WARNING: JUDGE_PROVIDER=anthropic overrides the explicitly requested "
                f"judge {model!r} with {ANTHROPIC_JUDGE_MODEL!r}. If that model was pinned "
                f"for neutrality, unset JUDGE_PROVIDER for this script.",
                file=sys.stderr,
            )
        cloud_model = model if (model or "").startswith("claude-") else ANTHROPIC_JUDGE_MODEL
        return _anthropic_generate(messages, model=cloud_model)
    return chat(messages=messages, format=format, model=model or JUDGE_MODEL)


# Contextualizer routing (2026-08-08). The query rewriter is the smallest model
# in the stack (qwen2.5:7b) doing the hardest inference in it - working out what
# "these values" refers to from a transcript - and it sits on the critical path
# for every follow-up. Tonight's numbers say that is where the system is weakest:
# under a frontier judge, primary turns score 4.24 mean and follow-ups 3.32
# (useful-answer 72.5% vs 51.9%). Round 5 already tried the local alternatives -
# qwen2.5:14b +2.5 (not worth 2x cost), gemma3:12b -6.2, qwen3:8b disqualified -
# so the ceiling looks like model capability, and a frontier model was never
# tested here.
#
# Cheap to run: input is a short transcript (~320 tokens) and output is one
# rewritten question (~30), so ~$0.001/call - about 6% of what a generation call
# costs. Separate from GENERATOR_PROVIDER and JUDGE_PROVIDER so each component
# can be swapped and measured on its own; changing two at once is what made
# attribution impossible earlier in this session.
CONTEXTUALIZE_PROVIDER = os.environ.get("CONTEXTUALIZE_PROVIDER", "").lower()
ANTHROPIC_CONTEXTUALIZE_MODEL = os.environ.get("ANTHROPIC_CONTEXTUALIZE_MODEL", "claude-sonnet-5")


def contextualize_chat(messages: list[dict], model: str | None = None) -> str:
    """Query-rewrite call. Routes to Anthropic under
    CONTEXTUALIZE_PROVIDER=anthropic, else the local CONTEXTUALIZE_MODEL."""
    if CONTEXTUALIZE_PROVIDER == "anthropic":
        cloud_model = model if (model or "").startswith("claude-") else ANTHROPIC_CONTEXTUALIZE_MODEL
        return _anthropic_generate(messages, model=cloud_model)
    return chat(messages=messages, model=model or CONTEXTUALIZE_MODEL)


def generate(messages: list[dict], on_token=None) -> str:
    """Answer-generation call. Routes to a cloud generator when
    GENERATOR_PROVIDER is set (else the local CHAT_MODEL via chat()). Kept
    separate from chat() so ONLY answer generation moves to the cloud while the
    contextualizer/judge/etc. stay local and free.

    `on_token` streams text deltas as they arrive and is honoured only by the
    Anthropic path. Other providers ignore it and return the whole answer at
    once - callers must not assume streaming happened, only that the return
    value is complete either way."""
    if GENERATOR_PROVIDER == "anthropic":
        return _anthropic_generate(messages, on_token=on_token)
    if not GENERATOR_PROVIDER:
        # local generation: the 14B production generator (LOCAL_GENERATOR_MODEL),
        # or a GENERATOR_MODEL override. CHAT_MODEL (7B) is untouched so the
        # misc local calls that use it (summary, relevance) stay on the 7B.
        return chat(messages=messages, model=GENERATOR_MODEL or LOCAL_GENERATOR_MODEL)
    if GENERATOR_PROVIDER not in _CLOUD_GENERATORS:
        raise ValueError(
            f"unknown GENERATOR_PROVIDER {GENERATOR_PROVIDER!r}; "
            f"known: {sorted(list(_CLOUD_GENERATORS) + ['anthropic'])}"
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
    if DETERMINISTIC and GENERATOR_PROVIDER in _SEED_SUPPORTED:
        payload["seed"] = 42
    if GENERATOR_REASONING_EFFORT:
        payload["reasoning_effort"] = GENERATOR_REASONING_EFFORT
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
        body = resp.json()
        choice = body["choices"][0]
        u = body.get("usage") or {}
        LAST_USAGE.clear()
        LAST_USAGE.update({
            "provider": GENERATOR_PROVIDER,
            "model": payload["model"],
            "input_tokens": u.get("prompt_tokens"),
            "output_tokens": u.get("completion_tokens"),
            # reasoning models bill invisible thinking inside completion_tokens -
            # broken out where the provider reports it (Groq's gpt-oss does)
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "stop_reason": choice.get("finish_reason"),
        })
        return choice["message"]["content"]
    raise RuntimeError(f"{GENERATOR_PROVIDER} generator rate-limited (429) after retries")


def embed(text: str, model: str = EMBED_MODEL) -> list[float]:
    response = ollama.embed(model=model, input=text)
    return response["embeddings"][0]


def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]
