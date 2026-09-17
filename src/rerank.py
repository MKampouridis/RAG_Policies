"""Reranking of the fused candidate pool.

The earlier LLM listwise reranker (asking qwen2.5:7b to pick the best 6 of
24 excerpts) made every retrieval metric worse - a generalist chat model
reasoning over a list of near-identical boilerplate chunks broke more
correct rankings than it fixed. A cross-encoder is a fundamentally different
mechanism: it scores each (query, passage) pair independently, purpose-built
for exactly this fine-grained relevance judgment, not multi-item reasoning.

Two backends, selected by BACKEND below - kept side by side (not one deleting
the other) so the working cross-encoder is one constant-flip away if the
ColBERT experiment (see eval/EXPERIMENTS.md) doesn't pan out.
"""

import os
from src.docid import top_family_count as _top_family_count

BACKEND = "colbert"  # "cross_encoder" (production) | "colbert" (experiment)

CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"
COLBERT_MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"
# how many of the fused candidates to actually score - failure analysis
# (eval/report.md) found relevant-but-mis-ranked documents as deep as rank 60
# in a top-50 dense+BM25 union, so this needs to be generous, not just N_RESULTS.
# Tried widening 30 -> 100 globally (J0b, eval/report.md): the J0 diagnostic
# found 4 of 12 misses in the fused pool at ranks 32-69, beyond this window.
# Widening DID rescue 2 of them - but lost 5 previously-correct turns (RoA
# hit@6 70%->62.5%) because the extra ~60 candidates per query are mostly
# near-duplicate boilerplate the reranker can't reliably distinguish from the
# right sibling, on queries that didn't need the extra depth at all.
# Env-overridable for the 2026-09-17 reranker investigation so arms differ by
# configuration, not by edited code between passes. Default unchanged at 30.
RERANK_POOL_SIZE = int(os.environ.get("RAG_RERANK_POOL", "30"))

# Idea 4 (targeted widening) - tried, regressed WORSE than J0b's naive
# global widening (eval/report.md "Code review round"): 0 rescues / 4 losses
# (RoA hit@6 70%->60%), vs J0b's 2 rescues / 5 losses. The pre-rerank
# family-fragmentation signal apparently doesn't correlate with "the right
# document is deeper in the pool" - it fired on queries where widening only
# added noise, and never once on the out-of-pool cases it was meant to
# catch. Off by default; kept for reference.
# The SPOILER arm for the reranker investigation (2026-09-17). Reranking has
# never been measured against its own absence - ColBERT was measured against a
# cross-encoder (it won, RoA hit@6 60%->70%) and both widening variants were
# measured against the shipped pool, but nothing ever asked whether reranking
# beats the fused order it replaces. It costs ~2GB of GPU footprint and most of
# the retrieval latency, so "is it earning that" is worth one free pass.
# RAG_RERANK=0 returns the fusion order untouched.
RERANK_ENABLED = os.environ.get("RAG_RERANK", "1") == "1"

# Per-document cap - TRIED AND ABANDONED (2026-09-17), kept with its
# falsification like this file's other rejected ideas.
#
# It is a NO-OP where it sits, and the arena measured that without noticing at
# first: cap2 and cap3 both scored EXACTLY 0.0000 against production across 146
# questions, which is the signature of a mechanism that never fires rather than
# one that does not help. Direct check on a question known to be monopolised:
# with cap=2, assessment-policies-summary.pdf still returned THREE chunks.
#
# Cause: `_adjacent_chunks` and `_complete_small_documents` run AFTER rerank()
# in retrieve() and both deliberately add chunks from a document already in the
# results - one so a rule split across a chunk boundary stays whole, the other
# so a small document can be enumerated completely. A cap applied inside
# rerank() is undone by them a few lines later.
#
# Testing it properly means capping AFTER those stages, i.e. deciding to
# override two mechanisms that were measured and shipped ON. That is a much
# larger change than the observation motivating it, and the existing evidence
# argues against it. Not pursued. Left at 0 with this note so the next person
# does not rediscover the no-op.
#
# Original observation, still true and still unaddressed: one page took 3 of 7
# slots on a real question, and across 146 real questions 122 return fewer than
# 6 distinct documents while 20 return chunks from a single document.
#
# (original note) Per-document cap on the FINAL results. Observed:
# one page took 3 of 7 slots on a real question, and across 146 real questions
# 122 return fewer than 6 DISTINCT documents while 20 return chunks from a
# single document. 0 = off (shipped default), N = at most N chunks per document.
#
# The obvious risk, and why this is measured rather than assumed: _adjacent_chunks
# and document completion deliberately pull MORE chunks from one document so an
# enumeration question can list every milestone. A cap fights those directly.
RERANK_MAX_PER_DOC = int(os.environ.get("RAG_MAX_PER_DOC", "0"))

TARGETED_WIDENING_ENABLED = False
WIDE_RERANK_POOL_SIZE = 100
FRAGMENTATION_THRESHOLD = 1

# Phase 4, experiment 1 (external code review round 2, 2026-07-21, Fable 5)
# - tried, rejected. Identity-enriched passages AT RERANK TIME ONLY, not
# re-embedding - avoided J2's corpus-wide embedding-displacement failure
# mode by construction (see eval/report.md, "Identity-first round"), but
# regressed anyway via a different mechanism: enrichment isn't neutral
# across candidates. Generic RoA "framework" documents that don't belong to
# one specific programme (masters-25.pdf, pgt-credit-framework-25.pdf) have
# thin-to-empty J1 identity records and get little/no enrichment, while
# programme-specific siblings (mres-gov-25.pdf: full programme_name,
# department, aliases) get a real content boost - on any query with loose
# semantic overlap to that added text, the enriched sibling's MaxSim/
# cross-encoder score rises while the correct-but-generic document's
# doesn't, regardless of true relevance. Full 80-turn eval: RoA hit@6
# 62.5%->57.5%, answer score 3.84->3.56, net 4 gained / 6 lost (5 of the 6
# losses on primary turns, concentrated in exactly this generic-vs-specific
# pattern). Reverted; kept for reference like the project's other rejected
# ideas - the mechanism (privileging identity-rich siblings regardless of
# relevance) would need a real fix (e.g. only enrich when the CANDIDATE and
# at least one COMPETITOR in the pool both have identity records, so a
# lone generic document isn't disadvantaged) before this is worth retrying.
IDENTITY_ENRICHED_RERANK_ENABLED = False

# Idea 1 (cached ColBERT embeddings, see eval/report.md "Code review round"):
# once build_colbert_index.py has run, reuse each candidate's precomputed
# token embedding (looked up by (source_url, chunk_index), which survives
# the whole fusion/dedup pipeline unchanged) instead of re-encoding its text
# from scratch on every single query - a chunk's embedding never changes
# between queries, so re-encoding it repeatedly is pure waste. Falls back to
# fresh encoding per-candidate when the index isn't built or a candidate
# isn't in it yet, so this is safe to leave on unconditionally - production
# behavior is byte-identical to before until the index actually exists.
# Env-overridable (2026-08-10). The cache is keyed by (source_url,
# chunk_index) with NO text validation, so against an index built with a
# DIFFERENT chunk size it would silently return the embedding of different
# text - reranking the experimental index on the production index's vectors
# and quietly invalidating any comparison. Set RAG_COLBERT_CACHE=0 on BOTH
# arms of a chunking comparison so each encodes its own text.
# DEFAULT FLIPPED TO OFF, 2026-08-12, on a measured retrieval improvement.
#
# The cache reuses ColBERT token embeddings precomputed by
# build_colbert_index.py. Those were built 2026-07-21; the corpus was
# re-ingested 2026-08-11. The index holds 20,477 chunks against Chroma's
# 21,709, and the overlapping ones were embedded from text that has since
# changed - so the reranker was scoring some chunks by their SUPERSEDED
# wording. Nothing detected the drift: run_ingest.py and reembed.py update
# Chroma and leave the ColBERT index alone.
#
# Measured (retrieval_replay, 160 turns, deterministic, paired per turn):
#   cache ON  121/160     cache OFF  126/160     +6 gained, -1 lost
# Three of the six gains are documents Round 22 had recorded as "the reranker
# demoted a top-of-pool document" - that anomaly was stale embeddings, not a
# reranker defect.
#
# Turning it off also stops _load() pulling the 3.7GB Voyager index into a 16GB
# machine, which Round 13 measured as degrading Chroma's filtered query from
# 81ms to 2543ms. So this is faster, lighter AND more accurate.
#
# Set RAG_COLBERT_CACHE=1 to re-enable after rebuilding the index; the
# staleness check below will tell you whether it is safe.
USE_CACHED_COLBERT_EMBEDDINGS = os.environ.get("RAG_COLBERT_CACHE", "0") == "1"

_cross_encoder = None




def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


def _identity_suffix(meta: dict) -> str:
    """J1 identity fields (programme/department/partner institution/awards/
    aliases) not already in the stored chunk_header, formatted the same way
    build_chunk_header() does for consistency with the J2 attempt this is
    deliberately NOT repeating the mistake of (see
    IDENTITY_ENRICHED_RERANK_ENABLED's comment above) - empty string when no
    identity record exists for this document (most policy documents), which
    is the common case and must be a no-op, not an error."""
    from src.ingest import _load_doc_identity

    identity = _load_doc_identity(meta.get("source_url", ""))
    if not identity:
        return ""
    parts = []
    if identity.get("programme_name"):
        parts.append(f"programme: {identity['programme_name']}")
    if identity.get("department"):
        parts.append(f"department: {identity['department']}")
    if identity.get("partner_institution"):
        parts.append(f"partner institution: {identity['partner_institution']}")
    if identity.get("awards"):
        parts.append(f"awards: {', '.join(identity['awards'])}")
    if identity.get("aliases"):
        parts.append(f"also known as: {', '.join(identity['aliases'])}")
    return " | ".join(parts)


def _passages(pool_docs: list[str], pool_metas: list[dict]) -> list[str]:
    # score against header+text, not the bare stored chunk - the document
    # identity (degree length, department, year) that actually disambiguates
    # near-identical RoA siblings lives only in chunk_header (prepended at
    # embedding time, never stored in `documents`); without it the reranker
    # sees strictly less signal than the embedder already had
    passages = [f"{meta.get('chunk_header', '')}\n{doc}" for doc, meta in zip(pool_docs, pool_metas)]
    if IDENTITY_ENRICHED_RERANK_ENABLED:
        passages = [
            f"{p}\n{suffix}" if (suffix := _identity_suffix(meta)) else p
            for p, meta in zip(passages, pool_metas)
        ]
    return passages


def _rerank_cross_encoder(query: str, pool_docs: list[str], pool_metas: list[dict], top_n: int) -> list[int]:
    passages = _passages(pool_docs, pool_metas)
    scores = _get_cross_encoder().predict([(query, p) for p in passages])
    return sorted(range(len(pool_docs)), key=lambda i: scores[i], reverse=True)[:top_n]


def rerank_many(query: str, pools: list[dict], top_n: int) -> list[dict]:
    """Rerank several candidate pools with ONE ColBERT encode pass.

    Multi-entity retrieval reranks each named department's pool separately. On
    a three-department question that is four encode calls - measured at 3.32s
    against 1.11s for an ordinary single-pool question, i.e. 3x the retrieval
    latency, on ~1 in 10 real questions (Round 30/31).

    MEASURED, AND NOT ADOPTED (Round 31). The hypothesis was that transformers
    batch well, so one encode of N passages would cost far less than four
    encodes of N/4. It does not: on four realistic pools,

        per-pool (4 calls)  5.15s
        batched (1 encode)  4.57s      -> 11%, not the large win expected

    ColBERT's cost is per-PASSAGE compute, not per-call overhead, so batching
    saves only the call overhead. Output was verified IDENTICAL, so the idea is
    sound and safe - it simply is not worth a second code path through the core
    retrieval for ~11% of a stage on ~1 in 10 questions (~1% overall).

    Kept, unused, with the measurement, so it is not re-proposed. Wire it into
    _multi_entity_results if per-entity reranking ever becomes the dominant
    cost - the equivalence test above is the thing to re-run first.

    Falls back to per-pool reranking on any error.
    """
    if BACKEND != "colbert" or len(pools) < 2:
        return [rerank(query, p, top_n) for p in pools]
    try:
        from pylate import rank
        from src import colbert_index

        model = colbert_index.get_model()
        q_emb = model.encode([query], is_query=True)

        spans, all_passages, kept = [], [], []
        for pool in pools:
            docs = pool.get("documents", [[]])[0][:RERANK_POOL_SIZE]
            metas = pool.get("metadatas", [[]])[0][:RERANK_POOL_SIZE]
            passages = _passages(docs, metas)
            spans.append((len(all_passages), len(passages)))
            all_passages.extend(passages)
            kept.append((docs, metas))
        if not all_passages:
            return [rerank(query, p, top_n) for p in pools]

        embeddings = model.encode(all_passages, is_query=False)   # ONE pass

        out = []
        for (start, count), (docs, metas) in zip(spans, kept):
            if not count:
                out.append({"documents": [[]], "metadatas": [[]]})
                continue
            d_emb = list(embeddings[start:start + count])
            ranked = rank.rerank(
                documents_ids=[list(range(count))],
                queries_embeddings=q_emb,
                documents_embeddings=[d_emb],
            )
            order = [r["id"] for r in ranked[0][:top_n]]
            out.append({"documents": [[docs[i] for i in order]],
                        "metadatas": [[metas[i] for i in order]]})
        return out
    except Exception:
        return [rerank(query, p, top_n) for p in pools]


def _rerank_colbert(query: str, pool_docs: list[str], pool_metas: list[dict], top_n: int) -> list[int]:
    from pylate import rank
    from src import colbert_index

    model = colbert_index.get_model()
    q_emb = model.encode([query], is_query=True)

    passages = _passages(pool_docs, pool_metas)
    cached = colbert_index.get_cached_embeddings_by_meta(pool_metas) if USE_CACHED_COLBERT_EMBEDDINGS else None
    if cached is None:
        cached = [None] * len(pool_metas)
    if IDENTITY_ENRICHED_RERANK_ENABLED:
        # The cache is keyed by (source_url, chunk_index), computed offline
        # from the non-enriched passage - stale, not just unavailable, for
        # any candidate _passages() actually appended an identity suffix to.
        # Force a cache miss for exactly those (most candidates have no
        # identity record and are genuinely unaffected, so still benefit
        # from the cache) rather than disabling caching wholesale.
        cached = [None if _identity_suffix(meta) else c for c, meta in zip(cached, pool_metas)]
    to_encode = [i for i, c in enumerate(cached) if c is None]
    fresh = iter(model.encode([passages[i] for i in to_encode], is_query=False)) if to_encode else iter([])
    d_emb = [c if c is not None else next(fresh) for c in cached]

    results = rank.rerank(
        documents_ids=[list(range(len(passages)))],
        queries_embeddings=q_emb,
        documents_embeddings=[d_emb],
    )
    return [r["id"] for r in results[0][:top_n]]


def _cap_per_document(order: list, metas: list, top_n: int, cap: int) -> list:
    """At most `cap` chunks per source document, refilled from the next best."""
    seen, kept, spill = {}, [], []
    for i in order:
        key = str((metas[i] or {}).get("source_url") or i)
        if seen.get(key, 0) < cap:
            seen[key] = seen.get(key, 0) + 1
            kept.append(i)
        else:
            spill.append(i)
        if len(kept) >= top_n:
            return kept
    # Not enough distinct documents to fill top_n - fall back to the capped
    # ones rather than returning a short list, so a narrow corpus area still
    # gets a full context window.
    return (kept + spill)[:top_n]


def rerank(query: str, results: dict, top_n: int) -> dict:
    """Rescores the top RERANK_POOL_SIZE candidates in `results` and returns
    the top_n reordered. Candidates beyond the rerank pool are dropped (they
    were already deep enough to be unlikely to matter, and keeping the scoring
    pass bounded keeps latency predictable)."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return results

    pool_size = RERANK_POOL_SIZE
    if TARGETED_WIDENING_ENABLED:
        if _top_family_count(metadatas[:RERANK_POOL_SIZE]) <= FRAGMENTATION_THRESHOLD:
            pool_size = WIDE_RERANK_POOL_SIZE

    pool_docs = documents[:pool_size]
    pool_metas = metadatas[:pool_size]

    if not RERANK_ENABLED:
        order = list(range(min(top_n, len(pool_docs))))   # fusion order, untouched
    elif BACKEND == "colbert":
        order = _rerank_colbert(query, pool_docs, pool_metas, top_n)
    else:
        order = _rerank_cross_encoder(query, pool_docs, pool_metas, top_n)

    if RERANK_MAX_PER_DOC > 0:
        # Applied AFTER scoring, so the cap changes which chunks survive but
        # never which are considered. Scored order is preserved among keepers,
        # and the freed slots are refilled from the next-best candidates the
        # reranker already ranked - so the pool is not simply shortened.
        order = _cap_per_document(order, pool_metas, top_n, RERANK_MAX_PER_DOC)

    return {
        "documents": [[pool_docs[i] for i in order]],
        "metadatas": [[pool_metas[i] for i in order]],
        "distances": [[None] * len(order)],
    }
