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

from src.docid import document_family as _document_family

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
RERANK_POOL_SIZE = 30

# Idea 4 (targeted widening) - tried, regressed WORSE than J0b's naive
# global widening (eval/report.md "Code review round"): 0 rescues / 4 losses
# (RoA hit@6 70%->60%), vs J0b's 2 rescues / 5 losses. The pre-rerank
# family-fragmentation signal apparently doesn't correlate with "the right
# document is deeper in the pool" - it fired on queries where widening only
# added noise, and never once on the out-of-pool cases it was meant to
# catch. Off by default; kept for reference.
TARGETED_WIDENING_ENABLED = False
WIDE_RERANK_POOL_SIZE = 100
FRAGMENTATION_THRESHOLD = 1

# Idea 1 (cached ColBERT embeddings, see eval/report.md "Code review round"):
# once build_colbert_index.py has run, reuse each candidate's precomputed
# token embedding (looked up by (source_url, chunk_index), which survives
# the whole fusion/dedup pipeline unchanged) instead of re-encoding its text
# from scratch on every single query - a chunk's embedding never changes
# between queries, so re-encoding it repeatedly is pure waste. Falls back to
# fresh encoding per-candidate when the index isn't built or a candidate
# isn't in it yet, so this is safe to leave on unconditionally - production
# behavior is byte-identical to before until the index actually exists.
USE_CACHED_COLBERT_EMBEDDINGS = True

_cross_encoder = None


def _top_family_count(pool_metas: list[dict]) -> int:
    if not pool_metas:
        return 0
    top_family = _document_family(pool_metas[0].get("source_url", ""))
    return sum(1 for m in pool_metas if _document_family(m.get("source_url", "")) == top_family)


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


def _passages(pool_docs: list[str], pool_metas: list[dict]) -> list[str]:
    # score against header+text, not the bare stored chunk - the document
    # identity (degree length, department, year) that actually disambiguates
    # near-identical RoA siblings lives only in chunk_header (prepended at
    # embedding time, never stored in `documents`); without it the reranker
    # sees strictly less signal than the embedder already had
    return [f"{meta.get('chunk_header', '')}\n{doc}" for doc, meta in zip(pool_docs, pool_metas)]


def _rerank_cross_encoder(query: str, pool_docs: list[str], pool_metas: list[dict], top_n: int) -> list[int]:
    passages = _passages(pool_docs, pool_metas)
    scores = _get_cross_encoder().predict([(query, p) for p in passages])
    return sorted(range(len(pool_docs)), key=lambda i: scores[i], reverse=True)[:top_n]


def _rerank_colbert(query: str, pool_docs: list[str], pool_metas: list[dict], top_n: int) -> list[int]:
    from pylate import rank
    from src import colbert_index

    model = colbert_index.get_model()
    q_emb = model.encode([query], is_query=True)

    passages = _passages(pool_docs, pool_metas)
    cached = colbert_index.get_cached_embeddings_by_meta(pool_metas) if USE_CACHED_COLBERT_EMBEDDINGS else None
    if cached is None:
        cached = [None] * len(pool_metas)
    to_encode = [i for i, c in enumerate(cached) if c is None]
    fresh = iter(model.encode([passages[i] for i in to_encode], is_query=False)) if to_encode else iter([])
    d_emb = [c if c is not None else next(fresh) for c in cached]

    results = rank.rerank(
        documents_ids=[list(range(len(passages)))],
        queries_embeddings=q_emb,
        documents_embeddings=[d_emb],
    )
    return [r["id"] for r in results[0][:top_n]]


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

    if BACKEND == "colbert":
        order = _rerank_colbert(query, pool_docs, pool_metas, top_n)
    else:
        order = _rerank_cross_encoder(query, pool_docs, pool_metas, top_n)

    return {
        "documents": [[pool_docs[i] for i in order]],
        "metadatas": [[pool_metas[i] for i in order]],
        "distances": [[None] * len(order)],
    }
