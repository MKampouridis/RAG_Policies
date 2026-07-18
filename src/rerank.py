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

BACKEND = "colbert"  # "cross_encoder" (production) | "colbert" (experiment)

CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"
COLBERT_MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"
# how many of the fused candidates to actually score - failure analysis
# (eval/report.md) found relevant-but-mis-ranked documents as deep as rank 60
# in a top-50 dense+BM25 union, so this needs to be generous, not just N_RESULTS
RERANK_POOL_SIZE = 30

_cross_encoder = None
_colbert = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _cross_encoder


def _get_colbert():
    global _colbert
    if _colbert is None:
        from pylate import models
        _colbert = models.ColBERT(model_name_or_path=COLBERT_MODEL_NAME)
    return _colbert


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
    passages = _passages(pool_docs, pool_metas)
    model = _get_colbert()
    q_emb = model.encode([query], is_query=True)
    d_emb = model.encode(passages, is_query=False)
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

    pool_docs = documents[:RERANK_POOL_SIZE]
    pool_metas = metadatas[:RERANK_POOL_SIZE]

    if BACKEND == "colbert":
        order = _rerank_colbert(query, pool_docs, pool_metas, top_n)
    else:
        order = _rerank_cross_encoder(query, pool_docs, pool_metas, top_n)

    return {
        "documents": [[pool_docs[i] for i in order]],
        "metadatas": [[pool_metas[i] for i in order]],
        "distances": [[None] * len(order)],
    }
