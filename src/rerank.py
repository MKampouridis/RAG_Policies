"""Cross-encoder reranking of the fused candidate pool.

The earlier LLM listwise reranker (asking qwen2.5:7b to pick the best 6 of
24 excerpts) made every retrieval metric worse - a generalist chat model
reasoning over a list of near-identical boilerplate chunks broke more
correct rankings than it fixed. A cross-encoder is a fundamentally different
mechanism: it scores each (query, passage) pair independently, purpose-built
for exactly this fine-grained relevance judgment, not multi-item reasoning.
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"
# how many of the fused candidates to actually score - failure analysis
# (eval/report.md) found relevant-but-mis-ranked documents as deep as rank 60
# in a top-50 dense+BM25 union, so this needs to be generous, not just N_RESULTS
RERANK_POOL_SIZE = 30

_model = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


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

    # score against header+text, not the bare stored chunk - the document
    # identity (degree length, department, year) that actually disambiguates
    # near-identical RoA siblings lives only in chunk_header (prepended at
    # embedding time, never stored in `documents`); without it the reranker
    # sees strictly less signal than the embedder already had
    passages = [f"{meta.get('chunk_header', '')}\n{doc}" for doc, meta in zip(pool_docs, pool_metas)]
    scores = _get_model().predict([(query, p) for p in passages])
    order = sorted(range(len(pool_docs)), key=lambda i: scores[i], reverse=True)[:top_n]

    return {
        "documents": [[pool_docs[i] for i in order]],
        "metadatas": [[pool_metas[i] for i in order]],
        "distances": [[None] * len(order)],
    }
