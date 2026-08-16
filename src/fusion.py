"""Fusing and de-duplicating ranked lists from the retrieval channels.

Split out of rag.py 2026-08-13. Pure move - no behaviour change.

Reciprocal Rank Fusion combines the dense and BM25 lists. Weighted score fusion
was tried as an alternative and rejected (_weighted_dense_bm25, off) - RRF wins
because the two channels' scores are not on a comparable scale, and normalising
them was worse than ignoring their magnitudes entirely.
"""

RRF_K = 60


def _dense_as_hits(dense: dict) -> list[tuple[str, str, dict]]:
    return list(zip(
        dense.get("ids", [[]])[0],
        dense.get("documents", [[]])[0],
        dense.get("metadatas", [[]])[0],
    ))


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1] within the given pool. Relative, not tied
    to a specific distance metric's absolute scale - works whether the
    incoming values are Chroma distances (lower=better, metric-dependent) or
    BM25 scores (higher=better, unbounded), as long as the caller flips the
    sign consistently before calling this."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _weighted_dense_bm25(dense: dict, bm25_hits: list[tuple[str, str, dict, float]],
                          dense_weight: float, bm25_weight: float) -> list[tuple[str, str, dict]]:
    """Combines one dense result and one BM25 result list via a normalized
    weighted score sum (Stage F) instead of reciprocal rank, per Bruch et al.
    2022's finding that tuned convex fusion outperforms RRF because it uses
    how much better a candidate scored, not just its rank position. Returns
    a single best-first (id, doc, meta) list, drop-in compatible with
    _rrf_fuse's inputs so it can still be combined with other signals
    (soft facet/year preference, SPLADE, embedding ensemble) upstream."""
    ids = dense.get("ids", [[]])[0]
    docs = dense.get("documents", [[]])[0]
    metas = dense.get("metadatas", [[]])[0]
    dists = dense.get("distances", [[]])[0]

    # lower distance = better match; negate before normalizing so higher
    # normalized value = better, matching BM25's own orientation
    dense_scores = _normalize({i: -d for i, d in zip(ids, dists)})
    entries: dict[str, tuple[str, dict]] = {i: (doc, meta) for i, doc, meta in zip(ids, docs, metas)}

    bm25_raw = {id_: score for id_, doc, meta, score in bm25_hits}
    for id_, doc, meta, _score in bm25_hits:
        entries.setdefault(id_, (doc, meta))
    bm25_scores = _normalize(bm25_raw)

    all_ids = set(dense_scores) | set(bm25_scores)
    combined = {
        i: dense_weight * dense_scores.get(i, 0.0) + bm25_weight * bm25_scores.get(i, 0.0)
        for i in all_ids
    }
    ordered = sorted(all_ids, key=lambda i: combined[i], reverse=True)
    return [(i, entries[i][0], entries[i][1]) for i in ordered]


def _rrf_fuse(*ranked_lists: list[tuple]) -> dict:
    """Reciprocal-rank fusion of any number of ranked (id, doc, meta, ...)
    lists, keyed by chunk id. Dense embeddings and BM25 fail on different
    queries (semantic paraphrase vs exact terms like "Capped Mark" or course
    codes), so the union ranked by combined reciprocal rank beats either
    alone. Items may carry extra trailing elements (e.g. BM25's raw score,
    used elsewhere for weighted fusion) - only the first three are used
    here, so both 3- and 4-tuple inputs work unchanged."""
    scores: dict[str, float] = {}
    entries: dict[str, tuple[str, dict]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            id_, doc, meta = item[0], item[1], item[2]
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank)
            entries.setdefault(id_, (doc, meta))

    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    # `ids` is carried forward so this output has the SAME shape as the Chroma
    # result `_dense_as_hits` expects. It did not, and the two functions sit in
    # one small module: feeding a fused dict back into `_dense_as_hits` hit its
    # `.get("ids", [[]])` default and returned an EMPTY list rather than raising.
    # No live path does that today - all five call sites pass fresh Chroma
    # results - so this closes a trap, it does not fix an observed defect.
    # Purely additive: every existing consumer reads documents/metadatas/
    # distances and is unaffected.
    return {
        "ids": [[i for i in ordered]],
        "documents": [[entries[i][0] for i in ordered]],
        "metadatas": [[entries[i][1] for i in ordered]],
        "distances": [[None] * len(ordered)],
    }


def _dedup_by_chunk(results: dict) -> dict:
    """Stage G's pseudo-query entries share a real chunk's (source_url,
    chunk_index) but carry a distinct id ("<chunk_id>_pqN"), so after fusion
    the same real content can appear twice under two different ids - once
    found via its own embedding, once via a pseudo-query's. Collapse to one
    entry per (source_url, chunk_index), keeping whichever occurrence ranked
    higher (results are already best-first at this point)."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[None] * len(documents)])[0]
    # Carried through for the same reason `_rrf_fuse` now emits it: so a dict
    # travelling this pipeline keeps one shape end to end. Defaulted by length
    # rather than assumed present, because this is also called on dicts built
    # elsewhere that have no ids.
    ids = results.get("ids", [[None] * len(documents)])[0]

    seen: set[tuple] = set()
    kept_ids, kept_docs, kept_metas, kept_dists = [], [], [], []
    for id_, doc, meta, dist in zip(ids, documents, metadatas, distances):
        key = (meta.get("source_url"), meta.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        kept_ids.append(id_)
        kept_docs.append(doc)
        kept_metas.append(meta)
        kept_dists.append(dist)

    return {"ids": [kept_ids], "documents": [kept_docs],
            "metadatas": [kept_metas], "distances": [kept_dists]}
