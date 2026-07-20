#!/usr/bin/env python3
"""J0 diagnostic (a): for each RoA miss in a results file, re-run the
PRE-RERANK retrieval using the exact retrieval_query the eval stored, and
report whether the expected document appears anywhere in the fused
48-candidate pool. In-pool misses are rerank problems (the candidate was
found but mis-ranked); out-of-pool misses are retrieval problems (no
reranker could have saved them).

Usage: PYTHONPATH=. python eval/diagnose_pool_recall.py [results_path]
"""

import json
import sys
from pathlib import Path

from src import lexical
from src import rag

RESULTS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results_stage_colbert.json")


def prerank_pool(retrieval_query: str) -> list[dict]:
    """Reproduces retrieve()'s default-branch candidate pool exactly as of
    stage_colbert (dense+BM25, is_current, RRF, recency dedupe), stopping
    BEFORE the rerank step."""
    pool_size = rag.N_RESULTS * rag.FETCH_POOL_MULTIPLIER
    asked_year = rag._mentioned_year(retrieval_query)
    if asked_year:
        year_dense = rag.vector_query(retrieval_query, n_results=pool_size,
                                      where={"academic_year_norm": asked_year})
        year_bm25 = lexical.query(retrieval_query, n_results=pool_size, year=asked_year)
        cur_dense = rag.vector_query(retrieval_query, n_results=pool_size, where={"is_current": True})
        cur_bm25 = lexical.query(retrieval_query, n_results=pool_size, current_only=True)
        candidates = rag._rrf_fuse(
            rag._dense_as_hits(year_dense), year_bm25,
            rag._dense_as_hits(cur_dense), cur_bm25,
        )
    else:
        dense = rag.vector_query(retrieval_query, n_results=pool_size, where={"is_current": True})
        bm25_hits = lexical.query(retrieval_query, n_results=pool_size, current_only=True)
        candidates = rag._prefer_most_recent_year(rag._rrf_fuse(rag._dense_as_hits(dense), bm25_hits))
    return candidates.get("metadatas", [[]])[0]


def main():
    data = json.loads(RESULTS_PATH.read_text())
    in_pool, out_of_pool = [], []
    for item in data:
        if item["doc_type"] != "rules_of_assessment":
            continue
        for tk in ["primary", "follow_up"]:
            turn = item[tk]
            if turn["retrieval"]["hit_at_6"]:
                continue
            rq = turn["retrieval"]["retrieval_query"]
            expected = item["source_url"]
            pool_metas = prerank_pool(rq)
            rank = None
            for i, meta in enumerate(pool_metas, 1):
                if meta.get("source_url") == expected:
                    rank = i
                    break
            label = f"{item['source_title'][:40]:41s} {tk:9s}"
            if rank is not None:
                in_pool.append((label, rank, len(pool_metas)))
                print(f"IN-POOL  rank {rank:3d}/{len(pool_metas):3d}  {label}", flush=True)
            else:
                out_of_pool.append((label, len(pool_metas)))
                print(f"OUT      ---/{len(pool_metas):3d}  {label}", flush=True)

    print(f"\nSummary: {len(in_pool)} in-pool (rerank problem), {len(out_of_pool)} out-of-pool (retrieval problem)")


if __name__ == "__main__":
    main()
