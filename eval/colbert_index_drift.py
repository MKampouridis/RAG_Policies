#!/usr/bin/env python3
"""Detect ColBERT index drift against the live corpus.

WHY: the reranker's embedding cache was three weeks stale and nothing noticed.
run_ingest.py and reembed.py update Chroma; neither rebuilds the ColBERT index,
so after any ingest the cached embeddings describe text that no longer exists.
That silently cost 5 turns of hit@6 (Round 27).

Run after every ingest, alongside audit_family_aliases.py and
eval/stale_index_audit.py.
"""
import sys


def main() -> int:
    from src import colbert_index as ci, ingest
    try:
        ci._load()
    except RuntimeError as exc:
        print(f"ColBERT index not built: {exc}")
        return 0
    n_index = len(ci._ids)
    n_chroma = ingest._get_collection().count()
    drift = n_chroma - n_index
    print(f"  ColBERT index : {n_index} chunks")
    print(f"  Chroma        : {n_chroma} chunks")
    print(f"  drift         : {drift:+d}")
    if drift:
        print("\n  STALE. The embedding cache describes a corpus that no longer matches.")
        print("  Either rebuild:      python build_colbert_index.py")
        print("  or leave it off:     RAG_COLBERT_CACHE=0 (current default)")
        return 1
    print("\n  in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
