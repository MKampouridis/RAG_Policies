#!/usr/bin/env python3
"""Which derived artifacts can go stale without anything noticing?

Two incidents, one class: the ColBERT embedding cache (three weeks stale, cost
5 turns of hit@6) and the eval set (9 items grading against superseded
documents). Both were derived from the corpus with nothing recording which
version they came from.

This lists every derived artifact and how it is protected:

  STAMPED    carries a corpus version, checked on read, fails closed
  REBUILDS   invalidates itself against read_corpus_version() at load
  UNPROTECTED  neither - can silently describe a corpus that no longer exists

Usage: PYTHONPATH=. python eval/artifact_provenance_audit.py
"""
import pathlib
import sys


def main() -> int:
    from src.provenance import describe, current_corpus_version

    corpus = current_corpus_version()
    print(f"\n  live corpus version: {str(corpus)[:16]}\n")

    # (path, how it is protected, what it is derived from)
    artifacts = [
        ("data/colbert_docs.json", "stamp", "chunk text + metadata"),
        ("data/colbert_index/", "rebuild-with-docs", "chunk text"),
        ("data/splade_matrix.npz", "none", "chunk text"),
        ("data/splade_docs.json", "none", "chunk text"),
        ("data/doc_identity/", "none", "per-document extraction"),
        ("eval/questions_regression.json", "stamp", "corpus snapshot at authoring time"),
        ("data/manifest.json", "source", "the crawl itself - this IS the input"),
    ]
    runtime = [
        ("src/lexical.py BM25 index", "rebuilds on read_corpus_version()"),
        ("src/doc_index.py", "rebuilds on read_corpus_version()"),
        ("contextualize._identity_anchor_index()", "process-lifetime cache, NO invalidation"),
        ("rag._uniform_parameter_terms() variance map", "process-lifetime cache, NO invalidation"),
    ]

    print(f"  {'artifact':<38}{'state':<26}derived from")
    unprotected = []
    for path, how, src in artifacts:
        p = pathlib.Path(path)
        if not p.exists():
            state = "absent"
        elif how == "stamp":
            d = describe(p)
            state = ("STAMPED, matches" if d["matches"]
                     else "STAMPED, MISMATCH" if d["stamped"] else "unstamped")
            if not d["stamped"]:
                unprotected.append(path)
        elif how == "source":
            state = "input, not derived"
        else:
            state = "UNPROTECTED"
            unprotected.append(path)
        print(f"  {path:<38}{state:<26}{src}")

    print(f"\n  {'runtime cache':<44}state")
    for name, state in runtime:
        print(f"  {name:<44}{state}")

    print(f"\n  UNPROTECTED artifacts on disk: {len(unprotected)}")
    for u in unprotected:
        print(f"    {u}")
    print("\n  The process-lifetime caches are lower risk - they die with the server,")
    print("  so the worst case is one stale process rather than three weeks. But a")
    print("  long-running production process IS three weeks, so they are not zero risk.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
