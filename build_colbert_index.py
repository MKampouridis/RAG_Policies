#!/usr/bin/env python3
"""Ideas 1+2 (see eval/report.md "Code review round"): precompute and
persist ColBERT token-level embeddings for every chunk in the corpus,
using PyLate's Voyager (HNSW) multi-vector index. Two payoffs from one
offline build:

1. First-stage retrieval (Idea 2): src/colbert_index.py's query() does a
   real ANN search over token embeddings across the WHOLE corpus, not just
   whatever dense+BM25 happened to surface - directly targets the 4/12
   out-of-pool misses J0 found, which no reranker can rescue since
   reranking only ever sees what earlier retrieval already picked.
2. Cached reranking (Idea 1): src/rerank.py's _rerank_colbert() currently
   re-encodes the same chunk text from scratch on every single query, pure
   waste since chunk text doesn't change between queries. Once every chunk
   has a persisted embedding, reranking becomes a lookup
   (index.get_documents_embeddings) instead of a fresh BERT forward pass.

Encodes header+text (matching src/rerank.py's _passages() convention -
document identity lives in chunk_header, not the stored chunk text alone).

Usage: PYTHONPATH=. python build_colbert_index.py
Resumable in the loose sense that Voyager's add_documents can be re-run
incrementally, but this script rebuilds from scratch each time (override=True)
since a partial index is worse than a clear signal to re-run after any
interruption - matches build_splade_index.py's mode.
"""

import json
import time

from pylate import indexes, models

from src.ingest import _get_collection

MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"
INDEX_FOLDER = "data/colbert_index"
INDEX_NAME = "chunks"
DOCS_PATH = "data/colbert_docs.json"
BATCH_SIZE = 32


def _passages(documents: list[str], metadatas: list[dict]) -> list[str]:
    # mirrors src/rerank.py's _passages() exactly - keep the two in sync if
    # either changes, since the whole point of caching is that the SAME text
    # gets embedded once here and looked up (not re-embedded) at query time
    return [f"{meta.get('chunk_header', '')}\n{doc}" for doc, meta in zip(documents, metadatas)]


def run() -> None:
    collection = _get_collection()
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    n = len(ids)
    print(f"Encoding {n} chunks with {MODEL_NAME} into a Voyager index ...", flush=True)

    passages = _passages(documents, metadatas)
    model = models.ColBERT(model_name_or_path=MODEL_NAME)
    index = indexes.Voyager(
        index_folder=INDEX_FOLDER,
        index_name=INDEX_NAME,
        override=True,
        embedding_size=128,
    )

    t0 = time.time()
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch_embeddings = model.encode(passages[start:end], is_query=False, batch_size=BATCH_SIZE)
        index.add_documents(documents_ids=ids[start:end], documents_embeddings=batch_embeddings)
        if end % (BATCH_SIZE * 10) == 0 or end == n:
            elapsed = time.time() - t0
            rate = end / elapsed
            eta = (n - end) / rate if rate else 0
            print(f"  [{end}/{n}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    with open(DOCS_PATH, "w") as f:
        json.dump({"ids": ids, "documents": documents, "metadatas": metadatas}, f, ensure_ascii=False)

    print(f"Done. Wrote Voyager index to {INDEX_FOLDER}/{INDEX_NAME} and {DOCS_PATH}")


if __name__ == "__main__":
    run()
