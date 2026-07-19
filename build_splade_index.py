#!/usr/bin/env python3
"""Precomputes SPLADE sparse vectors for every chunk in the active Chroma
collection and caches them to disk (data/splade_matrix.npz + data/splade_docs.json).

Unlike BM25 (src/lexical.py), which rebuilds from scratch in a few seconds on
every server start, a SPLADE forward pass is a ~66M-param BERT encode per
chunk - expensive enough that it must be a deliberate, one-off offline step,
not something src/splade.py silently redoes when the corpus version changes.
Re-run this script manually after any re-embed/ingest that should be
reflected in SPLADE retrieval.
"""

import json
import time
from pathlib import Path

from scipy import sparse

from src.ingest import _get_collection, read_corpus_version
from src.splade import SPLADE_MODEL_NAME

INDEX_PATH = Path("data/splade_matrix.npz")
DOCS_PATH = Path("data/splade_docs.json")
BATCH_SIZE = 64


def run() -> None:
    from sentence_transformers import SparseEncoder

    collection = _get_collection()
    data = collection.get(include=["documents", "metadatas"])
    ids = data["ids"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    n = len(ids)
    print(f"Encoding {n} chunks with {SPLADE_MODEL_NAME} ...", flush=True)

    model = SparseEncoder(SPLADE_MODEL_NAME)
    rows = []
    t0 = time.time()
    for start in range(0, n, BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]
        emb = model.encode_document(batch, convert_to_sparse_tensor=True)
        rows.append(sparse.csr_matrix(emb.to_dense().cpu().numpy()))
        done = min(start + BATCH_SIZE, n)
        if done % (BATCH_SIZE * 10) == 0 or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = (n - done) / rate if rate else 0
            print(f"  [{done}/{n}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining", flush=True)

    matrix = sparse.vstack(rows).tocsr()
    sparse.save_npz(INDEX_PATH, matrix)
    DOCS_PATH.write_text(json.dumps({
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "corpus_version": read_corpus_version(),
    }, ensure_ascii=False))
    print(f"Done. Wrote {INDEX_PATH} ({matrix.shape}) and {DOCS_PATH}")


if __name__ == "__main__":
    run()
