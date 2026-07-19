"""Stage G: queries the pseudo-query Chroma collection (built offline by
build_pseudo_query_index.py) as a fourth retrieval channel, fused via RRF in
src/rag.py alongside dense/BM25/facet/SPLADE/ensemble. Each entry's embedding
is a deterministic, metadata-filled question template, but its `documents`/
`metadatas` point back at the real underlying chunk - so a hit here surfaces
real content through an access path the primary chunk embedding alone might
miss (see build_pseudo_query_index.py's docstring)."""

import chromadb

from src.ingest import CHROMA_DIR
from src.llm import EMBED_MODEL, EMBED_QUERY_PREFIX, embed_batch

PSEUDO_COLLECTION_NAME = "pseudo_query_" + EMBED_MODEL.replace("-", "_")

_client = None


def _get_pseudo_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client.get_or_create_collection(PSEUDO_COLLECTION_NAME)


def query(text: str, n_results: int, where: dict | None = None) -> list[tuple[str, str, dict]]:
    """Returns [(chunk_id, document, metadata)] best-first, same shape as
    src/lexical.py's/src/splade.py's query() so it fuses identically via
    src/rag.py's _rrf_fuse(). Ids are the underlying real chunk id with a
    "_pqN" suffix - distinct from the primary collection's ids, so a
    post-fusion dedup by (source_url, chunk_index) is needed upstream to
    avoid the same real content being counted twice (see src/rag.py
    _dedup_by_chunk)."""
    collection = _get_pseudo_collection()
    query_embedding = embed_batch([EMBED_QUERY_PREFIX + text])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where)
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return list(zip(ids, documents, metadatas))
