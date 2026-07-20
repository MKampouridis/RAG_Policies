"""Stage E: embedding-model ensemble - queries the already-populated bge-m3
Chroma collection (left over from the Stage 3 experiment, see
eval/EXPERIMENTS.md "stage3_bgem3") alongside the primary nomic-embed-text
collection, fused via RRF in src/rag.py. Different embedding models make
different, often uncorrelated mistakes; fusing rankings from two models can
beat either alone even when neither wins outright as a standalone
replacement (bge-m3 was a wash/slight RoA regression solo).

NOTE: the bge-m3 Ollama model itself was removed during a 2026-07-20 disk
cleanup (this stage was already reverted, so the model was unused). The
Chroma collection's stored vectors are untouched, but re-enabling
EMBEDDING_ENSEMBLE_ENABLED now needs `ollama pull bge-m3` first, or query()
will fail immediately when it tries to embed the query text."""

import chromadb

from src.ingest import CHROMA_DIR
from src.llm import embed_batch

SECONDARY_EMBED_MODEL = "bge-m3"
SECONDARY_COLLECTION_NAME = "policies_bge-m3"  # no query/doc prefix needed for bge-m3

_client = None


def _get_secondary_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client.get_or_create_collection(SECONDARY_COLLECTION_NAME)


def query(text: str, n_results: int, where: dict | None = None) -> list[tuple[str, str, dict]]:
    """Returns [(chunk_id, document, metadata)] best-first, same shape as
    src/lexical.py's/src/splade.py's query() so it fuses identically via
    src/rag.py's _rrf_fuse(). Only the `is_current` filter is applied here
    (not degree_length/award_type or year) since the bge-m3 collection
    predates those facets and Stage E's role is supplementary evidence, not
    the primary correctness contract."""
    collection = _get_secondary_collection()
    query_embedding = embed_batch([text], model=SECONDARY_EMBED_MODEL)[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where)
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return list(zip(ids, documents, metadatas))
