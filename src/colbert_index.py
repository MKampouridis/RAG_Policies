"""Ideas 1+2 (see eval/report.md "Code review round"): query-time access to
the persisted ColBERT index built by build_colbert_index.py. One index,
two uses:

- query(): first-stage retrieval via Voyager's ANN token search + exact
  MaxSim rerank over the retrieved candidates (PyLate's retrieve.ColBERT),
  searching the FULL corpus rather than only whatever dense+BM25 already
  surfaced - directly targets the out-of-pool miss class J0 found (4/12
  misses whose correct document was never even in the candidate pool, so no
  reranker could have rescued them). Same (id, doc, meta) hit-tuple shape as
  src/lexical.py/src/splade.py/src/ensemble.py so it fuses identically via
  src/rag.py's _rrf_fuse().
- get_cached_embeddings_by_meta(): looks up precomputed token embeddings for
  chunks already in the index, so src/rerank.py's reranking step can stop
  re-encoding the same chunk text from scratch on every single query.

The ColBERT model instance is shared with src/rerank.py (get_model() here is
the single canonical loader both modules call) so it's loaded once per
process, not twice.
"""

import json
from pathlib import Path

from pylate import indexes, retrieve

MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"
INDEX_FOLDER = "data/colbert_index"
INDEX_NAME = "chunks"
DOCS_PATH = Path("data/colbert_docs.json")

_model = None
_index = None
_retriever = None
_ids = None
_documents = None
_metadatas = None
_id_to_pos = None
_meta_key_to_pos = None


def get_model():
    global _model
    if _model is None:
        from pylate import models
        _model = models.ColBERT(model_name_or_path=MODEL_NAME)
    return _model


def _load() -> None:
    global _index, _retriever, _ids, _documents, _metadatas, _id_to_pos, _meta_key_to_pos
    if _index is not None:
        return
    if not DOCS_PATH.exists():
        raise RuntimeError("ColBERT index not built yet - run `python build_colbert_index.py` first")
    _index = indexes.Voyager(index_folder=INDEX_FOLDER, index_name=INDEX_NAME, override=False)
    _retriever = retrieve.ColBERT(index=_index)
    cached = json.loads(DOCS_PATH.read_text())
    _ids = cached["ids"]
    _documents = cached["documents"]
    _metadatas = cached["metadatas"]
    _id_to_pos = {id_: i for i, id_ in enumerate(_ids)}
    _meta_key_to_pos = {
        (m.get("source_url"), m.get("chunk_index")): i for i, m in enumerate(_metadatas)
    }


def get_cached_embeddings_by_meta(pool_metas: list[dict]) -> list | None:
    """Idea 1: for each item in pool_metas, the cached embedding if that
    exact chunk (identified by (source_url, chunk_index), which survives
    src/rag.py's whole fusion/dedup pipeline unchanged - unlike a raw Chroma
    id, which doesn't) is in the persisted index, else None. Returns a list
    the same length as pool_metas; callers fall back to fresh encoding for
    the None entries. Returns None (not a list) if the index isn't built at
    all, so callers can short-circuit to encoding everything."""
    try:
        _load()
    except RuntimeError:
        return None

    keys = [(m.get("source_url"), m.get("chunk_index")) for m in pool_metas]
    positions = [_meta_key_to_pos.get(k) for k in keys]
    found_ids = [_ids[p] for p in positions if p is not None]
    if not found_ids:
        return [None] * len(pool_metas)

    found_embeddings = _index.get_documents_embeddings([found_ids])[0]
    id_to_embedding = dict(zip(found_ids, found_embeddings))

    return [id_to_embedding.get(_ids[p]) if p is not None else None for p in positions]


def query(text: str, n_results: int, current_only: bool = False, year: str = "") -> list[tuple[str, str, dict]]:
    """First-stage retrieval (Idea 2). Over-fetches before filtering since
    Voyager has no server-side metadata filter (is_current/year aren't
    index fields) - same pattern as the other channels' post-hoc filtering."""
    _load()
    model = get_model()
    q_emb = model.encode([text], is_query=True)
    raw = _retriever.retrieve(queries_embeddings=q_emb, k=min(n_results * 6, len(_ids)))

    hits = []
    for r in raw[0]:
        pos = _id_to_pos.get(r["id"])
        if pos is None:
            continue
        meta = _metadatas[pos]
        if current_only and not meta.get("is_current"):
            continue
        if year and meta.get("academic_year_norm") != year:
            continue
        hits.append((_ids[pos], _documents[pos], meta))
        if len(hits) >= n_results:
            break
    return hits
