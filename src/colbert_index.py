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

import itertools
import os
import threading
from pathlib import Path

from pylate import indexes, retrieve

MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"
INDEX_FOLDER = "data/colbert_index"
INDEX_NAME = "chunks"
DOCS_PATH = Path("data/colbert_docs.json")

# query()'s over-fetch (n_results * 6, up to n_results=48 in production ->
# k=288) exceeds Voyager's constructor default ef_search=200, and the
# underlying HNSW search requires ef_search >= k ("queryEf must be equal to
# or greater than the requested number of neighbors" - hit 40/40 turns in
# the first Idea 2 eval run). ef_search is a per-instance query-time knob
# (pylate/indexes/voyager.py: stored as self.ef_search, only read in
# __call__'s query_ef=self.ef_search), not baked into the persisted graph -
# safe to raise here without rebuilding the index built by
# build_colbert_index.py. Comfortable headroom over the 288 max, not tied
# exactly to it so a future pool_size bump doesn't silently reopen this.
EF_SEARCH = 400

_model = None
_index = None
_retriever = None
_ids = None
_documents = None
_metadatas = None
_id_to_pos = None
_meta_key_to_pos = None
_ids_to_embeddings = None   # memoised pylate map; see _get_documents_embeddings

# Both loaders below are check-then-set on module globals. That was safe while
# only request threads reached them and the first request was alone; the
# startup warmup in src/app.py now runs concurrently with real requests, so two
# threads can pass the `is None` check together and each load a full ColBERT
# model - several GB of duplicate allocation on a 16GB machine. The lock makes
# the second caller wait for the first rather than repeat the work.
_load_lock = threading.Lock()


# Which device the reranker runs on. UNSET (the default) keeps pylate's
# auto-selection, which on Apple silicon picks the GPU via MPS - what every
# measurement in eval/report.md was taken on. Set RAG_COLBERT_DEVICE=cpu to
# pin the CPU.
#
# This exists because the GPU appears to be a PESSIMISATION here, not an
# optimisation: MaxSim over a 30-candidate pool of short passages is too small
# a workload to amortise kernel launches and host/device transfers. A 16-query
# timing put CPU at a 1.84s median against MPS's 4.54s. Left unset until a
# retrieval eval confirms the ranking is unchanged - a device switch alters
# float precision, which can reorder near-ties.
COLBERT_DEVICE = os.environ.get("RAG_COLBERT_DEVICE", "").strip().lower() or None


def get_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:  # re-check: another thread may have loaded it
                from pylate import models
                kwargs = {"device": COLBERT_DEVICE} if COLBERT_DEVICE else {}
                _model = models.ColBERT(model_name_or_path=MODEL_NAME, **kwargs)
    return _model


def _load() -> None:
    if _index is not None:
        return
    with _load_lock:
        if _index is not None:  # re-check under the lock
            return
        _load_locked()


def _load_locked() -> None:
    global _index, _retriever, _ids, _documents, _metadatas, _id_to_pos, _meta_key_to_pos
    if not DOCS_PATH.exists():
        raise RuntimeError("ColBERT index not built yet - run `python build_colbert_index.py` first")
    _index = indexes.Voyager(
        index_folder=INDEX_FOLDER, index_name=INDEX_NAME, override=False, ef_search=EF_SEARCH
    )
    _retriever = retrieve.ColBERT(index=_index)
    # A mismatch is refused rather than warned about: this exact artifact went
    # three weeks stale and silently degraded retrieval. Unstamped files are
    # accepted (they predate stamping) but a stamp that DISAGREES stops us.
    from src.provenance import read_stamped
    cached = read_stamped(DOCS_PATH, required=False)
    _ids = cached["ids"]
    _documents = cached["documents"]
    _metadatas = cached["metadatas"]
    _id_to_pos = {id_: i for i, id_ in enumerate(_ids)}
    _meta_key_to_pos = {
        (m.get("source_url"), m.get("chunk_index")): i for i, m in enumerate(_metadatas)
    }


def _get_documents_embeddings(found_ids: list) -> list:
    """pylate's Voyager.get_documents_embeddings() calls
    _load_documents_ids_to_embeddings() first, which is an unconditional
    pickle.load of the WHOLE-corpus id->embedding map, from disk, on EVERY
    call. Measured on this index: document_ids_to_embeddings.pkl is 23MB and
    takes ~136ms to load - paid once per query, to save the ~0.2s of encoding
    that this cache exists to avoid.

    The file is immutable between index builds, so it is read once here and the
    mapping reused. Falls back to pylate's own path if its internals change.
    """
    global _ids_to_embeddings
    try:
        if _ids_to_embeddings is None:
            _ids_to_embeddings = _index._load_documents_ids_to_embeddings()
        emb_ids = list(itertools.chain.from_iterable(
            _ids_to_embeddings[doc_id] for doc_id in found_ids))
        vectors = _index.index.get_vectors(emb_ids)
        out, at = [], 0
        for doc_id in found_ids:
            n = len(_ids_to_embeddings[doc_id])
            out.append(vectors[at:at + n])
            at += n
        return out
    except Exception:
        # any mismatch with pylate's internals: use the supported call
        return _index.get_documents_embeddings([found_ids])[0]


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

    found_embeddings = _get_documents_embeddings(found_ids)
    id_to_embedding = dict(zip(found_ids, found_embeddings))

    return [id_to_embedding.get(_ids[p]) if p is not None else None for p in positions]


def query(text: str, n_results: int, current_only: bool = False, year: str = "") -> list[tuple[str, str, dict]]:
    """First-stage retrieval (Idea 2). Over-fetches before filtering since
    Voyager has no server-side metadata filter (is_current/year aren't
    index fields) - same pattern as the other channels' post-hoc filtering."""
    _load()
    model = get_model()
    q_emb = model.encode([text], is_query=True)
    k = min(n_results * 6, len(_ids))
    # Defensive per-query clamp (external code review, 2026-07-21): EF_SEARCH
    # was a fixed constant with comfortable headroom over today's k=288 max,
    # but headroom isn't a guarantee - a future n_results/pool_size increase
    # could silently reopen the exact "queryEf must be >= k" crash this was
    # first raised to fix. Bumping the live index's ef_search here (a
    # query-time-only attribute, not baked into the persisted graph, so this
    # is always safe) guarantees k is covered regardless of what future pool
    # sizing looks like.
    if k > _index.ef_search:
        _index.ef_search = k
    raw = _retriever.retrieve(queries_embeddings=q_emb, k=k)

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
