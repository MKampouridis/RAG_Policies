"""SPLADE (learned sparse retrieval) as a third retrieval channel, fused via
RRF alongside dense (Chroma) and lexical (BM25) in src/rag.py. Unlike BM25's
raw term-frequency matching (already tried boosting, see src/lexical.py),
SPLADE learns which terms a query/document should expand to - reported in
the literature to bridge terminology mismatches (e.g. "MSc" vs "MA") that
exact-match BM25 can't (eval/report.md, literature-grounded round).

Doc-side vectors are precomputed offline by build_splade_index.py (a BERT
forward pass per chunk is too expensive to redo lazily on every server start
the way BM25's plain tokenization is) and loaded here lazily, read-only.
"""

import json
from pathlib import Path

from scipy import sparse

SPLADE_MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
INDEX_PATH = Path("data/splade_matrix.npz")
DOCS_PATH = Path("data/splade_docs.json")

_model = None
_matrix = None
_ids = None
_documents = None
_metadatas = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SparseEncoder
        _model = SparseEncoder(SPLADE_MODEL_NAME)
    return _model


def _load_index() -> None:
    global _matrix, _ids, _documents, _metadatas
    if _matrix is not None:
        return
    if not INDEX_PATH.exists() or not DOCS_PATH.exists():
        raise RuntimeError(
            "SPLADE index not built yet - run `python build_splade_index.py` first"
        )
    _matrix = sparse.load_npz(INDEX_PATH)
    cached = json.loads(DOCS_PATH.read_text())
    _ids = cached["ids"]
    _documents = cached["documents"]
    _metadatas = cached["metadatas"]


def query(text: str, n_results: int, current_only: bool = False, year: str = "",
          degree_length: str = "", award_type: str = "") -> list[tuple[str, str, dict]]:
    """Returns [(chunk_id, document, metadata)] best-first, same shape and
    filter semantics as src/lexical.py's query() so both fuse identically
    into src/rag.py's _rrf_fuse()."""
    _load_index()
    if _matrix.shape[0] == 0:
        return []

    q_emb = _get_model().encode_query([text], convert_to_sparse_tensor=True)
    q_vec = sparse.csr_matrix(q_emb.to_dense().cpu().numpy())
    scores = (_matrix @ q_vec.T).toarray().ravel()

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = []
    for i in order:
        if scores[i] <= 0:
            break
        meta = _metadatas[i]
        if current_only and not meta.get("is_current"):
            continue
        if year and meta.get("academic_year_norm") != year:
            continue
        if degree_length and meta.get("degree_length") != degree_length:
            continue
        if award_type and meta.get("award_type") != award_type:
            continue
        hits.append((_ids[i], _documents[i], meta))
        if len(hits) >= n_results:
            break
    return hits
