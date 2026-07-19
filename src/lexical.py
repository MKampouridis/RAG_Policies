"""In-memory BM25 index over the chunk store, for hybrid (lexical + dense)
retrieval. Dense embeddings miss exact-term queries ("Capped Mark", "Model A",
course codes) that BM25 handles trivially; results are fused with the dense
ranking in src/rag.py via reciprocal-rank fusion.

The index is built lazily on first use from the same Chroma collection the
dense side queries (~12.6k chunks, a few seconds, held in memory). Staleness
is handled via the corpus version marker (src/ingest.py bump_corpus_version):
ingestion and flag recomputation - which may run in a different process -
bump the marker, and the next query here notices and rebuilds, so the BM25
side can't serve deleted chunks or stale is_current flags indefinitely.
"""

import re
import threading

from rank_bm25 import BM25Okapi

from src.ingest import _get_collection, read_corpus_version

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tried boosting this (repeating the header several times so identity terms
# like "CSEE"/"4yr" outweigh generic boilerplate body text) to help
# disambiguate near-identical RoA siblings - regressed RoA hit@6 in the full
# eval (60%->53%) despite improving hand-picked exemplars, because boosting
# amplifies the header's generic shared words ("masters", "rules", "year")
# right along with the genuinely distinguishing ones, and the corpus has more
# of the former. Reverted to 1x (see eval/EXPERIMENTS.md "stage2_header_boost").
HEADER_WEIGHT = 1

_index = None
_index_version = None
_lock = threading.Lock()


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class _BM25Index:
    def __init__(self):
        collection = _get_collection()
        data = collection.get(include=["documents", "metadatas"])
        self.ids = data["ids"]
        self.documents = data["documents"]
        self.metadatas = data["metadatas"]
        # index header (repeated HEADER_WEIGHT times) + body so document
        # identity is searchable and isn't drowned out by shared boilerplate
        corpus = [
            _tokenize(meta.get("chunk_header") or "") * HEADER_WEIGHT + _tokenize(doc)
            for doc, meta in zip(self.documents, self.metadatas)
        ]
        # BM25Okapi divides by corpus size; an empty collection (fresh setup,
        # or EMBED_MODEL switched before re-embedding) must not crash queries
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def query(self, text: str, n_results: int, current_only: bool, year: str,
              degree_length: str = "", award_type: str = "") -> list[tuple[str, str, dict, float]]:
        """Returns [(chunk_id, document, metadata, bm25_score)] best-first.
        `year` (canonical '2025-26' form) filters to chunks labeled with that
        academic year. `degree_length`/`award_type`, if given, filter to that
        closed-vocabulary facet value (see src/docid.py) - same hard-narrowing
        intent as `year`. The score is exposed (not just rank) so callers can
        do weighted score fusion (src/rag.py's _weighted_dense_bm25), not just
        reciprocal-rank fusion - a 3-tuple-only interface would discard it."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(text))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0:
                break
            meta = self.metadatas[i]
            if current_only and not meta.get("is_current"):
                continue
            if year and meta.get("academic_year_norm") != year:
                continue
            if degree_length and meta.get("degree_length") != degree_length:
                continue
            if award_type and meta.get("award_type") != award_type:
                continue
            hits.append((self.ids[i], self.documents[i], meta, float(scores[i])))
            if len(hits) >= n_results:
                break
        return hits


def query(text: str, n_results: int, current_only: bool = False, year: str = "",
          degree_length: str = "", award_type: str = "") -> list[tuple[str, str, dict, float]]:
    global _index, _index_version
    version = read_corpus_version()
    with _lock:
        if _index is None or _index_version != version:
            _index = _BM25Index()
            _index_version = version
        return _index.query(text, n_results, current_only, year, degree_length, award_type)
