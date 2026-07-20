"""J3: document-level identity index. One record per document - an
"identity card" built from the readable title plus J1's extracted identity
record (programme, department, partner institution, awards, aliases; see
extract_doc_identity.py) - embedded into its own small Chroma collection
(~1,200 vectors) and matched lexically via a tiny in-memory BM25.

The chunk-level store can't discriminate near-identical RoA siblings because
identity is a tiny fraction of each chunk's tokens; here identity is ALL of
the record's tokens, with no boilerplate dilution. Used by src/rag.py as a
soft document-level prior: documents whose identity cards match the query
get their chunks boosted via an extra RRF list - never a hard filter, and
chunk embeddings themselves are untouched (the J2 lesson: corpus-wide chunk
perturbation displaces unrelated queries; a separate index can't).
"""

import json
import re
import threading
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from src.ingest import CHROMA_DIR, _readable_title, read_corpus_version, url_hash
from src.llm import EMBED_DOCUMENT_PREFIX, EMBED_MODEL, EMBED_QUERY_PREFIX, embed_batch

DOC_INDEX_COLLECTION = "doc_identity_" + re.sub(r"[^a-zA-Z0-9_-]", "_", EMBED_MODEL)
IDENTITY_DIR = Path("data/doc_identity")
MANIFEST_PATH = Path("data/manifest.json")

TOKEN_RE = re.compile(r"[a-z0-9]+")

_client = None
_bm25 = None
_bm25_urls = None
_bm25_cards = None
_bm25_version = None
_lock = threading.Lock()


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client.get_or_create_collection(DOC_INDEX_COLLECTION)


def build_identity_card(url: str, metadata: dict) -> str:
    """One text record fully describing a document's identity - all signal,
    no boilerplate."""
    parts = [_readable_title(url.rsplit("/", 1)[-1])]
    if metadata.get("doc_type"):
        parts.append(metadata["doc_type"].replace("_", " "))
    identity_path = IDENTITY_DIR / f"{url_hash(url)}.json"
    if identity_path.exists():
        try:
            identity = json.loads(identity_path.read_text())
        except Exception:
            identity = {}
        for key in ("programme_name", "department", "partner_institution"):
            if identity.get(key):
                parts.append(identity[key])
        if identity.get("awards"):
            parts.append(", ".join(identity["awards"]))
        if identity.get("aliases"):
            parts.append(", ".join(identity["aliases"]))
    if metadata.get("academic_year"):
        parts.append(f"academic year {metadata['academic_year']}")
    return " | ".join(parts)


def build_index() -> int:
    """Offline build: one embedded identity card per kept document, plus the
    is_current flag copied over so retrieval-time filtering matches the
    chunk store's. Rebuild after any re-ingest that changes identity records
    or currency flags. Returns the number of documents indexed."""
    manifest = json.loads(MANIFEST_PATH.read_text())["documents"]
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    chunk_coll = client.get_or_create_collection(
        "policies_" + re.sub(r"[^a-zA-Z0-9_-]", "_", EMBED_MODEL))

    ids, texts, metadatas = [], [], []
    for url, doc in manifest.items():
        if not doc.get("keep"):
            continue
        # is_current lives on chunks, not the manifest - read it from chunk 0
        chunk0 = chunk_coll.get(where={"$and": [{"source_url": url}, {"chunk_index": 0}]},
                                 limit=1, include=["metadatas"])
        chunk_metas = chunk0.get("metadatas") or []
        is_current = bool(chunk_metas[0].get("is_current")) if chunk_metas else False
        meta = {"doc_type": doc.get("doc_type"), "academic_year": doc.get("academic_year")}
        card = build_identity_card(url, meta)
        ids.append(url_hash(url))
        texts.append(card)
        metadatas.append({
            "source_url": url,
            "identity_card": card,
            "is_current": is_current,
        })

    coll = _get_collection()
    batch = 128
    for start in range(0, len(ids), batch):
        embeddings = embed_batch([EMBED_DOCUMENT_PREFIX + t for t in texts[start:start + batch]])
        coll.upsert(ids=ids[start:start + batch], embeddings=embeddings,
                    documents=texts[start:start + batch], metadatas=metadatas[start:start + batch])
    return len(ids)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _load_bm25():
    """Rebuilds when the corpus version marker moves, same staleness
    mechanism as src/lexical.py's BM25 index. This index is only rebuilt
    offline (build_index()), not incrementally on every upsert like the
    chunk collection, so this is a conservative approximation - it reloads
    on any corpus change, not just ones that touched identity cards - but
    that only costs an extra rebuild, never serves stale data."""
    global _bm25, _bm25_urls, _bm25_cards, _bm25_version
    version = read_corpus_version()
    if _bm25 is not None and _bm25_version == version:
        return
    data = _get_collection().get(include=["metadatas"])
    metas = data["metadatas"]
    _bm25_urls = [m["source_url"] for m in metas]
    _bm25_cards = [(m.get("identity_card") or "", bool(m.get("is_current"))) for m in metas]
    corpus = [_tokenize(card) for card, _cur in _bm25_cards]
    _bm25 = BM25Okapi(corpus) if corpus else None
    _bm25_version = version


def query(text: str, n_results: int = 10, current_only: bool = True) -> list[str]:
    """Returns the source_urls of the documents whose identity cards best
    match the query, fusing the dense and BM25 views of the identity index
    by simple rank interleaving (dedup, dense first). Small enough that
    sophistication isn't warranted."""
    with _lock:
        _load_bm25()

    where = {"is_current": True} if current_only else None
    q_emb = embed_batch([EMBED_QUERY_PREFIX + text])[0]
    dense = _get_collection().query(query_embeddings=[q_emb], n_results=n_results, where=where)
    dense_urls = [m["source_url"] for m in dense.get("metadatas", [[]])[0]]

    bm25_urls: list[str] = []
    if _bm25 is not None:
        scores = _bm25.get_scores(_tokenize(text))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for i in order:
            if scores[i] <= 0 or len(bm25_urls) >= n_results:
                break
            card, is_cur = _bm25_cards[i]
            if current_only and not is_cur:
                continue
            bm25_urls.append(_bm25_urls[i])

    seen, merged = set(), []
    for pair in zip(dense_urls + [None] * len(bm25_urls), bm25_urls + [None] * len(dense_urls)):
        for u in pair:
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged[:n_results]
