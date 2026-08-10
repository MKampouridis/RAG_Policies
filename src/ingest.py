"""Chunk kept documents, embed them, and upsert into the persistent Chroma
vector store."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import chromadb

from src.docid import effective_year, extract_award_type, extract_degree_length
from src.llm import EMBED_DOCUMENT_PREFIX, EMBED_MODEL, EMBED_QUERY_PREFIX, embed_batch

# Env-overridable so an EXPERIMENTAL index can be built alongside production's
# without touching it (2026-08-10, chunk-size comparison). Defaults unchanged.
CHROMA_DIR = os.environ.get("RAG_CHROMA_DIR", "data/chroma")
CORPUS_VERSION_PATH = Path("data/corpus_version")
# Different embedding models produce different-dimension vectors that can't
# share a collection, and results aren't comparable across models anyway -
# keying the collection name to EMBED_MODEL keeps them cleanly separated and
# means switching models back and forth never requires re-embedding twice.
COLLECTION_NAME = "policies_" + re.sub(r"[^a-zA-Z0-9_-]", "_", EMBED_MODEL)

CHUNK_WORDS = int(os.environ.get("RAG_CHUNK_WORDS", "175"))
CHUNK_OVERLAP_WORDS = 30

DOT_LEADER_RE = re.compile(r"\.{4,}")
REPEATED_LINE_MIN_COUNT = 5
REPEATED_LINE_MAX_LEN = 120

# Repetition alone does NOT mean page furniture (bug found 2026-08-08). Rules-
# of-assessment documents are built from parallel per-year sections, so genuine
# policy clauses recur verbatim - and the old "short line repeated >=5 times"
# rule deleted them from EVERY occurrence. In
# roa-ug-integrated-masters-4yr-year-1 it stripped all six of its matches and
# not one was a header: "must withdraw from the University in any of the
# following situations:", "- Where the Year Mark is below 20;", "- Where a
# student cannot complete their studies within the maximum", "period.", "and",
# "or". The operative verb of the progression rule was removed from the index,
# so no retriever or generator could ever answer "what happens if I fail to
# progress" - which is exactly what a user reported. Corpus-wide the rule
# stripped 2351 distinct line-types, of which only ~10 matched any furniture
# pattern.
#
# A repeated line is now only removed when it also LOOKS like furniture: a page
# marker, a contents/nav line, or a bare number. Prose is additionally vetoed -
# anything ending in sentence punctuation or opening with a bullet/enumerator.
# Deliberately biased toward keeping text: a surviving running title costs a
# little chunk dilution (the original motivation), whereas a deleted clause is
# unanswerable and invisible.
_FURNITURE_RE = re.compile(
    r"(?:page\s*\d+|\d+\s*of\s*\d+|return\s+to\s+contents|back\s+to\s+contents"
    r"|^\s*contents\s*$|^\W*\d+\W*$)",
    re.I,
)
_PROSE_RE = re.compile(r"[.;:,]\s*$|^\s*[-•▪]|^\s*\(?[a-z0-9]{1,3}[).]\s")


def _is_page_furniture(line: str) -> bool:
    """True when a repeated line looks like a running header/footer rather than
    document content. Conservative by design - see the note above."""
    s = line.strip()
    if not s:
        return True
    if _PROSE_RE.search(s):
        return False
    return bool(_FURNITURE_RE.search(s))

_client = None


def bump_corpus_version() -> None:
    """Mark the chunk store as changed. src/lexical.py's in-memory BM25 index
    (possibly in a different process, e.g. the running server while a crawl
    executes) checks this marker per query and rebuilds when it moves - the
    only cross-process signal that upserts/deletes/flag-flips happened."""
    CORPUS_VERSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_VERSION_PATH.write_text(uuid.uuid4().hex)


def read_corpus_version() -> str | None:
    try:
        return CORPUS_VERSION_PATH.read_text()
    except FileNotFoundError:
        return None


def clean_text(text: str) -> str:
    """Strip PDF extraction noise before chunking: dot-leader runs from
    tables of contents, and short lines that repeat across many pages
    (running headers/footers like "Return to Contents Page 4 of 23"),
    which otherwise dilute every chunk with identical low-signal tokens."""
    text = DOT_LEADER_RE.sub(" ", text)
    lines = text.splitlines()
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) <= REPEATED_LINE_MAX_LEN:
            counts[stripped] = counts.get(stripped, 0) + 1
    repeated = {
        s for s, c in counts.items()
        if c >= REPEATED_LINE_MIN_COUNT and _is_page_furniture(s)
    }
    kept = [line for line in lines if line.strip() not in repeated]
    return "\n".join(kept)


# A3b (external code review round 3, 2026-07-22, Fable 5, verified): a few
# Essex filenames glue an award prefix to the programme name with no
# separator ("mscperiodontology_25" -> token "mscperiodontology"), so BM25 -
# whose tokenizer is [a-z0-9]+ - can't match it against a "MSc Periodontology"
# query while every cleanly-separated sibling can; the home document is thus
# lexically invisible on its own identity and crowded out of the pool
# entirely (this is the periodontology out-of-pool miss the A1/A2/A3a
# re-baseline confirmed hygiene alone doesn't fix). Repair is an EXPLICIT,
# audited map (a corpus audit confirmed naive award-prefix segmentation is
# unsafe: it would corrupt "masters"->ma+sters, "marking"->ma+rking). The
# spaced forms come from each document's J1 programme_name (e.g. "mscinursing"
# is "MSci Nursing" - an integrated masters - NOT "MSc in Nursing"). Applied
# here for future ingests and, for existing chunks, as a metadata-only
# chunk_header string-replace (no re-embed, so J2's corpus-wide dense-
# displacement failure mode structurally cannot occur - only BM25's view of
# these few documents changes).
_TITLE_REPAIRS = {
    "mscperiodontology": "msc periodontology",
    "mscinursing": "msci nursing",
    "pgcertpwp": "pgcert pwp",
}


def _readable_title(title: str) -> str:
    """Filenames-as-titles ("csee-ft-masters-accredited-variations-25.pdf")
    carry the degree/department identity that the chunk body lacks - turn
    them into plain words so the embedder and BM25 can use them."""
    title = re.sub(r"\.pdf$", "", title, flags=re.I)
    title = re.sub(r"[-_]+", " ", title).strip()
    for glued, spaced in _TITLE_REPAIRS.items():
        title = re.sub(rf"\b{glued}\b", spaced, title)
    return title


# J2 tried enriching chunk headers with the extracted identity records at
# embedding time - net regression (RoA hit@6 70%->60%, eval/report.md "J2")
# with a revealing mechanism: documents with EMPTY identity records (byte-
# identical headers and embeddings to baseline) still flipped hit->miss,
# because ~450 other documents' chunks moved in embedding space and crowded
# into queries they didn't previously win. Corpus-wide header changes perturb
# every query's neighborhood, not just the enriched documents'. The identity
# data itself was locally effective (rescued a target miss, improved MRR) -
# it now feeds the document-level identity index (J3) instead, which can't
# displace chunk embeddings. Off by default.
IDENTITY_HEADER_ENABLED = False


def _load_doc_identity(url: str) -> dict:
    """J1's per-document extracted identity record (programme name,
    department, partner institution, awards, aliases - see
    extract_doc_identity.py), or {} when absent. These fields carry the
    document identity that filenames don't (e.g. 'east15-25.pdf' says
    nothing about it being East 15 Acting School's MA/MSc rules) - the
    coverage gap behind every filename-derived facet attempt's failure."""
    path = Path("data/doc_identity") / f"{url_hash(url)}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def build_chunk_header(url: str, metadata: dict) -> str:
    """One-line document-identity header prepended to every chunk at
    embedding time. RoA chunk bodies are near-identical boilerplate across
    degree types/departments/years; the distinguishing facts live only on
    the title page (chunk 0) and in metadata, so without this header the
    embedder cannot tell sibling documents apart (see eval/report.md).
    Enriched (J2) with the per-document extracted identity record when one
    exists - programme names, partner institution, and user-style aliases
    that the filename-derived title lacks."""
    parts = [f"Document: {_readable_title(metadata.get('title') or url.rsplit('/', 1)[-1])}"]
    if metadata.get("doc_type"):
        parts.append(metadata["doc_type"].replace("_", " "))
    identity = _load_doc_identity(url) if IDENTITY_HEADER_ENABLED else {}
    if identity.get("programme_name"):
        parts.append(f"programme: {identity['programme_name']}")
    if identity.get("department") or metadata.get("department"):
        parts.append(f"department: {identity.get('department') or metadata['department']}")
    if identity.get("partner_institution"):
        parts.append(f"partner institution: {identity['partner_institution']}")
    if identity.get("awards"):
        parts.append(f"awards: {', '.join(identity['awards'])}")
    if identity.get("aliases"):
        parts.append(f"also known as: {', '.join(identity['aliases'])}")
    if metadata.get("academic_year"):
        parts.append(f"academic year {metadata['academic_year']}")
    return " | ".join(parts)


def _get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client.get_or_create_collection(COLLECTION_NAME)


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    step = max(chunk_words - overlap_words, 1)
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _sanitize_metadata(metadata: dict) -> dict:
    # Chroma metadata values must be str/int/float/bool, not None.
    return {k: v for k, v in metadata.items() if v is not None}


def upsert_document(url: str, text: str, metadata: dict, chunk_contexts: list[str] | None = None) -> int:
    """Chunk + embed + upsert a single document. Returns the number of
    chunks written. Existing chunks for this URL are deleted first so
    re-running ingestion on an updated document doesn't leave stale chunks
    behind.

    `chunk_contexts`, if given (one string per chunk, aligned by index, from
    generate_chunk_context.py's cache), is an LLM-written sentence situating
    that specific chunk within its document - richer, chunk-specific signal
    than the document-level chunk_header, for the near-identical-boilerplate
    RoA siblings header alone doesn't always disambiguate. Ignored (with a
    print warning) if its length doesn't match the chunk count, since a
    misaligned list would attach the wrong context to the wrong chunk."""
    collection = _get_collection()

    existing = collection.get(where={"source_url": url}, include=[])
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    chunks = chunk_text(clean_text(text))
    if not chunks:
        return 0

    if chunk_contexts is not None and len(chunk_contexts) != len(chunks):
        print(f"    WARNING: chunk_contexts length {len(chunk_contexts)} != {len(chunks)} chunks for {url}, ignoring")
        chunk_contexts = None

    header = build_chunk_header(url, metadata)
    embed_texts = []
    for i, c in enumerate(chunks):
        situating = chunk_contexts[i] if chunk_contexts else ""
        embed_texts.append(EMBED_DOCUMENT_PREFIX + header + "\n" + situating + "\n" + c)
    embeddings = embed_batch(embed_texts)
    doc_hash = url_hash(url)
    ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas = [
        _sanitize_metadata({
            **metadata,
            "source_url": url,
            "chunk_index": i,
            "chunk_header": header,
            "chunk_context": chunk_contexts[i] if chunk_contexts else None,
            "academic_year_norm": effective_year(url, metadata.get("academic_year")),
            "degree_length": extract_degree_length(header),
            "award_type": extract_award_type(header),
        })
        for i in range(len(chunks))
    ]

    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    bump_corpus_version()
    return len(chunks)


def delete_document(url: str) -> None:
    collection = _get_collection()
    existing = collection.get(where={"source_url": url}, include=[])
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])
        bump_corpus_version()


def query(text: str, n_results: int = 6, where: dict | None = None) -> dict:
    collection = _get_collection()
    query_embedding = embed_batch([EMBED_QUERY_PREFIX + text])[0]
    return collection.query(query_embeddings=[query_embedding], n_results=n_results, where=where)


def passages_for_documents(urls: list[str], query_text: str = "") -> list[dict]:
    """Read-only lookup for the UI's source modal: per document, its stored
    metadata and (when a query is given) the chunk of THAT document nearest the
    query embedding.

    NOT the exact chunk rag.answer() put in the generator's context - answer()
    returns source URLs only, and reading its real context would mean changing
    rag.py. This re-runs just the dense stage, scoped to one document, so the
    passage shown is real stored text from the cited document that matches the
    question; it is usually the same chunk, but that is not guaranteed and the
    UI must not claim otherwise.

    Embeds the query once for the whole batch. No writes, no cache mutation,
    nothing that can affect retrieval or an eval run.
    """
    collection = _get_collection()
    query_embedding = None
    if query_text.strip():
        query_embedding = embed_batch([EMBED_QUERY_PREFIX + query_text])[0]

    out = []
    for url in urls:
        record = {"url": url, "title": None, "doc_type": None, "academic_year": None, "excerpt": None}
        try:
            if query_embedding is not None:
                res = collection.query(
                    query_embeddings=[query_embedding], n_results=1, where={"source_url": url}
                )
                docs = (res.get("documents") or [[]])[0]
                metas = (res.get("metadatas") or [[]])[0]
                if docs:
                    record["excerpt"] = docs[0]
                meta = metas[0] if metas else {}
            else:
                got = collection.get(where={"source_url": url}, limit=1, include=["metadatas"])
                metas = got.get("metadatas") or []
                meta = metas[0] if metas else {}
            record["title"] = meta.get("title")
            record["doc_type"] = meta.get("doc_type")
            record["academic_year"] = meta.get("academic_year")
        except Exception:
            # a cited URL that is no longer in the index must not break the
            # answer that cited it - the modal falls back to the link alone
            pass
        out.append(record)
    return out
