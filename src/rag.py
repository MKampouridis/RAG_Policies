"""Retrieval-augmented generation: retrieve relevant chunks from Chroma,
assemble a prompt with retrieved context + conversation history, and
generate an answer via the local chat model."""

import re

from src import lexical
from src import rerank as _rerank
from src.docid import document_family as _document_family
from src.docid import normalize_year
from src.ingest import query as vector_query
from src.llm import chat

N_RESULTS = 6
# over-fetch so recency filtering AND reranking have real depth to work with -
# failure analysis (eval/report.md) found relevant-but-mis-ranked documents as
# deep as rank 60 in a wide dense+BM25 union, so 4x (24 candidates) wasn't
# enough room for a reranker to ever see them
FETCH_POOL_MULTIPLIER = 8
RRF_K = 60

# Academic-year mention: requires the paired "2025-26" / "2025/26" / "2025-2026"
# shape with word boundaries, so money ("£2000"), course codes ("CE2025"), and
# bare years don't trip it and silently degrade retrieval to the full archive.
YEAR_MENTION_RE = re.compile(r"\b(20\d{2})\s*[-/]\s*(20)?\d{2}\b")

SYSTEM_PROMPT = """You are a helpful assistant answering questions about University of Essex \
policies and rules of assessment, using only the provided context excerpts. Each excerpt is \
labeled with its source URL, title, document type, and (where known) department and academic year.

Rules:
- Answer using only the given context. If the context doesn't contain the answer, say so plainly \
rather than guessing.
- When multiple academic years of the same policy/rules document are relevant, prefer the most \
recent academic year unless the user asks about a specific past year.
- Always cite the source_url(s) you used, inline or in a short "Sources" list at the end.
- Be concise and direct.
"""

CONTEXTUALIZE_SYSTEM_PROMPT = """Given a conversation and a follow-up question, rewrite the \
follow-up question into a standalone question that contains all context needed to understand it \
without the conversation (e.g. replace "it"/"this policy"/"these" with the specific thing they \
refer to). Do not answer the question. Output ONLY the rewritten standalone question, nothing else."""

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or",
    "should", "that", "the", "their", "there", "these", "they", "this", "those", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def _is_faithful_rewrite(original: str, rewritten: str) -> bool:
    """Guards against a real failure mode of small local models on long/dense
    multi-topic conversation transcripts: instead of rewriting the new
    question, the contextualizer echoes a DIFFERENT question from earlier in
    the transcript (observed live: asked about "Professional Doctorate
    Director", got back a rewrite about "CSEE programmes" from six turns
    earlier - a completely unrelated retrieval followed). A faithful rewrite
    keeps most of the original's content words (replacing pronouns/references
    with specifics); a hijacked one shares almost none of them. Short
    questions with too few content words to judge are always trusted, since
    a heavily-abbreviated legitimate follow-up ("how about an independent
    chair?") can legitimately share little surface text with its expansion."""
    original_words = _content_words(original)
    if len(original_words) < 3:
        return True
    overlap = original_words & _content_words(rewritten)
    return len(overlap) / len(original_words) >= 0.3


def _contextualize_query(question: str, history: list[dict], summary: str = "") -> str:
    """Retrieval only sees the current turn's text, so a follow-up like "what
    happens after that?" carries no signal about what "that" is. Rewriting it
    into a standalone question before embedding fixes this; the answering
    model still gets the original question plus full history, since it can
    already resolve the reference itself."""
    if not history and not summary:
        return question

    parts = []
    if summary:
        parts.append(f"Earlier conversation summary: {summary}")
    if history:
        parts.append("\n".join(f"{m['role']}: {m['content']}" for m in history[-6:]))
    transcript = "\n".join(parts)

    rewritten = chat(messages=[
        {"role": "system", "content": CONTEXTUALIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{transcript}\n\nFollow-up question: {question}\n\nStandalone question:"},
    ]).strip()

    if not rewritten or not _is_faithful_rewrite(question, rewritten):
        return question
    return rewritten


def _mentioned_year(text: str) -> str:
    """Returns the canonical academic year mentioned in the text ('2025-26'),
    or '' if none."""
    m = YEAR_MENTION_RE.search(text)
    return normalize_year(m.group(1)) if m else ""


def _chunk_year(meta: dict) -> str:
    """Canonical academic year for a chunk: the backfilled academic_year_norm
    metadata when present, otherwise normalized on the fly."""
    return meta.get("academic_year_norm") or normalize_year(meta.get("academic_year"))


def _prefer_most_recent_year(results: dict) -> dict:
    """Within each document family in the candidate pool, drop chunks from
    editions older than the family's most recent academic year, preserving
    original relevance order. Chunks with no determinable year are always
    kept - a recency filter must not discard documents just because year
    extraction failed (the is_current pre-filter already owns currency).
    Distinct documents (different families) are all kept."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[None] * len(documents)])[0]

    best_year_per_family: dict[str, str] = {}
    for meta in metadatas:
        family = _document_family(meta.get("source_url", ""))
        year = _chunk_year(meta)
        if family not in best_year_per_family or year > best_year_per_family[family]:
            best_year_per_family[family] = year

    kept_docs, kept_metas, kept_dists = [], [], []
    for doc, meta, dist in zip(documents, metadatas, distances):
        family = _document_family(meta.get("source_url", ""))
        year = _chunk_year(meta)
        if not year or year == best_year_per_family[family]:
            kept_docs.append(doc)
            kept_metas.append(meta)
            kept_dists.append(dist)

    return {"documents": [kept_docs], "metadatas": [kept_metas], "distances": [kept_dists]}


def _dense_as_hits(dense: dict) -> list[tuple[str, str, dict]]:
    return list(zip(
        dense.get("ids", [[]])[0],
        dense.get("documents", [[]])[0],
        dense.get("metadatas", [[]])[0],
    ))


def _rrf_fuse(*ranked_lists: list[tuple[str, str, dict]]) -> dict:
    """Reciprocal-rank fusion of any number of ranked (id, doc, meta) lists,
    keyed by chunk id. Dense embeddings and BM25 fail on different queries
    (semantic paraphrase vs exact terms like "Capped Mark" or course codes),
    so the union ranked by combined reciprocal rank beats either alone."""
    scores: dict[str, float] = {}
    entries: dict[str, tuple[str, dict]] = {}

    for ranked in ranked_lists:
        for rank, (id_, doc, meta) in enumerate(ranked, 1):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank)
            entries.setdefault(id_, (doc, meta))

    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return {
        "documents": [[entries[i][0] for i in ordered]],
        "metadatas": [[entries[i][1] for i in ordered]],
        "distances": [[None] * len(ordered)],
    }


def _format_context(results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    blocks = []
    for doc, meta in zip(docs, metadatas):
        header = (
            f"[source_url: {meta.get('source_url')}] "
            f"[title: {meta.get('title')}] "
            f"[doc_type: {meta.get('doc_type')}] "
            f"[department: {meta.get('department', 'n/a')}] "
            f"[academic_year: {meta.get('academic_year', 'n/a')}]"
        )
        blocks.append(f"{header}\n{doc}")
    return "\n\n---\n\n".join(blocks)


def retrieve(question: str, history: list[dict], summary: str = "") -> tuple[dict, str]:
    """The full retrieval path used by answer() - query contextualization
    plus recency preference - exposed separately so eval/scoring code can
    measure exactly what production retrieves, not a simplified stand-in.
    Returns (results, retrieval_query)."""
    retrieval_query = _contextualize_query(question, history, summary)

    pool_size = N_RESULTS * FETCH_POOL_MULTIPLIER

    asked_year = _mentioned_year(retrieval_query)
    if asked_year:
        # a year is mentioned - but it may be an edition request ("rules for
        # 2021-22") or purely incidental (a cohort start year, a statistic
        # quoted from a document). Treat the year as a soft preference: fuse
        # the year-labeled pool with the default current pool, so edition
        # requests surface that year's documents while incidental mentions
        # can't exclude the current document that actually holds the answer.
        # No recency dedupe here - year-labeled docs are intentionally old.
        year_dense = vector_query(retrieval_query, n_results=pool_size,
                                  where={"academic_year_norm": asked_year})
        year_bm25 = lexical.query(retrieval_query, n_results=pool_size, year=asked_year)
        cur_dense = vector_query(retrieval_query, n_results=pool_size, where={"is_current": True})
        cur_bm25 = lexical.query(retrieval_query, n_results=pool_size, current_only=True)
        candidates = _rrf_fuse(
            _dense_as_hits(year_dense), year_bm25,
            _dense_as_hits(cur_dense), cur_bm25,
        )
    else:
        # default case: pre-filter the historical archive out of both pools
        # (~70% of chunks), fuse dense + BM25, then apply the family-level
        # recency dedupe as a safety net for docs the is_current flag missed
        dense = vector_query(retrieval_query, n_results=pool_size, where={"is_current": True})
        bm25_hits = lexical.query(retrieval_query, n_results=pool_size, current_only=True)
        candidates = _prefer_most_recent_year(_rrf_fuse(_dense_as_hits(dense), bm25_hits))

    results = _rerank.rerank(retrieval_query, candidates, N_RESULTS)

    return results, retrieval_query


def answer(question: str, history: list[dict], summary: str = "") -> tuple[str, list[str]]:
    """Returns (answer_text, source_urls_used)."""
    results, _ = retrieve(question, history, summary)
    context = _format_context(results)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if summary:
        messages.append({"role": "system", "content": f"Summary of earlier conversation:\n{summary}"})
    messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    response_text = chat(messages=messages)

    metadatas = results.get("metadatas", [[]])[0]
    sources = sorted({m.get("source_url") for m in metadatas if m.get("source_url")})
    return response_text, sources
