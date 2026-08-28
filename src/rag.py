"""Retrieval-augmented generation: retrieve relevant chunks from Chroma,
assemble a prompt with retrieved context + conversation history, and
generate an answer via the local chat model."""

import json
import os
import re
import time as _perf
from collections import Counter

from src import colbert_index as _colbert_index
from src import doc_index as _doc_index
from src import ensemble as _ensemble
from src import lexical
from src import pseudo_query as _pseudo_query
from src import rerank as _rerank
from src import splade as _splade
from src.docid import document_family as _document_family
from src.docid import top_family_count as _top_family_count
from src.docid import extract_award_type, extract_degree_length, normalize_year
from src.ingest import query as vector_query
from src.entities import detect_departments, department_filter_values
from src.llm import CONTEXTUALIZE_MODEL, chat, contextualize_chat, generate

N_RESULTS = 6
# over-fetch so recency filtering AND reranking have real depth to work with -
# failure analysis (eval/report.md) found relevant-but-mis-ranked documents as
# deep as rank 60 in a wide dense+BM25 union, so 4x (24 candidates) wasn't
# enough room for a reranker to ever see them
FETCH_POOL_MULTIPLIER = 8

# Stage D (SPLADE third retrieval channel) - regressed in the full eval
# (eval/report.md "Stage D"): RoA hit@6 70%->65%, overall 85%->82.5% (net
# -2 turns: +3/-5, almost entirely on follow-up retrieval) - the extra
# channel appears to add noise to the 3-way RRF fusion that disproportionately
# hurts follow-up queries. Combined with real cost (index build ~105 min,
# extra encode pass per query), not worth keeping. Off by default; kept for
# reference, not a dead end worth deleting.
SPLADE_ENABLED = False

# Stage E (embedding-model ensemble, nomic + bge-m3 RRF-fused) - the worst
# regression of the four new stages tried (eval/report.md "Stage E"): RoA
# hit@6 70%->57.5%, overall 85%->78.8%. Consistent with the earlier
# stage3_bgem3 finding (bge-m3 alone was a wash/slight regression on RoA) -
# fusing its weaker RoA rankings in via RRF introduces enough noise to
# displace nomic-embed-text's correct results from the top ranks rather than
# complementing them. Off by default; kept for reference.
EMBEDDING_ENSEMBLE_ENABLED = False

# Stage B (ambiguity detection + clarifying question) - same isolation
# discipline: off by default so Stage A can be evaluated on its own first.
AMBIGUITY_DETECTION_ENABLED = False

# Stage A / A2 (degree_length/award_type facet preference, hard then soft) -
# both regressed hit@6 in the full eval (eval/report.md "Stage A"/"Stage A2");
# off by default. degree_length/award_type metadata and extraction functions
# stay in place (src/docid.py, src/ingest.py) since they're harmless to keep
# computing, just not used for retrieval preference.
FACET_PREFERENCE_ENABLED = False

# Stage F: tuned weighted score fusion for the base dense+BM25 pair, as an
# alternative to reciprocal-rank fusion - Bruch et al. 2022 found a small
# amount of in-domain-tuned convex/weighted combination of normalized scores
# outperforms RRF, which only sees rank position and discards how much better
# one candidate scored than the next. Off by default (RRF is the proven,
# parameter-free baseline); DENSE_WEIGHT/BM25_WEIGHT are only read when on.
WEIGHTED_FUSION_ENABLED = False
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5

# Stage G: deterministic pseudo-query index (build_pseudo_query_index.py) as
# a fourth retrieval channel. Full-eval result was a net-zero wash (exact
# same hit@6 as baseline: 1 turn gained, 1 different turn lost - see
# eval/report.md "Stage G") - not harmful, but not worth the added
# complexity (extra collection, extra embed call per query) either. Off by
# default; kept for reference.
PSEUDO_QUERY_ENABLED = False

# Stage H: CRAG-style retrieval verification (Yan et al. 2024) - a lightweight
# LLM check on whether the retrieved context actually supports answering the
# question, surfacing uncertainty instead of a confident guess when it
# doesn't. Regressed in the full eval (eval/report.md "Stage H") for two
# reasons: (1) the verifier massively over-triggered (66/80 turns, 82.5%,
# including turns where retrieval had actually succeeded), tanking answer
# quality far beyond what abstention-on-genuine-misses would explain; (2)
# gating the primary turn's answer has a real knock-on cost in a
# conversational system - the follow-up turn's query contextualizer sees a
# generic uncertainty message instead of a real answer in history, which
# measurably regressed follow-up hit@6 (34/40->32/40) even though primary
# hit@6 was unaffected (retrieve() itself is untouched by this flag). Off by
# default; kept for reference, not a dead end worth deleting.
CRAG_VERIFICATION_ENABLED = False

VERIFICATION_SYSTEM_PROMPT = """You are checking whether a set of retrieved document excerpts \
contains enough information to confidently and specifically answer a question. Respond with \
ONLY a JSON object: {"supported": true or false, "reason": "<one short sentence>"}. Say false \
if the excerpts are off-topic, only tangentially related, or missing the key fact needed - not \
just because the wording differs from the question."""

# J3: document-level identity routing prior. A separate ~1,200-record index
# of per-document "identity cards" (title + J1's extracted programme/
# department/partner/aliases - src/doc_index.py) queried alongside chunk
# retrieval, softly boosting identity-matched documents' chunks via one extra
# RRF list. Unlike J2's header enrichment this left chunk embeddings
# untouched - and still regressed on every metric (eval/report.md "J3": RoA
# hit@6 70%->62.5%, 0 rescues / 3 losses). The routing prior never pulled a
# missing document into the top-6 (identity cards of true siblings - e.g.
# home vs partner-institution MSc Periodontology - are themselves near-
# identical), while the extra fused list diluted previously-correct results.
# Off by default; kept for reference.
DOC_ROUTING_ENABLED = False
DOC_ROUTING_TOP_DOCS = 5

# Idea 2 (ColBERT first-stage retrieval, see eval/report.md "Code review
# round") - rejected. src/colbert_index.py's persisted Voyager index (built
# by build_colbert_index.py) provides a genuine retrieval channel over the
# FULL corpus - token-level ANN search + exact MaxSim - not just a rerank of
# whatever dense+BM25 already surfaced. Targeted the out-of-pool miss class
# J0 found (4/12 misses whose correct document was never in the fused
# candidate pool at all, so no reranker downstream could have rescued it),
# but the 80-turn eval showed a net RoA regression (hit@6 70%->65%, answer
# score 3.80->3.55): adding 1-2 more RRF channels dilutes already-marginal
# (rank 4-6) correct documents, and the new channel's token-level MaxSim
# over-recalls topically-similar sibling/superseded-edition documents. None
# of the 4 known out-of-pool misses were rescued. See eval/report.md
# "Idea 2 eval result" for the full flip analysis. Off by default.
COLBERT_FIRST_STAGE_ENABLED = False

# Phase 4, experiment 2 (external code review round 2, 2026-07-21, Fable 5):
# home-institution tie-break. See _prefer_home_institution()'s docstring for
# the mechanism. Off by default pending the validation eval.
HOME_INSTITUTION_TIEBREAK_ENABLED = False

# Stage I: selective multi-hop query decomposition. Triggered only when the
# initial reranked top-6 is fragmented across many different document
# families (reusing Stage B's AMBIGUITY_FAMILY_COUNT_THRESHOLD signal). A
# pre-validation check (eval/report.md, "Pre-validation: facet-overlap graph
# killed...") predicted neither of the two dominant current failure modes
# (underspecified queries; same-family sibling confusion needing a
# finer-grained identifier) obviously calls for decomposition - tried anyway,
# and the full eval confirmed it: RoA hit@6 70%->62.5% (net -3 turns: +1/-4,
# see eval/report.md "Stage I"). It occasionally helped (recovered one
# genuine former miss) but more often diluted the rerank pool with a wrong
# hypothesis's candidates, displacing documents the single-shot retrieval
# had already found correctly. Off by default; kept for reference.
MULTIHOP_DECOMPOSITION_ENABLED = False

DECOMPOSE_SYSTEM_PROMPT = """A question was searched against a university policy/rules-of-\
assessment document corpus and the results were scattered across several different, seemingly \
unrelated documents - a sign the question may be ambiguous across multiple specific programmes, \
departments, or document types. Given the question and a list of the distinct candidate \
documents actually found, write up to 3 alternative, more specific versions of the SAME \
question, each one assuming it refers to one specific candidate document (use its title to make \
the rephrasing concrete). Respond with ONLY a JSON object: {"subqueries": ["...", "...", "..."]}."""

# Academic-year mention: requires the paired "2025-26" / "2025/26" / "2025-2026"
# shape with word boundaries, so money ("£2000"), course codes ("CE2025"), and
# bare years don't trip it and silently degrade retrieval to the full archive.
YEAR_MENTION_RE = re.compile(r"\b(20\d{2})\s*[-/]\s*(20)?\d{2}\b")

# D2 (review round 3): J7 keyphrase/verbatim-figures retry. J7 added a
# "quote specific numbers/thresholds verbatim" rule to raise keyphrase
# coverage (the keyphrases are mostly figures: "40", "50", "30 credits",
# "5 years") and reverted it - overall keyphrase +1.7pp but RoA keyphrase
# -1.4pp, answer -0.06, "7B doesn't reliably comply". But that eval predates
# both the determinism fix AND the num_ctx pin: with num_ctx unset the
# generation prompt could silently truncate (Fable 5's round-2 finding),
# which would look exactly like "doesn't follow the instruction". Retrying it
# now as a fair test - flag-gated so it's a clean A/B against current
# production. Targets the strict-vs-evidence gap (70% vs 87.5%: the system
# retrieves a sufficient document but the generator doesn't always surface
# its key figures).
# The answer prompt and its rules live in src/prompts.py (split out
# 2026-08-13). Imported by name so existing call sites are unchanged.
from src.prompts import (  # noqa: F401
    DETAIL_LEVELS, INLINE_CITATIONS, MULTI_ENTITY_COVERAGE, SYSTEM_PROMPT,
    USER_FACING_LANGUAGE, _scrub_plumbing, system_prompt_for)


# Re-exports. The 2026-08-13 refactor moved these into single-purpose modules
# (prompts, contextualize, institutions, fusion, instrumentation) but external
# callers still do `from src.rag import SYSTEM_PROMPT` / `_rrf_fuse` /
# `_is_partner_institution`, and eval scripts depend on that surface. Deleting
# the imports to silence pyflakes would break them.
#
# Declared via __all__ rather than `# noqa: F401` because PYFLAKES DOES NOT
# HONOUR noqa - that is flake8. The noqa comments were already there and all
# 29 messages were reported anyway, which is what made the one tool that would
# have caught the 503 too noisy to read. __all__ silences them and, unlike a
# comment, states the intent in something the language itself understands.
__all__ = [
    "CONTEXTUALIZE_SYSTEM_PROMPT", "DETAIL_LEVELS", "INLINE_CITATIONS", "MULTI_ENTITY_COVERAGE",
    "RRF_K", "SYSTEM_PROMPT", "USER_FACING_LANGUAGE", "_PARTNER_NAME_RE",
    "_PARTNER_NAME_TOKENS", "_TIMING_PATH", "_aliases", "_anchor_from_history",
    "_content_words", "_has_extraneous_family", "_identity_anchor_index", "_is_faithful_rewrite",
    "_is_partner_institution", "_log_rewrite_reject", "_normalize", "_only_partner_institutions",
]


def _mentioned_year(text: str) -> str:
    """Returns the FIRST canonical academic year mentioned in the text
    ('2025-26'), or '' if none. Prefer _mentioned_years() in retrieval - see
    its docstring for why first-match-only is a defect there."""
    m = YEAR_MENTION_RE.search(text)
    return normalize_year(m.group(1)) if m else ""


def _mentioned_years(text: str) -> list[str]:
    """Every canonical academic year mentioned, in order of appearance, without
    duplicates.

    `_mentioned_year` uses `.search()` and so sees only the first. That was
    invisible while the corpus held one edition per family, and became a
    reported defect the day nine years of PGRE milestones arrived
    (2026-08-28). Three user-reported failures, all reproduced:

      "Compare the 2025/26 CSEE PhD milestones with the 2020/21 ones"
          -> extracted '2025-26' only; retrieval returned six 2025-26
             documents and ZERO 2020/21, so the answer said it had no access
             to 2020/21 - while both editions sat in the index.
      "Now compare them to the most recent ones" (after a 2020/21 turn)
          -> the contextualizer rewrote it correctly naming both years, then
             the first-match rule picked '2020-21' and returned only 2020
             documents, so the answer could not find the CURRENT ones.

    Symmetric, and neither is a retrieval-quality problem: the pool was
    filtered to one year before ranking ever ran. A comparison question is
    unanswerable when half of what it compares is excluded by construction.
    """
    seen: list[str] = []
    for m in YEAR_MENTION_RE.finditer(text):
        year = normalize_year(m.group(1))
        if year and year not in seen:
            seen.append(year)
    return seen


# Department acronym -> the wording the CURRENT documents actually use.
#
# Essex renamed its departments and re-coded its filenames between editions, so
# the acronym a user types can be present in every ARCHIVED edition and absent
# from the current one. Reported case (2026-08-28): "CSEE" appears in
# `csee-phd-2020.pdf` (department field, filename, body text) and NOWHERE in
# `ce-phd-2025-26.pdf`, which spells out "School of Computer Science and
# Electronic Engineering" - so an exact-term match hands every slot to the
# superseded edition, and the right current document cannot be told apart from
# other departments' near-identical milestone files.
#
# Derived from the corpus, not guessed: each entry was checked to appear in 0
# of the 80 current milestone documents while present in archived ones. Terms
# that DO survive into the current wording are deliberately absent from this
# map - iser (3), sres (4), hsc (4), psychology (4), sociology (10), economics
# (3), government (5), history (7), philosophy (3), maths (1), law (4), art
# history (4) all still match and need no help. `msas` is the exception to the
# rule: it appears NOWHERE in the corpus, current or archived, yet users type
# it (seen in stored traffic), so it would otherwise match nothing at all.
#
# 17% of stored user turns contain one of these acronyms, CSEE dominating.
# Env-gated so both arms can be replayed on the SAME corpus - the corpus moved
# 20% the day this was written, so a comparison against any stored baseline
# would measure the ingest, not this. Default set from that measurement; see
# eval/report.md "Round 34c".
DEPARTMENT_ALIAS_ENABLED = os.environ.get("RAG_DEPARTMENT_ALIAS", "1") == "1"
DEPARTMENT_ALIASES = {
    "csee": "School of Computer Science and Electronic Engineering",
    "ebs": "Essex Business School",
    "lifts": "Literature, Film and Theatre Studies",
    "spah": "Philosophical, Historical, and Interdisciplinary Studies",
    "pps": "Psychosocial and Psychoanalytic Studies",
    "langling": "Language and Linguistics",
    "msas": "School of Mathematics, Statistics and Actuarial Sciences",
}
_DEPARTMENT_ALIAS_RE = re.compile(
    r"\b(" + "|".join(sorted(DEPARTMENT_ALIASES, key=len, reverse=True)) + r")\b", re.I
)


def _alias_expanded_query(text: str) -> str:
    """`text` plus the full department name for any acronym it uses, or '' when
    nothing applies.

    An expansion already spelled out in the query is not repeated - "CSEE
    (Computer Science and Electronic Engineering)" needs no help and doubling
    the phrase would just skew the embedding.

    Used ADDITIVELY: the caller retrieves an EXTRA pool with this query and
    fuses it alongside the original, rather than replacing the query. The
    original pools are therefore bit-identical to what they were, and this can
    only add candidates for the reranker to judge - it cannot displace a result
    that the unexpanded query would have found.
    """
    if not DEPARTMENT_ALIAS_ENABLED:
        return ""
    found = {m.group(1).lower() for m in _DEPARTMENT_ALIAS_RE.finditer(text)}
    lowered = text.lower()
    additions = [DEPARTMENT_ALIASES[a] for a in sorted(found)
                 if DEPARTMENT_ALIASES[a].lower() not in lowered]
    return f"{text} {' '.join(additions)}" if additions else ""


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


# Institution/partner filtering lives in src/institutions.py (split out
# 2026-08-13). Imported by name so existing call sites are unchanged.
# Follow-up query rewriting lives in src/contextualize.py (split out
# 2026-08-13). Imported by name so existing call sites are unchanged.
from src.contextualize import (  # noqa: F401
    CONTEXTUALIZE_SYSTEM_PROMPT, _anchor_from_history, _content_words,
    _contextualize_query, _has_extraneous_family, _identity_anchor_index,
    _is_faithful_rewrite, _log_rewrite_reject)
from src.institutions import (  # noqa: F401
    PARTNER_BOOST, PARTNER_EXCLUDE_WHEN_UNNAMED,
    PARTNER_INSTITUTION_DEMOTE_ENABLED, PARTNER_MODE,
    _PARTNER_NAME_RE, _PARTNER_NAME_TOKENS,
    _aliases, _boost_home_institution, _cap_partner_institutions,
    _demote_partner_institutions, _exclude_partner_institutions,
    _filter_by_institution, _is_partner_institution,
    _names_partner_institution, _only_partner_institutions,
    _prefer_home_institution)


# Ambiguity threshold: how few distinct document families in the top results
# count as "the answer may differ by programme". Belongs here with the
# disclosure logic that uses it - it was swept into fusion.py by a slice that
# happened to span it, which broke every answer until the NameError surfaced.
AMBIGUITY_FAMILY_COUNT_THRESHOLD = 1


# List fusion lives in src/fusion.py (split out 2026-08-13).
from src.fusion import (RRF_K, _dedup_by_chunk, _dense_as_hits,  # noqa: F401
                        _normalize, _rrf_fuse, _weighted_dense_bm25)




def _distinct_family_count(metadatas: list[dict], top_n: int = 6) -> int:
    """How many DISTINCT document families appear in the reranked top-N. High
    count = the pool is scattered across many unrelated documents with no single
    one dominating. The abstention-gate diagnostic (2026-07-23, eval/report.md)
    found this is the only retrieval signal carrying any hit/miss information,
    though a weak one (>=6 families -> 0.40 precision on misses; +an
    under-specified query -> 0.45). Used by the D3 clarify gate."""
    return len({_document_family(m.get("source_url", "")) for m in metadatas[:top_n]})


def _distinct_family_titles(metadatas: list[dict], limit: int = 4) -> list[str]:
    """Distinct document families in a candidate pool, most-relevant-first,
    named by title (falling back to the family key). Shared by the
    clarifying-question (Stage B) and query-decomposition (Stage I) paths,
    both of which need to name the actual candidate documents rather than
    speak generically about "a few different documents"."""
    seen_families: dict[str, str] = {}
    for meta in metadatas:
        family = _document_family(meta.get("source_url", ""))
        if family not in seen_families:
            seen_families[family] = meta.get("title") or family
        if len(seen_families) >= limit:
            break
    return list(seen_families.values())


def _clarifying_question(metadatas: list[dict]) -> str:
    """Built from the distinct document families in the ambiguous pool, most
    dominant first, so the question names the actual candidates instead of a
    generic "please clarify"."""
    listed = "; ".join(_distinct_family_titles(metadatas))
    return (
        "Your question could relate to a few different documents, and I want to point you to the "
        f"right one rather than guess: {listed}. Could you tell me which programme, department, or "
        "academic year you mean?"
    )


def _surrogate_hits(docs: list[str], metas: list[dict]) -> list[tuple[str, str, dict]]:
    """Re-keys (doc, meta) pairs by (source_url, chunk_index) instead of a
    Chroma embedding-store id, so the same real chunk found via two
    different representations (e.g. a decomposed subquery's own dense hit
    vs. the original unified pool) is recognized as the SAME candidate by
    _rrf_fuse's id-keyed accumulation, rather than double-counted under two
    different id strings. Needed because the pre-existing fused `candidates`
    dict (already an _rrf_fuse output) doesn't carry Chroma ids forward, only
    documents/metadatas - this is the uniform id scheme for combining it with
    freshly-queried lists that do have Chroma ids."""
    return [(f"{m.get('source_url')}::{m.get('chunk_index')}", d, m) for d, m in zip(docs, metas)]


def _decompose_query(question: str, candidate_titles: list[str]) -> list[str]:
    """Asks the local chat model to rewrite an ambiguous question into up to
    3 concrete, document-specific hypotheses, one per plausible candidate
    found in the initial fragmented pool - selective multi-hop decomposition
    (Consensus review's rank-4 suggestion), triggered only when initial
    retrieval shows genuine cross-document ambiguity, not on every query
    (always-on decomposition is reported to hurt ranking precision)."""
    titles_list = "\n".join(f"- {t}" for t in candidate_titles)
    raw = chat(
        messages=[
            {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nCandidate documents found:\n{titles_list}"},
        ],
        format="json",
    )
    try:
        subqueries = json.loads(raw).get("subqueries", [])
        return [s for s in subqueries if isinstance(s, str) and s.strip()][:3]
    except Exception:
        return []


def _context_supports_answer(question: str, context: str) -> bool:
    """CRAG-style lightweight retrieval evaluator (Yan et al. 2024): asks the
    same local chat model whether the retrieved excerpts actually contain
    what's needed to answer, as a corrective gate before generation - one
    short extra call, not the full answer-generation prompt. Fails open
    (treats unparseable output as "supported") so a judge-format hiccup
    doesn't block an otherwise-fine answer."""
    raw = chat(
        messages=[
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nRetrieved excerpts:\n{context}"},
        ],
        format="json",
    )
    try:
        return bool(json.loads(raw).get("supported", True))
    except Exception:
        return True


def _uncertainty_response(sources: list[str]) -> str:
    return (
        "I wasn't able to find information in the retrieved policy/rules-of-assessment excerpts "
        "that directly and confidently answers this question. You may want to check the source "
        "document(s) below directly, or rephrase your question with more specific details (e.g. "
        "programme, department, or academic year)."
    )


# Idea 3 (identity data in answer context) - tried, mixed but net negative
# on RoA specifically (eval/report.md "Code review round"): overall/policy
# answer score rose (3.89->3.95, 3.98->4.25) but that's likely noise from a
# feature that barely engages on policy docs (little identity data
# populated there); RoA - where it actually fires - moved the wrong way on
# BOTH quality metrics together (answer 3.80->3.65, keyphrase coverage
# 55.2%->53.4%), suggesting the extra context fields add clutter the 7B
# generator doesn't parse as precisely, rather than sharpening it. Off by
# default; kept for reference (e.g. worth retrying if the deferred
# stronger-generator phase changes this).
IDENTITY_CONTEXT_ENABLED = False


def _format_context(results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    blocks = []
    for doc, meta in zip(docs, metadatas):
        parts = [
            f"[source_url: {meta.get('source_url')}] "
            f"[title: {meta.get('title')}] "
            f"[doc_type: {meta.get('doc_type')}] "
            f"[department: {meta.get('department', 'n/a')}] "
            f"[academic_year: {meta.get('academic_year', 'n/a')}]"
        ]
        if IDENTITY_CONTEXT_ENABLED:
            from src.ingest import _load_doc_identity
            identity = _load_doc_identity(meta.get("source_url", ""))
            if identity.get("programme_name"):
                parts.append(f"[programme: {identity['programme_name']}]")
            if identity.get("partner_institution"):
                parts.append(f"[partner institution: {identity['partner_institution']}]")
            if identity.get("aliases"):
                parts.append(f"[also known as: {', '.join(identity['aliases'])}]")
        header = " ".join(parts)
        blocks.append(f"{header}\n{doc}")
    return "\n\n---\n\n".join(blocks)


_LAST_CANDIDATE_POOL = None  # set by retrieve() to the pre-rerank fused pool; read by the recall diagnostic


# Stage timing (2026-08-09). Out-of-process timing of this pipeline gave three
# mutually inconsistent answers in one session (dense retrieval measured at
# 0.15s, 2.73s and 0.57s) because single-query timings on this machine are
# dominated by cache state and memory pressure. Real end-to-end latency is
# ~9s median with a long tail (one 23s outlier in 8 requests), and the stages
# did not sum to the total - so the tail is worth attributing from real traffic
# rather than estimated again.
#
# Writes one JSON line per stage to data/latency.jsonl, gitignored, best-effort:
# instrumentation must never break a user's request. Off unless RAG_TIMING=1.
# Per-stage timing lives in src/instrumentation.py (split out 2026-08-13).
# Imported by name so existing call sites are unchanged.
from src.instrumentation import (RAG_TIMING, _TIMING_PATH,  # noqa: F401
                                 _stage_note, _stage_timer)


# Multi-entity RETRIEVAL (2026-08-10). MULTI_ENTITY_COVERAGE (above) fixed the
# HONESTY half of the six-school failure; this targets the COVERAGE half that
# note calls "not fixable by prompting". N_RESULTS caps CHUNKS, so the real
# question ("CSEE, MSAS, Psychology, HSC, SRES, Life Sciences") returned six
# chunks that resolved to THREE documents, all CSEE - one department in six.
#
# Reserves a slot budget per named department and fills it with a retrieval
# FILTERED to that department's metadata values, so coverage is guaranteed by
# construction rather than hoped for from one ranked list. Remaining slots go
# to the ordinary unfiltered ranking, so a question that names entities but
# whose answer lives elsewhere is not starved.
#
# Why this is not the rejected MULTIHOP_DECOMPOSITION (-7.5pts RoA hit@6):
# that asked a model to HYPOTHESISE candidate documents for a vague question,
# on an ambiguity trigger that fired on ordinary questions. Here the entities
# are named explicitly, the mapping is a curated lookup (src/entities.py), and
# the trigger requires >= 2 named departments - measured at 0/160 on the
# existing eval turns, so it cannot move any committed ledger number.
# ENABLED 2026-08-10 (eval/report.md Round 8g). On the real failing question:
# department coverage 1/6 -> 5/6, and the answer gains substantive content for
# CSEE, MSAS, SRES and HSC instead of CSEE alone, with CSEE's own accuracy
# preserved (8/8 accredited programmes, correct exit-award mapping) once the
# contiguity fix was in. Cost is ~11s on triggering questions only.
#
# The safety argument is the BLAST RADIUS, not the weight of evidence: the
# trigger fires on 0/160 existing eval turns, so no committed ledger number can
# move, and single-entity questions take the unchanged path. Evidence on the
# target case is one real user question plus one control - thin, and recorded
# as thin. A multi-entity question set is the outstanding follow-up.
MULTI_ENTITY_RETRIEVAL = os.environ.get("RAG_MULTI_ENTITY_RETRIEVAL", "1") == "1"
# Flag-gated so the fix can be A/B'd. retrieval_replay CANNOT measure it: 0 of
# its 160 turns name >=2 departments, so that harness reports "no change" for a
# defect it structurally cannot see. Measured on sets 5 and 6 instead.
MULTI_ENTITY_PARTNER_RECHECK = os.environ.get("RAG_MULTI_ENTITY_PARTNER_RECHECK", "1") == "1"
# Adds the BM25 channel to per-entity retrieval. OFF until an eval justifies it
# (the project rule that _has_extraneous_family was shipped in violation of, at
# a cost of -8.8 points). Measured on set 5's department coverage, which is the
# only metric this can move.
MULTI_ENTITY_LEXICAL = os.environ.get("RAG_MULTI_ENTITY_LEXICAL", "0") == "1"
MULTI_ENTITY_MIN_ENTITIES = 2
MULTI_ENTITY_PER_ENTITY = 2      # chunks reserved per named department
MULTI_ENTITY_MAX_RESULTS = 14    # hard ceiling on the widened result set


def _multi_entity_results(retrieval_query: str, aliases: list[str],
                          base_results: dict, pool_size: int) -> dict:
    """Per-department retrieval merged with the ordinary ranking.

    Departments with no `department` metadata (Life Sciences is the live case:
    11 current documents mention it, none carry it as a metadata value) fall
    back to a query-side hint. Filtering on a value the corpus never stores
    would return nothing and silently drop that entity."""
    # Scale the per-entity budget to the entity count. A faculty expansion can
    # name 11 departments (Arts, Humanities and Social Sciences); at a fixed 2
    # slots each that is 22 against a cap of 14, and the final slice would
    # silently drop the last four departments - the exact failure this
    # mechanism exists to prevent, reintroduced by its own budget.
    # At least 1 slot each, never more than MULTI_ENTITY_PER_ENTITY.
    per_entity = max(1, min(MULTI_ENTITY_PER_ENTITY,
                            MULTI_ENTITY_MAX_RESULTS // max(len(aliases), 1)))
    picked_docs: list[str] = []
    picked_metas: list[dict] = []
    seen: set[tuple] = set()
    _entity_fill: list[dict] = []

    def take(docs, metas, limit):
        n = 0
        for d, m in zip(docs, metas):
            key = (m.get("source_url"), m.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            picked_docs.append(d)
            picked_metas.append(m)
            n += 1
            if n >= limit:
                return

    for alias in aliases:
        values = department_filter_values([alias])
        if values:
            where = {"$and": [{"is_current": True}, {"department": {"$in": values}}]}
            entity_query = retrieval_query
            res = vector_query(entity_query, n_results=pool_size, where=where)
        else:
            # No metadata for this entity, so hint it in the query text. The
            # hint must be FOCUSED: prepending the alias to the full
            # multi-entity question ("life sciences What are the accredited
            # programmes offered by CSEE, MSAS, Psychology, HSC, SRES and Life
            # Sciences?") leaves the other five departments' terms dominating
            # the embedding and returns none of this entity's content -
            # measured, 0 Life Sciences chunks. Stripping the other named
            # entities restores the signal.
            focused = retrieval_query
            for other in aliases:
                if other != alias:
                    focused = re.sub(re.escape(other), " ", focused, flags=re.I)
            focused = re.sub(r"[,\s]{2,}", " ", focused).strip()
            entity_query = f"{alias} {focused}"
            res = vector_query(entity_query, n_results=pool_size,
                               where={"is_current": True})
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]

        # The per-entity retrieval above is DENSE-ONLY - no BM25, no RRF - on
        # the one query shape where exact name matching is most valuable, since
        # the entity is named literally in the question. The ordinary path
        # fuses both channels; these reserved slots were filled by a strictly
        # weaker retriever (external review, 2026-08-11).
        if MULTI_ENTITY_LEXICAL:
            bm = lexical.query(entity_query, n_results=pool_size, current_only=True)
            b_docs = [h[1] for h in bm]
            b_metas = [h[2] for h in bm]
            if values:
                # BM25 has no department filter, so apply the same restriction
                # the dense `where` clause applied - otherwise this channel
                # would widen the entity slot instead of filling it.
                keep = [i for i, m in enumerate(b_metas) if m.get("department") in values]
                b_docs = [b_docs[i] for i in keep]
                b_metas = [b_metas[i] for i in keep]
            if b_docs:
                fused = _dedup_by_chunk(_rrf_fuse(
                    _surrogate_hits(docs, metas), _surrogate_hits(b_docs, b_metas)))
                docs = fused.get("documents", [[]])[0]
                metas = fused.get("metadatas", [[]])[0]

        if not docs:
            continue
        # Same recency pass the unfiltered path applies to its candidate pool.
        # Without it this path surfaced csee_ft_masters_accredited_variations_24
        # (2024-25) above the 2025-26 edition: both are is_current=True because
        # the older one lives under /ug/current/, which the path rule treats as
        # authoritative and which therefore overrides the family-max rule.
        # is_current is a coarse pre-filter, not a within-family ordering.
        fresh = _prefer_most_recent_year({"documents": [docs], "metadatas": [metas]})
        ranked = _rerank.rerank(retrieval_query, fresh, per_entity)
        before = len(picked_docs)
        take(ranked.get("documents", [[]])[0], ranked.get("metadatas", [[]])[0],
             per_entity)
        # Starvation is otherwise invisible: an entity that contributes nothing
        # looks identical to one that was never asked for. With 11 aliases the
        # budget already leaves only 3 of 14 slots for the base ranking, so
        # knowing WHICH entity came up empty is the difference between a
        # metadata gap and a retrieval gap.
        if RAG_TIMING:
            _entity_fill.append({"alias": alias, "filtered": bool(values),
                                 "candidates": len(docs),
                                 "took": len(picked_docs) - before,
                                 "budget": per_entity})

    if RAG_TIMING and _entity_fill:
        starved = [e["alias"] for e in _entity_fill if e["took"] == 0]
        _stage_note("multi_entity_fill", {"entities": _entity_fill, "starved": starved})

    # fill the remainder from the ordinary ranking
    take(base_results.get("documents", [[]])[0],
         base_results.get("metadatas", [[]])[0],
         max(0, MULTI_ENTITY_MAX_RESULTS - len(picked_docs)))

    docs = picked_docs[:MULTI_ENTITY_MAX_RESULTS]
    metas = picked_metas[:MULTI_ENTITY_MAX_RESULTS]

    # Keep each document's chunks CONTIGUOUS, preserving first-appearance order
    # of the documents themselves.
    #
    # HONEST STATUS (corrected 2026-08-10). This was added because the
    # six-department question reported four genuinely accredited CSEE
    # programmes as non-accredited, and the interleaved context looked like the
    # cause: the reserved per-entity chunks sat at the top and the same
    # document's fill chunks at the bottom, split by other departments'
    # material. That causal claim is FALSIFIED. Round 8j tested ordering
    # directly on this exact context shape (14 chunks, multi-entity) by
    # reversing it, which shreds contiguity far more thoroughly than the
    # interleaving did: quality did not drop - it was nominally HIGHER
    # (4.05 vs 3.85), inside a +/-0.20 noise floor.
    #
    # The grouping is kept because it is a harmless default that makes the
    # assembled context easier to read and reason about, NOT because it was
    # shown to help. The original CSEE wrong answer is unexplained; with cloud
    # generation unpinnable it may simply have been variance. Anyone tempted to
    # build further on "chunk order matters" should read Round 8j first.
    order: dict[str, list[int]] = {}
    for i, m in enumerate(metas):
        order.setdefault(m.get("source_url") or "", []).append(i)
    idx = [i for group in order.values() for i in group]
    return {"documents": [[docs[i] for i in idx]],
            "metadatas": [[metas[i] for i in idx]]}


# Adjacent-chunk expansion (2026-08-10). Traced from a real user question:
# "in which cases is an independent chair required for the examination of PGR
# degrees?" answered "the materials available don't specify the exact
# triggering circumstances" - false, the policy lists seven in section 3.1.
# That list lives in ONE chunk (independent-chairs-policy.pdf index 2) and the
# query retrieved chunks 1 and 6 of the same document. hit@6 was True, so no
# metric in the ledger could see it.
#
# MEASURED BEFORE BUILDING (eval/report.md Round 8o): across 160 turns, 79% of
# turns already hold an answer-bearing chunk; of the 26 that do NOT, 31% have
# one sitting immediately beside a retrieved chunk. So the ceiling is ~8 turns
# in 160 - real but bounded, and the larger 11% "FAR" slice needs ranking work
# this cannot touch. Built on that basis, not on the single vivid case.
#
# ADDS chunks rather than displacing them: the neighbours are appended after
# the ranked results, so a turn that was already correct keeps its ordering and
# its top chunk. Cost is context length, which is why the count is capped.
# ENABLED 2026-08-10 (eval/report.md Round 8p).
#   Real case: "in which cases is an independent chair required" went from
#   0/7 circumstances AND a false "the materials don't specify" denial, to 5/7
#   with no denial.
#   A/B on 20 questions (sets 4+5, cloud generator): 4.17 -> 4.22, delta +0.05,
#   inside the +/-0.20 noise floor. So: no detectable harm, no detectable
#   broad benefit - which is the expected shape when the benefit concentrates
#   in the ~5% of turns that have an adjacent answer chunk.
ADJACENT_CHUNK_EXPANSION = os.environ.get("RAG_ADJACENT_CHUNKS", "1") == "1"
# Narrowed after measuring (2026-08-10). Expanding around the top THREE hits
# with a cap of 3 touched 97% of turns and added 2.62 chunks each - a
# system-wide context change for a benefit concentrated in ~5% of turns, which
# is the ratio that has cost this project points before. Expanding only around
# the RANK-1 chunk still fixes the real case while touching 81% of turns and
# adding 1.26 chunks: same fix, less than half the context growth.
#
# Note the A/B above measured the WIDER setting. The shipped setting is a
# strict subset of it - strictly fewer added chunks - so the no-harm result
# carries over a fortiori. That is an inference, not a measurement of this
# exact configuration.
ADJACENT_MAX_ADDED = 2          # hard cap on appended neighbours
ADJACENT_FROM_TOP_N = 1         # only expand around the rank-1 chunk

# Document completion (2026-08-28). Reported failure: "list a department's
# milestones" returns some and not all. Measured cause - N_RESULTS is 6 and
# `ce-phd-2025-26.pdf` is 9 chunks, so the complete list cannot fit in the
# context even when the right document wins every slot. Across 160 replay
# turns the generator sees 15% of the gold document's chunks (532/3487).
#
# Completing the document outright is NOT viable: on enumerative turns the
# median gold document is missing 18 chunks and the worst is missing 65, so
# "add the rest" would multiply context on exactly the queries that already
# retrieve well - the failure mode of "When More Documents Hurt RAG"
# (arXiv 2606.11350) that this codebase has paid for before.
#
# Bounded instead, and the bounds are measured rather than picked. Only
# documents SMALL enough to complete (<= MAX_DOC_CHUNKS) and already DOMINANT
# in the ranking (>= MIN_SLOTS of the six), which is a structural signal rather
# than a regex over question wording - "list"/"what are the" matches 32% of
# turns and says nothing about whether the answer is spread across chunks.
# Ceiling at these settings: 27 of 160 turns (17%), median 5 chunks added,
# worst case 9.
#
# OFF by default. hit@6 cannot validate this by construction - adding chunks of
# a document already in the top 6 cannot change whether it was retrieved - so
# justifying it ON needs a generation eval (keyphrase coverage / judge), which
# has not been run. See eval/report.md "Round 34e".
DOC_COMPLETION_ENABLED = os.environ.get("RAG_DOC_COMPLETION", "1") == "1"
# Sized to the corpus, not guessed (corrected 2026-08-28 after shipping 12).
# The 80 current milestone documents run 5-16 chunks, median 10, so a cap of 12
# silently excluded 12 of them (15%) - and excluded exactly the WRONG ones: the
# two largest, lw-phd-milestones{,-sustainable-transitions}, are 13 chunks and
# carry the most codes in the corpus (21 and 19). A cap that drops the
# documents most in need of completion is worse than no cap.
DOC_COMPLETION_MAX_DOC_CHUNKS = 16   # covers all 80; the largest is 16
DOC_COMPLETION_MIN_SLOTS = 2         # ...that already hold this many of the six
DOC_COMPLETION_MAX_ADDED = 16        # hard cap on appended chunks, all documents
# Complete ONE document, not every qualifying one (2026-08-28, after a probe).
# "List all the milestones for a Law PhD student" put BOTH lw-phd-milestones
# and its -sustainable-transitions variant over the slot threshold, so both were
# completed: 23 chunks of near-identical content, and the answer cited 3 of 19
# codes - far worse than the 15/16 this mechanism was built to fix. Near-
# duplicate documents in one context do not add information, they add confusion.
# Same narrowing _adjacent_chunks made when top-3 expansion beat itself.
DOC_COMPLETION_MAX_DOCS = 1

# Scope gate (user's call, and it is the reason this ships ON). Restricting
# completion to ONE document class rather than every small dominant document
# is what makes the change provably inert everywhere it has not been measured:
# these documents entered the corpus on 2026-08-28, so all 160 replay turns and
# both stored results files contain ZERO of them, and the mechanism cannot
# change a single previously-measured number.
#
# It also matches the shape of the defect. Milestone documents are an
# enumeration - 16 codes (M1.1 ... M3.3) spread across 9 chunks - so a partial
# retrieval yields a partial LIST, which reads as a complete answer and is not.
# A policy answers from the clause that matches; a milestone document only
# answers in full. Measured on "List all the milestones for a CSEE PhD
# student": 8 of 16 codes cited before, 15 of 16 after.
#
# Widening this to all small documents is a real option and is NOT justified
# yet: it would touch 27 of 160 turns (17%), which hit@6 cannot score, so it
# needs a generation eval first. Set RAG_DOC_COMPLETION_SCOPE="" to test that.
DOC_COMPLETION_SCOPE = os.environ.get("RAG_DOC_COMPLETION_SCOPE", "/pgre/milestones-")


def _complete_small_documents(results: dict) -> dict:
    """Append the missing chunks of any SMALL document that already dominates
    the ranking, so an enumerable answer is not truncated by the slot budget.

    Additive and order-preserving, like _adjacent_chunks: appended chunks go
    after the ranked ones and carry no distance, because they were never
    scored. Returns the input unchanged on any lookup failure - a context
    expansion must never cost a retrieval.
    """
    metas = results.get("metadatas", [[]])[0]
    docs = results.get("documents", [[]])[0]
    if not metas:
        return results

    slots = Counter(m.get("source_url") for m in metas if m.get("source_url"))
    candidates = [u for u, n in slots.items()
                  if n >= DOC_COMPLETION_MIN_SLOTS and DOC_COMPLETION_SCOPE in u]
    if not candidates:
        return results

    have = {(m.get("source_url"), m.get("chunk_index")) for m in metas}
    try:
        from src.ingest import _get_collection
        coll = _get_collection()
        # ONE call for every candidate document's chunks - the same batching
        # discipline as _adjacent_chunks, which was costing two full metadata
        # scans per query before it was fixed.
        got = coll.get(
            where={"source_url": {"$in": candidates}} if len(candidates) > 1
                  else {"source_url": candidates[0]},
            include=["documents", "metadatas"],
        )
        gd, gm = got.get("documents") or [], got.get("metadatas") or []
    except Exception:
        return results

    by_url: dict[str, list] = {}
    for d, m in zip(gd, gm):
        by_url.setdefault(m.get("source_url"), []).append((d, m))

    add_docs, add_metas = [], []
    # deterministic order: densest-in-the-ranking document first, then by URL,
    # so the same query yields the same context on every run
    for url in sorted(candidates, key=lambda u: (-slots[u], u))[:DOC_COMPLETION_MAX_DOCS]:
        chunks = by_url.get(url, [])
        if not chunks or len(chunks) > DOC_COMPLETION_MAX_DOC_CHUNKS:
            continue
        for d, m in sorted(chunks, key=lambda c: c[1].get("chunk_index") or 0):
            if (m.get("source_url"), m.get("chunk_index")) in have:
                continue
            if len(add_docs) >= DOC_COMPLETION_MAX_ADDED:
                break
            add_docs.append(d)
            add_metas.append(m)

    if not add_docs:
        return results
    out = {"documents": [docs + add_docs], "metadatas": [metas + add_metas]}
    dists = results.get("distances")
    if dists:
        out["distances"] = [list(dists[0]) + [None] * len(add_docs)]
    return out


def _adjacent_chunks(results: dict) -> dict:
    """Append the immediate neighbours (chunk_index +/-1) of the top-ranked
    chunks, skipping any already present. Returns the input unchanged if the
    collection lookup fails - an expansion failure must never cost a retrieval."""
    metas = results.get("metadatas", [[]])[0]
    docs = results.get("documents", [[]])[0]
    if not metas:
        return results

    have = {(m.get("source_url"), m.get("chunk_index")) for m in metas}
    wanted: list[tuple] = []
    for m in metas[:ADJACENT_FROM_TOP_N]:
        url, idx = m.get("source_url"), m.get("chunk_index")
        if url is None or idx is None:
            continue
        for nb in (idx - 1, idx + 1):
            if nb >= 0 and (url, nb) not in have and (url, nb) not in wanted:
                wanted.append((url, nb))
    if not wanted:
        return results

    wanted = wanted[:ADJACENT_MAX_ADDED]
    try:
        from src.ingest import _get_collection
        coll = _get_collection()
        # ONE metadata query for all neighbours instead of one per neighbour.
        # Each coll.get() with a `where` and no ids is a metadata scan over
        # 21.7k chunks, and this ran twice per query on ~81% of turns.
        got = coll.get(
            where={"$or": [{"$and": [{"source_url": u}, {"chunk_index": i}]}
                           for u, i in wanted]} if len(wanted) > 1 else
                  {"$and": [{"source_url": wanted[0][0]}, {"chunk_index": wanted[0][1]}]},
            include=["documents", "metadatas"],
        )
        gd = got.get("documents") or []
        gm = got.get("metadatas") or []
        add_docs, add_metas = [], []
        for d, m in zip(gd, gm):
            # $or returns matches in arbitrary order, and a metadata filter is
            # only as good as the metadata: verify each row is genuinely the
            # neighbour asked for, and belongs to the SAME document.
            if (m.get("source_url"), m.get("chunk_index")) in wanted:
                add_docs.append(d)
                add_metas.append(m)
    except Exception:
        return results

    if not add_docs:
        return results
    out = {"documents": [docs + add_docs], "metadatas": [metas + add_metas]}
    # `distances` is preserved by _exclude_partner_institutions but was dropped
    # here, so an expanded result silently lost a field the unexpanded one had.
    # Appended neighbours have no distance of their own - they were never
    # scored - so they carry None rather than a fabricated number.
    dists = results.get("distances")
    if dists:
        out["distances"] = [list(dists[0]) + [None] * len(add_docs)]
    return out


def retrieve(question: str, history: list[dict], summary: str = "",
             partner_mode: str | None = None) -> tuple[dict, str]:
    """The full retrieval path used by answer() - query contextualization
    plus recency preference - exposed separately so eval/scoring code can
    measure exactly what production retrieves, not a simplified stand-in.
    Returns (results, retrieval_query)."""
    _t0 = _perf.time()
    retrieval_query = _contextualize_query(question, history, summary)
    _stage_timer("contextualize", _t0)
    _t0 = _perf.time()

    pool_size = N_RESULTS * FETCH_POOL_MULTIPLIER

    asked_years = _mentioned_years(retrieval_query)
    if asked_years:
        # one or more years are mentioned - but they may be edition requests
        # ("rules for 2021-22") or purely incidental (a cohort start year, a
        # statistic quoted from a document). Treat them as a soft preference:
        # fuse a pool PER MENTIONED YEAR with the default current pool, so
        # edition requests surface those years' documents while incidental
        # mentions can't exclude the current document that actually holds the
        # answer. One pool per year is what makes "compare 2025/26 with
        # 2020/21" answerable at all - see _mentioned_years().
        # No recency dedupe here - year-labeled docs are intentionally old.
        # year pools first, then the current pool - preserving the original
        # single-year list order so RRF tie-breaking (which follows insertion
        # order) is byte-identical to before for one-year queries.
        ranked_lists = []
        for asked_year in asked_years:
            year_dense = vector_query(retrieval_query, n_results=pool_size,
                                      where={"academic_year_norm": asked_year})
            ranked_lists.append(_dense_as_hits(year_dense))
            ranked_lists.append(lexical.query(retrieval_query, n_results=pool_size, year=asked_year))
        cur_dense = vector_query(retrieval_query, n_results=pool_size, where={"is_current": True})
        cur_bm25 = lexical.query(retrieval_query, n_results=pool_size, current_only=True)
        ranked_lists.append(_dense_as_hits(cur_dense))
        ranked_lists.append(cur_bm25)
        if SPLADE_ENABLED:
            for asked_year in asked_years:
                ranked_lists.append(_splade.query(retrieval_query, n_results=pool_size, year=asked_year))
            ranked_lists.append(_splade.query(retrieval_query, n_results=pool_size, current_only=True))
        if EMBEDDING_ENSEMBLE_ENABLED:
            for asked_year in asked_years:
                ranked_lists.append(_ensemble.query(retrieval_query, n_results=pool_size,
                                                     where={"academic_year_norm": asked_year}))
            ranked_lists.append(_ensemble.query(retrieval_query, n_results=pool_size,
                                                 where={"is_current": True}))
        if PSEUDO_QUERY_ENABLED:
            ranked_lists.append(_pseudo_query.query(retrieval_query, n_results=pool_size,
                                                     where={"is_current": True}))
        if COLBERT_FIRST_STAGE_ENABLED:
            for asked_year in asked_years:
                ranked_lists.append(_colbert_index.query(retrieval_query, n_results=pool_size, year=asked_year))
            ranked_lists.append(_colbert_index.query(retrieval_query, n_results=pool_size, current_only=True))
        _aliased = _alias_expanded_query(retrieval_query)
        if _aliased:
            ranked_lists.append(_dense_as_hits(
                vector_query(_aliased, n_results=pool_size, where={"is_current": True})))
            ranked_lists.append(lexical.query(_aliased, n_results=pool_size, current_only=True))
        candidates = _dedup_by_chunk(_rrf_fuse(*ranked_lists))
    else:
        # default case: pre-filter the historical archive out of both pools
        # (~70% of chunks), fuse dense + BM25, then apply the family-level
        # recency dedupe as a safety net for docs the is_current flag missed.
        # degree_length/award_type are only consumed by FACET_PREFERENCE_ENABLED
        # and SPLADE_ENABLED below (both off by default) - skip the regex scan
        # when neither is on rather than computing it unconditionally.
        if FACET_PREFERENCE_ENABLED or SPLADE_ENABLED:
            degree_length = extract_degree_length(retrieval_query)
            award_type = extract_award_type(retrieval_query)
        else:
            degree_length = award_type = ""

        _ts = _perf.time()
        dense = vector_query(retrieval_query, n_results=pool_size, where={"is_current": True})
        _stage_timer("r_dense", _ts)
        bm25_hits = lexical.query(retrieval_query, n_results=pool_size, current_only=True)
        if WEIGHTED_FUSION_ENABLED:
            # one already-combined list, still handed to _rrf_fuse below
            # alongside the other heterogeneous preference signals (facet,
            # SPLADE, ensemble) - see _weighted_dense_bm25's docstring
            ranked_lists = [_weighted_dense_bm25(dense, bm25_hits, DENSE_WEIGHT, BM25_WEIGHT)]
        else:
            ranked_lists = [_dense_as_hits(dense), bm25_hits]

        if FACET_PREFERENCE_ENABLED and (degree_length or award_type):
            # soft facet preference, not a hard exclusion filter - a first
            # attempt at hard-filtering on these facets regressed hit@6
            # (eval/report.md, "Stage A") because degree_length/award_type
            # are not mutually-exclusive partitions of the corpus: a masters
            # document can legitimately hold the correct diploma-exit-award
            # answer, so excluding non-matching documents throws away real
            # answers. The soft version (this branch) regressed too, though
            # less badly (RoA hit@6 70%->60% vs 70%->57.5% hard-filtered,
            # see eval/report.md "Stage A2") - extraction gaps mean many
            # correct documents (filenames like "east15"/"mscperiodontology")
            # never get tagged with a facet at all, so they get no boost
            # while occasional false-positive matches on unrelated documents
            # do, net-negative even without ever excluding anyone. Off by
            # default; kept for reference, not a dead end worth deleting.
            facet_conditions = [{"is_current": True}]
            if degree_length:
                facet_conditions.append({"degree_length": degree_length})
            if award_type:
                facet_conditions.append({"award_type": award_type})
            facet_dense = vector_query(retrieval_query, n_results=pool_size, where={"$and": facet_conditions})
            facet_bm25 = lexical.query(retrieval_query, n_results=pool_size, current_only=True,
                                        degree_length=degree_length, award_type=award_type)
            ranked_lists.append(_dense_as_hits(facet_dense))
            ranked_lists.append(facet_bm25)

        if SPLADE_ENABLED:
            ranked_lists.append(_splade.query(retrieval_query, n_results=pool_size, current_only=True,
                                               degree_length=degree_length, award_type=award_type))
        if EMBEDDING_ENSEMBLE_ENABLED:
            ranked_lists.append(_ensemble.query(retrieval_query, n_results=pool_size,
                                                 where={"is_current": True}))
        if PSEUDO_QUERY_ENABLED:
            ranked_lists.append(_pseudo_query.query(retrieval_query, n_results=pool_size,
                                                     where={"is_current": True}))
        if DOC_ROUTING_ENABLED:
            # chunks of the top identity-matched documents, as one extra soft
            # RRF list - identity matching happens in the document index
            # (src/doc_index.py), then this pulls those documents' best chunks
            # into the fusion so they can outrank identity-less siblings
            routed_urls = _doc_index.query(retrieval_query, n_results=DOC_ROUTING_TOP_DOCS)
            if routed_urls:
                routed_dense = vector_query(retrieval_query, n_results=pool_size,
                                            where={"source_url": {"$in": routed_urls}})
                ranked_lists.append(_dense_as_hits(routed_dense))
        if COLBERT_FIRST_STAGE_ENABLED:
            ranked_lists.append(_colbert_index.query(retrieval_query, n_results=pool_size, current_only=True))
        _aliased = _alias_expanded_query(retrieval_query)
        if _aliased:
            # extra pool only - see _alias_expanded_query on why this is
            # additive rather than a query rewrite
            ranked_lists.append(_dense_as_hits(
                vector_query(_aliased, n_results=pool_size, where={"is_current": True})))
            ranked_lists.append(lexical.query(_aliased, n_results=pool_size, current_only=True))
        candidates = _prefer_most_recent_year(_dedup_by_chunk(_rrf_fuse(*ranked_lists)))

    global _LAST_CANDIDATE_POOL  # debug hook for the retrieval recall diagnostic (no behavior change)
    _LAST_CANDIDATE_POOL = candidates

    # STRICT modes are a user's explicit choice, so they ignore the
    # name-detection gate: "Essex only" means Essex only even when the question
    # names a college, and "Partner only" means partner even when it does not.
    # A switch the user can see should do what it says rather than negotiate
    # with a heuristic.
    _strict = partner_mode if partner_mode in ("essex_only", "partner_only") else None
    if _strict:
        # STRICT means strict. The heuristic path keeps a safeguard - if every
        # candidate would be dropped it returns them all, because a false
        # negative from the gate should not leave the user with nothing. That
        # safeguard is WRONG for an explicit user choice: measured on 60
        # partner documents, "Essex only" served partner documents on 25 of
        # them, which is the control silently doing the opposite of its label.
        # With no fallback the answer becomes "the policies I can see don't
        # cover this", which is the truthful response in that mode.
        candidates = _filter_by_institution(candidates, partner=(_strict == "partner_only"))
    elif PARTNER_EXCLUDE_WHEN_UNNAMED and not _names_partner_institution(retrieval_query, history):
        # before the rerank, so the freed slots are filled by Essex documents
        # rather than left empty
        # per-request override, falling back to the deployment default
        _mode = partner_mode if partner_mode in (
            "exclude", "cap1", "boost", "essex_only", "partner_only") else PARTNER_MODE
        if _mode == "exclude":
            candidates = _exclude_partner_institutions(candidates)
        elif _mode == "cap1":
            candidates = _cap_partner_institutions(candidates, cap=1)
        elif _mode == "boost":
            candidates = _boost_home_institution(candidates, places=PARTNER_BOOST)

    _ts = _perf.time()
    results = _rerank.rerank(retrieval_query, candidates, N_RESULTS)
    _stage_timer("r_rerank", _ts)

    if MULTI_ENTITY_RETRIEVAL:
        _entities = detect_departments(retrieval_query)
        if len(_entities) >= MULTI_ENTITY_MIN_ENTITIES:
            _ts = _perf.time()
            results = _multi_entity_results(retrieval_query, _entities, results, pool_size)
            _stage_timer("r_multientity", _ts)
            # The per-entity retrievals inside _multi_entity_results are FRESH
            # vector_query calls filtered only on is_current/department, so they
            # re-admit partner chunks that were excluded from `candidates`
            # above. Without this, any question naming >=2 departments quietly
            # bypassed the exclusion, and _demote_partner_institutions below
            # only reorders what is already there. Re-applied to the MERGED
            # result (external review, 2026-08-11).
            if (MULTI_ENTITY_PARTNER_RECHECK and PARTNER_EXCLUDE_WHEN_UNNAMED
                    and not _names_partner_institution(retrieval_query, history)):
                results = _exclude_partner_institutions(results)

    if MULTIHOP_DECOMPOSITION_ENABLED:
        prelim_metas = results.get("metadatas", [[]])[0]
        if _top_family_count(prelim_metas) <= AMBIGUITY_FAMILY_COUNT_THRESHOLD:
            candidate_titles = _distinct_family_titles(candidates.get("metadatas", [[]])[0], limit=5)
            subqueries = _decompose_query(retrieval_query, candidate_titles)
            if subqueries:
                expanded_lists = [_surrogate_hits(candidates.get("documents", [[]])[0],
                                                   candidates.get("metadatas", [[]])[0])]
                for sq in subqueries:
                    sq_dense = vector_query(sq, n_results=pool_size, where={"is_current": True})
                    sq_bm25 = lexical.query(sq, n_results=pool_size, current_only=True)
                    expanded_lists.append(_surrogate_hits(sq_dense.get("documents", [[]])[0],
                                                           sq_dense.get("metadatas", [[]])[0]))
                    expanded_lists.append(_surrogate_hits([h[1] for h in sq_bm25], [h[2] for h in sq_bm25]))
                expanded_candidates = _prefer_most_recent_year(_dedup_by_chunk(_rrf_fuse(*expanded_lists)))
                results = _rerank.rerank(retrieval_query, expanded_candidates, N_RESULTS)

    if HOME_INSTITUTION_TIEBREAK_ENABLED:
        results = _prefer_home_institution(results)

    if PARTNER_INSTITUTION_DEMOTE_ENABLED:
        results = _demote_partner_institutions(results)

    if ADJACENT_CHUNK_EXPANSION:
        # last, so neighbours are appended to the FINAL ordering rather than
        # competing in the rerank - the point is to add context, not to
        # re-rank on it
        _ts = _perf.time()
        results = _adjacent_chunks(results)
        _stage_timer("r_adjacent", _ts)

    if DOC_COMPLETION_ENABLED:
        # after adjacent expansion, so already-appended neighbours count as
        # present and are not added twice
        _ts = _perf.time()
        results = _complete_small_documents(results)
        _stage_timer("r_doc_completion", _ts)

    _stage_timer("retrieve", _t0)
    return results, retrieval_query


# J6: disclose-don't-gate. When the reranked top-6 is fragmented across many
# document families (the same imprecise ambiguity signal Stage B would have
# used to refuse/clarify, and Stage H to gate), answer anyway from the
# retrieved context but append a short disclosure naming the primary source
# document and inviting correction. Unlike gating (Stage H) the history keeps
# a real answer, so the follow-up contextualizer knock-on can't occur; unlike
# a clarifying question (Stage B) a false-positive trigger costs only an
# occasionally-unneeded caveat, not a wrong response type - which makes the
# signal's known 14% false-positive rate tolerable.
DISCLOSE_AMBIGUITY_ENABLED = True


# Variance-gated disclosure (round-6 Tier 1 mechanism, wired 2026-08-07).
# The J6 disclosure currently fires on every fragmented-pool turn, so a user
# is told "rules often differ by programme, tell me which one you mean" even
# when the answer is IDENTICAL across every programme - Merit is 60 corpus-
# wide, so for that question the caveat is pure noise that trains people to
# ignore it. eval/variance_map.py measured which parameters actually vary:
# Merit, Distinction and the top classification are UNIFORM; pass mark,
# condonement, reassessment cap, credits and further attempts VARY.
#
# So: suppress the disclosure when the question is clearly about a parameter
# known to be uniform, and keep it everywhere else. Deliberately conservative
# in the safe direction - an unrecognised question keeps the disclosure, and
# only an explicit uniform-parameter match with no varying-parameter match can
# suppress it. Worst case of a false suppression is a missing caveat on an
# answer that is the same across programmes anyway.
VARIANCE_GATED_DISCLOSURE = True
_VARIANCE_MAP_PATH = "eval/variance_map_result.json"
_variance_terms = None


def _uniform_parameter_terms() -> tuple[set, set]:
    """(uniform_terms, varying_terms) loaded from the measured variance map.
    Falls back to empty sets - i.e. disclosure behaves exactly as before - if
    the map is missing, so this can never harden into a hidden dependency."""
    global _variance_terms
    if _variance_terms is not None:
        return _variance_terms
    keywords = {
        "Merit threshold": ["merit"],
        "Distinction threshold": ["distinction"],
        "First-class / top classification": ["first class", "first-class"],
        "Module pass mark": ["pass mark", "pass a module", "passing mark"],
        "Condonement threshold": ["condone", "condonement", "compensat"],
        "Reassessment mark cap": ["capped", "cap on", "reassessment mark"],
        "Credits for the award": ["how many credits", "credits required", "credits must"],
        "Permitted further attempts": ["further attempt", "resit", "reassessment attempt"],
    }
    from pathlib import Path as _Path

    uniform, varying = set(), set()
    try:
        for row in json.loads(_Path(_VARIANCE_MAP_PATH).read_text()):
            bucket = uniform if row.get("verdict") == "UNIFORM" else varying
            bucket.update(keywords.get(row.get("parameter"), []))
    except Exception:
        pass  # no map -> no gating
    _variance_terms = (uniform, varying)
    return _variance_terms


def _answer_is_programme_invariant(question: str) -> bool:
    """True only when the question names a parameter measured as UNIFORM and
    names no varying one - so the retrieved sibling cannot change the answer."""
    uniform, varying = _uniform_parameter_terms()
    q = question.lower()
    return any(t in q for t in uniform) and not any(t in q for t in varying)


def _ambiguity_disclosure(metadatas: list[dict]) -> str:
    titles = _distinct_family_titles(metadatas, limit=3)
    primary = titles[0] if titles else "the retrieved document"
    # Idea 3 extension: name the actual differentiator (e.g. the specific
    # programme) when the J1 identity record has one, instead of only a
    # generic "tell me which programme" ask - post-retrieval, so it carries
    # none of J2/J3's retrieval-side risk.
    detail = ""
    if IDENTITY_CONTEXT_ENABLED and metadatas:
        from src.ingest import _load_doc_identity
        identity = _load_doc_identity(metadatas[0].get("source_url", ""))
        label = identity.get("programme_name") or identity.get("partner_institution")
        if label:
            detail = f" ({label})"
    return (
        f"\n\n_Note: this answer is based primarily on \"{primary}\"{detail}. Your question could "
        "also relate to other documents (rules often differ by programme, department, or academic "
        "year) - tell me which programme or year you mean if this isn't the right one._"
    )


# D3 (2026-07-23): generic clarify-on-underspecified gate. Fires when a query
# names no degree-length/award-type AND the reranked pool is fragmented across
# >= CLARIFY_FAMILY_THRESHOLD distinct families (no single document dominates,
# so the answer is programme-dependent and we don't know which). Then it STOPS
# and asks the user to name their programme instead of guessing. GENERIC ask
# only - it deliberately lists NO candidate options: on a retrieval miss the
# correct document is by definition absent from the pool, so any options sourced
# from it would all be wrong (proven by J8/NAMEABLE_CLARIFICATION below and by a
# logical certainty - a miss means hit@6=False). Trigger precision is only ~0.45
# (it interrupts some answerable general questions), and a clarifying question is
# scored as a MISS by the hit@6 eval by design, so this is OFF by default and
# meant to be judged on real conversations. See eval/report.md "Round 4, D3".
#
# DECLINED 2026-08-10 (user's call, after the trade was laid out): stays OFF.
# The ~0.45 trigger precision means roughly one interrupted answerable question
# for every two correct stops, and no metric here can settle it - a clarifying
# question scores as a MISS by construction, since hit@6 asks whether the right
# document was retrieved and asking a question retrieves nothing. Not a
# measurement question; a product one, and it has been answered.
CLARIFY_UNDERSPECIFIED_ENABLED = False
CLARIFY_FAMILY_THRESHOLD = 6


def _clarify_underspecified_response() -> str:
    # Offer BOTH branches: a specific programme (rescues the sibling-miss case,
    # validated) OR "in general" (Fable 5, round 5: the definitional questions
    # that also trigger this gate have no programme - the user saying "general"
    # lets the follow-up contextualizer retrieve the university-wide framework/
    # glossary instead of the gate asking again in a loop).
    return (
        "This depends on which programme or degree you're asking about - the rules of assessment "
        "differ across programmes, and your question doesn't name one. Tell me the specific "
        "programme or degree (and the academic year, if it matters) and I'll give you its exact "
        'rule - or say "in general" and I\'ll answer from the university-wide framework and glossary.'
    )


# J8: nameable-identity clarification - KILLED BY MANUAL PRE-VALIDATION,
# never run through a full eval (eval/report.md "J8"). Motivating idea: ask a
# clarifying question only when the candidate pool's J1 identity records
# contain >=2 distinct nameable labels, since a hand simulation confirmed
# supplying the RIGHT missing fact fixes retrieval cleanly (CSEE/MA Social
# Work both went to rank 1-2 after reformulation). But the candidate-sourcing
# step - scanning identity labels among documents retrieval ALREADY GOT
# WRONG - has no way to surface the correct option: tested on the MA Social
# Work miss, it offered 4 confidently-wrong programme names (MSc AI, East 15,
# Sport/Rehab, CSEE - none correct) as clarification choices, since none of
# retrieval's wrong picks happened to be the right one. Also tried sourcing
# candidates from the J3 document-identity index queried against the raw
# question text instead of the retrieved pool - same failure, for the same
# reason: a genuinely underspecified query has no signal for ANY index (chunk
# or document level) to match "Social Work" against. Conclusion: you can't
# auto-detect good clarification options for the queries that need them most
# - the missing information is only recoverable by asking a fully GENERIC
# question with no named guesses, which is what J6's disclosure already does
# without gating's demonstrated follow-up cost (Stage H). Left off; kept only
# as documented dead code, not wired to run.
NAMEABLE_CLARIFICATION_ENABLED = False


def _nameable_identity_labels(metadatas: list[dict], limit: int = 4) -> list[str]:
    """Distinct, non-empty identity labels (programme name, else partner
    institution, else department - the J1 fields, in specificity order)
    across the distinct document families in a candidate pool. Documents
    with an empty identity record (generic/university-wide) contribute
    nothing, which is exactly what lets this signal tell "ask which
    programme" apart from "there's no programme to ask about"."""
    from src.ingest import _load_doc_identity
    seen_families: set[str] = set()
    labels: list[str] = []
    for meta in metadatas:
        family = _document_family(meta.get("source_url", ""))
        if family in seen_families:
            continue
        seen_families.add(family)
        identity = _load_doc_identity(meta.get("source_url", ""))
        label = identity.get("programme_name") or identity.get("partner_institution") or identity.get("department")
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def _identity_clarifying_question(labels: list[str]) -> str:
    listed = "; ".join(labels)
    return (
        "Your question could relate to a few different programmes, and I want to give you the "
        f"right answer rather than guess: {listed}. Could you tell me which one you mean?"
    )


# Chunk ORDER (2026-08-10). Round 8g found that the SAME retrieved chunks in a
# different order produced a materially wrong answer - the multi-entity path
# reported 4 genuinely accredited CSEE programmes as non-accredited purely
# because that document's chunks were split across the context instead of
# contiguous. Nothing in this ledger measures ordering: hit@6, span coverage
# and evidence-sufficiency are all set-membership tests, so an ordering that
# halves answer quality scores identically to one that doubles it.
#
# RAG_CHUNK_ORDER re-orders the FINAL context without changing its membership,
# so an A/B isolates ordering alone:
#   "rank"     - reranker order (production default)
#   "grouped"  - chunks of the same document made contiguous
#   "reversed" - worst-ranked first; a deliberate spoiler. If quality is
#                unaffected by THIS, the pipeline is order-insensitive and the
#                Round 8g observation was something else.
CHUNK_ORDER = os.environ.get("RAG_CHUNK_ORDER", "rank")


def _apply_chunk_order(results: dict) -> dict:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if CHUNK_ORDER == "rank" or len(documents) < 2:
        return results
    if CHUNK_ORDER == "reversed":
        idx = list(range(len(documents)))[::-1]
    elif CHUNK_ORDER == "grouped":
        groups: dict[str, list[int]] = {}
        for i, m in enumerate(metadatas):
            groups.setdefault(m.get("source_url") or "", []).append(i)
        idx = [i for g in groups.values() for i in g]
    else:
        return results
    out = {"documents": [[documents[i] for i in idx]],
           "metadatas": [[metadatas[i] for i in idx]]}
    dists = results.get("distances")
    if dists and dists[0] and len(dists[0]) == len(documents):
        out["distances"] = [[dists[0][i] for i in idx]]
    return out


# Conversation titles (2026-08-10). The sidebar previously showed the first 60
# characters of the opening question, which truncates mid-word and reads badly
# ("In which cases is an independent chair required for the exam"). A short
# generated label is what makes a conversation list scannable.
#
# Generated AFTER the answer is returned (FastAPI BackgroundTasks) so it adds
# no latency to the turn the user is waiting on, and falls back to the old
# truncation on any error - a failed title must never cost an answer.
GENERATED_TITLES = os.environ.get("RAG_GENERATED_TITLES", "1") == "1"

_TITLE_PROMPT = (
    "Write a short title, 3 to 6 words, naming the TOPIC of this question about "
    "University of Essex policies. No quotation marks, no trailing full stop, "
    "no prefix like 'Title:'. Use the reader's own vocabulary. Reply with the "
    "title only."
)


def generate_title(question: str) -> str:
    """Short topical label for a conversation. Returns '' on any failure, so
    callers keep whatever title they already had."""
    if not GENERATED_TITLES:
        return ""
    try:
        out = contextualize_chat(messages=[
            {"role": "system", "content": _TITLE_PROMPT},
            {"role": "user", "content": question[:600]},
        ], model=CONTEXTUALIZE_MODEL).strip()
    except Exception:
        return ""
    out = out.strip().strip('"').strip("'").rstrip(".").strip()
    # a model that ignores the instruction and answers the question instead
    # would produce something long; fall back rather than show a paragraph
    if not out or len(out) > 70 or "\n" in out:
        return ""
    return out


# Enumeration repair (2026-08-28). The generator drops items from coded lists
# even when every item is in front of it - measured across four departments
# with retrieval verified complete each time: CSEE 17/17, Sociology 15/17,
# Psychology 13/15, Law 14/20. Persuasion was tried first and BACKFIRED
# (_ENUMERATION_RULE: 3/4 complete -> 2/4, and failures grew from one code to
# five), so this does not ask the model to behave. It checks.
#
# The check is arithmetic: the codes present in the context are known exactly,
# so the codes present in the answer can be subtracted from them. Anything left
# over was dropped. One retry naming the missing codes, then stop.
#
# WHY IT NEEDS A GATE. "What does milestone M2.1 require?" is correctly
# answered by citing one code, and demanding all seventeen would be wrong. The
# gate is structural rather than a regex over question wording: repair only
# when the answer already cites ENUMERATION_MIN_CITED codes, i.e. it is visibly
# attempting a list and fell short. One- and two-code answers are left alone.
#
# STREAMING. The retry does not stream; the corrected text is returned and the
# client re-renders from the returned text, so a streaming user can see the
# short answer replaced by the complete one. Same trade-off _scrub_plumbing
# already documents, and the same reason: correctness of the stored text wins.
ENUMERATION_REPAIR_ENABLED = os.environ.get("RAG_ENUMERATION_REPAIR", "1") == "1"
# [MC]: M milestones AND C completion milestones. C-codes appear in 52 of the
# 80 current milestone documents and an M-only pattern silently ignored them -
# every "16/16" measured before this was really out of 17.
ENUMERATION_CODE_RE = re.compile(r"\b[MC]\d+\.\d+\b")
ENUMERATION_MIN_CITED = 3
ENUMERATION_SCOPE = os.environ.get("RAG_ENUMERATION_SCOPE", "/pgre/milestones-")


def _missing_enumeration_codes(context: str, answer_text: str, metadatas: list[dict]) -> list[str]:
    """Codes present in the retrieved context but absent from the answer, or []
    when repair does not apply. Empty for a question about one specific
    milestone - see ENUMERATION_MIN_CITED."""
    if not any(ENUMERATION_SCOPE in (m.get("source_url") or "") for m in metadatas):
        return []
    in_ctx = set(ENUMERATION_CODE_RE.findall(context))
    in_ans = set(ENUMERATION_CODE_RE.findall(answer_text))
    if len(in_ans) < ENUMERATION_MIN_CITED:
        return []
    return sorted(in_ctx - in_ans)


def _repair_prompt(missing: list[str]) -> str:
    """The retry turn. Names the codes and asks for the WHOLE answer again -
    appending them as a postscript would read as an afterthought, and the point
    is an answer the reader can trust as a list."""
    names = ", ".join(missing)
    return (
        f"That answer left out {names}. Write the full answer again, covering every "
        f"milestone in the same style and order as before, including {names}. "
        "Do not mention this correction or that anything was missing."
    )


def answer(question: str, history: list[dict], summary: str = "", detail: str = "default",
           on_token=None, partner_mode: str | None = None) -> tuple[str, list[str], str, list[str]]:
    """Returns (answer_text, source_urls_used, retrieval_query, ranked_top_urls).

    The last two are the exact retrieval this call actually used, not a
    re-derived approximation - external code review (2026-07-21, see
    eval/report.md "Phase 1") found the eval harness previously called
    retrieve() a second, independent time (via ranked_retrieval() in
    eval/run_eval.py) to score retrieval quality, separately from this
    function's own internal retrieve() call that actually produced the
    context the answer was generated from. Since _contextualize_query()'s
    rewrite is an LLM sample, those two calls could diverge on follow-up
    turns - the eval would then be scoring a retrieval that wasn't the one
    the answer was actually generated from. Surfacing this call's own
    retrieval_query/ranked_top_urls lets callers score exactly what happened,
    with a single retrieve() invocation per turn."""
    _ta = _perf.time()
    results, retrieval_query = retrieve(question, history, summary, partner_mode=partner_mode)
    results = _apply_chunk_order(results)
    metadatas = results.get("metadatas", [[]])[0]
    ranked_top_urls = [m.get("source_url") for m in metadatas]

    if AMBIGUITY_DETECTION_ENABLED and _top_family_count(metadatas) <= AMBIGUITY_FAMILY_COUNT_THRESHOLD:
        sources = sorted({m.get("source_url") for m in metadatas if m.get("source_url")})
        return _clarifying_question(metadatas), sources, retrieval_query, ranked_top_urls

    if NAMEABLE_CLARIFICATION_ENABLED and _top_family_count(metadatas) <= AMBIGUITY_FAMILY_COUNT_THRESHOLD:
        labels = _nameable_identity_labels(metadatas)
        if len(labels) >= 2:
            sources = sorted({m.get("source_url") for m in metadatas if m.get("source_url")})
            return _identity_clarifying_question(labels), sources, retrieval_query, ranked_top_urls
        # no nameable identity among the candidates - nothing productive to
        # ask, fall through to a normal answer (+ J6 disclosure, if enabled)

    context = _format_context(results)

    if CRAG_VERIFICATION_ENABLED and not _context_supports_answer(question, context):
        sources = sorted({m.get("source_url") for m in metadatas if m.get("source_url")})
        return _uncertainty_response(sources), sources, retrieval_query, ranked_top_urls

    # D3: under-specified programme-rules question with a fragmented pool - ask
    # which programme instead of guessing (generic ask, no options; see flag).
    if (CLARIFY_UNDERSPECIFIED_ENABLED
            and _distinct_family_count(metadatas) >= CLARIFY_FAMILY_THRESHOLD
            and not extract_degree_length(retrieval_query)
            and not extract_award_type(retrieval_query)):
        sources = sorted({m.get("source_url") for m in metadatas if m.get("source_url")})
        return _clarify_underspecified_response(), sources, retrieval_query, ranked_top_urls

    messages = [{"role": "system", "content": system_prompt_for(detail)}]
    if summary:
        messages.append({"role": "system", "content": f"Summary of earlier conversation:\n{summary}"})
    messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    _tg = _perf.time()
    # on_token streams text to the caller as it arrives. The full text is still
    # returned, so storage and the non-streaming callers are unchanged - there
    # is ONE answer() and no parallel streaming implementation to drift from it.
    response_text = generate(messages=messages, on_token=on_token)
    _stage_timer("generate", _tg)
    # Deterministic backstop to the prompt rule, which only got plumbing-leaking
    # answers from 4/4 to 2/4. NOTE the streamed text is NOT scrubbed as it
    # flies past - substitutions need whole phrases, which a token boundary can
    # split. The client re-renders from the returned text, so what the user ends
    # up with is scrubbed; a leaked phrase can flicker mid-stream.
    response_text = _scrub_plumbing(response_text)

    if ENUMERATION_REPAIR_ENABLED:
        _missing = _missing_enumeration_codes(context, response_text, metadatas)
        if _missing:
            _tr = _perf.time()
            _repair = messages + [
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": _repair_prompt(_missing)},
            ]
            try:
                # no on_token: the retry is not streamed, the client re-renders
                # from the returned text (see ENUMERATION_REPAIR_ENABLED)
                _retried = _scrub_plumbing(generate(messages=_repair))
            except Exception:
                _retried = ""          # a failed repair must never cost the answer
            if _retried and not _missing_enumeration_codes(context, _retried, metadatas):
                response_text = _retried
            elif _retried:
                # one retry only, then be honest rather than silently short -
                # an incomplete list that looks complete is the actual defect
                _still = _missing_enumeration_codes(context, _retried, metadatas)
                response_text = _retried + (
                    f"\n\n_This list may be incomplete: the milestone document also "
                    f"defines {', '.join(_still)}._"
                )
            _stage_timer("enumeration_repair", _tr)

    if DISCLOSE_AMBIGUITY_ENABLED and _top_family_count(metadatas) <= AMBIGUITY_FAMILY_COUNT_THRESHOLD:
        # variance gate: skip the "rules differ by programme" caveat when the
        # measured answer does NOT differ by programme (see the variance map).
        if not (VARIANCE_GATED_DISCLOSURE and _answer_is_programme_invariant(question)):
            _disclosure = _ambiguity_disclosure(metadatas)
            response_text += _disclosure
            # it is appended, not interleaved, so a streaming caller can emit it
            # as the final chunk and still show exactly what gets stored
            if on_token is not None and _disclosure:
                on_token(_disclosure)

    sources = sorted({m.get("source_url") for m in metadatas if m.get("source_url")})
    _stage_timer("answer_total", _ta)
    return response_text, sources, retrieval_query, ranked_top_urls
