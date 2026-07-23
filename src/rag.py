"""Retrieval-augmented generation: retrieve relevant chunks from Chroma,
assemble a prompt with retrieved context + conversation history, and
generate an answer via the local chat model."""

import json
import re

from src import colbert_index as _colbert_index
from src import doc_index as _doc_index
from src import ensemble as _ensemble
from src import lexical
from src import pseudo_query as _pseudo_query
from src import rerank as _rerank
from src import splade as _splade
from src.docid import document_family as _document_family
from src.docid import extract_award_type, extract_degree_length, normalize_year
from src.ingest import query as vector_query
from src.llm import CONTEXTUALIZE_MODEL, chat, generate

N_RESULTS = 6
# over-fetch so recency filtering AND reranking have real depth to work with -
# failure analysis (eval/report.md) found relevant-but-mis-ranked documents as
# deep as rank 60 in a wide dense+BM25 union, so 4x (24 candidates) wasn't
# enough room for a reranker to ever see them
FETCH_POOL_MULTIPLIER = 8
RRF_K = 60

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
QUOTE_FIGURES_VERBATIM = False
_VERBATIM_RULE = (
    "\n- When the answer involves a specific number, mark, threshold, credit value, percentage, "
    "grade, or time limit, quote it exactly as it appears in the context - do not paraphrase, "
    "round, or omit it."
)

_SYSTEM_PROMPT_BASE = """You are a helpful assistant answering questions about University of Essex \
policies and rules of assessment, using only the provided context excerpts. Each excerpt is \
labeled with its source URL, title, document type, and (where known) department and academic year.

Rules:
- Answer using only the given context. If the context doesn't contain the answer, say so plainly \
rather than guessing.
- When multiple academic years of the same policy/rules document are relevant, prefer the most \
recent academic year unless the user asks about a specific past year.
- Always cite the source_url(s) you used, inline or in a short "Sources" list at the end.
- Be concise and direct."""

# Round 4 (user question, 2026-07-22): does per-claim INLINE citation reduce
# hallucination? The base prompt already asks for end-of-answer Sources; this
# stronger variant asks the model to attribute each specific factual claim to
# its source_url inline. RESULT (eval/results_inline_citations.json, full
# 80-turn A/B vs c1_anchor_v2 + hallucination_eval): REGRESSED groundedness
# 78.8% -> 67.5% (-11.3pts; every sub-metric worse - RoA 65->55, Policy
# 92.5->80, miss-turns 50->30.8). Answer_score was a wash (3.91->3.90) as D2
# predicted, but groundedness got WORSE: the per-claim citation becomes a new
# hallucination surface - the 7B confidently attributes facts to the WRONG
# filename ("pass mark is 50 [five-year-integrated-masters...]" when that doc
# says 40). Asking a small model to cite provenance per claim makes it
# fabricate provenance on top of facts. Reverted OFF; keep end-of-answer
# Sources only. Flag-gated.
INLINE_CITATIONS = False
_INLINE_CITATION_RULE = (
    "\n- Attribute every specific factual claim (a number, mark, threshold, credit value, "
    "percentage, time limit, or condition) to the exact source_url it came from, cited inline "
    "in square brackets immediately after the claim, e.g. \"the pass mark is 50 [<source_url>]\". "
    "Only state a claim if you can cite the context excerpt that supports it; if the context "
    "doesn't support it, say so instead of stating it."
)

SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_BASE
    + (_VERBATIM_RULE if QUOTE_FIGURES_VERBATIM else "")
    + (_INLINE_CITATION_RULE if INLINE_CITATIONS else "")
    + "\n"
)

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

# Conversation-reference / meta words: they point at PRIOR context ("going
# back to the very first thing I asked...") rather than carrying the current
# question's own topical content. A correct rewrite of a distant reference
# NECESSARILY drops these and substitutes the resolved topic in their place,
# so counting them in _is_faithful_rewrite's denominator penalizes exactly
# the rewrites that did their job (Phase 5 multi-turn probe found this - a
# fully-correct rewrite of "Back to the very first thing I asked about the
# credit limit - which department administers that programme?" was rejected
# at 27% overlap and fell back to the raw unresolved question, which then
# retrieved a completely unrelated document; see eval/report.md "Phase 5").
# Excluded from the ORIGINAL's word set in the faithfulness check only - a
# hijack still shares ~zero of the current question's real TOPICAL words, so
# stripping the scaffolding doesn't weaken hijack detection.
_REFERENTIAL_WORDS = {
    "about", "above", "again", "already", "asked", "asking", "back", "before", "discussed",
    "earlier", "first", "going", "initial", "initially", "just", "mentioned", "now", "originally",
    "point", "previous", "previously", "question", "raised", "said", "talked", "talking", "thing",
    "things", "told", "very",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def _is_faithful_rewrite(original: str, rewritten: str) -> bool:
    """Guards against a real failure mode of small local models on long/dense
    multi-topic conversation transcripts: instead of rewriting the new
    question, the contextualizer echoes a DIFFERENT question from earlier in
    the transcript (observed live: asked about "Professional Doctorate
    Director", got back a rewrite about "CSEE programmes" from six turns
    earlier - a completely unrelated retrieval followed). A faithful rewrite
    keeps most of the original's TOPICAL content words (replacing pronouns/
    references with specifics); a hijacked one shares almost none of them.

    The overlap is measured over topical words only - conversation-reference
    scaffolding (_REFERENTIAL_WORDS: "back", "earlier", "first", "asked"...)
    is excluded, because a correct resolution of a distant reference drops
    exactly those and substitutes the referenced topic, which the pre-Phase-5
    version mistook for an unfaithful rewrite. Short questions with too few
    topical words left to judge are always trusted, same as before, since a
    heavily-abbreviated or heavily-referential legitimate follow-up ("how
    about an independent chair?", "back to the first thing - who runs it?")
    can't be judged by surface overlap at all and relies on the transcript."""
    original_words = _content_words(original) - _REFERENTIAL_WORDS
    if len(original_words) < 3:
        return True
    overlap = original_words & _content_words(rewritten)
    return len(overlap) / len(original_words) >= 0.3


# J4: build the rewriter's transcript from the user's turns only, not the
# assistant's rendered answers. Two motivations (eval/report.md "J4"): the
# Stage H experiment showed that whatever the assistant says becomes the
# rewriter's input - a gated/uncertain answer measurably degraded follow-up
# retrieval - so coupling the rewriter to assistant prose makes retrieval
# hostage to generation; and the original live topic-drift bug came from the
# rewriter echoing the wrong part of a long mixed transcript, which a
# user-turns-only transcript halves. The user's own question sequence is
# usually what carries the topic thread.
# Tried always-on (J4, eval/report.md): small net regression (+1/-2 flips,
# follow-up-only hit@6 85%->82.5% - the very split it targeted). In normal
# operation the assistant's answers DO carry referents follow-ups point at
# ("what happens if a student fails that?" refers to something the answer
# introduced). Worth reconsidering only as a conditional fix if answer-gating
# (Stage H-style) ever returns. Off by default.
CONTEXTUALIZE_USER_TURNS_ONLY = False

# C1: alias-anchor guard (external code review round 3, 2026-07-22, Fable 5).
# The Phase-A re-baseline's one loss (east15 follow-up) was an identity-token-
# loss cascade: A3a reordered the primary pool -> the primary answer shifted ->
# the follow-up contextualizer's history changed -> its rewrite DROPPED the
# "East 15" identity anchor ("...at East 15 Acting School's Masters..." became
# "...non-core taught modules?"), and with no programme named, retrieval fell
# back to generic masters documents. This guard re-appends the active identity
# anchor when a follow-up rewrite loses it. CRITICALLY switch-safe: it fires
# ONLY when the rewrite is IDENTITY-LESS (names no distinctive programme/dept
# token from _identity_anchor_index at all) - a topic SWITCH names its new
# topic, so it's never identity-less and never gets the stale anchor appended
# (the Phase 5 probe showed switches work 19/19; this must not break them).
# Same deterministic-guard species as _is_faithful_rewrite - the only class of
# change that has survived evals here.
ALIAS_ANCHOR_GUARD_ENABLED = True

# identity tokens that are too generic to anchor on (appear across many
# programme families' identity records); on top of _STOPWORDS.
_ANCHOR_STOP = {
    "award", "awards", "certificate", "course", "courses", "degree", "degrees", "department",
    "diploma", "essex", "full", "graduate", "health", "integrated", "master", "masters", "module",
    "modules", "month", "months", "part", "postgraduate", "practice", "professional", "programme",
    "programmes", "registration", "rules", "school", "science", "sciences", "social", "student",
    "students", "taught", "time", "undergraduate", "university", "year",
}
_anchor_index = None


def _identity_anchor_index():
    """Cached (distinctive_tokens, families). A distinctive token is an
    identity word (from J1 programme_name/department/aliases) that appears in
    at most ANCHOR_DOCFREQ document FAMILIES - counting per family, not per
    file, so a programme's ~30 yearly editions/variants don't make its name
    ("periodontology", "acting") look common. families is [(label, tokenset)]
    for mapping a set of history-anchor tokens back to a clean label."""
    global _anchor_index
    if _anchor_index is not None:
        return _anchor_index
    from collections import Counter
    from pathlib import Path
    fam_toks: dict[str, set] = {}
    fam_label: dict[str, str] = {}
    for f in Path("data/doc_identity").glob("*.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        url = r.get("source_url", "")
        fam = _document_family(url)
        toks = {w for w in _content_words(
            " ".join([r.get("programme_name", ""), r.get("department", ""), " ".join(r.get("aliases") or [])])
        ) if w not in _ANCHOR_STOP}
        if not toks:
            continue
        fam_toks.setdefault(fam, set()).update(toks)
        # prefer a current edition's label (cleaner, e.g. the 25-26 wording)
        lab = r.get("programme_name") or r.get("department") or (r.get("aliases") or [""])[0]
        if lab and (fam not in fam_label or "-25" in url or "_25" in url or "/current/" in url):
            fam_label[fam] = lab
    docfreq = Counter()
    for toks in fam_toks.values():
        for t in toks:
            docfreq[t] += 1
    ANCHOR_DOCFREQ = 15  # famfreq<=15 keeps acting/east15/periodontology/nursing, drops the generic 16+ cluster
    # require len>=4: 3-char fragments that leak from identity phrases
    # ("non" from "non-standard", "pre" from "pre-registration") are common
    # English substrings that cause false "names a topic" positives - e.g.
    # "non-core taught modules" wrongly reads as naming a programme.
    distinctive = {t for t, c in docfreq.items() if c <= ANCHOR_DOCFREQ and len(t) >= 4}
    families = [(fam_label.get(fam, ""), toks & distinctive) for fam, toks in fam_toks.items()]
    families = [(lab, tk) for lab, tk in families if lab and tk]
    _anchor_index = (distinctive, families)
    return _anchor_index


def _anchor_from_history(history: list[dict]) -> tuple[str, set]:
    """The active identity anchor for a follow-up: the distinctive identity
    tokens present in the recent user turns, plus a clean label for the
    best-matching programme family. ('', set()) if the conversation names no
    distinctive identity yet."""
    distinctive, families = _identity_anchor_index()
    htoks: set = set()
    for m in [m for m in history if m.get("role") == "user"][-2:]:
        htoks |= _content_words(m.get("content", ""))
    hist_anchors = htoks & distinctive
    if not hist_anchors:
        return "", set()
    # score by (family-token overlap, then label-text-contains-anchor overlap):
    # the secondary term breaks ties toward the family whose own LABEL names
    # the anchor (e.g. prefer "East 15 Acting School" over a co-department
    # "Professional Code of Conduct" that shares the tokens but not the name).
    best_label, best_score = "", (0, 0)
    for label, toks in families:
        score = (len(toks & hist_anchors), len(_content_words(label) & hist_anchors))
        if score > best_score:
            best_score, best_label = score, label
    return best_label, hist_anchors


def _contextualize_query(question: str, history: list[dict], summary: str = "") -> str:
    """Retrieval only sees the current turn's text, so a follow-up like "what
    happens after that?" carries no signal about what "that" is. Rewriting it
    into a standalone question before embedding fixes this; the answering
    model still gets the original question plus full history, since it can
    already resolve the reference itself."""
    if not history and not summary:
        return question

    if CONTEXTUALIZE_USER_TURNS_ONLY:
        recent = [m for m in history if m.get("role") == "user"][-4:]
    else:
        recent = history[-6:]

    parts = []
    if summary:
        parts.append(f"Earlier conversation summary: {summary}")
    if recent:
        parts.append("\n".join(f"{m['role']}: {m['content']}" for m in recent))
    transcript = "\n".join(parts)

    rewritten = chat(messages=[
        {"role": "system", "content": CONTEXTUALIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{transcript}\n\nFollow-up question: {question}\n\nStandalone question:"},
    ], model=CONTEXTUALIZE_MODEL).strip()

    result = rewritten if (rewritten and _is_faithful_rewrite(question, rewritten)) else question

    if ALIAS_ANCHOR_GUARD_ENABLED:
        label, hist_anchors = _anchor_from_history(history)
        # Require >=2 overlapping distinctive tokens (external code review
        # round 4, 2026-07-22, Fable 5's false-anchor fix). A single
        # distinctive-token match is unreliable: common English words that
        # happen to appear in exactly one programme's identity card ("term"
        # from "what does the term...", "conditions", "principles",
        # "learning") register as distinctive because docfreq is computed over
        # identity records only, not over query/corpus frequency - so a
        # generic question would get a nonsensical programme anchor appended
        # (verified: the glossary/DipHE follow-ups were getting a
        # "musculoskeletal/public-health" anchor off the lone token "term"/
        # "conditions"). Two overlapping tokens is a real identity signal:
        # east15 still fires on {east, acting}, physiotherapy on {credit,
        # physiotherapy}; the spurious single-token cases stop firing.
        if label and len(hist_anchors) >= 2:
            result_tokens = _content_words(result)
            distinctive, _ = _identity_anchor_index()
            already_anchored = bool(result_tokens & hist_anchors)
            names_a_topic = bool(result_tokens & distinctive)  # a switch names its OWN new topic
            if not already_anchored and not names_a_topic:
                # identity-less continuation that dropped the anchor - re-append
                result = f"{result} ({label})"

    return result


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


def _is_partner_institution(meta: dict) -> bool:
    """True if this chunk's document is a partner-institution edition of a
    programme, using whichever signal is actually populated: the J1
    identity record's partner_institution field (only ~63% coverage -
    checked against the corpus, e.g. the Alexandria periodontology
    programme's own record has this blank despite genuinely being a
    partner edition) or the URL path (Essex's own site structure puts
    every partner-institution document under a /partner-institutions/
    folder - confirmed reliable structural signal, same category as the
    /previous-years/ and /current/ path overrides compute_current_flags
    already trusts)."""
    from src.ingest import _load_doc_identity

    if _load_doc_identity(meta.get("source_url", "")).get("partner_institution"):
        return True
    return "/partner-institutions/" in meta.get("source_url", "")


def _aliases(meta: dict) -> set[str]:
    from src.ingest import _load_doc_identity

    return {a.lower() for a in _load_doc_identity(meta.get("source_url", "")).get("aliases") or []}


def _prefer_home_institution(results: dict) -> dict:
    """Phase 4, experiment 2 (external code review round 2, 2026-07-21,
    Fable 5): when the final top-k contains both a partner-institution
    edition and a home (non-partner) edition of what looks like the same
    programme (sharing at least one J1 alias - e.g. both the home and
    Alexandria periodontology documents list "perio"), and the home
    edition currently ranks worse, promote it above the partner edition.
    Same species of deterministic, high-precision post-rerank rule as
    _prefer_most_recent_year - doesn't touch retrieval/reranking, just
    breaks a specific, identifiable tie the same way a human would default
    to "the home programme" absent the query naming a specific partner.
    Simplifying assumption for this first attempt: doesn't try to detect
    whether the query DOES name the partner institution specifically (the
    partner_institution field's coverage gaps make that unreliable too) -
    if that turns out to matter, the eval will show it as a loss."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[None] * len(documents)])[0]
    if len(documents) < 2:
        return results

    order = list(range(len(documents)))
    used = set()
    for i in range(len(order)):
        if i in used or not _is_partner_institution(metadatas[i]):
            continue
        partner_aliases = _aliases(metadatas[i])
        if not partner_aliases:
            continue
        for j in range(i + 1, len(order)):
            if j in used or _is_partner_institution(metadatas[j]):
                continue
            if partner_aliases & _aliases(metadatas[j]):
                order[i], order[j] = order[j], order[i]
                used.add(i)
                used.add(j)
                break

    return {
        "documents": [[documents[k] for k in order]],
        "metadatas": [[metadatas[k] for k in order]],
        "distances": [[distances[k] for k in order]],
    }


def _dense_as_hits(dense: dict) -> list[tuple[str, str, dict]]:
    return list(zip(
        dense.get("ids", [[]])[0],
        dense.get("documents", [[]])[0],
        dense.get("metadatas", [[]])[0],
    ))


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1] within the given pool. Relative, not tied
    to a specific distance metric's absolute scale - works whether the
    incoming values are Chroma distances (lower=better, metric-dependent) or
    BM25 scores (higher=better, unbounded), as long as the caller flips the
    sign consistently before calling this."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _weighted_dense_bm25(dense: dict, bm25_hits: list[tuple[str, str, dict, float]],
                          dense_weight: float, bm25_weight: float) -> list[tuple[str, str, dict]]:
    """Combines one dense result and one BM25 result list via a normalized
    weighted score sum (Stage F) instead of reciprocal rank, per Bruch et al.
    2022's finding that tuned convex fusion outperforms RRF because it uses
    how much better a candidate scored, not just its rank position. Returns
    a single best-first (id, doc, meta) list, drop-in compatible with
    _rrf_fuse's inputs so it can still be combined with other signals
    (soft facet/year preference, SPLADE, embedding ensemble) upstream."""
    ids = dense.get("ids", [[]])[0]
    docs = dense.get("documents", [[]])[0]
    metas = dense.get("metadatas", [[]])[0]
    dists = dense.get("distances", [[]])[0]

    # lower distance = better match; negate before normalizing so higher
    # normalized value = better, matching BM25's own orientation
    dense_scores = _normalize({i: -d for i, d in zip(ids, dists)})
    entries: dict[str, tuple[str, dict]] = {i: (doc, meta) for i, doc, meta in zip(ids, docs, metas)}

    bm25_raw = {id_: score for id_, doc, meta, score in bm25_hits}
    for id_, doc, meta, _score in bm25_hits:
        entries.setdefault(id_, (doc, meta))
    bm25_scores = _normalize(bm25_raw)

    all_ids = set(dense_scores) | set(bm25_scores)
    combined = {
        i: dense_weight * dense_scores.get(i, 0.0) + bm25_weight * bm25_scores.get(i, 0.0)
        for i in all_ids
    }
    ordered = sorted(all_ids, key=lambda i: combined[i], reverse=True)
    return [(i, entries[i][0], entries[i][1]) for i in ordered]


def _rrf_fuse(*ranked_lists: list[tuple]) -> dict:
    """Reciprocal-rank fusion of any number of ranked (id, doc, meta, ...)
    lists, keyed by chunk id. Dense embeddings and BM25 fail on different
    queries (semantic paraphrase vs exact terms like "Capped Mark" or course
    codes), so the union ranked by combined reciprocal rank beats either
    alone. Items may carry extra trailing elements (e.g. BM25's raw score,
    used elsewhere for weighted fusion) - only the first three are used
    here, so both 3- and 4-tuple inputs work unchanged."""
    scores: dict[str, float] = {}
    entries: dict[str, tuple[str, dict]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, 1):
            id_, doc, meta = item[0], item[1], item[2]
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (RRF_K + rank)
            entries.setdefault(id_, (doc, meta))

    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return {
        "documents": [[entries[i][0] for i in ordered]],
        "metadatas": [[entries[i][1] for i in ordered]],
        "distances": [[None] * len(ordered)],
    }


def _dedup_by_chunk(results: dict) -> dict:
    """Stage G's pseudo-query entries share a real chunk's (source_url,
    chunk_index) but carry a distinct id ("<chunk_id>_pqN"), so after fusion
    the same real content can appear twice under two different ids - once
    found via its own embedding, once via a pseudo-query's. Collapse to one
    entry per (source_url, chunk_index), keeping whichever occurrence ranked
    higher (results are already best-first at this point)."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[None] * len(documents)])[0]

    seen: set[tuple] = set()
    kept_docs, kept_metas, kept_dists = [], [], []
    for doc, meta, dist in zip(documents, metadatas, distances):
        key = (meta.get("source_url"), meta.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        kept_docs.append(doc)
        kept_metas.append(meta)
        kept_dists.append(dist)

    return {"documents": [kept_docs], "metadatas": [kept_metas], "distances": [kept_dists]}


AMBIGUITY_FAMILY_COUNT_THRESHOLD = 1


def _top_family_count(metadatas: list[dict]) -> int:
    """Among the reranked top-N results, how many chunks share the same
    document family as the #1 result. A low count means the pool is
    fragmented across many different documents with no single one
    dominating - the best-validated proxy found during Stage B
    pre-validation (eval/report.md) for "this query is ambiguous across
    genuinely different documents" (per arXiv 2603.24580's finding that
    genuine ambiguity needs surfacing, not more retrieval tuning). It is an
    imperfect signal (56% recall on known misses at 14% false-positive rate
    on known hits, measured on `stage1_rerank`) - not strong enough to have
    justified building this unprompted, but the least-bad option available."""
    if not metadatas:
        return 0
    top_family = _document_family(metadatas[0].get("source_url", ""))
    return sum(1 for m in metadatas if _document_family(m.get("source_url", "")) == top_family)


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
        ranked_lists = [
            _dense_as_hits(year_dense), year_bm25,
            _dense_as_hits(cur_dense), cur_bm25,
        ]
        if SPLADE_ENABLED:
            ranked_lists.append(_splade.query(retrieval_query, n_results=pool_size, year=asked_year))
            ranked_lists.append(_splade.query(retrieval_query, n_results=pool_size, current_only=True))
        if EMBEDDING_ENSEMBLE_ENABLED:
            ranked_lists.append(_ensemble.query(retrieval_query, n_results=pool_size,
                                                 where={"academic_year_norm": asked_year}))
            ranked_lists.append(_ensemble.query(retrieval_query, n_results=pool_size,
                                                 where={"is_current": True}))
        if PSEUDO_QUERY_ENABLED:
            ranked_lists.append(_pseudo_query.query(retrieval_query, n_results=pool_size,
                                                     where={"is_current": True}))
        if COLBERT_FIRST_STAGE_ENABLED:
            ranked_lists.append(_colbert_index.query(retrieval_query, n_results=pool_size, year=asked_year))
            ranked_lists.append(_colbert_index.query(retrieval_query, n_results=pool_size, current_only=True))
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

        dense = vector_query(retrieval_query, n_results=pool_size, where={"is_current": True})
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
        candidates = _prefer_most_recent_year(_dedup_by_chunk(_rrf_fuse(*ranked_lists)))

    results = _rerank.rerank(retrieval_query, candidates, N_RESULTS)

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


def answer(question: str, history: list[dict], summary: str = "") -> tuple[str, list[str], str, list[str]]:
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
    results, retrieval_query = retrieve(question, history, summary)
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if summary:
        messages.append({"role": "system", "content": f"Summary of earlier conversation:\n{summary}"})
    messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    response_text = generate(messages=messages)

    if DISCLOSE_AMBIGUITY_ENABLED and _top_family_count(metadatas) <= AMBIGUITY_FAMILY_COUNT_THRESHOLD:
        response_text += _ambiguity_disclosure(metadatas)

    sources = sorted({m.get("source_url") for m in metadatas if m.get("source_url")})
    return response_text, sources, retrieval_query, ranked_top_urls
