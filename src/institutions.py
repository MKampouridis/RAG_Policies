"""Institution-aware filtering of retrieval results.

Split out of rag.py 2026-08-13. Pure move - no behaviour change.

Everything here answers one question: when a query does not name a partner
college, what should happen to partner-edition documents? The measured answer
is "remove them" (4/17 real complaints -> 0/17), with the cost recorded twice
in eval/report.md - a question naming a partner PROGRAMME but not its
institution loses the document that answers it. The alternatives (cap at one
slot, soft boost) are implemented here too, both measured worse, both off.

The scope switch a user sees on the landing page is the strict pair -
essex_only / partner_only - which bypass the name-detection heuristic entirely,
because a control someone operates deliberately should do what it says.
"""

import os
import re

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


# Real production feedback (2026-08-07, eval/FEEDBACK_FINDINGS.md) surfaced
# three cases where a partner-institution document (Kaplan, Tavistock)
# outranked Essex's own document for a query that never named the partner -
# e.g. "exit awards for MSc Artificial Intelligence" put Kaplan's
# kol-pg-masters-roa-25.pdf at rank 1-2, ahead of all four CSEE (home)
# documents already present in the same top-6 pool. _prefer_home_institution
# above didn't fire: it requires J1 alias overlap to detect "same programme",
# but the Kaplan document's own identity extraction is entirely empty (no
# programme_name/aliases at all) - too fragile to rely on here. This is a
# simpler, unconditional post-rerank demotion: when the final top-k mixes
# partner and non-partner documents, partner documents sink below all
# non-partner ones (stable order otherwise). It never REMOVES a partner
# document, so a query where nothing non-partner is relevant still surfaces
# it unchanged - same simplifying assumption as _prefer_home_institution
# (doesn't try to detect the query explicitly naming the partner; if that
# matters in practice, a future eval/feedback pass will show it as a loss).
PARTNER_INSTITUTION_DEMOTE_ENABLED = True


# Partner EXCLUSION vs demotion (2026-08-10). _demote_partner_institutions
# (below) re-sorts WITHIN the already-chosen top 6, so it never frees a slot -
# which is why partner documents still reach the user on 4 of 17 real
# thumbs-down complaints ("Answers need to focus on Essex programmes, not
# partners", twice). Excluding them from the CANDIDATE POOL frees the slot for
# an Essex document.
#
# Gated on the question not naming a partner, because 9 of 120 eval questions
# have a partner edition as their gold. Measured: only 4 of those 9 name their
# partner, so this gate does lose the labelled gold on 5. Inspect those 5
# before reading that as damage - they are generic questions ("what are the
# general principles for reassessment?", "what is the requirement for passing
# Year One?") whose gold is a partner edition by test-set assignment rather
# than because a partner document is what the user needs. That is the
# gold-multiplicity problem eval/gold_multiplicity.py already documents.
# Strict hit@6 therefore UNDERSTATES this change by construction.
# ENABLED 2026-08-10 after measurement (eval/report.md Round 8i).
#   Real complaints with a partner doc in the top 6: 4/17 -> 0/17
#   Partner-held slots on those questions:            5.5% -> 0.0%
#   hit@6 on 160 turns (history-aware replay):        123 -> 124, 0 lost
PARTNER_EXCLUDE_WHEN_UNNAMED = os.environ.get("RAG_PARTNER_EXCLUDE", "1") == "1"

_PARTNER_NAME_TOKENS = (
    "tavistock", "portman", "aegean", "omiros", "laksamana", "skku", "kaplan",
    "kol", "south essex", "colchester institute", "portobello", "chula",
    "writtle", "edge hotel", "east 15", "east15", "sak", "eput", "northwest",
    "north west", "alexandria", "partner institution", "partner-institution",
)

# Substring matching on these tokens fired on ordinary governance vocabulary:
# "eput" is inside *deputy* and *reputation*, "sak" inside *for the sake of*.
# Each false positive silently switched PARTNER_EXCLUDE_WHEN_UNNAMED OFF, and
# because the gate reads the whole conversation, one "deputy" disabled
# exclusion for the rest of it. In a corpus of university governance documents
# "deputy" is not a rare word. Word-boundary matching, compiled once at import.
# Multi-word tokens keep internal spaces; \b works at both ends regardless.
_PARTNER_NAME_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _PARTNER_NAME_TOKENS) + r")\b"
)


def _names_partner_institution(query: str, history: list[dict] | None = None) -> bool:
    """Checks the CONVERSATION, not just this turn. Measured: the Tavistock
    follow-up ("what actions should new Professional Doctorate students take
    after receiving...") drops the institution name, because the
    contextualizer rewrites for topic continuity rather than for this gate.
    Reading the current query alone excluded partner documents on a turn whose
    own conversation was explicitly about a partner - a lost hit@6 caused
    entirely by the gate, not by the exclusion being wrong."""
    haystacks = [query]
    for m in history or []:
        if m.get("role") == "user" and m.get("content"):
            haystacks.append(m["content"])
    low = " ".join(haystacks).lower()
    return _PARTNER_NAME_RE.search(low) is not None


def _exclude_partner_institutions(results: dict) -> dict:
    """Drop partner-edition chunks entirely. Unlike the demotion this FREES
    slots. Returns the input unchanged if everything would be dropped - an
    empty context is worse than a partner-sourced answer."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    keep = [i for i, m in enumerate(metadatas) if not _is_partner_institution(m)]
    if not keep or len(keep) == len(documents):
        return results
    out = {"documents": [[documents[i] for i in keep]],
           "metadatas": [[metadatas[i] for i in keep]]}
    dists = results.get("distances")
    if dists:
        out["distances"] = [[dists[0][i] for i in keep]]
    return out


# How partner-edition documents are handled when the query names no partner.
# "exclude" (default) drops them entirely - the behaviour the user's complaint
# asked for, measured at 4/17 real complaints -> 0/17, and measured TWICE to
# cost the inverse case: a question naming a partner PROGRAMME but not its
# institution loses the document that answers it (set 6, Rounds 8r and 32).
#
#   exclude  - drop all partner chunks (current)
#   cap1     - demote them below every Essex document AND keep at most one, so
#              a partner answer stays reachable at the bottom
#   boost    - no gate at all; Essex documents move up by PARTNER_BOOST places
#
# cap1 and boost are the review's softening proposals. Both change retrieval,
# so both are off until measured.
PARTNER_MODE = os.environ.get("RAG_PARTNER_MODE", "exclude")
PARTNER_BOOST = int(os.environ.get("RAG_PARTNER_BOOST", "3"))


def _cap_partner_institutions(results: dict, cap: int = 1) -> dict:
    """Demote partner chunks below all Essex ones and keep at most `cap`.

    The difference from exclusion is one slot: an Essex question still gets
    Essex documents in every position that matters, while a question whose only
    correct answer is a partner document can still find it at the bottom."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[None] * len(documents)])[0]
    if not documents:
        return results
    essex = [i for i, m in enumerate(metadatas) if not _is_partner_institution(m)]
    partner = [i for i, m in enumerate(metadatas) if _is_partner_institution(m)]
    if not partner:
        return results
    order = essex + partner[:cap]
    return {"documents": [[documents[k] for k in order]],
            "metadatas": [[metadatas[k] for k in order]],
            "distances": [[distances[k] for k in order]]}


def _boost_home_institution(results: dict, places: int = 3) -> dict:
    """Soft preference: move Essex documents up by `places` rather than gating
    partners at all. Cannot guarantee Essex dominance - which is what the
    original complaint asked for - so it trades certainty for recall."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[None] * len(documents)])[0]
    if len(documents) < 2:
        return results
    keyed = sorted(range(len(documents)),
                   key=lambda i: i + (places if _is_partner_institution(metadatas[i]) else 0))
    if keyed == list(range(len(documents))):
        return results
    return {"documents": [[documents[k] for k in keyed]],
            "metadatas": [[metadatas[k] for k in keyed]],
            "distances": [[distances[k] for k in keyed]]}


def _filter_by_institution(results: dict, partner: bool) -> dict:
    """Hard filter for the user-facing scope switch. No fallback: if nothing
    matches, the context is empty and the assistant says so."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    keep = [i for i, m in enumerate(metadatas)
            if _is_partner_institution(m) == partner]
    out = {"documents": [[documents[i] for i in keep]],
           "metadatas": [[metadatas[i] for i in keep]]}
    dists = results.get("distances")
    if dists:
        out["distances"] = [[dists[0][i] for i in keep]]
    return out


def _only_partner_institutions(results: dict) -> dict:
    """Keep ONLY partner-edition chunks. The mirror of
    _exclude_partner_institutions, and it takes the same safeguard: if that
    would empty the context, return the input unchanged. An empty context
    produces a confident "I have no document on this", which is worse than an
    Essex answer to someone who asked for a partner one."""
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    keep = [i for i, m in enumerate(metadatas) if _is_partner_institution(m)]
    if not keep or len(keep) == len(documents):
        return results
    out = {"documents": [[documents[i] for i in keep]],
           "metadatas": [[metadatas[i] for i in keep]]}
    dists = results.get("distances")
    if dists:
        out["distances"] = [[dists[0][i] for i in keep]]
    return out


def _demote_partner_institutions(results: dict) -> dict:
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[None] * len(documents)])[0]
    if len(documents) < 2:
        return results

    order = sorted(range(len(documents)), key=lambda i: _is_partner_institution(metadatas[i]))
    if order == list(range(len(documents))):
        return results

    return {
        "documents": [[documents[k] for k in order]],
        "metadatas": [[metadatas[k] for k in order]],
        "distances": [[distances[k] for k in order]],
    }
