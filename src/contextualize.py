"""Follow-up query contextualisation: turning "what about the second year?"
into a standalone question before it reaches retrieval.

Split out of rag.py 2026-08-13. Pure move - no behaviour change.

This is the only LLM call left in the retrieval path, and the one place a model
choice touches WHICH documents come back rather than how they are described -
which is why it is pinned separately from the answer generator and measured on
follow-up hit@6 alone.

Contains the faithfulness guard that cost -8.8 points when it shipped enabled
on two hand-checked cases (_has_extraneous_family, now off), and the identity
anchor index that a later round found the startup warmup was failing to warm.
"""

import json
import re

from src.docid import document_family as _document_family
from src.llm import CONTEXTUALIZE_MODEL, contextualize_chat

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


_REFERENTIAL_WORDS = {
    "about", "above", "again", "already", "asked", "asking", "back", "before", "discussed",
    "earlier", "first", "going", "initial", "initially", "just", "mentioned", "now", "originally",
    "point", "previous", "previously", "question", "raised", "said", "talked", "talking", "thing",
    "things", "told", "very",
}


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def _log_rewrite_reject(original: str, rewritten: str) -> None:
    """Best-effort append of a guard-discarded (topic-drifted) rewrite to a
    gitignored log, so these low-confidence rewrites can be reviewed alongside
    user feedback. Swallows every error - logging must never break retrieval."""
    try:
        import json as _json
        from datetime import datetime, timezone
        from pathlib import Path as _Path
        p = _Path("data/contextualizer_rejects.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(_json.dumps({"ts": datetime.now(timezone.utc).isoformat(),
                                 "original": original, "rewritten": rewritten}, ensure_ascii=False) + "\n")
    except Exception:
        pass


# Award-type category words (real feedback bug, 2026-07-27: "duration of
# phd" after a Professional Doctorate question got answered from
# Professional Doctorate documents). These name a genuinely different set of
# regulations from each other, but the identity-anchor index can't tell them
# apart: "phd" and "doctorate" appear across so many different department
# records that their docfreq exceeds ANCHOR_DOCFREQ and they're filtered out
# as too generic to anchor ON - but that same genericness means a question
# that names one is already self-sufficient and must NOT be treated as
# identity-less (which is what let the stale Professional Doctorate anchor
# get re-appended). Only used to broaden the "does this already name its own
# topic" check below, never as something to anchor OTHER questions to.
_AWARD_TYPE_TERMS = {"phd", "mphil"}
_AWARD_TYPE_PHRASES = ("professional doctorate", "doctor of philosophy")


def _names_award_type(text: str) -> bool:
    low = text.lower()
    if _content_words(text) & _AWARD_TYPE_TERMS:
        return True
    return any(p in low for p in _AWARD_TYPE_PHRASES)


def _family_labels_named(text: str) -> set[str]:
    """Distinct programme-family labels whose distinctive identity tokens
    appear in text (possibly several, if multiple programmes are named)."""
    distinctive, families = _identity_anchor_index()
    hit = _content_words(text) & distinctive
    if not hit:
        return set()
    return {lab for lab, toks in families if lab and (toks & hit)}


EXTRANEOUS_FAMILY_GUARD = False  # see _has_extraneous_family: caused a -8.8pt follow-up regression


def _has_extraneous_family(original: str, rewritten: str, history: list[dict]) -> bool:
    """True if the rewrite introduced identity tokens for a programme family
    the original question never named and that isn't the single active
    history anchor - the failure mode where the small rewriter model
    free-associates from a dense transcript and bolts several prior-turn
    programme names onto an unrelated new question (real example: "what is
    the minimum and maximum duration for a phd?" came back rewritten with
    five unrelated MSc programme names appended from two turns earlier).

    DISABLED 2026-08-08 - it did more harm than the bug it fixed, and the
    design is unsound. _family_labels_named counts document FAMILIES, but one
    programme spans many (a "three-year Honours Degree" mention matches
    roa-ug-3yr-year-1/-2/-3 and their variations), so naming ONE programme is
    indistinguishable from naming several. Raising the threshold to >=2
    recovered only 3 of 33 wrongly-rejected rewrites, confirming the counting -
    not the threshold - is wrong.

    Cost/benefit is decisive: it caught one rare hallucination (a rewrite that
    appended five unrelated MSc names) while rejecting 33 correct rewrites in a
    day against 1 in all prior history, each falling back to the raw
    context-free question - follow-up hit@6 75.0% -> 66.2% (-8.8pts) with
    primary turns unaffected (+1.2). The other reported topic-switch bug
    ("duration of phd" answered about Professional Doctorates) is fixed
    independently by _names_award_type, which is verified and stays on.

    Re-enable only with a counting scheme that resolves families to programmes
    first, and validate on FOLLOW-UP hit@6 - the 30-turn multi-turn probe is
    topic-switch-heavy and did not catch this."""
    if not EXTRANEOUS_FAMILY_GUARD:
        return False
    new_families = _family_labels_named(rewritten) - _family_labels_named(original)
    if not new_families:
        return False
    anchor_label, _ = _anchor_from_history(history)
    new_families.discard(anchor_label)
    return len(new_families) >= 2


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

    rewritten = contextualize_chat(messages=[
        {"role": "system", "content": CONTEXTUALIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"{transcript}\n\nFollow-up question: {question}\n\nStandalone question:"},
    ], model=CONTEXTUALIZE_MODEL).strip()

    faithful = bool(
        rewritten
        and _is_faithful_rewrite(question, rewritten)
        and not _has_extraneous_family(question, rewritten, history)
    )
    if rewritten and not faithful:
        # The guard discarded a topic-drifted rewrite (the postfix3->postfix4 bug
        # class). Log it best-effort so these low-confidence rewrites can be
        # reviewed alongside user feedback (round-6 review, Grok). Never fatal.
        _log_rewrite_reject(question, rewritten)
    result = rewritten if faithful else question

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
            # a switch names its OWN new topic - either a specific programme
            # (distinctive) or an award-type category too generic to be
            # "distinctive" itself but still self-sufficient (_names_award_type)
            names_a_topic = bool(result_tokens & distinctive) or _names_award_type(result)
            if not already_anchored and not names_a_topic:
                # identity-less continuation that dropped the anchor - re-append
                result = f"{result} ({label})"

    return result
