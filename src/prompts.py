"""The answer prompt, and the rules layered onto it.

Split out of rag.py 2026-08-13. Pure move - no behaviour change.

Every rule here was measured before it shipped, and two went the wrong way:
INLINE_CITATIONS cost 11 points of groundedness, QUOTE_FIGURES_VERBATIM was
rejected. Both stay in the code, off, with their falsifications - that is the
convention that stops them being re-proposed.

_scrub_plumbing is the deterministic backstop to USER_FACING_LANGUAGE, which as
a prompt rule alone only got leaks from 4/4 to 2/4. It rewrites retrieval
vocabulary AFTER generation, so unlike a prompt rule it cannot cost
groundedness - and it took three attempts, because substituting a plural phrase
forced verb agreement and produced worse sentences than the leak.
"""

import os
import re

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

# Multi-entity coverage (real user feedback, 2026-08-07). ~5 of 17 thumbs-down
# were questions naming SEVERAL entities that got answered for one: "what are
# the accredited programmes offered by CSEE, MSAS, Psychology, HSC, SRES and
# Life Sciences" came back about CSEE only, and the follow-up complaint ("I've
# asked information on 6 schools, which I've explicitly listed, but I'm still
# only getting info on one") got CSEE again.
#
# There are two separable failures and this rule only targets the second:
#   (a) COVERAGE - N_RESULTS=6 is a hard ceiling, so a six-school question
#       cannot retrieve enough chunks to answer six schools. Not fixable by
#       prompting; needs per-entity retrieval or a widened k for this shape.
#   (b) HONESTY - the answer silently presents one entity's information as
#       though it were the whole answer, giving no signal that five are
#       missing. That is a prompt-addressable failure and it is what the
#       user actually complained about.
#
# ON since 2026-08-08, on targeted evidence in both directions - the project's
# one prior base-prompt rule (INLINE_CITATIONS, above) regressed groundedness by
# 11 points, so this was not enabled until measured.
#
# BENEFIT, on the real failing question ("accredited programmes offered by CSEE,
# MSAS, Psychology, HSC, SRES, and Life Sciences"):
#   OFF -> "I am sorry, but the provided context does not list the accredited
#          programs..." and then gives NOTHING, withholding even the CSEE data
#          it had retrieved.
#   ON  -> names the five it lacks explicitly AND lists the actual CSEE
#          programmes. Strictly more useful and strictly more honest.
# COLLATERAL: 6 ordinary single-entity questions, cloud-judged, mean 5.00 with
# the rule vs 5.00 without - no regression.
#
# VALIDATED 2026-08-09 on the full 160 turns (both sets, ON vs OFF, local
# generator and judge): pooled mean 3.80 ON vs 3.77 OFF; primary hit@6 and
# useful-answer rate IDENTICAL (80.0% / 76.2%); every difference within noise.
# So the rule causes no collateral damage - which was the actual risk, and
# replaces the earlier 6-question sample that was saturated at the judge's
# ceiling and could only ever show "no obvious harm".
#
# It still does not show the rule HELPS: neither question set contains a
# multi-entity question, the same structural gap that makes the partner-
# institution fix unmeasurable. It stays on for the real-user evidence
# (the CSEE/MSAS/Psychology complaint), with that limit recorded not assumed.
# Revert with = False.
# Enumeration fidelity (2026-08-28). Retrieval was fixed first: milestone
# documents are now completed into the context, so all 16 codes (M1.1 ... M3.3)
# of ce-phd-2025-26.pdf are present on EVERY run. The generator still dropped
# one on 3 of 6 runs, always a trailing item of a group - M2.7 is the last row
# of its chunk, in a flattened table, immediately before a new section header.
#
# Detail level is NOT the lever: concise scored 15/16, 16, 16 and detailed
# 15, 15, 16, so lengthening the answer did not help. Cloud generation cannot
# be temperature-pinned, so this is measured over repeats, never one run.
# FALSIFIED 2026-08-28, kept OFF with the measurement rather than deleted.
# Paired arms, 4 runs each, same session and same context:
#   rule OFF: 3/4 runs complete, worst run 15/16 (dropped M2.7 only)
#   rule ON : 2/4 runs complete, worst run 11/16 (dropped M1.3-M1.6 AND M3.3)
# Telling the model not to omit items made it omit MORE, and in blocks -
# the same direction INLINE_CITATIONS went (-11 points of groundedness).
# Two base-prompt rules have now helped (MULTI_ENTITY) and two have hurt.
ENUMERATION_COMPLETENESS = os.environ.get("RAG_ENUMERATION_RULE", "0") == "1"
_ENUMERATION_RULE = (
    "\n- When the documents present an itemised or coded list (milestone codes like M2.7, "
    "numbered clauses, lettered sub-paragraphs) and the question asks what the list contains, "
    "reproduce EVERY item, including the last one in each group. Do not summarise a list, "
    "abbreviate it with \"etc.\", or stop at a representative sample - a list that silently "
    "omits an item reads as complete and is wrong."
)

MULTI_ENTITY_COVERAGE = True
_MULTI_ENTITY_RULE = (
    "\n- If the question names several specific things (multiple programmes, departments, "
    "schools, or years), address each one by name. For any of them the context does not cover, "
    "say so explicitly - e.g. \"the context has nothing on X or Y\" - rather than answering only "
    "for the ones you found and leaving the rest unmentioned."
)

# User-facing language (2026-08-10). The retrieval plumbing is an
# implementation detail the user never sees, but the answers describe it: real
# production answers said "the context you've provided across both turns" and
# offered "if you have excerpts from their rules of assessment, please share
# them and I can identify the programmes". The user supplied neither - the
# retriever did - so this reads as either a mistake or a request they cannot
# act on. Cost is confusion and lost trust, not a wrong fact, which is why no
# accuracy metric would ever have caught it.
#
# Scoped to PHRASING only: it must not change WHETHER the model declines, just
# how it says so. That distinction is the risk - abstention is measured (set 3,
# 6/6 correct under cloud) and a rule about how to phrase "I don't have this"
# sits directly on top of it. Flag-gated and measured before enabling, per the
# convention that cost -8.8 points when ignored.
# ENABLED 2026-08-10 after measurement (eval/report.md Round 8f). Like-for-like
# A/B on the same 4 turns: plumbing-leaking answers 4/4 -> 2/4, abstention
# unchanged at 4/4 correct. Residual is "the excerpts I can see" in secondary
# sentences - the opening framing is fixed in all 4, the model does not fully
# comply with the explicit "never say excerpts" instruction.
USER_FACING_LANGUAGE = os.environ.get("RAG_USER_FACING_LANGUAGE", "1") == "1"

# Deterministic backstop for USER_FACING_LANGUAGE. The prompt rule cut
# plumbing-leaking answers from 4/4 to 2/4, not to zero - "the excerpts I can
# see" survives an explicit instruction not to say it. A prompt asks; a
# substitution guarantees. Applied AFTER generation, so it cannot cost
# groundedness the way INLINE_CITATIONS did (-11 points).
#
# Deliberately narrow: only phrases whose replacement is unambiguous. It does
# NOT try to rewrite sentences that ask the user to paste documents - that
# needs meaning, not string replacement, and a bad rewrite is worse than the
# leak. Those stay the prompt rule's job, and the residual stays measured.
SCRUB_PLUMBING_LANGUAGE = os.environ.get("RAG_SCRUB_PLUMBING", "1") == "1"

# Replacements use a SINGULAR subject ("the guidance I can see") on purpose.
# An earlier version substituted the plural "the policies I can see", which
# forced verb agreement fixes - "the context does not cover" became "the
# policies I can see does not cover" - and patching that with a verb-rewriting
# table produced sentences worse than the leak it replaced ("the policies I can
# see access to does not cover this"). A singular phrase needs no agreement
# work at all, so the whole class of bug disappears.
_PLUMBING_SUBS = [
    # whole-phrase forms first, so nothing is left stranded
    (r"\bthe (?:provided |supplied )?context (?:that )?(?:I have |I was )?"
     r"(?:access to|available to me|given|provided to me|I can see)\b",
     "the guidance I can see"),
    # PLURAL source -> PLURAL replacement, so agreement survives in both
    # directions. "the excerpts ... do not mention" must not become "the
    # guidance ... do not mention".
    (r"\b(?:the |these |those )?(?:provided |supplied )?excerpts (?:I can see|provided|"
     r"you(?:'ve| have) provided)\b", "the policies I can see"),
    (r"\bthe (?:provided |supplied )excerpts\b", "the policies I can see"),
    (r"\bthe documents (?:you(?:'ve| have) provided|provided)\b", "the policies I can see"),
    # singular "excerpt" keeps the singular replacement
    (r"\bthe (?:provided |supplied )?excerpt (?:I can see|provided)\b",
     "the guidance I can see"),
    (r"\baccording to the (?:provided |supplied )?context\b(?! of\b)",
     "according to the guidance I can see"),
    (r"\bbased on the (?:provided |supplied )?context\b(?! of\b)",
     "based on the guidance I can see"),
    # "in the context of X" is the commonest ordinary-English form of all -
    # same guard as the catch-all below, for the same reason.
    (r"\bin the (?:provided |supplied )?context\b(?! of\b)", "in the guidance I can see"),
    # catch-all last
    # NOT followed by "of": "the context of the field" is ordinary English and
    # belongs to the POLICY being quoted, not to our retrieval plumbing.
    # Without this guard the scrub rewrote a milestone the documents actually
    # define - "understanding of chosen topic within the context of the field"
    # was served to users as "within the guidance I can see of the field".
    # 504 of the 511 occurrences of "the context" in the corpus are "the
    # context of", across 389 documents, so the unguarded rule was wrong far
    # more often than it was right whenever an answer quoted its source.
    (r"\bthe (?:provided |supplied )?context\b(?! of\b)", "the guidance I can see"),
]
_PLUMBING_RES = [(re.compile(pat, re.I), rep) for pat, rep in _PLUMBING_SUBS]


def _scrub_plumbing(text: str) -> str:
    """Replace retrieval-plumbing vocabulary with user-facing wording."""
    if not SCRUB_PLUMBING_LANGUAGE or not text:
        return text

    def _sub(m):
        rep = m.expand(_current_rep[0])
        # Replacements are written lowercase, but a match can start a sentence.
        # Without this, "The context does not contain X" became "the policies I
        # can see don't cover X" - a mid-sentence lowercase word opening a line.
        return rep[:1].upper() + rep[1:] if m.group(0)[:1].isupper() else rep

    for rx, rep in _PLUMBING_RES:
        _current_rep[0] = rep
        text = rx.sub(_sub, text)

    return text


_current_rep = [""]   # bound per substitution for _sub above


_USER_FACING_RULE = (
    "\n- Write for a reader who cannot see the retrieval machinery. Never mention \"the context\", "
    "\"excerpts\", \"the provided excerpts\", \"the documents provided\", or what the user did or "
    "did not supply - "
    "they did not supply anything, the system retrieved it. Say \"the policies I can see don't "
    "cover X\" or \"I don't have a document covering X\" instead of \"the context does not contain "
    "X\". Never ask the user to paste, share or provide documents or excerpts; if something is "
    "missing, say what is missing and, where useful, name the document or team likely to hold it."
)

# Answer detail level (2026-08-11). A per-request preference, not a global
# flag: the user picks it in Settings and it travels with the message. The
# DEFAULT is unchanged behaviour, so a user who never touches the control gets
# exactly what production gives today.
#
# CONCISE is the risky one and is why this is measured rather than assumed. The
# project's two prior base-prompt rules went in opposite directions -
# INLINE_CITATIONS cost 11 points of groundedness, MULTI_ENTITY_COVERAGE helped
# - and a brevity instruction plausibly damages exactly the enumeration
# questions the user already complained about ("I gave you 6 schools and you
# answered about one"). The rule below therefore protects enumeration and
# source-citing explicitly rather than just asking for brevity.
DETAIL_LEVELS = ("default", "concise", "detailed")

_CONCISE_RULE = (
    "\n- Answer briefly: lead with the rule itself and stop. Omit background, "
    "restatement of the question, and caveats the user did not ask for. Do NOT "
    "drop any of the following to save space: an entity the question named, an "
    "item of a list the document gives, or the source citation - brevity must "
    "never turn a complete answer into a partial one."
)
_DETAILED_RULE = (
    "\n- Give the rule and then the context around it: which programmes or "
    "cases it applies to, any exceptions or thresholds stated in the document, "
    "and anything adjacent the reader would otherwise have to ask next. Stay "
    "within what the retrieved documents say."
)


# What "default" MEANS, changed 2026-08-11 on the user's instruction. Generation
# is the dominant latency term and output length is the generation cost - a real
# answer ran 996 output tokens in ~10s (Round 19) - so concise is worth ~2-3s on
# every turn. It was measured to lose no named entities and no list items
# (Round 10), and the rule explicitly forbids dropping either.
#
# "detailed" still exists and still lengthens; a user who wants the old
# behaviour picks it in Settings.
DEFAULT_DETAIL = os.environ.get("RAG_DEFAULT_DETAIL", "concise")


def system_prompt_for(detail: str = "default") -> str:
    """SYSTEM_PROMPT with a detail-level rule appended. Unknown values fall back
    to the default rather than raising - a bad preference must not cost an
    answer. Note "default" is a POINTER to DEFAULT_DETAIL, not a third style."""
    if detail not in ("concise", "detailed"):
        detail = DEFAULT_DETAIL      # covers "default" AND any unknown value
    if detail == "concise":
        return SYSTEM_PROMPT + _CONCISE_RULE + "\n"
    if detail == "detailed":
        return SYSTEM_PROMPT + _DETAILED_RULE + "\n"
    return SYSTEM_PROMPT


SYSTEM_PROMPT = (
    _SYSTEM_PROMPT_BASE
    + (_VERBATIM_RULE if QUOTE_FIGURES_VERBATIM else "")
    + (_INLINE_CITATION_RULE if INLINE_CITATIONS else "")
    + (_MULTI_ENTITY_RULE if MULTI_ENTITY_COVERAGE else "")
    + (_ENUMERATION_RULE if ENUMERATION_COMPLETENESS else "")
    + (_USER_FACING_RULE if USER_FACING_LANGUAGE else "")
    + "\n"
)

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
