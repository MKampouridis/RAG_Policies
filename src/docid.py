"""Canonical document identity: one shared definition of "which document
family is this" and "which academic year is this", used by ingestion
(run_ingest.py, reembed.py), retrieval (src/rag.py), and evaluation
(eval/score_summary.py). These previously lived in separate modules with
raw-string comparisons drifting between them - every year/family decision
must go through here so the definitions can't diverge again."""

import re

_FAMILY_YEAR_SUFFIX_RE = re.compile(r"[-_]?(20)?\d{2}(-\d{2,4})?\.pdf$")
_YEAR_START_RE = re.compile(r"20(\d{2})")
_YEAR_DIR_RE = re.compile(r"/(20\d{2}-\d{2,4})/")

# Rename-split alias map (external code review round 3, 2026-07-22, Fable 5,
# independently verified). Essex changes a document's FILENAME FORMAT between
# academic years - separator flips ("sres-modular-25" vs "sres_modular_24"),
# added version suffixes ("msc-ot_24-v1"), extra hyphens ("tavistock-roa---
# pg-diploma-24"), reformatted years ("25-26" vs "24") - so document_family()'s
# suffix-stripping regex lands two editions of the SAME document in DIFFERENT
# families. Each then becomes its own family maximum and the superseded prior
# edition keeps is_current=True (via the one-year grace window) alongside its
# successor, polluting the current-only retrieval pool with stale editions
# (verified: east15_24, the 2024-25 Tavistock/KOL/CSEE editions, etc. all sat
# current next to their 2025-26 versions). A cleverer regex can't fix this
# (Essex's renames aren't systematic - the mandatory-separator variant tried
# earlier broke 45 correct bare-suffix groupings), so per Fable 5 this is an
# EXPLICIT, AUDITED map: superseded family key -> canonical (newest-edition)
# family key. Regenerate after any ingest with `python audit_family_aliases.py`
# and review before pasting - the audit lists only groups that share a
# structural-token-preserving normalized stem AND currently have >1
# simultaneously-current edition, so genuinely-distinct siblings (4yr-year-1
# vs 4yr-year-2) are never merged. Verified: applying this demotes exactly 22
# superseded editions to is_current=False, 0 spurious promotions.
_FAMILY_ALIASES = {
    "csee_ft_masters_-accredited_variations.pdf": "csee-ft-masters-accredited-variations.pdf",
    "csee_ft_masters_accredited_variations.pdf": "csee-ft-masters-accredited-variations.pdf",
    "csee_pt_masters_accredited_variations.pdf": "csee-pt-masters-accredited-variations.pdf",
    "east15.pdf": "east.pdf",
    "eput-roa---apprenticeship-route.pdf": "eput-roa-apprenticeship-route.pdf",
    "five_year_integrated_masters.pdf": "five-year-integrated-masters-21-v7.pdf",
    "five_year_integrated_masters_17-v4.pdf": "five-year-integrated-masters-21-v7.pdf",
    "five_year_integrated_masters_19-v5.pdf": "five-year-integrated-masters-21-v7.pdf",
    "five_year_integrated_masters_20-v7.pdf": "five-year-integrated-masters-21-v7.pdf",
    "four_year_integrated_masters.pdf": "four-year-integrated-masters-22-v4.pdf",
    "four_year_integrated_masters_17-v4.pdf": "four-year-integrated-masters-22-v4.pdf",
    "four_year_integrated_masters_20-v5.pdf": "four-year-integrated-masters-22-v4.pdf",
    "kol_pg-masters-roa.pdf": "kol-pg-masters-roa.pdf",
    "kol_pgcert-roa.pdf": "kol-pgcert-roa.pdf",
    "kol_pgdip-roa.pdf": "kol-pgdip-roa.pdf",
    "msc-periodontology-science-alexandria.pdf": "msc-periodontology-science-(alexandria).pdf",
    "ma-psychodynamic-counselling_psychotherapy-2-year.pdf": "ma-psychodynamic-counselling-psychotherapy-2-year.pdf",
    "ma-psychodynamic-counselling_psychotherapy-3year.pdf": "ma-psychodynamic-counselling-psychotherapy-3year.pdf",
    "ma-psychodynamic-counselling_psychotherapy-4year.pdf": "ma-psychodynamic-counselling-psychotherapy-4year.pdf",
    "modular_24-v1.pdf": "modular.pdf",
    "msc--physiotherapy.pdf": "msc-physiotherapy.pdf",
    "msc-ot_24-v1.pdf": "msc-ot.pdf",
    "msc_physiotherapy.pdf": "msc-physiotherapy.pdf",
    "pgt_credit_framework.pdf": "pgt-credit-framework.pdf",
    "roa-ug-glossary": "roa-ug-glossary.pdf",
    "speechtherapy_24-v1.pdf": "speechtherapy.pdf",
    "sres_modular.pdf": "sres-modular.pdf",
    "tavistock-roa---graduate-certificate.pdf": "tavistock-roa-graduate-certificate.pdf",
    "tavistock-roa---graduate-diploma.pdf": "tavistock-roa-graduate-diploma.pdf",
    "tavistock-roa---pg-certificate.pdf": "tavistock-roa-pg-certificate.pdf",
    "tavistock-roa---pg-diploma.pdf": "tavistock-roa-pg-diploma.pdf",
    "tavistock_taught_masters.pdf": "tavistock-taught-masters.pdf",
}

# Closed, small vocabularies (unlike the open-ended `department` field that
# failed on query-text coverage before) - deliberately deterministic
# regex/keyword matching, not an LLM pass, since the vocabulary is this
# tractable. Order matters: first match wins, most specific first.
_DEGREE_LENGTH_PATTERNS = [
    ("foundation", re.compile(r"\bfoundation\b", re.I)),
    # document filenames abbreviate this as "3yr"/"4yr"/"5yr" (no space before
    # "yr"), while questions tend to say "3-year"/"three-year" in full - both
    # forms need to match or the doc-side and query-side extractions silently
    # disagree (see eval/report.md, Stage A first attempt)
    ("5yr", re.compile(r"\b5[\s-]?year\b|\bfive[\s-]?year\b|\b5yr\b", re.I)),
    ("4yr", re.compile(r"\b4[\s-]?year\b|\bfour[\s-]?year\b|\b4yr\b", re.I)),
    ("3yr", re.compile(r"\b3[\s-]?year\b|\bthree[\s-]?year\b|\b3yr\b", re.I)),
]

_AWARD_TYPE_PATTERNS = [
    ("phd", re.compile(r"\bphd\b|\bdoctorate\b", re.I)),
    ("certificate", re.compile(r"\bcertificate\b|\b(?:grad(?:uate)?|pg)\s*cert\b", re.I)),
    ("diploma", re.compile(r"\bdiploma\b|\b(?:grad(?:uate)?|pg)\s*dip\b", re.I)),
    ("masters", re.compile(r"\bmasters?\b|\bmsc\b|\bm\.?a\.?(?=\s)", re.I)),
]


def document_family(source_url: str) -> str:
    """Groups yearly reissues of the same document together by filename with
    the year suffix stripped, e.g. "csee-ft-masters-accredited-variations-24.pdf"
    and "...-25.pdf" both map to "...-variations.pdf". Filename-based rather
    than path-based because "current" and "previous-years" archives use
    different folder structures but keep the same filename. Heuristic, not
    exact - good enough to stop an older year's chunk from crowding out the
    current year's in the top-k.

    The leading `[-_]?` in _FAMILY_YEAR_SUFFIX_RE is deliberately optional,
    not a bug left unfixed: external code review (2026-07-21) flagged that
    an optional separator lets a bare-digit identity token that isn't a year
    (e.g. "east15" in "east15-25.pdf", where "15" names the institution -
    East 15 Acting School - not an edition) get swallowed into the year
    match, and hypothesized this could merge two genuinely different
    document families that happen to share a base name plus bare digits.
    Tried making the separator mandatory and audited the corpus-wide effect
    before shipping (same methodology as the is_current fix above): it
    changed 45 documents' computed family, and the audit showed Essex's
    DOMINANT real filename convention is exactly a bare 2-digit year suffix
    with no separator ("ug-dip-he22.pdf", "variations22.pdf", "mlang20.pdf",
    etc, confirmed via manifest inspection to be genuine same-document
    yearly reissues, not distinct documents) - making the separator
    mandatory broke correct grouping for dozens of real families to guard
    against a hypothetical (an "east16" sibling) that doesn't currently
    exist in this corpus. Reverted; left optional. Real latent risk, but a
    proper fix needs to distinguish "bare year suffix" from "bare
    identity-bearing digits" (e.g. a small denylist of known non-year
    prefixes like "east15"), which isn't worth building for a currently
    zero-impact case - revisit only if a real cross-family collision like
    this is ever actually observed.

    Note the east15 collision the 2026-07-21 note called "zero-impact" turned
    out NOT to be: the two east15 editions land in different families
    ('east.pdf' vs 'east15.pdf') and BOTH stayed current - that's exactly the
    rename-split defect _FAMILY_ALIASES now corrects (2026-07-22). The alias
    lookup runs AFTER the regex, mapping a superseded edition's family key
    onto its canonical successor's."""
    filename = source_url.rsplit("/", 1)[-1]
    key = _FAMILY_YEAR_SUFFIX_RE.sub(".pdf", filename).lower()
    return _FAMILY_ALIASES.get(key, key)


def normalize_year(raw: str | None) -> str:
    """LLM-extracted academic_year values are messy ('2025-2026',
    '2025-26 onwards', '2025-26, 2024-25, ...'). Canonicalize to '2025-26'
    (first start-year mentioned) so string comparisons are meaningful.
    Returns '' when no year is present."""
    m = _YEAR_START_RE.search(raw or "")
    if not m:
        return ""
    start = 2000 + int(m.group(1))
    return f"{start}-{str(start + 1)[-2:]}"


def effective_year(source_url: str, raw_academic_year: str | None) -> str:
    """normalize_year(), capped (never raised) at the document's own URL
    year-folder for rules-of-assessment/pgt documents. Found via the Idea 2
    investigation (eval/report.md): PGT "January starts" documents describe
    the academic year the cohort finishes in, not the edition/publish year,
    so content-extracted academic_year can overstate a superseded edition's
    year enough to tie with the true current edition (e.g. a document filed
    under .../pgt/2024-25/... but whose extracted academic_year reads
    "2025-26") - silently marking a stale document is_current alongside the
    real one. Only ever lowers the year, matching compute_current_flags's
    existing convention that overrides only ever force False, never True -
    the safe direction, since correcting the opposite mismatch (content
    understating a doc's year relative to its folder) is not clearly a bug
    and a first attempt at correcting both directions introduced a NEW
    false-tie elsewhere (part-time-taught-masters family) that this
    one-directional version doesn't."""
    content_year = normalize_year(raw_academic_year)
    if "/rules-of-assessment/pgt/" in source_url:
        m = _YEAR_DIR_RE.search(source_url)
        if m:
            dir_year = normalize_year(m.group(1))
            if dir_year and content_year:
                return min(content_year, dir_year)
    return content_year


def previous_year(normalized: str) -> str:
    """'2026-27' -> '2025-26'. Empty input stays empty."""
    if not normalized:
        return ""
    start = int(normalized[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def extract_degree_length(text: str) -> str:
    """Closed-vocabulary degree-length facet ('3yr'/'4yr'/'5yr'/'foundation'),
    used both at ingest time (against the document title/header) and at
    retrieval time (against the contextualized query) so a hard `where`
    filter can narrow the search space the way "When More Documents Hurt
    RAG" (arXiv 2606.11350) recommends for near-identical sibling documents
    that differ mainly in programme length. Returns '' when no facet is
    mentioned - callers must treat that as "unknown", not "none of these"."""
    for label, pattern in _DEGREE_LENGTH_PATTERNS:
        if pattern.search(text or ""):
            return label
    return ""


def extract_award_type(text: str) -> str:
    """Closed-vocabulary award-type facet ('certificate'/'diploma'/'masters'/
    'phd'). Same rationale and same '' = unknown convention as
    extract_degree_length."""
    for label, pattern in _AWARD_TYPE_PATTERNS:
        if pattern.search(text or ""):
            return label
    return ""
