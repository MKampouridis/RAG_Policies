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
    current year's in the top-k."""
    filename = source_url.rsplit("/", 1)[-1]
    filename = _FAMILY_YEAR_SUFFIX_RE.sub(".pdf", filename)
    return filename.lower()


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
