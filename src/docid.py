"""Canonical document identity: one shared definition of "which document
family is this" and "which academic year is this", used by ingestion
(run_ingest.py, reembed.py), retrieval (src/rag.py), and evaluation
(eval/score_summary.py). These previously lived in separate modules with
raw-string comparisons drifting between them - every year/family decision
must go through here so the definitions can't diverge again."""

import re

_FAMILY_YEAR_SUFFIX_RE = re.compile(r"[-_]?(20)?\d{2}(-\d{2,4})?\.pdf$")
_YEAR_START_RE = re.compile(r"20(\d{2})")


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


def previous_year(normalized: str) -> str:
    """'2026-27' -> '2025-26'. Empty input stays empty."""
    if not normalized:
        return ""
    start = int(normalized[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"
