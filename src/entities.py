"""Department/school entity detection for multi-entity questions (2026-08-10).

WHY THIS EXISTS
Real user feedback: "what are the accredited programmes offered by CSEE, MSAS,
Psychology, HSC, SRES and Life Sciences" returned six chunks from three CSEE
documents - one of the six departments. `N_RESULTS = 6` caps CHUNKS, not
documents, so a six-department question cannot physically retrieve six
departments' worth of evidence. `MULTI_ENTITY_COVERAGE` (src/rag.py) makes the
answer honest about the gap; only per-entity retrieval can close it.

WHY A CURATED MAP AND NOT A MODEL
The corpus carries a `department` metadata field with 62 distinct values, but
they are inconsistent - case variants ("government"/"Government"), singular vs
plural ("...Exercise Science"/"...Exercise Sciences"), and parenthesised
abbreviations ("Health and Human Sciences (HSC)"). Users type the abbreviation.
Mapping abbreviation -> the set of metadata values is a lookup, not an
inference, so it is written down and auditable rather than delegated to a
model. This is the same reasoning as `_FAMILY_ALIASES` in src/docid.py.

The adjacent `MULTIHOP_DECOMPOSITION_ENABLED` was rejected at -7.5pts RoA
hit@6. It asked a model to HYPOTHESISE which documents a vague question might
mean. This does not: the entities are named explicitly in the question, so
there is nothing to guess, and the trigger requires two or more of them.
"""

# alias (lowercase, matched on word boundaries) -> department metadata values.
# An empty list means "no department metadata carries this entity" - detection
# still fires so the question can be handled, but retrieval must fall back to
# a query-side hint rather than a metadata filter. Life Sciences is the live
# example: it appears in document TEXT (11 current documents) but never as a
# department value, so filtering on it would silently return nothing.
DEPARTMENT_ALIASES: dict[str, list[str]] = {
    "csee": ["CSEE"],
    "computer science and electronic engineering": ["CSEE"],
    "msas": [
        "School of Mathematics, Statistics and Actuarial Science",
        "Department of Mathematical Sciences",
        "Mathematical Sciences",
    ],
    "mathematical sciences": [
        "Department of Mathematical Sciences", "Mathematical Sciences",
        "School of Mathematics, Statistics and Actuarial Science",
    ],
    "hsc": [
        "Health and Social Care", "HEALTH AND SOCIAL CARE",
        "School of Health and Social Care", "Health and Human Sciences (HSC)",
    ],
    "health and social care": [
        "Health and Social Care", "HEALTH AND SOCIAL CARE",
        "School of Health and Social Care",
    ],
    "hhs": ["Health and Human Sciences", "Health and Human Sciences (HHS)"],
    "health and human sciences": [
        "Health and Human Sciences", "Health and Human Sciences (HHS)",
    ],
    "sres": [
        "School of Sport, Rehabilitation and Exercise Science",
        "School of Sport, Rehabilitation and Exercise Sciences",
    ],
    "sport, rehabilitation and exercise": [
        "School of Sport, Rehabilitation and Exercise Science",
        "School of Sport, Rehabilitation and Exercise Sciences",
    ],
    "psychology": ["Psychology", "Psychology (C800 PJ)"],
    "psychosocial and psychoanalytic studies": [
        "Psychosocial and Psychoanalytic Studies",
        "Department of Psychosocial and Psychoanalytic Studies",
        "DEPARTMENT OF PSYCHOSOCIAL AND PSYCHOANALYTIC STUDIES",
    ],
    "life sciences": [],          # in document text only - see note above
    "biomedical science": ["Biomedical Science"],
    "essex business school": [
        "ESSEX BUSINESS SCHOOL", "Essex Business School",
        "Business School (ESSEX BUSINESS SCHOOL)",
    ],
    "ebs": ["ESSEX BUSINESS SCHOOL", "Essex Business School"],
    "government": ["Government", "government", "Department of Government"],
    "history": ["History", "Department of History"],
    "law": ["Law", "School of Law"],
    "modern languages": [
        "Modern Languages", "Modern Languages (Translation)",
        "Integrated Masters in Modern Languages (Translation)",
    ],
    "east 15": ["East 15 Acting School", "Acting School"],
    "edge hotel school": [
        "Edge Hotel School", "EDGE HOTEL SCHOOL", "EDGE Hotel School",
        "EDGE HOTEL SCHOOL (EHS)",
    ],
    "human resource management": ["Human Resource Management"],
    "early childhood care and education": ["Early Childhood Care and Education"],
}

# Longest aliases first so "health and social care" wins over a bare "hsc"
# substring inside another word, and so multi-word names are not shadowed.
_ALIASES_BY_LENGTH = sorted(DEPARTMENT_ALIASES, key=len, reverse=True)

import re

_BOUNDARY = {a: re.compile(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", re.I)
             for a in DEPARTMENT_ALIASES}


def detect_departments(text: str) -> list[str]:
    """Aliases explicitly named in `text`, de-duplicated by the department set
    they resolve to so "MSAS" and "Mathematical Sciences" in one question count
    once, not twice. Order follows first appearance in the text.

    Word-boundary matched, so "law" does not fire on "allow" and "hsc" does not
    fire inside a filename. Returns alias keys, not metadata values - the
    caller decides whether to filter or fall back."""
    low = text.lower()
    found: list[str] = []
    seen_targets: set[tuple[str, ...]] = set()
    claimed: list[tuple[int, int]] = []
    for alias in _ALIASES_BY_LENGTH:
        m = _BOUNDARY[alias].search(low)
        if not m:
            continue
        # skip an alias already covered by a longer one at the same position
        # ("health and social care" claims the span that bare "hsc" would want)
        if any(s <= m.start() < e for s, e in claimed):
            continue
        target = tuple(sorted(DEPARTMENT_ALIASES[alias]))
        if target and target in seen_targets:
            continue
        if target:
            seen_targets.add(target)
        claimed.append((m.start(), m.end()))
        found.append((m.start(), alias))
    return [a for _, a in sorted(found)]


def department_filter_values(aliases: list[str]) -> list[str]:
    """Metadata values for these aliases, for a Chroma `where` clause."""
    out: list[str] = []
    for a in aliases:
        out.extend(DEPARTMENT_ALIASES.get(a, []))
    return sorted(set(out))
