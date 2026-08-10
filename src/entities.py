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
    # School of Life Sciences has no department value of its own, but
    # Biomedical Science is one of its programme areas and IS carried as a
    # department value, so filtering on it reaches part of the school rather
    # than nothing. The rest of the school's rules live as a "BS - Life
    # Sciences" SECTION inside multi-department UG variations documents, which
    # carry no department at all - those are reached by the query-side
    # fallback, not by filtering.
    "life sciences": ["Biomedical Science"],
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
    # Faculty roster members added 2026-08-10. Empty list = named on the
    # University's faculty page but carrying NO department metadata in the
    # corpus, so detection fires (the entity is real and the user may name it)
    # while retrieval falls back to a query-side hint rather than filtering on
    # a value that would match nothing.
    "economics": ["BE ESSEX BUSINESS SCHOOL and EC ECONOMICS", "Business and Economics"],
    "sociology": [],
    "criminology": [],
    "philosophical, historical and interdisciplinary studies": [],
    "language, literature and media": ["Modern Languages", "Modern Languages (Translation)"],
    "essex law school": ["Law", "School of Law"],
}

# Faculty -> member departments, from the University's own faculty pages
# (essex.ac.uk/about/university/faculties/..., retrieved 2026-08-10).
#
# WHY: a question naming a FACULTY rather than its departments - "what are the
# accredited programmes offered by Schools/Departments in the Faculty of
# Science and Health" - was a real thumbs-down, and multi-entity retrieval
# could not help because it triggers on named DEPARTMENTS and that question
# names none. Expanding the faculty to its roster makes such a question
# behave exactly like the explicit six-department version the user also asked.
#
# The rosters are recorded verbatim from the source pages even where the
# corpus has no matching documents (Sociology, Criminology, ISER, UK Data
# Archive): a roster that silently omits members would be wrong as a fact, and
# the empty-alias convention above already handles "named but not filterable".
FACULTY_DEPARTMENTS: dict[str, list[str]] = {
    "science and health": [
        "life sciences", "csee", "hsc", "msas", "psychology", "sres",
    ],
    "arts, humanities and social sciences": [
        "east 15", "economics", "essex business school", "edge hotel school",
        "essex law school", "government", "language, literature and media",
        "philosophical, historical and interdisciplinary studies",
        "psychosocial and psychoanalytic studies", "sociology", "criminology",
    ],
}

_FACULTY_PATTERNS = {
    "science and health": ("faculty of science and health", "science and health faculty",
                           "science & health"),
    "arts, humanities and social sciences": (
        "faculty of arts, humanities and social sciences", "arts, humanities and social sciences",
        "arts and humanities faculty", "ahss",
    ),
}


def detect_faculties(text: str) -> list[str]:
    low = text.lower()
    return [f for f, pats in _FACULTY_PATTERNS.items() if any(p in low for p in pats)]

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
    # A named FACULTY stands in for its member departments, so "departments in
    # the Faculty of Science and Health" behaves like naming all six.
    faculty_expanded: list[str] = []
    for fac in detect_faculties(text):
        faculty_expanded.extend(FACULTY_DEPARTMENTS.get(fac, []))
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
    named = [a for _, a in sorted(found)]
    # faculty members appended after explicitly-named departments, de-duplicated,
    # so an explicit mention keeps its position and priority
    for a in faculty_expanded:
        if a not in named:
            named.append(a)
    return named


def department_filter_values(aliases: list[str]) -> list[str]:
    """Metadata values for these aliases, for a Chroma `where` clause."""
    out: list[str] = []
    for a in aliases:
        out.extend(DEPARTMENT_ALIASES.get(a, []))
    return sorted(set(out))
