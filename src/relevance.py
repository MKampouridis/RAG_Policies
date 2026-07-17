"""LLM-based keep/reject classification for crawled documents. Every
candidate is judged against one criterion: is it a University policy
document, or a rules-of-assessment document?"""

import json
import re

from src.llm import chat

RELEVANCE_SYSTEM_PROMPT = """You are helping build a personal knowledge base of University of Essex \
policy documents and rules-of-assessment documents. Given a document's title, URL, and an excerpt \
of its text, decide whether it should be KEPT.

Keep a document if it is either:
- a University policy, code of practice, procedure, regulation, or governance document, OR
- a rules-of-assessment document (undergraduate or postgraduate, general or department/course specific).

Reject anything else: news articles, course listings, staff profiles, event pages, generic \
navigation/hub pages with no substantive policy content of their own, marketing content, etc.

Respond with ONLY a JSON object with these keys:
- "keep": true or false
- "doc_type": "policy", "rules_of_assessment", or "none"
- "department": a department/school code or name if the document is department-specific, else null
- "academic_year": the academic year the document applies to, formatted like "2025-26", if determinable, else null
- "reason": one short sentence explaining the decision
"""

YEAR_RE = re.compile(r"(20\d{2})[-_/](\d{2,4})")

REQUIRED_KEYS = {
    "keep": False,
    "doc_type": "none",
    "department": None,
    "academic_year": None,
    "reason": "",
}


def _guess_academic_year(*sources: str) -> str | None:
    for source in sources:
        if not source:
            continue
        m = YEAR_RE.search(source)
        if m:
            start, end = m.groups()
            return f"{start}-{end[-2:]}"
    return None


def classify(title: str, url: str, text: str) -> dict:
    excerpt = text[:3000].strip()
    if not excerpt:
        return {**REQUIRED_KEYS, "reason": "no extractable text"}

    user_prompt = f"Title: {title}\nURL: {url}\nExcerpt:\n{excerpt}"
    raw = chat(
        messages=[
            {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
    )

    try:
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("not a dict")
    except (json.JSONDecodeError, TypeError, ValueError):
        return {**REQUIRED_KEYS, "reason": "LLM returned unparseable output"}

    merged = {**REQUIRED_KEYS, **result}
    if not merged.get("academic_year"):
        merged["academic_year"] = _guess_academic_year(title, url)

    return merged
