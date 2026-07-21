#!/usr/bin/env python3
"""Aggregate a results_*.json file into summary statistics.

Reports two views of retrieval quality:
- strict: the retrieved URL must equal the question's exact source document
- lenient: a retrieved URL counts if it is the same document family (filename
  stem with year suffix stripped, src.rag._document_family) AND the same
  academic year as the expected document. This is fairer for boilerplate-heavy
  corpora where several near-identical siblings could serve the user equally,
  without crediting wrong-year or wrong-family retrievals.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.docid import document_family, effective_year

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.json"

K_MAX = 6

_year_by_url: dict[str, str] = {}
_dated_families: set[str] = set()


def _load_years() -> None:
    """Uses effective_year(), not raw normalize_year() - external code review
    (2026-07-21) found this module had drifted from production's own
    currency logic (reembed.py/src/ingest.py switched to effective_year() to
    fix a real is_current bug earlier the same day; this file was missed).
    Matters concretely for the two PGT "January starts" families the bug
    affected - without this, a lenient-match check could still credit a
    retrieved document as "the current year" using the same mis-extracted
    year that caused the original bug."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    for doc in manifest["documents"].values():
        year = effective_year(doc["url"], doc.get("academic_year"))
        _year_by_url[doc["url"]] = year
        if year:
            _dated_families.add(document_family(doc["url"]))


def _lenient_rank(top_urls: list[str], expected_url: str) -> int | None:
    """Lenient hit: same document family AND same canonical academic year.
    Years are normalized so format variants ('2025-2026' vs '2025-26') of the
    same year match. Two unknown years count as a match only in families with
    no dated members at all (evergreen policies) - in a dated family an
    unknown-year sibling could be any edition and must not be credited."""
    expected_family = document_family(expected_url)
    expected_year = _year_by_url.get(expected_url, "")
    for i, url in enumerate(top_urls[:K_MAX], 1):
        if document_family(url) != expected_family:
            continue
        year = _year_by_url.get(url, "")
        if year == expected_year and (year or expected_family not in _dated_families):
            return i
    return None


def summarize(results: list[dict]) -> dict:
    if not _year_by_url:
        _load_years()

    def turns(doc_type_filter=None):
        out = []
        for r in results:
            if doc_type_filter and r["doc_type"] != doc_type_filter:
                continue
            out.append((r, r["primary"]))
            out.append((r, r["follow_up"]))
        return out

    def stats_for(turn_list):
        n = len(turn_list)
        if not n:
            return {"n": 0}

        strict_ranks = [t["retrieval"].get("rank") for _, t in turn_list]
        lenient_ranks = [
            _lenient_rank(t["retrieval"]["top_urls"], r["source_url"]) for r, t in turn_list
        ]

        def hit_curve(ranks):
            return {f"hit@{k}": sum(1 for rk in ranks if rk is not None and rk <= k) / n
                    for k in range(1, K_MAX + 1)}

        def mrr(ranks):
            return statistics.mean((1.0 / rk) if rk else 0.0 for rk in ranks)

        scores = [t["judge"]["score"] for _, t in turn_list if t["judge"]["score"] is not None]
        kp = [t["keyphrase_coverage"] for _, t in turn_list if t["keyphrase_coverage"] is not None]
        return {
            "n": n,
            "strict": {**hit_curve(strict_ranks), "mrr": mrr(strict_ranks)},
            "lenient": {**hit_curve(lenient_ranks), "mrr": mrr(lenient_ranks)},
            "answer_score_mean": statistics.mean(scores) if scores else None,
            "answer_score_stdev": statistics.stdev(scores) if len(scores) > 1 else None,
            "keyphrase_coverage_mean": statistics.mean(kp) if kp else None,
        }

    return {
        "overall": stats_for(turns()),
        "primary_only": stats_for([(r, r["primary"]) for r in results]),
        "follow_up_only": stats_for([(r, r["follow_up"]) for r in results]),
        "policy": stats_for(turns("policy")),
        "rules_of_assessment": stats_for(turns("rules_of_assessment")),
    }


if __name__ == "__main__":
    path = Path(sys.argv[1])
    results = json.loads(path.read_text())
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
