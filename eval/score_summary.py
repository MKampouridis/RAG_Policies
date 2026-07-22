#!/usr/bin/env python3
"""Aggregate a results_*.json file into summary statistics.

Reports three views of retrieval quality:
- strict: the retrieved URL must equal the question's exact source document
- lenient: a retrieved URL counts if it is the same document family (filename
  stem with year suffix stripped, src.rag._document_family) AND the same
  academic year as the expected document. This is fairer for boilerplate-heavy
  corpora where several near-identical siblings could serve the user equally,
  without crediting wrong-year or wrong-family retrievals.
- evidence_sufficient@6: at least one top-6 document's full text contains at
  least half the turn's expected keyphrases, regardless of which exact
  document it is (J5a, eval/report.md - promoted from a one-off diagnostic
  script to a standing headline column per external code review, 2026-07-21:
  it's the number that tracks what a user actually experiences, and reframes
  strict hit@6 deltas that are really about test-set construction rather
  than retrieval quality).

Needs eval/questions.json (for keyphrases) - pass a different path as the
optional second CLI arg if scoring against a different question set than the
default.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.docid import document_family, effective_year

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "manifest.json"
DEFAULT_QUESTIONS_PATH = ROOT / "eval" / "questions.json"

K_MAX = 6

_year_by_url: dict[str, str] = {}
_dated_families: set[str] = set()
_doc_text_cache: dict[str, str] = {}


def _doc_text(url: str, manifest: dict) -> str:
    if url not in _doc_text_cache:
        doc = manifest.get(url) or {}
        path = Path(doc.get("text_cache_path", ""))
        _doc_text_cache[url] = path.read_text(encoding="utf-8").lower() if path.exists() else ""
    return _doc_text_cache[url]


def _evidence_sufficient(top_urls: list[str], keyphrases: list[str], manifest: dict) -> bool | None:
    """Same convention as run_eval.py's answer-side keyphrase_coverage:
    case-insensitive substring match, "sufficient" at >= half the
    keyphrases. Returns None (not scoreable) when the question has no
    keyphrases annotated."""
    if not keyphrases:
        return None
    for url in dict.fromkeys(top_urls):  # dedup, keep order
        text = _doc_text(url, manifest)
        if not text:
            continue
        found = sum(1 for kp in keyphrases if kp.lower() in text)
        if found >= (len(keyphrases) + 1) // 2:
            return True
    return False


def _load_questions_by_url(questions_path: Path) -> dict[str, list[dict]]:
    # per-URL queue, not a flat {source_url: question} dict - a future
    # question set with 2+ questions on one document would otherwise
    # silently drop all but the last (external code review, 2026-07-21)
    by_url: dict[str, list[dict]] = {}
    for q in json.loads(questions_path.read_text()):
        by_url.setdefault(q["source_url"], []).append(q)
    return by_url


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


def summarize(results: list[dict], questions_path: Path = DEFAULT_QUESTIONS_PATH) -> dict:
    if not _year_by_url:
        _load_years()

    manifest = json.loads(MANIFEST_PATH.read_text())["documents"]
    questions_by_url = _load_questions_by_url(questions_path)

    # annotate each turn in-place with its evidence-sufficiency verdict,
    # matched via the same per-URL queue as score_evidence_sufficiency.py
    for r in results:
        queue = questions_by_url.get(r["source_url"])
        q = queue.pop(0) if queue else None
        for tk, kp_key in (("primary", "keyphrases"), ("follow_up", "follow_up_keyphrases")):
            turn = r[tk]
            keyphrases = (q.get(kp_key) if q else None) or []
            turn["_evidence_sufficient"] = _evidence_sufficient(turn["retrieval"]["top_urls"], keyphrases, manifest)

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
        # strict hit implies the evidence was retrievable, even on the rare
        # chance the keyphrase substring check itself misses on the exact
        # gold document's own text
        ev = [
            (t["_evidence_sufficient"] or t["retrieval"]["hit_at_6"])
            for _, t in turn_list if t["_evidence_sufficient"] is not None
        ]
        return {
            "n": n,
            # HEADLINE retrieval metric (round-4 metric rework, eval/report.md):
            # evidence-sufficient@6 - did the top-6 contain ANY document with
            # the answer - is what users experience and what the gold-
            # multiplicity analysis (eval/gold_multiplicity.py) shows is the
            # honest target. Strict hit@6 is now at its achievable single-gold
            # ceiling (RoA 70% actual vs 68.6% achievable) and kept only as an
            # attribution diagnostic, not the headline.
            "evidence_sufficient_at_6": (sum(ev) / len(ev)) if ev else None,
            "answer_score_mean": statistics.mean(scores) if scores else None,
            "answer_score_stdev": statistics.stdev(scores) if len(scores) > 1 else None,
            "keyphrase_coverage_mean": statistics.mean(kp) if kp else None,
            # diagnostics (single-gold; see note above):
            "strict": {**hit_curve(strict_ranks), "mrr": mrr(strict_ranks)},
            "lenient": {**hit_curve(lenient_ranks), "mrr": mrr(lenient_ranks)},
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
    q_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_QUESTIONS_PATH
    results = json.loads(path.read_text())
    summary = summarize(results, q_path)
    print(json.dumps(summary, indent=2))
