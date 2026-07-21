#!/usr/bin/env python3
"""J5(a): evidence-sufficiency scoring - separates "did we retrieve the exact
gold document" (the strict hit@6 every eval reports) from "did we retrieve
ANY document whose full text contains the expected key facts". In a corpus
where hundreds of near-identical siblings state the same rule, the strict
metric marks a turn as a miss even when the retrieved sibling contains a
perfectly sufficient answer - the J0 diagnostic found all 12 strict misses
still earned judge scores of 3-4 for exactly this reason.

A turn is "evidence-sufficient" if at least one top-6 retrieved document's
cached full text contains at least half of that turn's expected keyphrases
(case-insensitive substring match, same convention as the answer-side
keyphrase_coverage metric in run_eval.py).

Usage: PYTHONPATH=. python eval/score_evidence_sufficiency.py [results_path] [questions_path]
"""

import json
import sys
from pathlib import Path

RESULTS_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/results_stage_colbert.json")
QUESTIONS_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("eval/questions.json")
MANIFEST_PATH = Path("data/manifest.json")

_text_cache: dict[str, str] = {}


def _doc_text(url: str, manifest: dict) -> str:
    if url not in _text_cache:
        doc = manifest.get(url) or {}
        path = Path(doc.get("text_cache_path", ""))
        _text_cache[url] = path.read_text(encoding="utf-8").lower() if path.exists() else ""
    return _text_cache[url]


def evidence_sufficient(top_urls: list[str], keyphrases: list[str], manifest: dict) -> bool | None:
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


def main():
    results = json.loads(RESULTS_PATH.read_text())
    # A plain {source_url: question} dict silently drops all but the last
    # question if a future set ever has 2+ questions on the same document
    # (external code review, 2026-07-21) - keep a queue per URL instead and
    # consume in order, so duplicates match up correctly rather than one
    # overwriting another. Safe for both old results files (which could have
    # gaps from the pre-Phase-1 run_eval.py silently dropping failed
    # questions - entries are dropped, never reordered, so per-URL order is
    # preserved) and new ones (guaranteed complete, same reasoning applies).
    questions_by_url: dict[str, list[dict]] = {}
    for q in json.loads(QUESTIONS_PATH.read_text()):
        questions_by_url.setdefault(q["source_url"], []).append(q)
    manifest = json.loads(MANIFEST_PATH.read_text())["documents"]

    stats = {"overall": [0, 0, 0], "rules_of_assessment": [0, 0, 0], "policy": [0, 0, 0]}
    # [n_scored, strict_hits, evidence_hits]
    upgraded = []  # strict miss but evidence-sufficient

    for item in results:
        queue = questions_by_url.get(item["source_url"])
        if not queue:
            continue
        q = queue.pop(0)
        for tk, kp_key in [("primary", "keyphrases"), ("follow_up", "follow_up_keyphrases")]:
            turn = item[tk]
            keyphrases = q.get(kp_key) or []
            verdict = evidence_sufficient(turn["retrieval"]["top_urls"], keyphrases, manifest)
            if verdict is None:
                continue
            strict = turn["retrieval"]["hit_at_6"]
            for group in ("overall", item["doc_type"]):
                stats[group][0] += 1
                stats[group][1] += strict
                stats[group][2] += verdict or strict  # strict hit implies evidence was retrievable
            if verdict and not strict:
                upgraded.append((tk, item["source_title"]))

    for group, (n, s, e) in stats.items():
        if n:
            print(f"{group:24s} n={n:3d}  strict hit@6={s / n * 100:5.1f}%  evidence-sufficient@6={e / n * 100:5.1f}%")
    print("\nStrict misses that were evidence-sufficient (retrieved a sibling that contains the key facts):")
    for tk, title in upgraded:
        print(f"  {tk:9s} {title}")


if __name__ == "__main__":
    main()
