#!/usr/bin/env python3
"""Read the user-feedback log (data/feedback.jsonl) and produce a ROUTED action
plan: group thumbs-down by the failure-taxonomy tags so each bucket points at a
specific lever (generator / retrieval / D3 / data-hygiene), and list the actual
questions + comments to act on.

Optionally (--replay) re-runs retrieval on each thumbs-down and prints the top
document families retrieved, so you can eyeball whether the issue was retrieval
(scattered / wrong doc) or generation (right doc, wrong figure) even when the
user didn't tag it.

Usage:
    python feedback_report.py            # tag-based routed summary
    python feedback_report.py --replay   # + replay retrieval per thumbs-down
"""
import sys
from collections import Counter

from src.feedback import TAGS, load_feedback

TAG_LEVER = {
    "wrong_programme": "retrieval / D3 clarification (try CLARIFY_UNDERSPECIFIED_ENABLED)",
    "wrong_figure": "generator (already on local 14B; consider cloud / bigger local)",
    "outdated": "data-hygiene (add stale URL to run_ingest.py _EXCLUDED_URLS, re-crawl)",
    "no_answer": "abstention / under-specified -> D3, or retrieval",
    "other": "read the comment",
}


def main() -> None:
    replay = "--replay" in sys.argv
    fb = load_feedback()
    if not fb:
        print("No feedback yet (data/feedback.jsonl is empty or absent).")
        return

    ups = [f for f in fb if f.get("rating") == "up"]
    downs = [f for f in fb if f.get("rating") == "down"]
    print(f"=== Feedback summary ({len(fb)} ratings) ===")
    sat = len(ups) / len(fb) * 100 if fb else 0
    print(f"  up {len(ups)}   down {len(downs)}   satisfaction {sat:.0f}%")

    if not downs:
        print("\nNo thumbs-down yet - nothing to route.")
        return

    tagc = Counter()
    for f in downs:
        tags = f.get("tags") or ["(untagged)"]
        for t in tags:
            tagc[t] += 1
    print("\n=== Thumbs-down by failure category -> lever ===")
    for tag, n in tagc.most_common():
        label = TAGS.get(tag, tag)
        lever = TAG_LEVER.get(tag, "")
        print(f"  {n:3d}  {label:34s} {('-> ' + lever) if lever else ''}")

    print("\n=== Thumbs-down detail (act on these) ===")
    retrieve = document_family = None
    if replay:
        from src.docid import document_family  # noqa
        from src.rag import retrieve  # noqa
    for i, f in enumerate(downs, 1):
        tags = ", ".join(f.get("tags") or []) or "untagged"
        print(f"\n{i}. [{tags}]  {f.get('question', '')}")
        if f.get("comment"):
            print(f"     comment: {f['comment']}")
        if f.get("sources"):
            print(f"     sources shown: {', '.join(s.split('/')[-1] for s in f['sources'][:3])}")
        if replay:
            q = f.get("retrieval_query") or f.get("question") or ""
            if q:
                res, _ = retrieve(q, [])
                metas = res.get("metadatas", [[]])[0]
                fams = [document_family(m.get("source_url", "")).replace(".pdf", "") for m in metas[:6]]
                distinct = len(set(fams))
                flag = "  (scattered pool - likely retrieval/under-specified)" if distinct >= 5 else ""
                print(f"     replay top-6 families: {fams}{flag}")


if __name__ == "__main__":
    main()
