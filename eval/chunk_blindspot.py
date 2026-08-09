#!/usr/bin/env python3
"""The hit@6 blind spot (2026-08-07): hit@6 is scored by comparing document
URLs (score_retrieval in run_eval.py), so it answers "was the right DOCUMENT
retrieved" - not "were the answer-bearing FACTS put in front of the
generator". A turn can therefore score a clean hit while the user gets a
wrong or empty answer, and the whole failure class is invisible to the eval.

Traced from real user feedback: three thumbs-down on "when is an independent
chair required" all scored hit@6=True. The policy lists SIX qualifying
circumstances in one chunk; the answers surfaced one or two. Manual replay
showed two different causes wearing the same costume - one turn never
retrieved the chunk holding the list (a chunk-level retrieval miss), while
others retrieved it and under-reported anyway (a generation failure).

This script separates those causes automatically. For every turn that scored
hit@6=True but was judged <= SCORE_THRESHOLD, it re-runs retrieval with the
turn's own stored retrieval_query and asks where the gold keyphrases are:

  CHUNK_MISS  - keyphrases are in the gold DOCUMENT but not in any RETRIEVED
                chunk. Right document, wrong chunk. No generator can fix it.
  GENERATOR   - keyphrases ARE in the retrieved chunks. The facts were in
                context and the answer still failed: synthesis, not retrieval.
  WEAK_TEST   - keyphrases appear nowhere in the gold document, so the test
                item itself can't be satisfied. Not a system failure; report
                separately rather than blaming either component.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/chunk_blindspot.py [results.json ...]
Writes eval/chunk_blindspot_result.json
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import _get_collection
from src.rag import retrieve

# RAG_BLINDSPOT_THRESHOLD lets a run include partial failures (score 3), not
# just outright ones. Default stays 2 so existing ledger numbers reproduce.
SCORE_THRESHOLD = int(os.environ.get("RAG_BLINDSPOT_THRESHOLD", "2"))
RESULTS = sys.argv[1:] or ["eval/results_gemma3_e2e_main.json", "eval/results_gemma3_e2e_set2.json"]
QUESTION_FILES = os.environ.get(
    "RAG_BLINDSPOT_QUESTIONS", "eval/questions.json,eval/questions_set2.json"
).split(",")

# Keyed on (source_url, QUESTION TEXT, turn), not (source_url, turn)
# (2026-08-09). Set 3 has four separate questions on
# code-practice-postgraduate-research.pdf; a url-only key keeps whichever was
# loaded last and would silently score every one of them against another
# question's keyphrases - producing confident CHUNK_MISS/GENERATOR verdicts
# from the wrong gold facts.
QUESTIONS = {}
for qf in QUESTION_FILES:
    qp = Path(qf.strip())
    if qp.is_file():
        for q in json.loads(qp.read_text()):
            QUESTIONS[(q["source_url"], q["question"], "primary")] = q.get("keyphrases") or []
            QUESTIONS[(q["source_url"], q["follow_up_question"], "follow_up")] = (
                q.get("follow_up_keyphrases") or []
            )


def gold_document_text(url: str) -> str:
    """All chunk text for the gold document, to tell 'not retrieved' apart
    from 'not present in the document at all'."""
    coll = _get_collection()
    res = coll.get(include=["documents", "metadatas"])
    return " ".join(
        d for d, m in zip(res["documents"], res["metadatas"]) if m.get("source_url") == url
    ).lower()


def covered(keyphrases: list[str], haystack: str) -> float:
    if not keyphrases:
        return None
    return sum(1 for k in keyphrases if k.lower() in haystack) / len(keyphrases)


rows = []
for rf in RESULTS:
    p = Path(rf)
    if not p.is_file():
        print(f"skip (missing): {rf}", flush=True)
        continue
    data = json.loads(p.read_text())
    print(f"\n=== {p.name} ===", flush=True)
    for r in data:
        url = r["source_url"]
        for turn in ("primary", "follow_up"):
            t = r.get(turn)
            if not t:
                continue
            score = t["judge"]["score"]
            if r.get("expects_abstention") or not url:
                continue  # no gold document: chunk-vs-document is undefined
            if not t["retrieval"]["hit_at_6"] or score is None or score > SCORE_THRESHOLD:
                continue

            keyphrases = QUESTIONS.get((url, t["question"], turn)) or []
            query = t["retrieval"].get("retrieval_query") or t["question"]
            res, _ = retrieve(query, [])
            retrieved_text = " ".join(res.get("documents", [[]])[0]).lower()

            in_retrieved = covered(keyphrases, retrieved_text)
            in_gold_doc = covered(keyphrases, gold_document_text(url))

            if in_gold_doc is None:
                verdict = "NO_KEYPHRASES"
            elif in_gold_doc == 0:
                verdict = "WEAK_TEST"
            elif in_retrieved is not None and in_retrieved < in_gold_doc:
                verdict = "CHUNK_MISS"
            else:
                verdict = "GENERATOR"

            rows.append({
                "results_file": p.name, "source_url": url, "turn": turn, "judge_score": score,
                "keyphrases": keyphrases, "keyphrase_in_retrieved": in_retrieved,
                "keyphrase_in_gold_doc": in_gold_doc, "verdict": verdict,
                "question": t["question"], "retrieval_query": query,
            })
            print(
                f"  {verdict:13s} score={score} kp_in_retrieved="
                f"{in_retrieved if in_retrieved is None else round(in_retrieved,2)} "
                f"kp_in_gold_doc={in_gold_doc if in_gold_doc is None else round(in_gold_doc,2)} "
                f"-- {url.split('/')[-1][:46]} [{turn}]",
                flush=True,
            )

Path("eval/chunk_blindspot_result.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

print(f"\n=== SUMMARY ({len(rows)} hit-but-failed turns) ===")
for v in ("CHUNK_MISS", "GENERATOR", "WEAK_TEST", "NO_KEYPHRASES"):
    n = sum(1 for r in rows if r["verdict"] == v)
    if n:
        print(f"  {v:13s} {n:3d}  ({n/len(rows)*100:.0f}% of the blind spot)")
print("\nCHUNK_MISS is invisible to hit@6 AND unfixable by the generator - it needs")
print("chunk-level retrieval work. GENERATOR turns had the facts in context already.")
