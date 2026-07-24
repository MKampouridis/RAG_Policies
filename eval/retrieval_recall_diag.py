#!/usr/bin/env python3
"""Retrieval recall diagnostic: for each RoA turn, split misses into RANKING
failures (gold family IS in the pre-rerank candidate pool but not in the top-6 ->
a stronger reranker could fix it) vs RECALL failures (gold family not even in the
pool -> only a better embedder/first-stage can fix it). This decides whether the
retrieval bake-off needs the expensive embedder sweep or just the reranker sweep.

Uses family-match (confound-robust). Reads gold + questions from the reference
run; re-runs retrieve() which stashes the pre-rerank pool in rag._LAST_CANDIDATE_POOL.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/retrieval_recall_diag.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.rag as rag
from src.docid import document_family as fam
from src.rag import retrieve

REF = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "eval/results_qwen14b_full.json").read_text())

hit, ranking_fail, recall_fail = [], [], []
for r in REF:
    if r["doc_type"] != "rules_of_assessment":
        continue
    goldfam = fam(r["source_url"])
    history = []
    for turn in ("primary", "follow_up"):
        t = r[turn]
        res, _ = retrieve(t["question"], list(history))
        top6 = {fam(m.get("source_url", "")) for m in res.get("metadatas", [[]])[0]}
        pool = rag._LAST_CANDIDATE_POOL or {"metadatas": [[]]}
        poolfams = {fam(m.get("source_url", "")) for m in pool.get("metadatas", [[]])[0]}
        label = f"{r['source_title']}[{turn}]"
        if goldfam in top6:
            hit.append(label)
        elif goldfam in poolfams:
            ranking_fail.append(label)   # in pool, reranker didn't surface it
        else:
            recall_fail.append(label)    # not even in pool
        history += [{"role": "user", "content": t["question"]},
                    {"role": "assistant", "content": t["actual_answer"]}]

n = len(hit) + len(ranking_fail) + len(recall_fail)
print(f"\n=== RoA recall diagnostic ({n} turns) ===")
print(f"HIT (gold family in top-6):                              {len(hit)}")
print(f"MISS - RANKING failure (in pool, not top-6 -> RERANKER): {len(ranking_fail)}")
for x in ranking_fail:
    print("     ", x)
print(f"MISS - RECALL failure (not in pool -> EMBEDDER):         {len(recall_fail)}")
for x in recall_fail:
    print("     ", x)
misses = len(ranking_fail) + len(recall_fail)
if misses:
    print(f"\n=> of {misses} RoA misses: {len(ranking_fail)} ({len(ranking_fail)/misses*100:.0f}%) are RANKING "
          f"(reranker sweep can help), {len(recall_fail)} ({len(recall_fail)/misses*100:.0f}%) are RECALL "
          f"(need embedder sweep).")
