#!/usr/bin/env python3
"""Reranker sweep: the recall diagnostic showed 69% of RoA misses are RANKING
failures (gold doc in the pool, current ColBERT ranks it below wrong siblings).
So re-rank the SAME captured candidate pools with different rerankers and measure
family-hit@6 + how many of the in-pool ranking-failures each rescues. Cheap: no
re-embedding, pools captured once.

Baseline = current production ColBERT: 27/40 family-hit; 9 in-pool ranking
failures (ceiling if all rescued = 36/40 = 90%).

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/reranker_sweep.py <hf_reranker> [<hf_reranker> ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.rag as rag
from src.docid import document_family as fam
from src.rag import retrieve
from src.rerank import _passages

POOLS = Path("eval/reranker_pools.json")


SETS = {"main": "eval/results_qwen14b_full.json", "set2": "eval/results_set2_14b.json"}


def build_pools():
    if POOLS.exists():
        return json.loads(POOLS.read_text())
    pools = []
    for setname, path in SETS.items():
        ref = json.loads(Path(path).read_text())
        for r in ref:
            if r["doc_type"] != "rules_of_assessment":
                continue
            goldfam = fam(r["source_url"])
            history = []
            for turn in ("primary", "follow_up"):
                t = r[turn]
                res, rq = retrieve(t["question"], list(history))
                pool = rag._LAST_CANDIDATE_POOL or {"documents": [[]], "metadatas": [[]]}
                docs = pool.get("documents", [[]])[0]
                metas = pool.get("metadatas", [[]])[0]
                pools.append({
                    "set": setname,
                    "label": f"{r['source_title']}[{turn}]",
                    "query": rq,
                    "goldfam": goldfam,
                    "passages": _passages(docs, metas),
                    "poolfams": [fam(m.get("source_url", "")) for m in metas],
                    "cur_top6": [fam(m.get("source_url", "")) for m in res.get("metadatas", [[]])[0]],
                })
                history += [{"role": "user", "content": t["question"]},
                            {"role": "assistant", "content": t["actual_answer"]}]
            print(f"  [{setname}] pool captured: {r['source_title']}", flush=True)
    POOLS.write_text(json.dumps(pools, ensure_ascii=False))
    return pools


def eval_reranker(model_name, pools, top_n=6):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model_name, trust_remote_code=True, max_length=512)
    stats = {}  # set -> [hit, n, rescued, in_pool_miss]
    for p in pools:
        s = stats.setdefault(p["set"], [0, 0, 0, 0])
        s[1] += 1
        gold_in_pool = p["goldfam"] in p["poolfams"]
        cur_hit = p["goldfam"] in p["cur_top6"]
        scores = ce.predict([(p["query"], pas) for pas in p["passages"]])
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        new_hit = p["goldfam"] in {p["poolfams"][i] for i in order}
        if new_hit:
            s[0] += 1
        if gold_in_pool and not cur_hit:
            s[3] += 1
            if new_hit:
                s[2] += 1
    for setname, (hit, n, rescued, ipm) in stats.items():
        print(f"RESULT {model_name:46s} [{setname}] family-hit@6 {hit}/{n} = {hit / n * 100:.1f}% "
              f"| ranking-failures rescued {rescued}/{ipm}", flush=True)


if __name__ == "__main__":
    pools = build_pools()
    print(f"pools ready: {len(pools)} RoA turns (baseline ColBERT 27/40 = 67.5%)\n", flush=True)
    for m in sys.argv[1:]:
        print(f"=== {m} ===", flush=True)
        eval_reranker(m, pools)
