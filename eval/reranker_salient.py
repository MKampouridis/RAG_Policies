#!/usr/bin/env python3
"""Identity-salience test (#3): the sibling wall is that near-identical documents
read the same to a reranker - the ONLY discriminator is the programme/degree/year
identity, which sits buried in the chunk header. So make it PROMINENT: prepend a
humanized identity line (from the doc family) + repeat it, then re-rank. Tests
whether a DATA-side formatting change beats swapping reranker models. Reuses the
cached pools; re-ranks with the CURRENT ColBERT and bge-reranker-base.

Baselines (non-salient): ColBERT main 27/40, set2 26/40; bge-base main 30, set2 27.

Usage: RAG_DETERMINISTIC=1 PYTHONPATH=. python eval/reranker_salient.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
POOLS = json.loads(Path("eval/reranker_pools.json").read_text())


def humanize(f):
    return f.replace(".pdf", "").replace("-", " ").replace("_", " ").strip()


def make_salient(p):
    return [f"Programme/document: {humanize(f)}. {humanize(f)}.\n{pas}"
            for f, pas in zip(p["poolfams"], p["passages"])]


def report(name, orders_by_pool):
    stats = {}
    for p, order in zip(POOLS, orders_by_pool):
        s = stats.setdefault(p["set"], [0, 0, 0, 0])
        s[1] += 1
        new_hit = p["goldfam"] in {p["poolfams"][i] for i in order}
        if new_hit:
            s[0] += 1
        if p["goldfam"] in p["poolfams"] and p["goldfam"] not in p["cur_top6"]:
            s[3] += 1
            if new_hit:
                s[2] += 1
    for sn, (h, n, r, i) in stats.items():
        print(f"RESULT {name:34s} [{sn}] family-hit@6 {h}/{n} = {h / n * 100:.1f}% | rescued {r}/{i}", flush=True)


def colbert_salient():
    from pylate import rank
    from src import colbert_index
    m = colbert_index.get_model()
    orders = []
    for p in POOLS:
        pas = make_salient(p)
        q = m.encode([p["query"]], is_query=True)
        d = m.encode(pas, is_query=False)
        res = rank.rerank(documents_ids=[list(range(len(pas)))], queries_embeddings=q, documents_embeddings=[d])
        orders.append([r["id"] for r in res[0][:6]])
    return orders


def crossenc_salient(model):
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(model, max_length=512)
    orders = []
    for p in POOLS:
        pas = make_salient(p)
        sc = ce.predict([(p["query"], x) for x in pas])
        orders.append(sorted(range(len(sc)), key=lambda j: sc[j], reverse=True)[:6])
    return orders


if __name__ == "__main__":
    print("SALIENT-passage reranking (identity prepended + repeated)\n", flush=True)
    print("=== ColBERT (current) + salient ===", flush=True)
    report("ColBERT(current)+salient", colbert_salient())
    print("=== bge-reranker-base + salient ===", flush=True)
    report("bge-base+salient", crossenc_salient("BAAI/bge-reranker-base"))
