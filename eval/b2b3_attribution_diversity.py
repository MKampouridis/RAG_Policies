"""Phase B2 + B3 (review round 3): two niche-lever ceilings on the 12 current
misses, offline, using each miss's logged deterministic retrieval query.

B2 attribution ceiling (Fable 5): fraction of misses whose reranked top-6 contains
a chunk whose NORMALISED BODY is byte-identical to a chunk in the GOLD document.
Those are shared-boilerplate chunks a citation-layer tie-break could re-attribute
to the gold doc - so this is the exact rescue ceiling of that scheme.

B3 diversity-cap ceiling: re-rank each miss's candidate pool with a per-document
cap (max 2 chunks/doc) and check whether the gold document enters top-6 - the
rescue ceiling of freeing duplicate-filled slots (mt8's top-6 held only 3
distinct docs).
"""
import hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, "/Users/mkampo/RAG_Policies")
from src import lexical, rerank as _rerank
from src.ingest import query as vector_query, _get_collection
from src.rag import (_dense_as_hits, _rrf_fuse, _dedup_by_chunk, _prefer_most_recent_year,
                     N_RESULTS, FETCH_POOL_MULTIPLIER, _document_family)

norm = lambda t: hashlib.sha256(" ".join((t or "").split()).lower().encode()).hexdigest()

# gold-doc body-hash sets
coll = _get_collection()
alldata = coll.get(include=["documents", "metadatas"])
gold_hashes = {}
for d, m in zip(alldata["documents"], alldata["metadatas"]):
    gold_hashes.setdefault(m.get("source_url", ""), set()).add(norm(d))

def reranked(query, top_n=30):
    pool = N_RESULTS * FETCH_POOL_MULTIPLIER
    dense = vector_query(query, n_results=pool, where={"is_current": True})
    bm25 = lexical.query(query, n_results=pool, current_only=True)
    cands = _prefer_most_recent_year(_dedup_by_chunk(_rrf_fuse(_dense_as_hits(dense), bm25)))
    res = _rerank.rerank(query, cands, top_n)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip([m.get("source_url", "") for m in metas], docs))

RESULTS = json.loads(Path("eval/results_c1_anchor.json").read_text())
misses = []
for r in RESULTS:
    for turn in ("primary", "follow_up"):
        if not r[turn]["retrieval"]["hit_at_6"]:
            misses.append((f"{r['source_title']}[{turn}]", r["source_url"], r[turn]["retrieval"]["retrieval_query"]))

b2_rescue, b3_rescue = [], []
for lbl, gold, q in misses:
    ranked = reranked(q, 30)
    top6 = ranked[:6]
    # B2: any top-6 chunk body identical to a gold-doc chunk?
    gh = gold_hashes.get(gold, set())
    if any(norm(text) in gh for _, text in top6):
        b2_rescue.append(lbl)
    # B3: per-doc cap (max 2), does gold enter top-6?
    seen, capped = {}, []
    for url, text in ranked:
        seen[url] = seen.get(url, 0) + 1
        if seen[url] <= 2:
            capped.append(url)
        if len(capped) >= 6:
            break
    if gold in capped[:6]:
        b3_rescue.append(lbl)

print(f"misses analysed: {len(misses)}")
print(f"\nB2 attribution ceiling: {len(b2_rescue)}/{len(misses)} misses have a top-6 chunk byte-identical to a gold-doc chunk")
for x in b2_rescue: print("   +", x)
print(f"\nB3 diversity-cap ceiling: {len(b3_rescue)}/{len(misses)} misses would surface gold into top-6 under a max-2-per-doc cap")
for x in b3_rescue: print("   +", x)
